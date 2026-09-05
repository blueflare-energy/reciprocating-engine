//! Single-head scaled-dot-product attention as ONE fused SynapseAI recipe
//! (`gemm -> softmax -> gemm`), built directly through the C API.
//!
//! The PyTorch frameworks miscompute this exact composition on 1.24; our direct
//! fused graph computes it correctly. Two stack constraints shape it: (1) build
//! the whole thing as one graph with one launch (separate launches per op race
//! to zeros), and (2) the launched kernel must run long enough to clear a
//! readback coherency race - a fast kernel's HBM writeback is not yet coherent
//! when the readback DMA fires, and no stream/device sync we tried prevents it.
//! Reliability scales with kernel size: seq=dim=64 always races, 128 is
//! marginal (fails intermittently), 256 is solid (8/8), and real-model shapes
//! (seq in the thousands, hidden >= 2048) are far into the safe regime. We
//! reject below 128 outright; realistic sizes are reliable.

use crate::ffi::*;
use crate::{bf16_to_f32, f32_to_bf16};
use core::ffi::c_void;
use reng_core::{Error, Result};
use std::ffi::CString;

macro_rules! syn {
    ($call:expr) => {{
        let st = unsafe { $call };
        if st != SYN_SUCCESS {
            return Err(Error::Other(format!(
                concat!(stringify!($call), " -> synStatus {}"),
                st
            )));
        }
    }};
}

/// Create a 2D bf16 tensor with FCD-first sizes `[fcd, outer]`.
fn t2d(
    graph: synGraphHandle,
    name: &CString,
    fcd: u64,
    outer: u64,
    persistent: bool,
) -> Result<synTensor> {
    let mut t: synTensor = core::ptr::null_mut();
    syn!(synTensorHandleCreate(
        &mut t,
        graph,
        SYN_TENSOR_DATA,
        name.as_ptr()
    ));
    if persistent {
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        syn!(synTensorAssignToSection(t, sec, 0));
    }
    let mut geo = synTensorGeometry {
        sizes: [0; HABANA_DIM_MAX],
        dims: 2,
    };
    geo.sizes[0] = fcd;
    geo.sizes[1] = outer;
    syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
    syn!(synTensorSetDeviceDataType(t, SYN_TYPE_BF16));
    Ok(t)
}

/// Single-head attention `softmax((Q*scale) @ K^T) @ V` in bf16 on the HPU, as
/// one fused recipe. `q`, `k`, `v` are row-major `[seq, dim]`; returns the
/// `[seq, dim]` output as `f32`.
///
/// `seq` and `dim` must be at least 128 (see the module docs); smaller shapes
/// hit a stack readback race and are rejected.
///
/// # Errors
///
/// Returns an error if `seq` or `dim` is below 128, or if any SynapseAI call
/// fails.
///
/// # Panics
///
/// Panics if any input length is not `seq*dim`.
#[allow(clippy::too_many_lines)]
pub fn attention_bf16(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    assert_eq!(q.len(), seq * dim);
    assert_eq!(k.len(), seq * dim);
    assert_eq!(v.len(), seq * dim);
    if seq < 128 || dim < 128 {
        return Err(Error::Other(format!(
            "attention_bf16 requires seq>=128 and dim>=128 (got seq={seq}, dim={dim}); \
             smaller shapes hit the stack readback race"
        )));
    }
    let (s, d) = (seq as u64, dim as u64);

    // Host prep: pre-scale Q, and transpose K into K^T [dim, seq].
    let q_scaled: Vec<f32> = q.iter().map(|x| x * scale).collect();
    let mut kt = vec![0.0f32; dim * seq];
    for i in 0..seq {
        for c in 0..dim {
            kt[c * seq + i] = k[i * dim + c];
        }
    }

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    let (n_q, n_kt, n_v, n_out) = (
        CString::new("Q").unwrap(),
        CString::new("KT").unwrap(),
        CString::new("V").unwrap(),
        CString::new("OUT").unwrap(),
    );
    // FCD-first sizes: Q[dim,seq] KT[seq,dim] V[dim,seq] scores/probs[seq,seq] OUT[dim,seq].
    let t_q = t2d(graph, &n_q, d, s, true)?;
    let t_kt = t2d(graph, &n_kt, s, d, true)?;
    let t_v = t2d(graph, &n_v, d, s, true)?;
    let t_out = t2d(graph, &n_out, d, s, true)?;
    let t_scores = t2d(graph, &CString::new("scores").unwrap(), s, s, false)?;
    let t_probs = t2d(graph, &CString::new("probs").unwrap(), s, s, false)?;

    let gemm_p = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let sm_p = synSoftmaxParams { dim: 0 };
    let guid_gemm = CString::new("gemm").unwrap();
    let guid_sm = CString::new("softmax_fwd_bf16").unwrap();

    // scores = Q @ KT -> [seq, seq] (fcd=key, outer=query)
    let in0 = [t_q, t_kt];
    let out0 = [t_scores];
    syn!(synNodeCreate(
        graph,
        in0.as_ptr(),
        out0.as_ptr(),
        2,
        1,
        (&raw const gemm_p).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        guid_gemm.as_ptr(),
        CString::new("qk").unwrap().as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    // probs = softmax(scores) over FCD (the key axis)
    let in1 = [t_scores];
    let out1 = [t_probs];
    syn!(synNodeCreate(
        graph,
        in1.as_ptr(),
        out1.as_ptr(),
        1,
        1,
        (&raw const sm_p).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
        guid_sm.as_ptr(),
        CString::new("sm").unwrap().as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    // out = probs @ V -> [seq, dim] (fcd=embedding, outer=query)
    let in2 = [t_probs, t_v];
    let out2 = [t_out];
    syn!(synNodeCreate(
        graph,
        in2.as_ptr(),
        out2.as_ptr(),
        2,
        1,
        (&raw const gemm_p).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        guid_gemm.as_ptr(),
        CString::new("av").unwrap().as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));

    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        CString::new("attn").unwrap().as_ptr(),
        core::ptr::null()
    ));

    let names = [n_q.as_ptr(), n_kt.as_ptr(), n_v.as_ptr(), n_out.as_ptr()];
    let mut ids: [u64; 4] = [0; 4];
    syn!(synTensorRetrieveIds(
        recipe,
        names.as_ptr(),
        ids.as_mut_ptr(),
        4
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));

    let sd_bytes = (seq * dim * 2) as u64;
    let (mut dq, mut dkt, mut dv, mut dout) = (0u64, 0u64, 0u64, 0u64);
    syn!(synDeviceMalloc(dev, sd_bytes, 0, 0, &mut dq));
    syn!(synDeviceMalloc(dev, sd_bytes, 0, 0, &mut dkt));
    syn!(synDeviceMalloc(dev, sd_bytes, 0, 0, &mut dv));
    syn!(synDeviceMalloc(dev, sd_bytes, 0, 0, &mut dout));
    let mut ws = 0u64;
    syn!(synWorkspaceGetSize(&mut ws, recipe));
    let mut dws = 0u64;
    if ws > 0 {
        syn!(synDeviceMalloc(dev, ws, 0, 0, &mut dws));
    }

    let (mut hq, mut hkt, mut hv, mut hout): (*mut c_void, *mut c_void, *mut c_void, *mut c_void) = (
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    syn!(synHostMalloc(dev, sd_bytes, 0, &mut hq));
    syn!(synHostMalloc(dev, sd_bytes, 0, &mut hkt));
    syn!(synHostMalloc(dev, sd_bytes, 0, &mut hv));
    syn!(synHostMalloc(dev, sd_bytes, 0, &mut hout));

    // SAFETY: each host buffer holds seq*dim bf16 elements.
    unsafe {
        let fill = |dst: *mut c_void, src: &[f32]| {
            let p = dst.cast::<u16>();
            for (i, &x) in src.iter().enumerate() {
                *p.add(i) = f32_to_bf16(x);
            }
        };
        fill(hq, &q_scaled);
        fill(hkt, &kt);
        fill(hv, v);
    }

    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    for (h, dptr) in [(hq, dq), (hkt, dkt), (hv, dv)] {
        syn!(synMemCopyAsync(
            stream,
            h as u64,
            sd_bytes,
            dptr,
            SYN_HOST_TO_DRAM
        ));
    }
    syn!(synStreamSynchronize(stream));

    let mk = |name: &CString, addr: u64, id: u64, fcd: u64, outer: u64| {
        let mut ti = synLaunchTensorInfo {
            tensor_name: name.as_ptr(),
            tensor_address: addr,
            tensor_type: SYN_TENSOR_DATA,
            tensor_size: [0; HABANA_DIM_MAX],
            tensor_id: id,
        };
        ti.tensor_size[0] = fcd;
        ti.tensor_size[1] = outer;
        ti
    };
    let infos = [
        mk(&n_q, dq, ids[0], d, s),
        mk(&n_kt, dkt, ids[1], s, d),
        mk(&n_v, dv, ids[2], d, s),
        mk(&n_out, dout, ids[3], d, s),
    ];
    syn!(synLaunch(stream, infos.as_ptr(), 4, dws, recipe, 0));
    syn!(synStreamSynchronize(stream));
    syn!(synMemCopyAsync(
        stream,
        dout,
        sd_bytes,
        hout as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; seq * dim];
    // SAFETY: hout holds seq*dim bf16 elements just copied back.
    unsafe {
        let p = hout.cast::<u16>();
        for (i, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*p.add(i));
        }
    }

    unsafe {
        for h in [hq, hkt, hv, hout] {
            synHostFree(dev, h, 0);
        }
        for dptr in [dq, dkt, dv, dout] {
            synDeviceFree(dev, dptr, 0);
        }
        if dws != 0 {
            synDeviceFree(dev, dws, 0);
        }
        synStreamDestroy(stream);
        synGraphDestroy(graph);
        synDeviceRelease(dev);
        synDestroy();
    }
    Ok(out)
}

/// CPU reference for `softmax((Q*scale) @ K^T) @ V`, row-major f32.
#[must_use]
pub fn attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * dim];
    for i in 0..seq {
        let mut scores = vec![0.0f32; seq];
        for (j, sc) in scores.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..dim {
                acc += q[i * dim + c] * k[j * dim + c];
            }
            *sc = acc * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for sc in &mut scores {
            *sc = (*sc - m).exp();
            sum += *sc;
        }
        for sc in &mut scores {
            *sc /= sum;
        }
        for (j, &p) in scores.iter().enumerate() {
            for e in 0..dim {
                out[i * dim + e] += p * v[j * dim + e];
            }
        }
    }
    out
}
