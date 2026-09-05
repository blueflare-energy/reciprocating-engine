//! SwiGLU MLP block as ONE fused SynapseAI recipe:
//! `out = down_proj( silu(gate_proj(x)) * up_proj(x) ) + x`.
//!
//! This is the first multi-gemm on-device-dataflow milestone: three matmuls, an
//! activation, two elementwise multiplies, and a residual add composed in a
//! single graph with a single launch (activations never leave HBM; only the
//! final `[tokens, hidden]` output is read back). It exercises the same fused
//! pattern a full transformer layer uses, with only already-verified kernels.

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

/// Add a `gemm` node `c = a @ b` (no transpose) to `graph`.
fn gemm_node(
    graph: synGraphHandle,
    a: synTensor,
    b: synTensor,
    c: synTensor,
    name: &str,
) -> Result<()> {
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let guid = CString::new("gemm").unwrap();
    let ins = [a, b];
    let outs = [c];
    let nm = CString::new(name).unwrap();
    syn!(synNodeCreate(
        graph,
        ins.as_ptr(),
        outs.as_ptr(),
        2,
        1,
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        guid.as_ptr(),
        nm.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    Ok(())
}

/// Add a binary elementwise node (`guid` with 2 inputs, 1 output, no params).
fn binary_node(
    graph: synGraphHandle,
    guid: &str,
    a: synTensor,
    b: synTensor,
    out: synTensor,
    name: &str,
) -> Result<()> {
    let g = CString::new(guid).unwrap();
    let ins = [a, b];
    let outs = [out];
    let nm = CString::new(name).unwrap();
    syn!(synNodeCreate(
        graph,
        ins.as_ptr(),
        outs.as_ptr(),
        2,
        1,
        core::ptr::null(),
        0,
        g.as_ptr(),
        nm.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    Ok(())
}

/// Add a unary elementwise node (`guid` with 1 input, 1 output, no params).
fn unary_node(
    graph: synGraphHandle,
    guid: &str,
    a: synTensor,
    out: synTensor,
    name: &str,
) -> Result<()> {
    let g = CString::new(guid).unwrap();
    let ins = [a];
    let outs = [out];
    let nm = CString::new(name).unwrap();
    syn!(synNodeCreate(
        graph,
        ins.as_ptr(),
        outs.as_ptr(),
        1,
        1,
        core::ptr::null(),
        0,
        g.as_ptr(),
        nm.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    Ok(())
}

/// Fused SwiGLU MLP `out = down @ (silu(x @ wg) * (x @ wu)) + x`, bf16 on the
/// HPU as one recipe. All matrices are row-major: `x` is `[tokens, hidden]`,
/// `wgate` and `wup` are `[hidden, inter]` (already transposed, input dim
/// first), and `wdown` is `[inter, hidden]`. Returns `out` `[tokens, hidden]`
/// as f32.
///
/// For reliability every dimension should be at least 128 (see [`crate::Device`]).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if any matrix length disagrees with `tokens`, `hidden`, `inter`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn swiglu_mlp_bf16(
    x: &[f32],
    wgate: &[f32],
    wup: &[f32],
    wdown: &[f32],
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(wgate.len(), hidden * inter);
    assert_eq!(wup.len(), hidden * inter);
    assert_eq!(wdown.len(), inter * hidden);
    let (t, h, i) = (tokens as u64, hidden as u64, inter as u64);

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    let (n_x, n_wg, n_wu, n_wd, n_out) = (
        CString::new("X").unwrap(),
        CString::new("WG").unwrap(),
        CString::new("WU").unwrap(),
        CString::new("WD").unwrap(),
        CString::new("OUT").unwrap(),
    );
    // Persistent I/O, FCD-first (gemm C[m,n]=A@B => A[k,m], B[n,k], C[n,m]):
    // X[hidden,tokens]  WG[inter,hidden]  WU[inter,hidden]  WD[hidden,inter]  OUT[hidden,tokens]
    let t_x = t2d(graph, &n_x, h, t, true)?;
    let t_wg = t2d(graph, &n_wg, i, h, true)?;
    let t_wu = t2d(graph, &n_wu, i, h, true)?;
    let t_wd = t2d(graph, &n_wd, h, i, true)?;
    let t_out = t2d(graph, &n_out, h, t, true)?;
    // Graph-internal intermediates (all [inter,tokens] except down[hidden,tokens]).
    let t_gate = t2d(graph, &CString::new("gate").unwrap(), i, t, false)?;
    let t_up = t2d(graph, &CString::new("up").unwrap(), i, t, false)?;
    let t_sg = t2d(graph, &CString::new("sg").unwrap(), i, t, false)?;
    let t_silu = t2d(graph, &CString::new("silu").unwrap(), i, t, false)?;
    let t_gated = t2d(graph, &CString::new("gated").unwrap(), i, t, false)?;
    let t_down = t2d(graph, &CString::new("down").unwrap(), h, t, false)?;

    gemm_node(graph, t_x, t_wg, t_gate, "gate_proj")?;
    gemm_node(graph, t_x, t_wu, t_up, "up_proj")?;
    unary_node(graph, "sigmoid_fwd_bf16", t_gate, t_sg, "sig")?;
    binary_node(graph, "mult_fwd_bf16", t_gate, t_sg, t_silu, "silu")?;
    binary_node(graph, "mult_fwd_bf16", t_silu, t_up, t_gated, "gate_x_up")?;
    gemm_node(graph, t_gated, t_wd, t_down, "down_proj")?;
    binary_node(graph, "add_fwd_bf16", t_down, t_x, t_out, "residual")?;

    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        CString::new("mlp").unwrap().as_ptr(),
        core::ptr::null()
    ));

    let names = [
        n_x.as_ptr(),
        n_wg.as_ptr(),
        n_wu.as_ptr(),
        n_wd.as_ptr(),
        n_out.as_ptr(),
    ];
    let mut ids: [u64; 5] = [0; 5];
    syn!(synTensorRetrieveIds(
        recipe,
        names.as_ptr(),
        ids.as_mut_ptr(),
        5
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));

    let x_bytes = (tokens * hidden * 2) as u64;
    let w_bytes = (hidden * inter * 2) as u64;
    let (mut dx, mut dwg, mut dwu, mut dwd, mut dout) = (0u64, 0u64, 0u64, 0u64, 0u64);
    syn!(synDeviceMalloc(dev, x_bytes, 0, 0, &mut dx));
    syn!(synDeviceMalloc(dev, w_bytes, 0, 0, &mut dwg));
    syn!(synDeviceMalloc(dev, w_bytes, 0, 0, &mut dwu));
    syn!(synDeviceMalloc(dev, w_bytes, 0, 0, &mut dwd));
    syn!(synDeviceMalloc(dev, x_bytes, 0, 0, &mut dout));
    let mut ws = 0u64;
    syn!(synWorkspaceGetSize(&mut ws, recipe));
    let mut dws = 0u64;
    if ws > 0 {
        syn!(synDeviceMalloc(dev, ws, 0, 0, &mut dws));
    }

    let (mut hx, mut hwg, mut hwu, mut hwd, mut hout): (
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) = (
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    syn!(synHostMalloc(dev, x_bytes, 0, &mut hx));
    syn!(synHostMalloc(dev, w_bytes, 0, &mut hwg));
    syn!(synHostMalloc(dev, w_bytes, 0, &mut hwu));
    syn!(synHostMalloc(dev, w_bytes, 0, &mut hwd));
    syn!(synHostMalloc(dev, x_bytes, 0, &mut hout));

    // SAFETY: each host buffer holds the element count matched to its byte size.
    unsafe {
        let fill = |dst: *mut c_void, src: &[f32]| {
            let p = dst.cast::<u16>();
            for (j, &v) in src.iter().enumerate() {
                *p.add(j) = f32_to_bf16(v);
            }
        };
        fill(hx, x);
        fill(hwg, wgate);
        fill(hwu, wup);
        fill(hwd, wdown);
    }

    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    for (hh, dptr, bytes) in [
        (hx, dx, x_bytes),
        (hwg, dwg, w_bytes),
        (hwu, dwu, w_bytes),
        (hwd, dwd, w_bytes),
    ] {
        syn!(synMemCopyAsync(
            stream,
            hh as u64,
            bytes,
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
        mk(&n_x, dx, ids[0], h, t),
        mk(&n_wg, dwg, ids[1], i, h),
        mk(&n_wu, dwu, ids[2], i, h),
        mk(&n_wd, dwd, ids[3], h, i),
        mk(&n_out, dout, ids[4], h, t),
    ];
    syn!(synLaunch(stream, infos.as_ptr(), 5, dws, recipe, 0));
    syn!(synStreamSynchronize(stream));
    syn!(synMemCopyAsync(
        stream,
        dout,
        x_bytes,
        hout as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; tokens * hidden];
    // SAFETY: hout holds tokens*hidden bf16 elements just copied back.
    unsafe {
        let p = hout.cast::<u16>();
        for (j, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*p.add(j));
        }
    }

    unsafe {
        for hh in [hx, hwg, hwu, hwd, hout] {
            synHostFree(dev, hh, 0);
        }
        for dptr in [dx, dwg, dwu, dwd, dout] {
            synDeviceFree(dev, dptr, 0);
        }
        if dws != 0 {
            synDeviceFree(dev, dws, 0);
        }
        synStreamDestroy(stream);
        synRecipeDestroy(recipe);
        synGraphDestroy(graph);
        synDeviceRelease(dev);
        synDestroy();
    }
    Ok(out)
}

/// CPU reference for [`swiglu_mlp_bf16`]. Same row-major layouts; returns
/// `[tokens, hidden]`.
#[must_use]
pub fn swiglu_mlp_cpu(
    x: &[f32],
    wgate: &[f32],
    wup: &[f32],
    wdown: &[f32],
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        // gate/up: [inter]
        let mut gated = vec![0.0f32; inter];
        for col in 0..inter {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for hh in 0..hidden {
                let xv = x[t * hidden + hh];
                g += xv * wgate[hh * inter + col];
                u += xv * wup[hh * inter + col];
            }
            let silu = g / (1.0 + (-g).exp());
            gated[col] = silu * u;
        }
        // down: [hidden], + residual
        for hh in 0..hidden {
            let mut d = 0.0f32;
            for col in 0..inter {
                d += gated[col] * wdown[col * hidden + hh];
            }
            out[t * hidden + hh] = d + x[t * hidden + hh];
        }
    }
    out
}
