//! Head split / merge along the feature (FCD) axis via the graph compiler's
//! `split` and `concat` logical nodes. Multi-head attention is built by
//! splitting `[hidden, tokens]` activations into `n_heads` slices of
//! `[head_dim, tokens]`, running the verified 2D attention per head inside the
//! same recipe, and concatenating the results. This module pins the axis
//! semantics of those two nodes with a round trip that rotates head order.

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

/// `synAxisParams`: the axis a split/concat operates on (FCD-first indexing).
#[repr(C)]
pub struct AxisParams {
    pub axis: u32,
}

fn t2d(
    graph: synGraphHandle,
    name: &str,
    fcd: u64,
    outer: u64,
    persistent: bool,
) -> Result<(synTensor, CString)> {
    let cname = CString::new(name).unwrap();
    let mut t: synTensor = core::ptr::null_mut();
    syn!(synTensorHandleCreate(
        &mut t,
        graph,
        SYN_TENSOR_DATA,
        cname.as_ptr()
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
    Ok((t, cname))
}

/// Split `x` (`[tokens, hidden]` row-major, hidden contiguous) into `n_heads`
/// slices along the feature axis and concatenate them back rotated by one
/// (output head `j` is input head `(j + 1) % n_heads`). One recipe, two node
/// types; returns `[tokens, hidden]` f32. A pure data-movement round trip that
/// proves the split/concat axis semantics used for multi-head attention.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `x.len() != tokens*hidden` or `hidden % n_heads != 0`.
#[allow(clippy::too_many_lines)]
pub fn split_rotate_concat_bf16(
    x: &[f32],
    tokens: usize,
    hidden: usize,
    n_heads: usize,
) -> Result<Vec<f32>> {
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(hidden % n_heads, 0);
    let head_dim = hidden / n_heads;
    let (t, h, hd) = (tokens as u64, hidden as u64, head_dim as u64);

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    let (t_x, n_x) = t2d(graph, "X", h, t, true)?;
    let (t_out, n_out) = t2d(graph, "OUT", h, t, true)?;
    let mut heads: Vec<synTensor> = Vec::with_capacity(n_heads);
    for i in 0..n_heads {
        heads.push(t2d(graph, &format!("head{i}"), hd, t, false)?.0);
    }

    let axis = AxisParams { axis: 0 };
    let ap = (&raw const axis).cast::<c_void>();
    let asz = core::mem::size_of::<AxisParams>() as u32;
    let g_split = CString::new("split").unwrap();
    let g_concat = CString::new("concat").unwrap();
    let ins = [t_x];
    syn!(synNodeCreate(
        graph,
        ins.as_ptr(),
        heads.as_ptr(),
        1,
        n_heads as u32,
        ap,
        asz,
        g_split.as_ptr(),
        CString::new("split_heads").unwrap().as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    let rotated: Vec<synTensor> = (0..n_heads).map(|j| heads[(j + 1) % n_heads]).collect();
    let outs = [t_out];
    syn!(synNodeCreate(
        graph,
        rotated.as_ptr(),
        outs.as_ptr(),
        n_heads as u32,
        1,
        ap,
        asz,
        g_concat.as_ptr(),
        CString::new("merge_heads").unwrap().as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));

    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        CString::new("heads").unwrap().as_ptr(),
        core::ptr::null()
    ));
    let names = [n_x.as_ptr(), n_out.as_ptr()];
    let mut ids: [u64; 2] = [0; 2];
    syn!(synTensorRetrieveIds(
        recipe,
        names.as_ptr(),
        ids.as_mut_ptr(),
        2
    ));

    let dev = crate::device::acquire_device()?;
    let bytes = (tokens * hidden * 2) as u64;
    let (mut dx, mut dout) = (0u64, 0u64);
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dx));
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dout));
    let mut ws = 0u64;
    syn!(synWorkspaceGetSize(&mut ws, recipe));
    let mut dws = 0u64;
    if ws > 0 {
        syn!(synDeviceMalloc(dev, ws, 0, 0, &mut dws));
    }
    let (mut hx, mut hout): (*mut c_void, *mut c_void) =
        (core::ptr::null_mut(), core::ptr::null_mut());
    syn!(synHostMalloc(dev, bytes, 0, &mut hx));
    syn!(synHostMalloc(dev, bytes, 0, &mut hout));
    // SAFETY: hx holds tokens*hidden bf16 elements.
    unsafe {
        let p = hx.cast::<u16>();
        for (j, &v) in x.iter().enumerate() {
            *p.add(j) = f32_to_bf16(v);
        }
    }
    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    syn!(synMemCopyAsync(
        stream,
        hx as u64,
        bytes,
        dx,
        SYN_HOST_TO_DRAM
    ));
    syn!(synStreamSynchronize(stream));

    let mk = |name: &CString, addr: u64, id: u64| {
        let mut ti = synLaunchTensorInfo {
            tensor_name: name.as_ptr(),
            tensor_address: addr,
            tensor_type: SYN_TENSOR_DATA,
            tensor_size: [0; HABANA_DIM_MAX],
            tensor_id: id,
        };
        ti.tensor_size[0] = h;
        ti.tensor_size[1] = t;
        ti
    };
    let infos = [mk(&n_x, dx, ids[0]), mk(&n_out, dout, ids[1])];
    syn!(synLaunch(stream, infos.as_ptr(), 2, dws, recipe, 0));
    syn!(synStreamSynchronize(stream));
    syn!(synMemCopyAsync(
        stream,
        dout,
        bytes,
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
        synHostFree(dev, hx, 0);
        synHostFree(dev, hout, 0);
        synDeviceFree(dev, dx, 0);
        synDeviceFree(dev, dout, 0);
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

/// CPU reference for [`split_rotate_concat_bf16`].
#[must_use]
pub fn split_rotate_concat_cpu(
    x: &[f32],
    tokens: usize,
    hidden: usize,
    n_heads: usize,
) -> Vec<f32> {
    let hd = hidden / n_heads;
    let mut out = vec![0.0f32; tokens * hidden];
    for tk in 0..tokens {
        for j in 0..n_heads {
            let src = (j + 1) % n_heads;
            for d in 0..hd {
                out[tk * hidden + j * hd + d] = x[tk * hidden + src * hd + d];
            }
        }
    }
    out
}
