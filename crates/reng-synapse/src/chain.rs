//! A chained bf16 matmul graph (`A * W^depth`) built as a single SynapseAI
//! recipe. Used to test whether a multi-node graph composed directly through
//! the C API computes correctly, independent of the PyTorch/vLLM lowering.

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

fn square_tensor(
    graph: synGraphHandle,
    name: &CString,
    d: u64,
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
    geo.sizes[0] = d;
    geo.sizes[1] = d;
    syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
    syn!(synTensorSetDeviceDataType(t, SYN_TYPE_BF16));
    Ok(t)
}

/// Compute `A * W^depth` (square `d x d`, row-major) in bf16 on the Gaudi2 MME
/// as a single compiled graph of `depth` chained gemm nodes, returning the
/// result as `f32`.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `a.len() != d*d`, `w.len() != d*d`, or `depth == 0`.
pub fn matmul_chain_bf16(a: &[f32], w: &[f32], d: usize, depth: usize) -> Result<Vec<f32>> {
    assert_eq!(a.len(), d * d);
    assert_eq!(w.len(), d * d);
    assert!(depth >= 1);
    let dd = d as u64;

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    let name_a = CString::new("A").unwrap();
    let name_w = CString::new("W").unwrap();
    let name_out = CString::new("OUT").unwrap();

    let ta = square_tensor(graph, &name_a, dd, true)?;
    let tw = square_tensor(graph, &name_w, dd, true)?;
    let tout = square_tensor(graph, &name_out, dd, true)?;

    // Intermediates h_0 .. h_{depth-2} are graph-internal (non-persistent).
    let mut inter: Vec<synTensor> = Vec::new();
    for i in 0..depth.saturating_sub(1) {
        let nm = CString::new(format!("h{i}")).unwrap();
        inter.push(square_tensor(graph, &nm, dd, false)?);
    }

    let params = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let guid = CString::new("gemm").unwrap();
    // node k: prev @ W -> next
    for k in 0..depth {
        let prev = if k == 0 { ta } else { inter[k - 1] };
        let next = if k == depth - 1 { tout } else { inter[k] };
        let inputs = [prev, tw];
        let outputs = [next];
        let nm = CString::new(format!("mm{k}")).unwrap();
        syn!(synNodeCreate(
            graph,
            inputs.as_ptr(),
            outputs.as_ptr(),
            2,
            1,
            (&raw const params).cast::<c_void>(),
            core::mem::size_of::<synGEMMParams>() as u32,
            guid.as_ptr(),
            nm.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
        ));
    }

    let recipe_name = CString::new("chain").unwrap();
    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        recipe_name.as_ptr(),
        core::ptr::null()
    ));

    let name_ptrs: [*const core::ffi::c_char; 3] =
        [name_a.as_ptr(), name_w.as_ptr(), name_out.as_ptr()];
    let mut ids: [u64; 3] = [0; 3];
    syn!(synTensorRetrieveIds(
        recipe,
        name_ptrs.as_ptr(),
        ids.as_mut_ptr(),
        3
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));

    let bytes = (d * d * 2) as u64;
    let (mut dev_a, mut dev_w, mut dev_out) = (0u64, 0u64, 0u64);
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dev_a));
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dev_w));
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dev_out));

    let mut ws_size = 0u64;
    syn!(synWorkspaceGetSize(&mut ws_size, recipe));
    let mut ws_addr = 0u64;
    if ws_size > 0 {
        syn!(synDeviceMalloc(dev, ws_size, 0, 0, &mut ws_addr));
    }

    let (mut host_a, mut host_w, mut host_out): (*mut c_void, *mut c_void, *mut c_void) = (
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    syn!(synHostMalloc(dev, bytes, 0, &mut host_a));
    syn!(synHostMalloc(dev, bytes, 0, &mut host_w));
    syn!(synHostMalloc(dev, bytes, 0, &mut host_out));

    // SAFETY: host buffers hold d*d bf16 elements each.
    unsafe {
        let ha = host_a.cast::<u16>();
        for (i, &v) in a.iter().enumerate() {
            *ha.add(i) = f32_to_bf16(v);
        }
        let hw = host_w.cast::<u16>();
        for (i, &v) in w.iter().enumerate() {
            *hw.add(i) = f32_to_bf16(v);
        }
    }

    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    syn!(synMemCopyAsync(
        stream,
        host_a as u64,
        bytes,
        dev_a,
        SYN_HOST_TO_DRAM
    ));
    syn!(synMemCopyAsync(
        stream,
        host_w as u64,
        bytes,
        dev_w,
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
        ti.tensor_size[0] = dd;
        ti.tensor_size[1] = dd;
        ti
    };
    let infos = [
        mk(&name_a, dev_a, ids[0]),
        mk(&name_w, dev_w, ids[1]),
        mk(&name_out, dev_out, ids[2]),
    ];
    syn!(synLaunch(stream, infos.as_ptr(), 3, ws_addr, recipe, 0));
    syn!(synStreamSynchronize(stream));
    syn!(synMemCopyAsync(
        stream,
        dev_out,
        bytes,
        host_out as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; d * d];
    // SAFETY: host_out holds d*d bf16 elements just copied from the device.
    unsafe {
        let ho = host_out.cast::<u16>();
        for (i, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*ho.add(i));
        }
    }

    unsafe {
        synHostFree(dev, host_a, 0);
        synHostFree(dev, host_w, 0);
        synHostFree(dev, host_out, 0);
        synDeviceFree(dev, dev_a, 0);
        synDeviceFree(dev, dev_w, 0);
        synDeviceFree(dev, dev_out, 0);
        if ws_addr != 0 {
            synDeviceFree(dev, ws_addr, 0);
        }
        synStreamDestroy(stream);
        synGraphDestroy(graph);
        synDeviceRelease(dev);
        synDestroy();
    }
    Ok(out)
}

/// CPU reference for `A * W^depth` (square `d x d`, row-major, f32).
#[must_use]
pub fn matmul_chain_cpu(a: &[f32], w: &[f32], d: usize, depth: usize) -> Vec<f32> {
    let mut cur = a.to_vec();
    for _ in 0..depth {
        let mut next = vec![0.0f32; d * d];
        for i in 0..d {
            for k in 0..d {
                let cik = cur[i * d + k];
                for j in 0..d {
                    next[i * d + j] += cik * w[k * d + j];
                }
            }
        }
        cur = next;
    }
    cur
}
