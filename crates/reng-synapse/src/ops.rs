//! Individual transformer ops via the direct SynapseAI C API, to test each on
//! the 1.24 stack where the PyTorch frameworks miscompute. Softmax first: it is
//! the prime suspect for the `!!!!` miscompute (fused-softmax overflow).

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

/// Create a 2D bf16 tensor with FCD-first sizes `[cols, rows]`.
fn tensor2d(
    graph: synGraphHandle,
    name: &CString,
    cols: u64,
    rows: u64,
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
    geo.sizes[0] = cols;
    geo.sizes[1] = rows;
    syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
    syn!(synTensorSetDeviceDataType(t, SYN_TYPE_BF16));
    Ok(t)
}

/// Row-wise softmax of a `rows x cols` matrix in bf16 on the HPU (softmax over
/// the `cols` axis), via the direct C API `softmax_fwd_bf16` kernel. Returns
/// `f32`.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `input.len() != rows*cols`.
pub fn softmax_bf16(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    assert_eq!(input.len(), rows * cols);
    let (c, r) = (cols as u64, rows as u64);

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    let name_in = CString::new("IN").unwrap();
    let name_out = CString::new("OUT").unwrap();
    let tin = tensor2d(graph, &name_in, c, r, true)?;
    let tout = tensor2d(graph, &name_out, c, r, true)?;

    // dim 0 = FCD = the `cols` axis (row-wise softmax).
    let params = synSoftmaxParams { dim: 0 };
    let inputs = [tin];
    let outputs = [tout];
    let guid = CString::new("softmax_fwd_bf16").unwrap();
    let node_name = CString::new("sm").unwrap();
    syn!(synNodeCreate(
        graph,
        inputs.as_ptr(),
        outputs.as_ptr(),
        1,
        1,
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
        guid.as_ptr(),
        node_name.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));

    let recipe_name = CString::new("sm").unwrap();
    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        recipe_name.as_ptr(),
        core::ptr::null()
    ));

    let name_ptrs: [*const core::ffi::c_char; 2] = [name_in.as_ptr(), name_out.as_ptr()];
    let mut ids: [u64; 2] = [0; 2];
    syn!(synTensorRetrieveIds(
        recipe,
        name_ptrs.as_ptr(),
        ids.as_mut_ptr(),
        2
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));

    let bytes = (rows * cols * 2) as u64;
    let (mut dev_in, mut dev_out) = (0u64, 0u64);
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dev_in));
    syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut dev_out));
    let mut ws_size = 0u64;
    syn!(synWorkspaceGetSize(&mut ws_size, recipe));
    let mut ws_addr = 0u64;
    if ws_size > 0 {
        syn!(synDeviceMalloc(dev, ws_size, 0, 0, &mut ws_addr));
    }

    let (mut host_in, mut host_out): (*mut c_void, *mut c_void) =
        (core::ptr::null_mut(), core::ptr::null_mut());
    syn!(synHostMalloc(dev, bytes, 0, &mut host_in));
    syn!(synHostMalloc(dev, bytes, 0, &mut host_out));

    // SAFETY: host_in holds rows*cols bf16 elements.
    unsafe {
        let hi = host_in.cast::<u16>();
        for (i, &v) in input.iter().enumerate() {
            *hi.add(i) = f32_to_bf16(v);
        }
    }

    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    syn!(synMemCopyAsync(
        stream,
        host_in as u64,
        bytes,
        dev_in,
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
        ti.tensor_size[0] = c;
        ti.tensor_size[1] = r;
        ti
    };
    let infos = [mk(&name_in, dev_in, ids[0]), mk(&name_out, dev_out, ids[1])];
    syn!(synLaunch(stream, infos.as_ptr(), 2, ws_addr, recipe, 0));
    syn!(synStreamSynchronize(stream));

    syn!(synMemCopyAsync(
        stream,
        dev_out,
        bytes,
        host_out as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; rows * cols];
    // SAFETY: host_out holds rows*cols bf16 elements just copied back.
    unsafe {
        let ho = host_out.cast::<u16>();
        for (i, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*ho.add(i));
        }
    }

    unsafe {
        synHostFree(dev, host_in, 0);
        synHostFree(dev, host_out, 0);
        synDeviceFree(dev, dev_in, 0);
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

/// CPU reference row-wise softmax over `cols`, f32.
#[must_use]
pub fn softmax_cpu(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..r * cols + cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &v) in out[r * cols..r * cols + cols].iter_mut().zip(row) {
            let e = (v - m).exp();
            *o = e;
            sum += e;
        }
        for o in &mut out[r * cols..r * cols + cols] {
            *o /= sum;
        }
    }
    out
}
