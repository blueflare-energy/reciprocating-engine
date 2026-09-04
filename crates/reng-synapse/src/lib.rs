//! Rust interface to the Intel Gaudi SynapseAI graph compiler.
//!
//! The public surface is small: bf16 conversion helpers, a CPU reference
//! matmul, and (behind the `link-synapse` feature) [`matmul_bf16`], which runs
//! `C = A @ B` on the Gaudi2 MME through the SynapseAI C API. The feature gates
//! everything that links `libSynapse`, so the crate still builds as a plain
//! rlib on hosts without the Habana libraries.

/// Convert an `f32` to bf16 bits, rounding to nearest even.
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0; // NaN
    }
    let round_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round_bias) >> 16) as u16
}

/// Convert bf16 bits to an `f32`.
#[must_use]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

/// CPU reference matmul: `C[m,n] = A[m,k] @ B[k,n]`, row-major `f32`.
#[must_use]
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aik = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for (cj, bj) in crow.iter_mut().zip(brow) {
                *cj += aik * bj;
            }
        }
    }
    c
}

#[cfg(feature = "link-synapse")]
mod ffi;

/// Run `C[m,n] = A[m,k] @ B[k,n]` in bf16 on the Gaudi2 MME and return `C` as
/// `f32`.
///
/// Inputs are row-major `f32`; they are rounded to bf16, computed on the HPU
/// (with FP32 accumulation), and converted back. Requires `libSynapse` at link
/// time (the `link-synapse` feature) and a Gaudi2 device at run time.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `a.len() != m*k` or `b.len() != k*n`.
#[cfg(feature = "link-synapse")]
pub fn matmul_bf16(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> reng_core::Result<Vec<f32>> {
    use core::ffi::c_void;
    use ffi::*;
    use reng_core::Error;
    use std::ffi::CString;

    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

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

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    // Persistent bf16 tensors. Synapse sizes are FCD-first (fastest dim first):
    // A[m,k] -> [k,m], B[k,n] -> [n,k], C[m,n] -> [n,m].
    let names = [
        CString::new("A").unwrap(),
        CString::new("B").unwrap(),
        CString::new("C").unwrap(),
    ];
    let sizes: [[u64; 2]; 3] = [
        [k as u64, m as u64],
        [n as u64, k as u64],
        [n as u64, m as u64],
    ];
    let mut tensors: Vec<synTensor> = Vec::with_capacity(3);
    for (name, dims) in names.iter().zip(sizes.iter()) {
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        let mut t: synTensor = core::ptr::null_mut();
        syn!(synTensorHandleCreate(
            &mut t,
            graph,
            SYN_TENSOR_DATA,
            name.as_ptr()
        ));
        syn!(synTensorAssignToSection(t, sec, 0));
        let mut geo = synTensorGeometry {
            sizes: [0; HABANA_DIM_MAX],
            dims: 2,
        };
        geo.sizes[0] = dims[0];
        geo.sizes[1] = dims[1];
        syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
        syn!(synTensorSetDeviceDataType(t, SYN_TYPE_BF16));
        tensors.push(t);
    }

    let params = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let inputs = [tensors[0], tensors[1]];
    let outputs = [tensors[2]];
    let guid = CString::new("gemm").unwrap();
    let node_name = CString::new("mm").unwrap();
    syn!(synNodeCreate(
        graph,
        inputs.as_ptr(),
        outputs.as_ptr(),
        2,
        1,
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        guid.as_ptr(),
        node_name.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));

    let recipe_name = CString::new("mm").unwrap();
    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        recipe_name.as_ptr(),
        core::ptr::null()
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));

    let bytes_a = (m * k * 2) as u64;
    let bytes_b = (k * n * 2) as u64;
    let bytes_c = (m * n * 2) as u64;

    let (mut dev_a, mut dev_b, mut dev_c) = (0u64, 0u64, 0u64);
    syn!(synDeviceMalloc(dev, bytes_a, 0, 0, &mut dev_a));
    syn!(synDeviceMalloc(dev, bytes_b, 0, 0, &mut dev_b));
    syn!(synDeviceMalloc(dev, bytes_c, 0, 0, &mut dev_c));

    let mut ws_size = 0u64;
    syn!(synWorkspaceGetSize(&mut ws_size, recipe));
    let mut ws_addr = 0u64;
    if ws_size > 0 {
        syn!(synDeviceMalloc(dev, ws_size, 0, 0, &mut ws_addr));
    }

    let (mut host_a, mut host_b, mut host_c): (*mut c_void, *mut c_void, *mut c_void) = (
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    syn!(synHostMalloc(dev, bytes_a, 0, &mut host_a));
    syn!(synHostMalloc(dev, bytes_b, 0, &mut host_b));
    syn!(synHostMalloc(dev, bytes_c, 0, &mut host_c));

    // SAFETY: host buffers were just allocated with the matching byte sizes.
    unsafe {
        let ha = host_a.cast::<u16>();
        for (idx, &v) in a.iter().enumerate() {
            *ha.add(idx) = f32_to_bf16(v);
        }
        let hb = host_b.cast::<u16>();
        for (idx, &v) in b.iter().enumerate() {
            *hb.add(idx) = f32_to_bf16(v);
        }
    }

    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));
    syn!(synMemCopyAsync(
        stream,
        host_a as u64,
        bytes_a,
        dev_a,
        SYN_HOST_TO_DRAM
    ));
    syn!(synMemCopyAsync(
        stream,
        host_b as u64,
        bytes_b,
        dev_b,
        SYN_HOST_TO_DRAM
    ));
    syn!(synStreamSynchronize(stream));

    let mk = |name: &CString, addr: u64, dims: [u64; 2]| {
        let mut ti = synLaunchTensorInfo {
            tensor_name: name.as_ptr(),
            tensor_address: addr,
            tensor_type: SYN_TENSOR_DATA,
            tensor_size: [0; HABANA_DIM_MAX],
            tensor_id: 0,
        };
        ti.tensor_size[0] = dims[0];
        ti.tensor_size[1] = dims[1];
        ti
    };
    let infos = [
        mk(&names[0], dev_a, sizes[0]),
        mk(&names[1], dev_b, sizes[1]),
        mk(&names[2], dev_c, sizes[2]),
    ];
    syn!(synLaunch(stream, infos.as_ptr(), 3, ws_addr, recipe, 0));
    syn!(synStreamSynchronize(stream));

    syn!(synMemCopyAsync(
        stream,
        dev_c,
        bytes_c,
        host_c as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; m * n];
    // SAFETY: host_c holds m*n bf16 values just copied from the device.
    unsafe {
        let hc = host_c.cast::<u16>();
        for (idx, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*hc.add(idx));
        }
    }

    // Best-effort teardown; ignore errors on the way out.
    unsafe {
        synHostFree(dev, host_a, 0);
        synHostFree(dev, host_b, 0);
        synHostFree(dev, host_c, 0);
        synDeviceFree(dev, dev_a, 0);
        synDeviceFree(dev, dev_b, 0);
        synDeviceFree(dev, dev_c, 0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_is_close() {
        for &x in &[0.0f32, 1.0, -1.0, 3.5, 0.125, 100.0, -2.75] {
            let r = bf16_to_f32(f32_to_bf16(x));
            assert!((r - x).abs() <= x.abs() * 0.01 + 1e-3, "{x} -> {r}");
        }
    }

    #[test]
    fn cpu_matmul_2x2() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(matmul_cpu(&a, &b, 2, 2, 2), vec![19.0, 22.0, 43.0, 50.0]);
    }
}
