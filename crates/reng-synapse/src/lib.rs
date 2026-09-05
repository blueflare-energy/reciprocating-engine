//! Rust interface to the Intel Gaudi SynapseAI graph compiler.
//!
//! The public surface: bf16 conversion helpers, a CPU reference matmul, and
//! (behind the `link-synapse` feature) [`MatmulHpu`], a compile-once /
//! launch-many bf16 matmul on the Gaudi2 MME, plus the [`matmul_bf16`]
//! convenience wrapper. The feature gates everything that links `libSynapse`,
//! so the crate still builds as a plain rlib on hosts without the Habana
//! libraries.

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

/// Convert a slice to bf16 (round-to-nearest-even), the device's weight
/// format.
#[must_use]
pub fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
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
pub(crate) mod ffi;

#[cfg(feature = "link-synapse")]
mod device;

#[cfg(feature = "link-synapse")]
pub use device::Device;

#[cfg(feature = "link-synapse")]
mod chain;

#[cfg(feature = "link-synapse")]
pub use chain::{matmul_chain_bf16, matmul_chain_cpu};

#[cfg(feature = "link-synapse")]
mod mm;

#[cfg(feature = "link-synapse")]
pub use mm::{gemm_bf16, gemm_cpu};

#[cfg(feature = "link-synapse")]
mod ops;

#[cfg(feature = "link-synapse")]
pub use ops::{
    rms_norm_bf16, rms_norm_cpu, rope_bf16, rope_cpu, silu_bf16, silu_cpu, softmax_bf16,
    softmax_cpu,
};

#[cfg(feature = "link-synapse")]
mod attn;

#[cfg(feature = "link-synapse")]
pub use attn::{attention_bf16, attention_cpu};

#[cfg(feature = "link-synapse")]
mod mlp;

#[cfg(feature = "link-synapse")]
pub use mlp::{swiglu_mlp_bf16, swiglu_mlp_cpu};

#[cfg(feature = "link-synapse")]
mod layer;

#[cfg(feature = "link-synapse")]
pub use layer::{LayerWeights, decoder_layer_bf16, decoder_layer_cpu};

#[cfg(feature = "link-synapse")]
mod heads;

#[cfg(feature = "link-synapse")]
pub use heads::{AxisParams, split_rotate_concat_bf16, split_rotate_concat_cpu};

#[cfg(feature = "link-synapse")]
mod model;

#[cfg(feature = "link-synapse")]
pub use model::{
    ModelWeights, layer_cpu, model_forward_bf16, model_forward_cpu, model_probe_bf16,
    model_probe_cpu,
};

#[cfg(feature = "link-synapse")]
mod runtime;

#[cfg(feature = "link-synapse")]
mod cached;

#[cfg(feature = "link-synapse")]
pub use cached::CachedModel;

#[cfg(feature = "link-synapse")]
mod batched;

#[cfg(feature = "link-synapse")]
pub use batched::BatchedModel;

#[cfg(feature = "link-synapse")]
mod probe;

#[cfg(feature = "link-synapse")]
pub use probe::{NodeInput, SYN_TYPE_INT32, bench_node, run_node, run_node_i32};

/// Vendor parameter structs, for [`run_node`] probes.
#[cfg(feature = "link-synapse")]
pub use ffi::{synGEMMParams, synSoftmaxParams};

#[cfg(feature = "link-synapse")]
pub use hpu::MatmulHpu;

/// Run `C[m,n] = A[m,k] @ B[k,n]` in bf16 on the Gaudi2 MME and return `C` as
/// `f32`. Convenience one-shot; use [`MatmulHpu`] to compile once and launch
/// many times.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
#[cfg(feature = "link-synapse")]
pub fn matmul_bf16(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> reng_core::Result<Vec<f32>> {
    MatmulHpu::new(m, k, n)?.run(a, b)
}

#[cfg(feature = "link-synapse")]
mod hpu {
    use super::{bf16_to_f32, f32_to_bf16};
    use crate::ffi::*;
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

    /// A compiled bf16 matmul `C[m,n] = A[m,k] @ B[k,n]` on the Gaudi2 MME.
    ///
    /// The graph is compiled to a recipe once in [`MatmulHpu::new`]; each
    /// [`MatmulHpu::run`] copies inputs in, launches the recipe, and copies the
    /// result out. Device and host buffers are held for the lifetime of the
    /// handle and released on drop.
    pub struct MatmulHpu {
        m: usize,
        k: usize,
        n: usize,
        dev: synDeviceId,
        graph: synGraphHandle,
        recipe: synRecipeHandle,
        stream: synStreamHandle,
        dev_a: u64,
        dev_b: u64,
        dev_c: u64,
        ws_addr: u64,
        host_a: *mut c_void,
        host_b: *mut c_void,
        host_c: *mut c_void,
        names: [CString; 3],
        ids: [u64; 3],
        sizes: [[u64; 2]; 3],
    }

    impl MatmulHpu {
        /// Build the graph and compile the recipe, then acquire a device and
        /// allocate the input, output, and workspace buffers.
        ///
        /// # Errors
        ///
        /// Returns an error if any SynapseAI call fails.
        ///
        /// # Panics
        ///
        /// Never panics; dimensions are validated by [`MatmulHpu::run`].
        pub fn new(m: usize, k: usize, n: usize) -> Result<Self> {
            syn!(synInitialize());
            let mut graph: synGraphHandle = core::ptr::null_mut();
            syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

            // Synapse sizes are FCD-first: A[m,k]->[k,m], B[k,n]->[n,k], C->[n,m].
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

            let name_ptrs: [*const core::ffi::c_char; 3] =
                [names[0].as_ptr(), names[1].as_ptr(), names[2].as_ptr()];
            let mut ids: [u64; 3] = [0; 3];
            syn!(synTensorRetrieveIds(
                recipe,
                name_ptrs.as_ptr(),
                ids.as_mut_ptr(),
                3
            ));

            let dev = crate::device::acquire_device()?;

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

            Ok(Self {
                m,
                k,
                n,
                dev,
                graph,
                recipe,
                stream: core::ptr::null_mut(),
                dev_a,
                dev_b,
                dev_c,
                ws_addr,
                host_a,
                host_b,
                host_c,
                names,
                ids,
                sizes,
            })
            .and_then(Self::with_stream)
        }

        fn with_stream(mut self) -> Result<Self> {
            let mut stream: synStreamHandle = core::ptr::null_mut();
            syn!(synStreamCreateGeneric(&mut stream, self.dev, 0));
            self.stream = stream;
            Ok(self)
        }

        fn infos(&self) -> [synLaunchTensorInfo; 3] {
            let addrs = [self.dev_a, self.dev_b, self.dev_c];
            core::array::from_fn(|i| {
                let mut ti = synLaunchTensorInfo {
                    tensor_name: self.names[i].as_ptr(),
                    tensor_address: addrs[i],
                    tensor_type: SYN_TENSOR_DATA,
                    tensor_size: [0; HABANA_DIM_MAX],
                    tensor_id: self.ids[i],
                };
                ti.tensor_size[0] = self.sizes[i][0];
                ti.tensor_size[1] = self.sizes[i][1];
                ti
            })
        }

        /// Launch the compiled recipe once, reusing whatever inputs are already
        /// resident on the device. Use it to time steady-state throughput after
        /// a [`MatmulHpu::run`].
        ///
        /// # Errors
        ///
        /// Returns an error if the launch or synchronization fails.
        pub fn launch_only(&self) -> Result<()> {
            let infos = self.infos();
            syn!(synLaunch(
                self.stream,
                infos.as_ptr(),
                3,
                self.ws_addr,
                self.recipe,
                0
            ));
            syn!(synStreamSynchronize(self.stream));
            Ok(())
        }

        /// Copy `a` and `b` to the device, launch, and return `C` as `f32`.
        ///
        /// # Errors
        ///
        /// Returns an error if any SynapseAI call fails.
        ///
        /// # Panics
        ///
        /// Panics if `a.len() != m*k` or `b.len() != k*n`.
        pub fn run(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            assert_eq!(a.len(), self.m * self.k);
            assert_eq!(b.len(), self.k * self.n);
            let bytes_a = (self.m * self.k * 2) as u64;
            let bytes_b = (self.k * self.n * 2) as u64;
            let bytes_c = (self.m * self.n * 2) as u64;

            // SAFETY: host buffers were allocated in `new` with these sizes.
            unsafe {
                let ha = self.host_a.cast::<u16>();
                for (idx, &v) in a.iter().enumerate() {
                    *ha.add(idx) = f32_to_bf16(v);
                }
                let hb = self.host_b.cast::<u16>();
                for (idx, &v) in b.iter().enumerate() {
                    *hb.add(idx) = f32_to_bf16(v);
                }
            }

            syn!(synMemCopyAsync(
                self.stream,
                self.host_a as u64,
                bytes_a,
                self.dev_a,
                SYN_HOST_TO_DRAM
            ));
            syn!(synMemCopyAsync(
                self.stream,
                self.host_b as u64,
                bytes_b,
                self.dev_b,
                SYN_HOST_TO_DRAM
            ));
            syn!(synStreamSynchronize(self.stream));

            self.launch_only()?;

            syn!(synMemCopyAsync(
                self.stream,
                self.dev_c,
                bytes_c,
                self.host_c as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(self.stream));

            let mut out = vec![0.0f32; self.m * self.n];
            // SAFETY: host_c holds m*n bf16 values just copied from the device.
            unsafe {
                let hc = self.host_c.cast::<u16>();
                for (idx, o) in out.iter_mut().enumerate() {
                    *o = bf16_to_f32(*hc.add(idx));
                }
            }
            Ok(out)
        }
    }

    impl Drop for MatmulHpu {
        fn drop(&mut self) {
            // Best-effort teardown; ignore errors on the way out.
            unsafe {
                if !self.host_a.is_null() {
                    synHostFree(self.dev, self.host_a, 0);
                }
                if !self.host_b.is_null() {
                    synHostFree(self.dev, self.host_b, 0);
                }
                if !self.host_c.is_null() {
                    synHostFree(self.dev, self.host_c, 0);
                }
                synDeviceFree(self.dev, self.dev_a, 0);
                synDeviceFree(self.dev, self.dev_b, 0);
                synDeviceFree(self.dev, self.dev_c, 0);
                if self.ws_addr != 0 {
                    synDeviceFree(self.dev, self.ws_addr, 0);
                }
                if !self.stream.is_null() {
                    synStreamDestroy(self.stream);
                }
                synGraphDestroy(self.graph);
                synDeviceRelease(self.dev);
                synDestroy();
            }
        }
    }
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
