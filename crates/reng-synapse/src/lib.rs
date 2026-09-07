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

/// The gate activation of a decoder layer's MLP (a model-level fact, so
/// it lives outside the SynapseAI-linked modules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `x * sigmoid(x)` (Llama and most others).
    Silu,
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, the HF
    /// `gelu_pytorch_tanh` of Gemma.
    GeluTanh,
}

/// What `RENG_SDPA` (its value, `None` when unset) says about the fused
/// attention node of the decoder recipes: `Some(false)` for `0`, `off`,
/// `false`, `no` or an empty value, `Some(true)` for any other value, and
/// `None` when the variable is unset, which leaves the choice to the
/// recipe (fused in the single-sequence decode recipe, the four-node
/// chain elsewhere; see `model.rs`).
#[must_use]
pub fn sdpa_switch(value: Option<&str>) -> Option<bool> {
    value.map(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "off" | "false" | "no"
        )
    })
}

/// `w * scale` in bf16: one rounding of the f32 product per element, as
/// the device would see a scalar folded into a weight. Exact for
/// power-of-two scales.
#[must_use]
pub fn scale_bf16(w: &[u16], scale: f32) -> Vec<u16> {
    w.iter()
        .map(|&b| f32_to_bf16(bf16_to_f32(b) * scale))
        .collect()
}

/// A column window of a row-major bf16 matrix on the host: `rows` runs of
/// `cols` elements, `pitch` elements apart (`pitch >= cols`), starting at
/// the slice an input was given. The device gets the contiguous `[rows,
/// cols]` matrix; the host keeps a view of the whole checkpoint matrix
/// (a tensor-parallel shard of an `o_proj` or `down_proj` weight).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride {
    pub rows: usize,
    pub cols: usize,
    pub pitch: usize,
}

/// Gather the column window `st` of `data` into a contiguous `[rows,
/// cols]` matrix (the CPU-side view of a strided weight).
///
/// # Panics
///
/// Panics if `data` is too short for the window.
#[must_use]
pub fn gather_columns(data: &[u16], st: Stride) -> Vec<u16> {
    let mut v = Vec::with_capacity(st.rows * st.cols);
    for r in 0..st.rows {
        v.extend_from_slice(&data[r * st.pitch..r * st.pitch + st.cols]);
    }
    v
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
    EmbedTable, ModelWeights, RopeTables, layer_cpu, model_forward_bf16, model_forward_cpu,
    model_probe_bf16, model_probe_cpu,
};

/// How many wide recipes a model may hold, one per key bucket (see
/// [`key_step`]). Each costs its own mask input and gather indices and
/// borrows everything else (weights, KV cache, workspace) from the first
/// one, so the ceiling is compile time, not memory.
const MAX_KEY_BUCKETS: usize = 16;

/// Granularity of the key buckets: `rows`, doubled until at most
/// [`MAX_KEY_BUCKETS`] buckets span `keys_full` slots.
///
/// A block of `rows` queries at position `p` sees keys `0 .. p + rows` and
/// no more, so a recipe compiled for `p + rows` keys does half the work of
/// one compiled for the whole capacity, summed over a full prefill. One
/// recipe per block would be exact; a bucket wastes at most `step` key
/// columns per block, 11% over an exact per-block range at 256-row blocks
/// and a capacity of 32832 (see the unit test), against a factor of 1.77
/// saved on the whole-capacity form.
pub fn key_step(rows: usize, keys_full: usize) -> usize {
    let mut step = rows.max(1);
    while keys_full.div_ceil(step) > MAX_KEY_BUCKETS {
        step *= 2;
    }
    step
}

/// The bucket that holds `need` key slots.
pub fn key_bucket(need: usize, step: usize, keys_full: usize) -> usize {
    need.max(1)
        .div_ceil(step)
        .saturating_mul(step)
        .min(keys_full)
}

/// Smallest KV-cache bucket the batched decoder compiles, in positions.
/// `RENG_MIN_CAP` overrides it (a huge value disables bucketing, a tiny
/// one exercises growth in tests).
pub const MIN_CACHE_BUCKET: usize = 256;

/// The cache bucket `BatchedModel` holds `need` positions in: the smallest
/// power-of-two multiple of [`MIN_CACHE_BUCKET`] (or `RENG_MIN_CAP`) that
/// holds them, capped at `capacity`.
///
/// The batched path's `capacity` is a ceiling, not an allocation. The
/// model starts at `cache_bucket(1, capacity)` and doubles only when a
/// sequence actually reaches the current bucket, copying the used rows
/// across, so a run that stops at position 544 never allocates for more
/// than 1024 positions whatever `--capacity` says. Anything that budgets
/// device memory for this path has to ask what bucket the run reaches,
/// not what the ceiling is.
#[must_use]
pub fn cache_bucket(need: usize, capacity: usize) -> usize {
    let min_cap: usize = std::env::var("RENG_MIN_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_CACHE_BUCKET);
    let mut c = min_cap.max(1);
    while c < need {
        c *= 2;
    }
    c.min(capacity)
}

/// The largest tensor the engine will build, in bytes.
///
/// A SynapseAI tensor stride is a 32-bit field in the TPC descriptor, and
/// the largest stride of a dense tensor is its own byte size (the stride
/// past its outermost dimension). Above `2^32` the graph compiler prints
/// `non-irf44 tpc node with a tensor with non-0 high32 bits in tensor
/// stride ... (Will not be set in the descriptor)` at critical level and
/// computes with the truncated stride, so any TPC node reading such a
/// tensor is one layout change away from silently reading the wrong rows.
/// Measured on SynapseAI 1.24.1: a `[t, t, heads]` bf16 score tensor is
/// silent at `t = 8192` (exactly `2^32` bytes) and prints one message per
/// layer per tensor from `t = 16384` (`2^34` bytes) up.
pub const MAX_TENSOR_BYTES: u64 = 1 << 32;

/// The byte size of a dense tensor of `sizes` elements of `elem` bytes,
/// and whether it stays inside [`MAX_TENSOR_BYTES`].
#[must_use]
pub fn tensor_bytes(sizes: &[u64], elem: usize) -> u64 {
    sizes.iter().product::<u64>() * elem as u64
}

/// Whether a tensor of these sizes can be built: see [`MAX_TENSOR_BYTES`].
#[must_use]
pub fn tensor_fits(sizes: &[u64], elem: usize) -> bool {
    tensor_bytes(sizes, elem) <= MAX_TENSOR_BYTES
}

/// Tensor-parallel decoding over the cards of one HCCL communicator.
#[cfg(feature = "link-synapse")]
pub mod tp;

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

/// FP8 operands on the MME: fp8 tensor creation with the vendor's
/// quantization metadata, and the probes that decide the gemm form.
#[cfg(feature = "link-synapse")]
pub mod fp8;

#[cfg(feature = "link-synapse")]
mod probe;

/// HCCL collectives (one card per process) and the `reng-hccl-test` worker.
#[cfg(feature = "link-synapse")]
pub mod hccl;

#[cfg(feature = "link-synapse")]
pub use probe::{
    NodeInput, SYN_TYPE_INT32, bench_node, run_node, run_node_extra, run_node_extra_typed,
    run_node_i32, run_node_pick,
};

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
    fn scale_bf16_power_of_two_is_exact() {
        let w = to_bf16(&[1.0, -3.0, 0.4375, 1000.0, -0.001]);
        let back: Vec<f32> = scale_bf16(&w, 0.125)
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();
        let want: Vec<f32> = w.iter().map(|&b| bf16_to_f32(b) * 0.125).collect();
        assert_eq!(back, want);
        // A non-power-of-two scale rounds once, to within half a bf16 ulp.
        let back: Vec<f32> = scale_bf16(&w, 0.22)
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();
        for (&b, &x) in back.iter().zip(&w) {
            let exact = bf16_to_f32(x) * 0.22;
            assert!(
                (b - exact).abs() <= exact.abs() * (1.0 / 256.0),
                "{b} vs {exact}"
            );
        }
    }

    #[test]
    fn sdpa_switch_reads_the_variable() {
        assert_eq!(sdpa_switch(None), None);
        for off in ["", "0", "off", "OFF", "false", "no", " 0 "] {
            assert_eq!(sdpa_switch(Some(off)), Some(false), "{off:?}");
        }
        for on in ["1", "on", "yes", "true", "2"] {
            assert_eq!(sdpa_switch(Some(on)), Some(true), "{on:?}");
        }
    }

    #[test]
    fn cpu_matmul_2x2() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(matmul_cpu(&a, &b, 2, 2, 2), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn tensor_stride_limit_matches_the_compiler() {
        // The no-cache prefill score tensor [tokens, tokens, heads] of a
        // 32-head model: silent at 8192 tokens (exactly 2^32 bytes),
        // "non-0 high32 bits in tensor stride" at 16384 and beyond.
        assert_eq!(tensor_bytes(&[8192, 8192, 32], 2), MAX_TENSOR_BYTES);
        assert!(tensor_fits(&[8192, 8192, 32], 2));
        assert!(!tensor_fits(&[16384, 16384, 32], 2));
        assert!(!tensor_fits(&[32768, 32768, 32], 2));
        // The cached path's block score tensor [keys, 256, heads] stays
        // inside it even at a capacity of 131072 positions.
        assert!(tensor_fits(&[131_137, 256, 32], 2));
        // And so do the two KV cache buffers of that capacity.
        assert!(tensor_fits(&[128, 131_137, 1, 8], 2));
        // f32 counts four bytes per element.
        assert!(tensor_fits(&[1 << 30], 4));
        assert!(!tensor_fits(&[(1 << 30) + 1], 4));
    }

    #[test]
    fn key_buckets_span_the_capacity() {
        // 256-row blocks over a 32832-position cache: the step doubles
        // from the block size until at most sixteen buckets cover it.
        let (rows, full) = (256, 32833);
        let step = key_step(rows, full);
        assert_eq!(step, 4096);
        assert!(full.div_ceil(step) <= 16);
        assert_eq!(key_bucket(1, step, full), 4096);
        assert_eq!(key_bucket(4096, step, full), 4096);
        assert_eq!(key_bucket(4097, step, full), 8192);
        assert_eq!(key_bucket(full, step, full), full);
        // Never past the cache, never below one bucket.
        for need in [0, 1, full, full + 1] {
            let b = key_bucket(need, step, full);
            assert!(b <= full && b >= need.min(full).max(1));
        }
        // A small cache keeps the block size as its step.
        assert_eq!(key_step(256, 513), 256);
        assert_eq!(key_bucket(256, 256, 513), 256);
        assert_eq!(key_bucket(257, 256, 513), 512);
        // Summed over a full prefill the buckets cost well under the
        // whole-capacity form (2392129 key columns against 4235457, a
        // factor of 1.77) and not much over the exact prefix (2146369,
        // a factor of 1.11); the exact prefix itself is the factor of two
        // the causal mask throws away.
        let (mut bucketed, mut exact) = (0usize, 0usize);
        let mut pos = 0;
        while pos < full - 1 {
            bucketed += key_bucket((pos + rows).min(full), step, full);
            exact += (pos + rows).min(full);
            pos += rows;
        }
        let whole = full * (full - 1).div_ceil(rows);
        assert!(bucketed * 5 < whole * 3, "{bucketed} vs {whole}");
        assert!(bucketed < exact * 6 / 5, "{bucketed} vs {exact}");
    }
}
