//! A full decoder-only transformer forward pass as ONE fused SynapseAI recipe:
//! `L` decoder layers, a final RMSNorm, and the LM head, with a causal mask.
//! One graph, one launch; activations stay in HBM and only the
//! `[tokens, vocab]` logits are read back.
//!
//! The graph is assembled by a small builder that appends layers, so the same
//! code path serves a 2-layer synthetic model and a real 30-layer one. The
//! embedding gather is done on the host (a row copy per token; trivially exact
//! and cheap), so the graph starts from `[tokens, hidden]` activations.
//! Grouped-query attention is supported: query head `j` uses K/V head
//! `j / (n_heads / n_kv_heads)`, the HF `repeat_kv` convention. Attention
//! is batched over heads in 4-D tensors (`[.., heads-per-group, groups]`)
//! so each step of it is one node for all heads.
//!
//! With a KV cache (see `cached.rs`) each layer updates its cache tensors
//! `[head_dim, capacity + 1, 1, kv_heads]` in place: an ONNX ScatterND
//! update writes the block's rotated keys and values at their positions
//! (index tensor uploaded per step). Attention runs over the whole cache
//! with a mask that admits positions up to each query's own. No DMA node
//! touches the cache.
//!
//! Launch and readback (including the completion protocol) live in
//! `runtime.rs`. Diagnostic environment switches (all off by default):
//! `RENG_DEVSYNC` (device-wide sync after launch), `RENG_SETTLE_MS` (sleep
//! before readback), `RENG_EVBRIDGE` (event-gated readback on a second
//! stream), `RENG_SERIALIZE` (explicit dependency chain over all nodes),
//! `RENG_PERSIST_LAYERS` (every layer output is a persistent tensor instead
//! of a workspace tensor), `RENG_READBACK_TRACE` (print readback poll counts).

use crate::LayerWeights;
use crate::ffi::*;
use crate::runtime::{Out, OutKind, Runtime};
use crate::{bf16_to_f32, scale_bf16};
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

#[repr(C)]
struct RmsNormParams {
    epsilon: f32,
    fused_gamma_beta: bool,
    use_stages: bool,
    /// Explicit padding, zeroed: the raw bytes go into the recipe cache
    /// key, and compiler-inserted padding would carry stack garbage.
    _pad: [u8; 2],
    bwd_mode: i32,
}

#[repr(C)]
struct RopeParams {
    offset: u32,
    mode: i32,
}

/// Additive attention-mask value for disallowed keys; `exp` of it underflows
/// to exactly zero in bf16 softmax while staying representable.
pub(crate) const MASK_NEG: f32 = -30000.0;

/// `wq` with the attention scale folded in: a scaled bf16 copy, or the
/// checkpoint slice itself when the model has a q norm, which would divide
/// the scale out of `wq` again (it rides on the norm gain instead).
fn scaled_wq<'a>(w: &LayerWeights<'a>) -> std::borrow::Cow<'a, [u16]> {
    if w.qn.is_empty() {
        std::borrow::Cow::Owned(scale_bf16(w.wq, w.scale))
    } else {
        std::borrow::Cow::Borrowed(w.wq)
    }
}

fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok()
}

pub(crate) fn make_tensor(
    graph: synGraphHandle,
    name: &str,
    sizes: &[u64],
    dtype: core::ffi::c_int,
    persistent: bool,
) -> Result<(synTensor, CString)> {
    let section = if persistent {
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        sec
    } else {
        core::ptr::null_mut()
    };
    make_tensor_in(graph, name, sizes, dtype, section)
}

/// Create a tensor; a non-null `section` makes it persistent in that section
/// (at offset 0), so two tensors in one section alias the same memory.
pub(crate) fn make_tensor_in(
    graph: synGraphHandle,
    name: &str,
    sizes: &[u64],
    dtype: core::ffi::c_int,
    section: synSectionHandle,
) -> Result<(synTensor, CString)> {
    let cname = CString::new(name).unwrap();
    let mut t: synTensor = core::ptr::null_mut();
    syn!(synTensorHandleCreate(
        &mut t,
        graph,
        SYN_TENSOR_DATA,
        cname.as_ptr()
    ));
    if !section.is_null() {
        syn!(synTensorAssignToSection(t, section, 0));
    }
    let mut geo = synTensorGeometry {
        sizes: [0; HABANA_DIM_MAX],
        dims: sizes.len() as u32,
    };
    geo.sizes[..sizes.len()].copy_from_slice(sizes);
    syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
    syn!(synTensorSetDeviceDataType(t, dtype));
    Ok((t, cname))
}

/// Accumulates a graph: persistent inputs (with their host data), persistent
/// scratch tensors (device-resident, not read back), internal tensors, and
/// nodes. Launch plumbing lives in [`Runtime`].
pub(crate) struct Gb<'a> {
    pub graph: synGraphHandle,
    pub names: Vec<CString>,
    pub sizes: Vec<Vec<u64>>,
    /// Host data per bf16 input: a checkpoint slice borrowed as is, or an
    /// owned conversion (per-step inputs, scaled copies).
    pub data: Vec<std::borrow::Cow<'a, [u16]>>,
    /// Raw device bytes for non-bf16 inputs (else `data` is converted).
    pub raw: Vec<Option<Vec<u8>>>,
    pub scratch_names: Vec<CString>,
    pub scratch_sizes: Vec<Vec<u64>>,
    /// Whether each scratch tensor is f32 (else bf16); sizes count elements.
    pub scratch_f32: Vec<bool>,
    /// For a scratch tensor that shares another scratch tensor's section
    /// (in-place update), that tensor's name; the runtime binds both to one
    /// buffer.
    pub scratch_alias: Vec<Option<String>>,
    /// Section of each scratch tensor (by name), for aliasing.
    scratch_sections: std::collections::HashMap<String, synSectionHandle>,
    /// Node ids in creation order (a valid topological order for our graphs).
    node_ids: Vec<synNodeId>,
    /// Running digest of the graph's structure (every tensor's name, sizes,
    /// dtype and persistence, every node's guid, name, operands and params),
    /// the key of the on-disk recipe cache (see `Runtime`).
    digest: std::hash::DefaultHasher,
    digest2: std::hash::DefaultHasher,
}

impl<'a> Gb<'a> {
    pub fn new() -> Result<Self> {
        // A second graph in the same process finds the library initialised.
        let st = unsafe { synInitialize() };
        if st != SYN_SUCCESS && st != SYN_OBJECT_ALREADY_INITIALIZED {
            return Err(Error::Other(format!("synInitialize -> synStatus {st}")));
        }
        let mut graph: synGraphHandle = core::ptr::null_mut();
        syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));
        Ok(Self {
            graph,
            names: Vec::new(),
            sizes: Vec::new(),
            data: Vec::new(),
            raw: Vec::new(),
            scratch_names: Vec::new(),
            scratch_sizes: Vec::new(),
            scratch_f32: Vec::new(),
            scratch_alias: Vec::new(),
            scratch_sections: std::collections::HashMap::new(),
            node_ids: Vec::new(),
            digest: std::hash::DefaultHasher::new(),
            digest2: {
                let mut h = std::hash::DefaultHasher::new();
                std::hash::Hash::hash(&0x5eed_u64, &mut h);
                h
            },
        })
    }

    /// Fold `bytes` into the structural digest.
    fn note(&mut self, bytes: &[u8]) {
        use std::hash::Hash;
        bytes.hash(&mut self.digest);
        bytes.hash(&mut self.digest2);
    }

    /// Record a tensor's declaration in the digest: kind, name, sizes,
    /// dtype and, for aliased scratch tensors, the tensor they alias.
    fn note_tensor(
        &mut self,
        kind: &str,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
        of: &str,
    ) {
        let mut b = Vec::new();
        b.extend_from_slice(kind.as_bytes());
        b.push(0);
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        for d in sizes {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&dtype.to_le_bytes());
        b.extend_from_slice(of.as_bytes());
        self.note(&b);
    }

    /// The structural digest as 32 hex digits. Two graphs with the same
    /// digest compile to interchangeable recipes: the compiled program
    /// depends on the structure only, never on the weights' values.
    pub fn cache_key(&self) -> String {
        use std::hash::Hasher;
        format!(
            "{:016x}{:016x}",
            self.digest.finish(),
            self.digest2.finish()
        )
    }

    /// A persistent output tensor that is not an input (read back, or
    /// written by the graph and read through [`Runtime::read_bf16_range`]).
    pub fn output(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
    ) -> Result<(synTensor, CString)> {
        self.note_tensor("out", name, sizes, dtype, "");
        make_tensor(self.graph, name, sizes, dtype, true)
    }

    /// A persistent bf16 input tensor whose host data (converted from f32
    /// here) is uploaded at launch.
    pub fn input(&mut self, name: &str, sizes: &[u64], data: &[f32]) -> Result<synTensor> {
        self.input_bf16(name, sizes, std::borrow::Cow::Owned(crate::to_bf16(data)))
    }

    /// A persistent bf16 input tensor from bf16 host data, borrowed when the
    /// caller keeps it alive (checkpoint weights) so nothing is copied
    /// before the upload.
    pub fn input_bf16(
        &mut self,
        name: &str,
        sizes: &[u64],
        data: std::borrow::Cow<'a, [u16]>,
    ) -> Result<synTensor> {
        assert_eq!(
            sizes.iter().product::<u64>() as usize,
            data.len(),
            "input {name}: sizes {sizes:?} against {} elements",
            data.len()
        );
        self.note_tensor("in", name, sizes, SYN_TYPE_BF16, "");
        let (t, cname) = make_tensor(self.graph, name, sizes, SYN_TYPE_BF16, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(data);
        self.raw.push(None);
        Ok(t)
    }

    /// A persistent input tensor of another dtype whose bytes are uploaded
    /// as given (index tensors for scatter/gather probes).
    pub fn input_raw(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
        bytes: &[u8],
    ) -> Result<synTensor> {
        self.note_tensor("in", name, sizes, dtype, "");
        let (t, cname) = make_tensor(self.graph, name, sizes, dtype, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(std::borrow::Cow::Owned(Vec::new()));
        self.raw.push(Some(bytes.to_vec()));
        Ok(t)
    }

    /// A persistent bf16 tensor that gets its own device buffer at launch but
    /// is neither uploaded nor read back (device-resident intermediate).
    pub fn scratch(&mut self, name: &str, sizes: &[u64]) -> Result<synTensor> {
        self.scratch_typed(name, sizes, SYN_TYPE_BF16)
    }

    fn scratch_typed(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
    ) -> Result<synTensor> {
        self.note_tensor("scratch", name, sizes, dtype, "");
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, self.graph));
        syn!(synSectionSetPersistent(sec, true));
        let (t, cname) = make_tensor_in(self.graph, name, sizes, dtype, sec)?;
        self.scratch_sections.insert(name.to_owned(), sec);
        self.scratch_names.push(cname);
        self.scratch_sizes.push(sizes.to_vec());
        self.scratch_f32.push(dtype == SYN_TYPE_F32);
        self.scratch_alias.push(None);
        Ok(t)
    }

    /// A persistent bf16 tensor in the same section (same memory) as the
    /// scratch tensor named `of`: the output side of an in-place update.
    ///
    /// # Panics
    ///
    /// Panics if `of` is not a scratch tensor of this graph.
    pub fn scratch_alias(&mut self, name: &str, sizes: &[u64], of: &str) -> Result<synTensor> {
        let sec = *self
            .scratch_sections
            .get(of)
            .unwrap_or_else(|| panic!("no scratch tensor named {of}"));
        self.note_tensor("alias", name, sizes, SYN_TYPE_BF16, of);
        let (t, cname) = make_tensor_in(self.graph, name, sizes, SYN_TYPE_BF16, sec)?;
        self.scratch_names.push(cname);
        self.scratch_sizes.push(sizes.to_vec());
        self.scratch_f32.push(false);
        self.scratch_alias.push(Some(of.to_owned()));
        Ok(t)
    }

    /// A graph-internal tensor. Diagnostic `RENG_PERSIST_ALL` makes every
    /// intermediate a persistent scratch tensor instead, so a run can be
    /// dumped tensor by tensor (`RENG_DUMP_SCRATCH`).
    pub fn mid(&mut self, name: &str, sizes: &[u64], dtype: core::ffi::c_int) -> Result<synTensor> {
        if env_on("RENG_PERSIST_ALL") {
            return self.scratch_typed(name, sizes, dtype);
        }
        self.note_tensor("mid", name, sizes, dtype, "");
        Ok(make_tensor(self.graph, name, sizes, dtype, false)?.0)
    }

    /// Append a node and return its id (for explicit dependency edges).
    pub fn node(
        &mut self,
        guid: &str,
        name: &str,
        ins: &[synTensor],
        outs: &[synTensor],
        params: *const c_void,
        params_size: u32,
    ) -> Result<synNodeId> {
        let g = CString::new(guid).unwrap();
        let n = CString::new(name).unwrap();
        // Digest: guid, name, operand tensor names, raw params.
        let mut b = Vec::new();
        b.extend_from_slice(guid.as_bytes());
        b.push(0);
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        for (which, list) in [(b'i', ins), (b'o', outs)] {
            for &t in list {
                let mut buf = [0u8; 256];
                syn!(synTensorGetName(
                    t,
                    buf.len() as u64,
                    buf.as_mut_ptr().cast::<core::ffi::c_char>()
                ));
                let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
                b.push(which);
                b.extend_from_slice(&buf[..end]);
                b.push(0);
            }
        }
        if params_size > 0 {
            // SAFETY: the caller passes `params_size` readable bytes.
            b.extend_from_slice(unsafe {
                core::slice::from_raw_parts(params.cast::<u8>(), params_size as usize)
            });
        }
        self.note(&b);
        let mut id: synNodeId = 0;
        syn!(synNodeCreateWithId(
            self.graph,
            ins.as_ptr(),
            outs.as_ptr(),
            ins.len() as u32,
            outs.len() as u32,
            params,
            params_size,
            g.as_ptr(),
            n.as_ptr(),
            &mut id,
            core::ptr::null(),
            core::ptr::null(),
        ));
        self.node_ids.push(id);
        Ok(id)
    }

    /// Diagnostic (`RENG_SERIALIZE`): an explicit control dependency from every
    /// node to the next one in creation order, forcing sequential execution.
    pub fn serialize_if_requested(&mut self) -> Result<()> {
        if !env_on("RENG_SERIALIZE") {
            return Ok(());
        }
        self.note(b"RENG_SERIALIZE");
        for w in self.node_ids.windows(2) {
            syn!(synNodeDependencySet(self.graph, &w[0], &w[1], 1, 1));
        }
        Ok(())
    }
}

/// Shared per-graph tensors a layer needs besides its own weights.
pub(crate) struct Shared {
    pub sin: synTensor,
    pub cos: synTensor,
    /// Additive mask laid out like the score matrix, `[keys, queries, 1, 1]`
    /// (broadcast over heads).
    pub mask: Option<synTensor>,
    /// KV-cache capacity in positions. When set, every layer gets cache
    /// input and output tensors (see [`cache_names`]) of `capacity + 1`
    /// positions (the last is a trash slot padded rows are written to),
    /// which is also the key axis of the scores.
    pub cache: Option<usize>,
    /// The int32 scatter indices input, `[3, tokens * groups]` device sizes:
    /// update `r + tokens * g` (row r of KV head g) goes to ONNX index
    /// `(g, 0, position)` of the `[hd, keys, 1, groups]` cache.
    pub kidx: Option<synTensor>,
    /// Whether the cache update writes in place (output aliasing input).
    /// In-place ScatterND runs its rows serially, so a block of many rows
    /// (prefill) is written into a separate output buffer that the caller
    /// copies back; a one-row decode step stays in place.
    pub inplace: bool,
}

/// Persistent tensor names of layer `li`'s cache state:
/// `(k_cache_in, v_cache_in, k_cache_out, v_cache_out)`, each
/// `[head_dim, capacity + 1, 1, n_kv_heads]` (keys after RoPE); the "out"
/// tensors alias the "in" ones (in-place scatter).
#[must_use]
pub fn cache_names(li: usize) -> (String, String, String, String) {
    (
        format!("l{li}_kci"),
        format!("l{li}_vci"),
        format!("l{li}_kco"),
        format!("l{li}_vco"),
    )
}

/// `synTransposeParams`: `permutation[i]` is the source dim of output dim
/// `i` (five entries), then the tensor's dim count.
#[repr(C)]
struct TransposeParams {
    permutation: [u32; 5],
    tensor_dim: u32,
}

/// `ns_ScatterNdUpdateKernel::Params`: 0 = non-deterministic on duplicate
/// indices (the padded rows all hit the trash slot, whose value is unused).
#[repr(C)]
struct ScatterNdUpdateParams {
    mode: i32,
}

/// Append one decoder layer reading `x` and return its output tensor. When
/// `out` is given (a persistent tensor) the layer writes into it instead of a
/// graph-internal tensor.
///
/// Attention is batched over heads: with `hpg = n_heads / n_kv_heads` query
/// heads per KV group, queries live in `[hd, tokens, hpg, groups]` and keys
/// and values in `[hd, keys, 1, groups]`, so one `batch_gemm` per step
/// serves every head (the size-1 dim broadcasts a KV head over its group).
/// Projections are `batch_gemm`s of the normalised input (broadcast) against
/// per-head weight blocks, RoPE runs on the batched tensors with one 2-D
/// table, and the per-head outputs go through one transpose back to
/// `[hidden, tokens]` for the output projection.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn build_layer<'a>(
    gb: &mut Gb<'a>,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'a>,
    sh: &Shared,
    tokens: usize,
    hidden: usize,
    inter: usize,
    out: Option<synTensor>,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && w.head_dim >= 1 && nkv >= 1 && nh % nkv == 0);
    let hd_us = w.head_dim;
    let hpg_us = nh / nkv;
    // Width of the query projection (all heads); `hidden` unless the
    // config decouples `head_dim` from `hidden / n_heads`.
    let qw_us = nh * hd_us;
    let (t, h, i, hd, hpg, groups, qw) = (
        tokens as u64,
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        qw_us as u64,
    );
    // Key axis of the score matrices: the block alone, or the whole cache
    // (its usable positions plus the trash slot).
    let keys = sh.cache.map_or(t, |c| c as u64 + 1);
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");

    // Weights are the checkpoint's bf16 `[out, in]` matrices, borrowed:
    // as the gemms' transposed B operand (`[K = in, N = out]`) they need
    // no copy; the per-head form is a free reshape of the same matrix.
    // Every MME node runs at the same rate whatever its N, except that a
    // batch_gemm over per-head blocks runs each head as an N = hd gemm at
    // a fraction of the rate (12% of a 1.7B prefill), so the plain form is
    // the default and diagnostic `RENG_HEAD_BLOCKS` keeps the per-head one.
    let head_blocks = env_on("RENG_HEAD_BLOCKS");
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    // With a per-head q norm the scale cannot ride on `wq` (the norm would
    // divide it out again); it goes on the norm gain instead.
    let q_scale = if w.qn.is_empty() { w.scale } else { 1.0 };
    let wq_scaled = scaled_wq(w);
    let (t_wq, t_wk, t_wv) = if head_blocks {
        (
            gb.input_bf16(&p("wq"), &[h, hd, hpg, groups], wq_scaled)?,
            gb.input_bf16(
                &p("wk"),
                &[h, hd, 1, groups],
                std::borrow::Cow::Borrowed(w.wk),
            )?,
            gb.input_bf16(
                &p("wv"),
                &[h, hd, 1, groups],
                std::borrow::Cow::Borrowed(w.wv),
            )?,
        )
    } else {
        (
            gb.input_bf16(&p("wq2"), &[h, qw], wq_scaled)?,
            gb.input_bf16(
                &p("wk2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wk),
            )?,
            gb.input_bf16(
                &p("wv2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wv),
            )?,
        )
    };
    let t_wo = gb.input_bf16(&p("wo"), &[qw, h], std::borrow::Cow::Borrowed(w.wo))?;
    let t_wg = gb.input_bf16(&p("wg"), &[h, i], std::borrow::Cow::Borrowed(w.wg))?;
    let t_wu = gb.input_bf16(&p("wu"), &[h, i], std::borrow::Cow::Borrowed(w.wu))?;
    let t_wd = gb.input_bf16(&p("wd"), &[i, h], std::borrow::Cow::Borrowed(w.wd))?;

    let t_n1 = gb.mid(&p("n1"), &[h, t], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, t], SYN_TYPE_F32)?;
    let t_n1_4 = gb.mid(&p("n1_4"), &[h, t, 1, 1], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, t, hpg, groups], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, t, 1, groups], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, t, 1, groups], bf)?;
    let t_sc = gb.mid(&p("scores"), &[keys, t, hpg, groups], bf)?;
    let t_pr = gb.mid(&p("probs"), &[keys, t, hpg, groups], bf)?;
    let t_at = gb.mid(&p("at"), &[hd, t, hpg, groups], bf)?;
    let t_at3 = gb.mid(&p("at3"), &[hd, t, hpg * groups], bf)?;
    let t_att = gb.mid(&p("att"), &[hd, hpg * groups, t], bf)?;
    let t_attn = gb.mid(&p("attn"), &[qw, t], bf)?;
    let t_o = gb.mid(&p("o"), &[h, t], bf)?;
    let t_h = gb.mid(&p("h"), &[h, t], bf)?;
    let t_n2 = gb.mid(&p("n2"), &[h, t], bf)?;
    let t_inv2 = gb.mid(&p("inv2"), &[1, t], SYN_TYPE_F32)?;
    let t_gate = gb.mid(&p("gate"), &[i, t], bf)?;
    let t_up = gb.mid(&p("up"), &[i, t], bf)?;
    let t_sg = gb.mid(&p("sg"), &[i, t], bf)?;
    let t_silu = gb.mid(&p("silu"), &[i, t], bf)?;
    let t_gated = gb.mid(&p("gated"), &[i, t], bf)?;
    let t_down = gb.mid(&p("down"), &[h, t], bf)?;
    let t_out = match out {
        Some(o) => o,
        None => gb.mid(&p("out"), &[h, t], bf)?,
    };

    let rms = RmsNormParams {
        epsilon: w.eps,
        fused_gamma_beta: false,
        use_stages: false,
        _pad: [0; 2],
        bwd_mode: 0,
    };
    let rope = RopeParams { offset: 0, mode: 0 };
    let gemm = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let sm = synSoftmaxParams { dim: 0 };
    let tr = TransposeParams {
        permutation: [0, 2, 1, 0, 0],
        tensor_dim: 3,
    };
    let prm = (
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    );
    let pr = (
        (&raw const rope).cast::<c_void>(),
        core::mem::size_of::<RopeParams>() as u32,
    );
    let pg = (
        (&raw const gemm).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let pgt = (
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let ps = (
        (&raw const sm).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
    );
    let ptr_ = (
        (&raw const tr).cast::<c_void>(),
        core::mem::size_of::<TransposeParams>() as u32,
    );
    let scatter = ScatterNdUpdateParams { mode: 0 };
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let none = (core::ptr::null::<c_void>(), 0u32);

    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm1"),
        &[x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    if head_blocks {
        gb.node("reshape", &p("n1_4d"), &[t_n1], &[t_n1_4], none.0, none.1)?;
        gb.node(
            "batch_gemm",
            &p("q_proj"),
            &[t_n1_4, t_wq],
            &[t_q],
            pgt.0,
            pgt.1,
        )?;
        gb.node(
            "batch_gemm",
            &p("k_proj"),
            &[t_n1_4, t_wk],
            &[t_k],
            pgt.0,
            pgt.1,
        )?;
        gb.node(
            "batch_gemm",
            &p("v_proj"),
            &[t_n1_4, t_wv],
            &[t_v],
            pgt.0,
            pgt.1,
        )?;
    } else {
        // `[features, t]` is `[hd, heads, t]`; the head layout wants the
        // token dim inside the heads, so each projection ends in a
        // transpose of its two outer dims (the inverse of `heads_last`).
        let tr_in = TransposeParams {
            permutation: [0, 2, 1, 0, 0],
            tensor_dim: 3,
        };
        let ptr_in = (
            (&raw const tr_in).cast::<c_void>(),
            core::mem::size_of::<TransposeParams>() as u32,
        );
        for (name, wt, n_out, heads, out) in [
            ("q", t_wq, qw, hpg * groups, t_q),
            ("k", t_wk, hd * groups, groups, t_k),
            ("v", t_wv, hd * groups, groups, t_v),
        ] {
            let flat = gb.mid(&p(&format!("{name}2")), &[n_out, t], bf)?;
            let by_head = gb.mid(&p(&format!("{name}_heads")), &[hd, heads, t], bf)?;
            let tokens_in = gb.mid(&p(&format!("{name}_tokens")), &[hd, t, heads], bf)?;
            gb.node(
                "gemm",
                &p(&format!("{name}_proj")),
                &[t_n1, wt],
                &[flat],
                pgt.0,
                pgt.1,
            )?;
            gb.node(
                "reshape",
                &p(&format!("{name}_3d")),
                &[flat],
                &[by_head],
                none.0,
                none.1,
            )?;
            gb.node(
                "transpose",
                &p(&format!("{name}_tokens_in")),
                &[by_head],
                &[tokens_in],
                ptr_in.0,
                ptr_in.1,
            )?;
            gb.node(
                "reshape",
                &p(&format!("{name}_4d")),
                &[tokens_in],
                &[out],
                none.0,
                none.1,
            )?;
        }
    }
    // Attention biases (when present) broadcast over the token dim.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input(&p("bq"), &[hd, 1, hpg, groups], &bq_scaled)?;
        let t_bk = gb.input(&p("bk"), &[hd, 1, 1, groups], w.bk)?;
        let t_bv = gb.input(&p("bv"), &[hd, 1, 1, groups], w.bv)?;
        let qb = gb.mid(&p("qb"), &[hd, t, hpg, groups], bf)?;
        let kb = gb.mid(&p("kb"), &[hd, t, 1, groups], bf)?;
        let vb = gb.mid(&p("vb"), &[hd, t, 1, groups], bf)?;
        gb.node(
            "add_fwd_bf16",
            &p("q_bias"),
            &[t_q, t_bq],
            &[qb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("k_bias"),
            &[t_k, t_bk],
            &[kb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("v_bias"),
            &[t_v, t_bv],
            &[vb],
            none.0,
            none.1,
        )?;
        (qb, kb, vb)
    };
    // Qwen3 q/k norms (when present): an RMSNorm over the head dim (the
    // FCD) of every query and key head, the gain `[hd]` broadcast over the
    // outer dims, exactly like the layer norms over `[hidden, t]`.
    let (t_q, t_k) = if w.qn.is_empty() {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * w.scale).collect();
        let t_qn = gb.input(&p("qn"), &[hd], &qn_scaled)?;
        let t_kn = gb.input(&p("kn"), &[hd], w.kn)?;
        let qn = gb.mid(&p("qn_out"), &[hd, t, hpg, groups], bf)?;
        let qn_inv = gb.mid(&p("qn_inv"), &[1, t, hpg, groups], SYN_TYPE_F32)?;
        let kn = gb.mid(&p("kn_out"), &[hd, t, 1, groups], bf)?;
        let kn_inv = gb.mid(&p("kn_inv"), &[1, t, 1, groups], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("q_norm"),
            &[t_q, t_qn],
            &[qn, qn_inv],
            prm.0,
            prm.1,
        )?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("k_norm"),
            &[t_k, t_kn],
            &[kn, kn_inv],
            prm.0,
            prm.1,
        )?;
        (qn, kn)
    };
    // RoPE on q and k, or neither for a NoPE layer.
    let (t_qr, t_kr) = if w.use_rope {
        let t_qr = gb.mid(&p("qr"), &[hd, t, hpg, groups], bf)?;
        let t_kr = gb.mid(&p("kr"), &[hd, t, 1, groups], bf)?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_q"),
            &[t_q, sh.sin, sh.cos],
            &[t_qr],
            pr.0,
            pr.1,
        )?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_k"),
            &[t_k, sh.sin, sh.cos],
            &[t_kr],
            pr.0,
            pr.1,
        )?;
        (t_qr, t_kr)
    } else {
        (t_q, t_k)
    };

    // Keys and values attention reads: the block's own, or the cache updated
    // in place with the block: an ONNX ScatterND update writes each real row
    // of the rotated keys and of the values at its position (padded rows go
    // to the trash slot); the "out" tensor aliases the "in" one, so only the
    // written rows move (see cached.rs).
    let (k_full, v_full) = if let Some(kidx) = sh.kidx {
        let (n_kci, n_vci, n_kco, n_vco) = cache_names(li);
        let kci = gb.scratch(&n_kci, &[hd, keys, 1, groups])?;
        let vci = gb.scratch(&n_vci, &[hd, keys, 1, groups])?;
        let (kco, vco) = if sh.inplace {
            (
                gb.scratch_alias(&n_kco, &[hd, keys, 1, groups], &n_kci)?,
                gb.scratch_alias(&n_vco, &[hd, keys, 1, groups], &n_vci)?,
            )
        } else {
            (
                gb.scratch(&n_kco, &[hd, keys, 1, groups])?,
                gb.scratch(&n_vco, &[hd, keys, 1, groups])?,
            )
        };
        let kru = gb.mid(&p("kru"), &[hd, t * groups], bf)?;
        let vu = gb.mid(&p("vu"), &[hd, t * groups], bf)?;
        gb.node("reshape", &p("kr_updates"), &[t_kr], &[kru], none.0, none.1)?;
        gb.node("reshape", &p("v_updates"), &[t_v], &[vu], none.0, none.1)?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("k_scatter"),
            &[kci, kidx, kru],
            &[kco],
            psc.0,
            psc.1,
        )?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("v_scatter"),
            &[vci, kidx, vu],
            &[vco],
            psc.0,
            psc.1,
        )?;
        (kco, vco)
    } else {
        (t_kr, t_v)
    };
    // scores[key, query] per head = q @ k^T with K in its natural layout.
    gb.node(
        "batch_gemm",
        &p("qk"),
        &[t_qr, k_full],
        &[t_sc],
        pgt.0,
        pgt.1,
    )?;
    let sm_in = if let Some(mask) = sh.mask {
        let masked = gb.mid(&p("masked"), &[keys, t, hpg, groups], bf)?;
        gb.node(
            "add_fwd_bf16",
            &p("mask"),
            &[t_sc, mask],
            &[masked],
            none.0,
            none.1,
        )?;
        masked
    } else {
        t_sc
    };
    gb.node(
        "softmax_fwd_bf16",
        &p("softmax"),
        &[sm_in],
        &[t_pr],
        ps.0,
        ps.1,
    )?;
    gb.node("batch_gemm", &p("av"), &[t_pr, v_full], &[t_at], pg.0, pg.1)?;
    // [hd, t, hpg, groups] -> [hd, t, heads] -> [hd, heads, t] = [hidden, t].
    gb.node("reshape", &p("at_3d"), &[t_at], &[t_at3], none.0, none.1)?;
    gb.node(
        "transpose",
        &p("heads_last"),
        &[t_at3],
        &[t_att],
        ptr_.0,
        ptr_.1,
    )?;
    gb.node(
        "reshape",
        &p("attn_2d"),
        &[t_att],
        &[t_attn],
        none.0,
        none.1,
    )?;
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
    gb.node(
        "add_fwd_bf16",
        &p("res1"),
        &[x, t_o],
        &[t_h],
        none.0,
        none.1,
    )?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm2"),
        &[t_h, t_g2],
        &[t_n2, t_inv2],
        prm.0,
        prm.1,
    )?;
    gb.node(
        "gemm",
        &p("gate_proj"),
        &[t_n2, t_wg],
        &[t_gate],
        pgt.0,
        pgt.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_n2, t_wu], &[t_up], pgt.0, pgt.1)?;
    gb.node(
        "sigmoid_fwd_bf16",
        &p("sig"),
        &[t_gate],
        &[t_sg],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("silu"),
        &[t_gate, t_sg],
        &[t_silu],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gate_x_up"),
        &[t_silu, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    gb.node(
        "gemm",
        &p("down_proj"),
        &[t_gated, t_wd],
        &[t_down],
        pgt.0,
        pgt.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("res2"),
        &[t_h, t_down],
        &[t_out],
        none.0,
        none.1,
    )?;
    Ok(t_out)
}

/// Per-graph tensors of the batched decode layer: one row per sequence,
/// everything per-sequence carried in the outermost (fifth) dimension.
pub(crate) struct SharedBatched {
    /// RoPE rows per sequence, `[hd, 1, 1, 1, B]`.
    pub sin: synTensor,
    pub cos: synTensor,
    /// Additive mask per sequence, `[keys, 1, 1, 1, B]`.
    pub mask: synTensor,
    /// Int32 scatter indices, `[4, groups * B]`: update `g + groups * b`
    /// goes to ONNX index `(b, g, 0, position_b)` of the
    /// `[hd, keys, 1, groups, B]` cache.
    pub kidx: synTensor,
    pub capacity: usize,
    pub batch: usize,
}

/// Append one decoder layer for `B` sequences of one token each, reading `x`
/// (`[hidden, B]`) and returning the layer output in the same shape. The
/// attention path is the 4-D batched one of [`build_layer`] with the
/// sequence batch as a fifth, outermost dimension: the projections are 2-D
/// gemms with `M = B` whose `[hidden, B]` outputs are free reshapes of the
/// head layout, every sequence has its own RoPE row, cache
/// slot, placement column and mask, and with one query row per sequence the
/// context `[hd, 1, hpg, groups, B]` already is `[hidden, B]` in memory.
/// Weights carry the same names and element counts as in [`build_layer`],
/// so a runtime can share them with a prefill recipe.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn build_layer_batched<'a>(
    gb: &mut Gb<'a>,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'a>,
    sh: &SharedBatched,
    hidden: usize,
    inter: usize,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && w.head_dim >= 1 && nkv >= 1 && nh % nkv == 0);
    let hd_us = w.head_dim;
    let hpg_us = nh / nkv;
    let (h, i, hd, hpg, groups, keys, b, qw) = (
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        sh.capacity as u64 + 1,
        sh.batch as u64,
        (nh * hd_us) as u64,
    );
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");
    // See `build_layer`: the scale rides on the q norm gain when there is one.
    let q_scale = if w.qn.is_empty() { w.scale } else { 1.0 };
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    // With one row per sequence the projections are plain gemms with
    // M = B over the natural `[in, out]` weights: `[hidden, B]` is already
    // `[hd, 1, hpg, groups, B]` in memory (head-major features, sequence
    // outermost), so the head layout is a free reshape. These weights are
    // laid out differently from the wide recipe's per-head blocks, so they
    // get their own names and buffers.
    let t_wq = gb.input_bf16(&p("wq2"), &[h, qw], scaled_wq(w))?;
    let t_wk = gb.input_bf16(
        &p("wk2"),
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wk),
    )?;
    let t_wv = gb.input_bf16(
        &p("wv2"),
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wv),
    )?;
    let t_wo = gb.input_bf16(&p("wo"), &[qw, h], std::borrow::Cow::Borrowed(w.wo))?;
    let t_wg = gb.input_bf16(&p("wg"), &[h, i], std::borrow::Cow::Borrowed(w.wg))?;
    let t_wu = gb.input_bf16(&p("wu"), &[h, i], std::borrow::Cow::Borrowed(w.wu))?;
    let t_wd = gb.input_bf16(&p("wd"), &[i, h], std::borrow::Cow::Borrowed(w.wd))?;

    let t_n1 = gb.mid(&p("n1"), &[h, b], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, b], SYN_TYPE_F32)?;
    let t_q2 = gb.mid(&p("q2"), &[qw, b], bf)?;
    let t_k2 = gb.mid(&p("k2"), &[hd * groups, b], bf)?;
    let t_v2 = gb.mid(&p("v2"), &[hd * groups, b], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, 1, hpg, groups, b], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, 1, 1, groups, b], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, 1, 1, groups, b], bf)?;
    let (n_kci, n_vci, n_kco, n_vco) = cache_names(li);
    let kci = gb.scratch(&n_kci, &[hd, keys, 1, groups, b])?;
    let vci = gb.scratch(&n_vci, &[hd, keys, 1, groups, b])?;
    let kco = gb.scratch_alias(&n_kco, &[hd, keys, 1, groups, b], &n_kci)?;
    let vco = gb.scratch_alias(&n_vco, &[hd, keys, 1, groups, b], &n_vci)?;
    let t_kru = gb.mid(&p("kru"), &[hd, groups * b], bf)?;
    let t_vu = gb.mid(&p("vu"), &[hd, groups * b], bf)?;
    let t_sc = gb.mid(&p("scores"), &[keys, 1, hpg, groups, b], bf)?;
    let t_masked = gb.mid(&p("masked"), &[keys, 1, hpg, groups, b], bf)?;
    let t_pr = gb.mid(&p("probs"), &[keys, 1, hpg, groups, b], bf)?;
    let t_at = gb.mid(&p("at"), &[hd, 1, hpg, groups, b], bf)?;
    let t_attn = gb.mid(&p("attn"), &[qw, b], bf)?;
    let t_o = gb.mid(&p("o"), &[h, b], bf)?;
    let t_h = gb.mid(&p("h"), &[h, b], bf)?;
    let t_n2 = gb.mid(&p("n2"), &[h, b], bf)?;
    let t_inv2 = gb.mid(&p("inv2"), &[1, b], SYN_TYPE_F32)?;
    let t_gate = gb.mid(&p("gate"), &[i, b], bf)?;
    let t_up = gb.mid(&p("up"), &[i, b], bf)?;
    let t_sg = gb.mid(&p("sg"), &[i, b], bf)?;
    let t_silu = gb.mid(&p("silu"), &[i, b], bf)?;
    let t_gated = gb.mid(&p("gated"), &[i, b], bf)?;
    let t_down = gb.mid(&p("down"), &[h, b], bf)?;
    let t_out = gb.mid(&p("out"), &[h, b], bf)?;

    let rms = RmsNormParams {
        epsilon: w.eps,
        fused_gamma_beta: false,
        use_stages: false,
        _pad: [0; 2],
        bwd_mode: 0,
    };
    let rope = RopeParams { offset: 0, mode: 0 };
    let gemm = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let sm = synSoftmaxParams { dim: 0 };
    let prm = (
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    );
    let pr = (
        (&raw const rope).cast::<c_void>(),
        core::mem::size_of::<RopeParams>() as u32,
    );
    let pg = (
        (&raw const gemm).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let pgt = (
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let ps = (
        (&raw const sm).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
    );
    let scatter = ScatterNdUpdateParams { mode: 0 };
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let none = (core::ptr::null::<c_void>(), 0u32);

    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm1"),
        &[x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    gb.node("gemm", &p("q_proj"), &[t_n1, t_wq], &[t_q2], pgt.0, pgt.1)?;
    gb.node("gemm", &p("k_proj"), &[t_n1, t_wk], &[t_k2], pgt.0, pgt.1)?;
    gb.node("gemm", &p("v_proj"), &[t_n1, t_wv], &[t_v2], pgt.0, pgt.1)?;
    gb.node("reshape", &p("q_5d"), &[t_q2], &[t_q], none.0, none.1)?;
    gb.node("reshape", &p("k_5d"), &[t_k2], &[t_k], none.0, none.1)?;
    gb.node("reshape", &p("v_5d"), &[t_v2], &[t_v], none.0, none.1)?;
    // Attention biases (when present) broadcast over the sequence batch.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input(&p("bq"), &[hd, 1, hpg, groups, 1], &bq_scaled)?;
        let t_bk = gb.input(&p("bk"), &[hd, 1, 1, groups, 1], w.bk)?;
        let t_bv = gb.input(&p("bv"), &[hd, 1, 1, groups, 1], w.bv)?;
        let qb = gb.mid(&p("qb"), &[hd, 1, hpg, groups, b], bf)?;
        let kb = gb.mid(&p("kb"), &[hd, 1, 1, groups, b], bf)?;
        let vb = gb.mid(&p("vb"), &[hd, 1, 1, groups, b], bf)?;
        gb.node(
            "add_fwd_bf16",
            &p("q_bias"),
            &[t_q, t_bq],
            &[qb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("k_bias"),
            &[t_k, t_bk],
            &[kb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("v_bias"),
            &[t_v, t_bv],
            &[vb],
            none.0,
            none.1,
        )?;
        (qb, kb, vb)
    };
    // Qwen3 q/k norms (when present), as in `build_layer`.
    let (t_q, t_k) = if w.qn.is_empty() {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * w.scale).collect();
        let t_qn = gb.input(&p("qn"), &[hd], &qn_scaled)?;
        let t_kn = gb.input(&p("kn"), &[hd], w.kn)?;
        let qn = gb.mid(&p("qn_out"), &[hd, 1, hpg, groups, b], bf)?;
        let qn_inv = gb.mid(&p("qn_inv"), &[1, 1, hpg, groups, b], SYN_TYPE_F32)?;
        let kn = gb.mid(&p("kn_out"), &[hd, 1, 1, groups, b], bf)?;
        let kn_inv = gb.mid(&p("kn_inv"), &[1, 1, 1, groups, b], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("q_norm"),
            &[t_q, t_qn],
            &[qn, qn_inv],
            prm.0,
            prm.1,
        )?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("k_norm"),
            &[t_k, t_kn],
            &[kn, kn_inv],
            prm.0,
            prm.1,
        )?;
        (qn, kn)
    };
    let (t_qr, t_kr) = if w.use_rope {
        let t_qr = gb.mid(&p("qr"), &[hd, 1, hpg, groups, b], bf)?;
        let t_kr = gb.mid(&p("kr"), &[hd, 1, 1, groups, b], bf)?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_q"),
            &[t_q, sh.sin, sh.cos],
            &[t_qr],
            pr.0,
            pr.1,
        )?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_k"),
            &[t_k, sh.sin, sh.cos],
            &[t_kr],
            pr.0,
            pr.1,
        )?;
        (t_qr, t_kr)
    } else {
        (t_q, t_k)
    };
    gb.node(
        "reshape",
        &p("kr_updates"),
        &[t_kr],
        &[t_kru],
        none.0,
        none.1,
    )?;
    gb.node("reshape", &p("v_updates"), &[t_v], &[t_vu], none.0, none.1)?;
    // Diagnostic `RENG_NO_SCATTER`: leave the cache stale (wrong results) to
    // time the step without the two ScatterND nodes.
    let (kco, vco) = if env_on("RENG_NO_SCATTER") {
        (kci, vci)
    } else {
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("k_scatter"),
            &[kci, sh.kidx, t_kru],
            &[kco],
            psc.0,
            psc.1,
        )?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("v_scatter"),
            &[vci, sh.kidx, t_vu],
            &[vco],
            psc.0,
            psc.1,
        )?;
        (kco, vco)
    };
    gb.node("batch_gemm", &p("qk"), &[t_qr, kco], &[t_sc], pgt.0, pgt.1)?;
    gb.node(
        "add_fwd_bf16",
        &p("mask"),
        &[t_sc, sh.mask],
        &[t_masked],
        none.0,
        none.1,
    )?;
    gb.node(
        "softmax_fwd_bf16",
        &p("softmax"),
        &[t_masked],
        &[t_pr],
        ps.0,
        ps.1,
    )?;
    gb.node("batch_gemm", &p("av"), &[t_pr, vco], &[t_at], pg.0, pg.1)?;
    gb.node("reshape", &p("attn_2d"), &[t_at], &[t_attn], none.0, none.1)?;
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
    gb.node(
        "add_fwd_bf16",
        &p("res1"),
        &[x, t_o],
        &[t_h],
        none.0,
        none.1,
    )?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm2"),
        &[t_h, t_g2],
        &[t_n2, t_inv2],
        prm.0,
        prm.1,
    )?;
    gb.node(
        "gemm",
        &p("gate_proj"),
        &[t_n2, t_wg],
        &[t_gate],
        pgt.0,
        pgt.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_n2, t_wu], &[t_up], pgt.0, pgt.1)?;
    gb.node(
        "sigmoid_fwd_bf16",
        &p("sig"),
        &[t_gate],
        &[t_sg],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("silu"),
        &[t_gate, t_sg],
        &[t_silu],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gate_x_up"),
        &[t_silu, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    gb.node(
        "gemm",
        &p("down_proj"),
        &[t_gated, t_wd],
        &[t_down],
        pgt.0,
        pgt.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("res2"),
        &[t_h, t_down],
        &[t_out],
        none.0,
        none.1,
    )?;
    Ok(t_out)
}

/// `ns_Reduction::Params` for the argmax over the FCD.
#[repr(C)]
struct ReductionParams {
    reduction_dimension: u32,
}

/// Append the final RMSNorm and LM head reading `cur`. Without `ids_out`
/// the logits `[vocab, tokens]` are the graph's read-back tensor; with it
/// the logits stay device-resident (the persistent scratch tensor
/// `LOGITS`, readable on demand) and an argmax over the vocabulary produces
/// the read-back tensor `IDS`, int32 `[1, tokens]`, so a decode step moves
/// four bytes per token over the bus.
pub(crate) fn build_head<'a>(
    gb: &mut Gb<'a>,
    cur: synTensor,
    m: &ModelWeights<'a>,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    ids_out: bool,
) -> Result<Out> {
    let (t, h, v) = (tokens as u64, hidden as u64, vocab as u64);
    let bf = SYN_TYPE_BF16;
    let t_gf = gb.input("GF", &[h], m.final_gamma)?;
    let t_lm = gb.input_bf16("LM", &[h, v], std::borrow::Cow::Borrowed(m.lm_head))?;
    let t_nf = gb.mid("nf", &[h, t], bf)?;
    let t_invf = gb.mid("invf", &[1, t], SYN_TYPE_F32)?;
    let (t_logits, n_logits) = if ids_out {
        let t = gb.scratch("LOGITS", &[v, t])?;
        (t, CString::new("LOGITS").unwrap())
    } else {
        gb.output("LOGITS", &[v, t], bf)?
    };
    // Diagnostic `RENG_TPC_OUT`: route the logits through a TPC identity
    // (add 0) so a TPC kernel, not the MME, is the output's last writer.
    let tpc_out = env_on("RENG_TPC_OUT");
    let t_lg = if tpc_out {
        gb.mid("lg", &[v, t], bf)?
    } else {
        t_logits
    };
    let rms = RmsNormParams {
        epsilon: m.layers[0].eps,
        fused_gamma_beta: false,
        use_stages: false,
        _pad: [0; 2],
        bwd_mode: 0,
    };
    gb.node(
        "rms_norm_fwd_bf16",
        "final_norm",
        &[cur, t_gf],
        &[t_nf, t_invf],
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    )?;
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    gb.node(
        "gemm",
        "lm_head",
        &[t_nf, t_lm],
        &[t_lg],
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    if tpc_out {
        let t_zero = gb.input("ZERO", &[v, t], &vec![0.0; vocab * tokens])?;
        gb.node(
            "add_fwd_bf16",
            "logits_out",
            &[t_lg, t_zero],
            &[t_logits],
            core::ptr::null(),
            0,
        )?;
    }
    if !ids_out {
        return Ok(Out {
            name: n_logits,
            sizes: vec![v, t],
            kind: OutKind::Bf16,
        });
    }
    let (t_ids, n_ids) = gb.output("IDS", &[1, t], SYN_TYPE_INT32)?;
    let red = ReductionParams {
        reduction_dimension: 0,
    };
    // `argmax_fwd_bf16` is wrong for a single-row input (the decode shape)
    // whenever the row's maximum is small or negative: it returns 0 or an
    // index past the end (Phi-3 at 200 to 300 tokens; reng-argmax-test).
    // The f32 kernel is right in every case, so cast first.
    let t_lf32 = gb.mid("logits_f32", &[v, t], SYN_TYPE_F32)?;
    gb.node(
        "cast_bf16_to_f32",
        "logits_cast",
        &[t_logits],
        &[t_lf32],
        core::ptr::null(),
        0,
    )?;
    gb.node(
        "argmax_fwd_f32",
        "argmax",
        &[t_lf32],
        &[t_ids],
        (&raw const red).cast::<c_void>(),
        core::mem::size_of::<ReductionParams>() as u32,
    )?;
    Ok(Out {
        name: n_ids,
        sizes: vec![1, t],
        kind: OutKind::I32,
    })
}

/// Whole-model weights. All layers share `layers[0]`'s head counts, `eps`, and
/// RoPE caches (`sin`/`cos`, `[tokens, head_dim]`).
#[derive(Clone)]
pub struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    /// Final RMSNorm gain, length `hidden`.
    pub final_gamma: &'a [f32],
    /// LM head, bf16 `[vocab, hidden]` (the checkpoint's layout; tied
    /// embeddings as they are).
    pub lm_head: &'a [u16],
}

/// Build the shared inputs (activations, RoPE caches, causal mask) and the
/// decoder stack up to and including layer `upto`, returning the graph and the
/// last layer's output tensor. `probe_out` makes layer `upto` write into that
/// persistent tensor.
#[allow(clippy::too_many_arguments)]
fn build_stack<'a>(
    x: &[f32],
    m: &ModelWeights<'a>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    window: Option<usize>,
    upto: usize,
    probe_out: Option<synTensor>,
) -> Result<(Gb<'a>, synTensor)> {
    let l0 = &m.layers[0];
    let hd = l0.head_dim;
    assert_eq!(l0.sin.len(), tokens * hd);
    let (t, h, hd64) = (tokens as u64, hidden as u64, hd as u64);
    let mut gb = Gb::new()?;
    let t_x = gb.input("X", &[h, t], x)?;
    let t_sin = gb.input("SIN", &[hd64, t], l0.sin)?;
    let t_cos = gb.input("COS", &[hd64, t], l0.cos)?;
    // Causal mask laid out like the score matrix: [key (FCD), query].
    let mask_host: Vec<f32> = (0..tokens * tokens)
        .map(|idx| {
            let (q, k) = (idx / tokens, idx % tokens);
            if k <= q && window.is_none_or(|w| q - k < w) {
                0.0
            } else {
                MASK_NEG
            }
        })
        .collect();
    let t_mask = if causal {
        Some(gb.input("MASK", &[t, t, 1, 1], &mask_host)?)
    } else {
        None
    };
    let sh = Shared {
        sin: t_sin,
        cos: t_cos,
        mask: t_mask,
        cache: None,
        kidx: None,
        inplace: true,
    };
    let persist = env_on("RENG_PERSIST_LAYERS");
    let mut cur = t_x;
    for (li, lw) in m.layers.iter().enumerate().take(upto + 1) {
        let out = if li == upto && probe_out.is_some() {
            probe_out
        } else if persist {
            Some(gb.scratch(&format!("l{li}_res"), &[h, t])?)
        } else {
            None
        };
        cur = build_layer(&mut gb, li, cur, lw, &sh, tokens, hidden, inter, out)?;
    }
    Ok((gb, cur))
}

/// [`model_forward_bf16_window`] without a sliding window.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn model_forward_bf16(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    model_forward_bf16_window(x, m, tokens, hidden, inter, vocab, causal, None)
}

/// Run the full forward pass on `x` (`[tokens, hidden]` embeddings, row-major)
/// and return logits `[tokens, vocab]` as f32. With `causal`, key positions
/// after the query are masked out in every layer. `hidden`, `inter`, and
/// `vocab` should be at least 128 (per-head sizes may be smaller).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails or the output never completes.
///
/// # Panics
///
/// Panics if `layers` is empty or any buffer length disagrees with the sizes.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_bf16_window(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    assert!(!m.layers.is_empty());
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(m.final_gamma.len(), hidden);
    assert_eq!(m.lm_head.len(), hidden * vocab);
    let last = m.layers.len() - 1;
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, window, last, None)?;
    let out = build_head(&mut gb, cur, m, tokens, hidden, vocab, false)?;
    Runtime::new(gb, out)?.launch_and_read(tokens)
}

/// Diagnostic: build layers `0..=upto` and read back the residual stream after
/// layer `upto` (`[tokens, hidden]`, f32).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails or the output never completes.
///
/// # Panics
///
/// Panics if `upto >= layers.len()` or a buffer length disagrees with the sizes.
pub fn model_probe_bf16(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
) -> Result<Vec<f32>> {
    assert!(upto < m.layers.len());
    assert_eq!(x.len(), tokens * hidden);
    let (t, h) = (tokens as u64, hidden as u64);
    // Build the stack, then expose the last layer's output through a TPC
    // identity into a dedicated persistent tensor that is read back.
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, None, upto, None)?;
    let (t_probe, n_probe) = gb.output("PROBE", &[h, t], SYN_TYPE_BF16)?;
    let t_zero = gb.input("ZERO", &[h, t], &vec![0.0; hidden * tokens])?;
    gb.node(
        "add_fwd_bf16",
        "probe_out",
        &[cur, t_zero],
        &[t_probe],
        core::ptr::null(),
        0,
    )?;
    let out = Out {
        name: n_probe,
        sizes: vec![h, t],
        kind: OutKind::Bf16,
    };
    Runtime::new(gb, out)?.launch_and_read(tokens)
}

/// One decoder layer on the CPU (f32), with optional causal masking and GQA.
#[must_use]
pub fn layer_cpu(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
) -> Vec<f32> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    let hd = w.head_dim;
    let qw = nh * hd;
    let kvd = nkv * hd;
    let n_rep = nh / nkv;
    let rmsnorm = |src: &[f32], g: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; tokens * hidden];
        for tk in 0..tokens {
            let b = tk * hidden;
            let ms = src[b..b + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + w.eps).sqrt();
            for f in 0..hidden {
                o[b + f] = src[b + f] * inv * g[f];
            }
        }
        o
    };
    // a[tokens, kin] @ W^T for W stored [kout, kin] (bf16) -> [tokens, kout]
    let matmul = |a: &[f32], mtx: &[u16], kin: usize, kout: usize| -> Vec<f32> {
        let wf: Vec<f32> = mtx.iter().map(|&b| bf16_to_f32(b)).collect();
        let mut o = vec![0.0f32; tokens * kout];
        for tk in 0..tokens {
            for c in 0..kout {
                let row = &wf[c * kin..(c + 1) * kin];
                o[tk * kout + c] = a[tk * kin..(tk + 1) * kin]
                    .iter()
                    .zip(row)
                    .map(|(x, y)| x * y)
                    .sum();
            }
        }
        o
    };
    // Rotate-half RoPE on head `head` of a [tokens, stride] tensor, written
    // into `out` (same layout).
    let rope_head = |src: &[f32], stride: usize, head: usize, out: &mut [f32]| {
        let half = hd / 2;
        for tk in 0..tokens {
            let b = tk * stride + head * hd;
            let c = tk * hd;
            for d in 0..hd {
                let rot = if d < half {
                    -src[b + d + half]
                } else {
                    src[b + d - half]
                };
                out[b + d] = src[b + d] * w.cos[c + d] + rot * w.sin[c + d];
            }
        }
    };

    // Per-head RMSNorm over `hd` of every head of a `[tokens, stride]`
    // tensor (Qwen3 q/k norms), in place; a no-op without a gain.
    let head_norm = |m: &mut [f32], stride: usize, g: &[f32]| {
        if g.is_empty() {
            return;
        }
        for row in m.chunks_exact_mut(stride) {
            for head in row.chunks_exact_mut(hd) {
                let ms = head.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + w.eps).sqrt();
                for (v, gain) in head.iter_mut().zip(g) {
                    *v *= inv * gain;
                }
            }
        }
    };
    let n1 = rmsnorm(x, w.g1);
    // The scale rides on `wq` (and `bq`), or on the q norm gain when the
    // model has one (the norm would divide it out of `wq`).
    let q_scale = if w.qn.is_empty() { w.scale } else { 1.0 };
    let wq_scaled = scaled_wq(w);
    let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * w.scale).collect();
    let bias = |mut m: Vec<f32>, b: &[f32], scale: f32| -> Vec<f32> {
        if !b.is_empty() {
            let cols = b.len();
            for (i, v) in m.iter_mut().enumerate() {
                *v += b[i % cols] * scale;
            }
        }
        m
    };
    let mut q = bias(matmul(&n1, &wq_scaled, hidden, qw), w.bq, q_scale);
    let mut k = bias(matmul(&n1, w.wk, hidden, kvd), w.bk, 1.0);
    let v = bias(matmul(&n1, w.wv, hidden, kvd), w.bv, 1.0);
    head_norm(&mut q, qw, &qn_scaled);
    head_norm(&mut k, kvd, w.kn);
    let mut qr = vec![0.0f32; tokens * qw];
    let mut kr = vec![0.0f32; tokens * kvd];
    if w.use_rope {
        for head in 0..nh {
            rope_head(&q, qw, head, &mut qr);
        }
        for g in 0..nkv {
            rope_head(&k, kvd, g, &mut kr);
        }
    } else {
        qr.copy_from_slice(&q);
        kr.copy_from_slice(&k);
    }
    let mut attn = vec![0.0f32; tokens * qw];
    let mut scores = vec![0.0f32; tokens];
    for head in 0..nh {
        let g = head / n_rep;
        let qoff = head * hd;
        let koff = g * hd;
        for qi in 0..tokens {
            let limit = if causal { qi + 1 } else { tokens };
            for (ki, s) in scores.iter_mut().enumerate().take(limit) {
                *s = (0..hd)
                    .map(|d| qr[qi * qw + qoff + d] * kr[ki * kvd + koff + d])
                    .sum();
            }
            let mx = scores[..limit]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores[..limit] {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for (ki, s) in scores[..limit].iter().enumerate() {
                let pr = s / sum;
                for d in 0..hd {
                    attn[qi * qw + qoff + d] += pr * v[ki * kvd + koff + d];
                }
            }
        }
    }
    let o = matmul(&attn, w.wo, qw, hidden);
    let hres: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
    let n2 = rmsnorm(&hres, w.g2);
    let gate = matmul(&n2, w.wg, hidden, inter);
    let up = matmul(&n2, w.wu, hidden, inter);
    let gated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let down = matmul(&gated, w.wd, inter, hidden);
    hres.iter().zip(&down).map(|(a, b)| a + b).collect()
}

/// CPU reference for [`model_forward_bf16`]: logits `[tokens, vocab]`.
#[must_use]
pub fn model_forward_cpu(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for lw in &m.layers {
        cur = layer_cpu(&cur, lw, tokens, hidden, inter, causal);
    }
    let eps = m.layers[0].eps;
    let mut logits = vec![0.0f32; tokens * vocab];
    for tk in 0..tokens {
        let b = tk * hidden;
        let ms = cur[b..b + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for p in 0..hidden {
            let nv = cur[b + p] * inv * m.final_gamma[p];
            for c in 0..vocab {
                logits[tk * vocab + c] += nv * bf16_to_f32(m.lm_head[c * hidden + p]);
            }
        }
    }
    logits
}

/// CPU reference for [`model_probe_bf16`]: residual stream after layer `upto`.
#[must_use]
pub fn model_probe_cpu(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for lw in m.layers.iter().take(upto + 1) {
        cur = layer_cpu(&cur, lw, tokens, hidden, inter, causal);
    }
    cur
}
