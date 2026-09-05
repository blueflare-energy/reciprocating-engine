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
pub(crate) struct Gb {
    pub graph: synGraphHandle,
    pub names: Vec<CString>,
    pub sizes: Vec<Vec<u64>>,
    pub data: Vec<Vec<f32>>,
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
}

impl Gb {
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
        })
    }

    /// A persistent bf16 input tensor whose host data is uploaded at launch.
    pub fn input(&mut self, name: &str, sizes: &[u64], data: &[f32]) -> Result<synTensor> {
        debug_assert_eq!(sizes.iter().product::<u64>() as usize, data.len());
        let (t, cname) = make_tensor(self.graph, name, sizes, SYN_TYPE_BF16, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(data.to_vec());
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
        let (t, cname) = make_tensor(self.graph, name, sizes, dtype, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(Vec::new());
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
    pub fn serialize_if_requested(&self) -> Result<()> {
        if !env_on("RENG_SERIALIZE") {
            return Ok(());
        }
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
pub(crate) fn build_layer(
    gb: &mut Gb,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'_>,
    sh: &Shared,
    tokens: usize,
    hidden: usize,
    inter: usize,
    out: Option<synTensor>,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && hidden % nh == 0 && nkv >= 1 && nh % nkv == 0);
    let hd_us = hidden / nh;
    let hpg_us = nh / nkv;
    let (t, h, i, hd, hpg, groups) = (
        tokens as u64,
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
    );
    // Key axis of the score matrices: the block alone, or the whole cache
    // (its usable positions plus the trash slot).
    let keys = sh.cache.map_or(t, |c| c as u64 + 1);
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");

    // Per-head weight blocks, `[hd, hidden, heads-in-group, groups]` on the
    // device: block (j, g) is the columns of head `g * hpg + j`.
    let blocks = |m: &[f32], cols: usize, per_group: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(hidden * cols);
        for g in 0..nkv {
            for j in 0..per_group {
                let head = g * per_group + j;
                for r in 0..hidden {
                    v.extend_from_slice(&m[r * cols + head * hd_us..r * cols + (head + 1) * hd_us]);
                }
            }
        }
        v
    };
    let wq_scaled: Vec<f32> = w.wq.iter().map(|v| v * w.scale).collect();
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    let t_wq = gb.input(
        &p("wq"),
        &[hd, h, hpg, groups],
        &blocks(&wq_scaled, hidden, hpg_us),
    )?;
    let t_wk = gb.input(&p("wk"), &[hd, h, 1, groups], &blocks(w.wk, nkv * hd_us, 1))?;
    let t_wv = gb.input(&p("wv"), &[hd, h, 1, groups], &blocks(w.wv, nkv * hd_us, 1))?;
    let t_wo = gb.input(&p("wo"), &[h, h], w.wo)?;
    let t_wg = gb.input(&p("wg"), &[i, h], w.wg)?;
    let t_wu = gb.input(&p("wu"), &[i, h], w.wu)?;
    let t_wd = gb.input(&p("wd"), &[h, i], w.wd)?;

    let t_n1 = gb.mid(&p("n1"), &[h, t], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, t], SYN_TYPE_F32)?;
    let t_n1_4 = gb.mid(&p("n1_4"), &[h, t, 1, 1], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, t, hpg, groups], bf)?;
    let t_qr = gb.mid(&p("qr"), &[hd, t, hpg, groups], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, t, 1, groups], bf)?;
    let t_kr = gb.mid(&p("kr"), &[hd, t, 1, groups], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, t, 1, groups], bf)?;
    let t_sc = gb.mid(&p("scores"), &[keys, t, hpg, groups], bf)?;
    let t_pr = gb.mid(&p("probs"), &[keys, t, hpg, groups], bf)?;
    let t_at = gb.mid(&p("at"), &[hd, t, hpg, groups], bf)?;
    let t_at3 = gb.mid(&p("at3"), &[hd, t, hpg * groups], bf)?;
    let t_att = gb.mid(&p("att"), &[hd, hpg * groups, t], bf)?;
    let t_attn = gb.mid(&p("attn"), &[h, t], bf)?;
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
    gb.node("reshape", &p("n1_4d"), &[t_n1], &[t_n1_4], none.0, none.1)?;
    gb.node(
        "batch_gemm",
        &p("q_proj"),
        &[t_n1_4, t_wq],
        &[t_q],
        pg.0,
        pg.1,
    )?;
    gb.node(
        "batch_gemm",
        &p("k_proj"),
        &[t_n1_4, t_wk],
        &[t_k],
        pg.0,
        pg.1,
    )?;
    gb.node(
        "batch_gemm",
        &p("v_proj"),
        &[t_n1_4, t_wv],
        &[t_v],
        pg.0,
        pg.1,
    )?;
    // Attention biases (when present) broadcast over the token dim.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * w.scale).collect();
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

    // Keys and values attention reads: the block's own, or the cache updated
    // in place with the block: an ONNX ScatterND update writes each real row
    // of the rotated keys and of the values at its position (padded rows go
    // to the trash slot); the "out" tensor aliases the "in" one, so only the
    // written rows move (see cached.rs).
    let (k_full, v_full) = if let Some(kidx) = sh.kidx {
        let (n_kci, n_vci, n_kco, n_vco) = cache_names(li);
        let kci = gb.scratch(&n_kci, &[hd, keys, 1, groups])?;
        let vci = gb.scratch(&n_vci, &[hd, keys, 1, groups])?;
        let kco = gb.scratch_alias(&n_kco, &[hd, keys, 1, groups], &n_kci)?;
        let vco = gb.scratch_alias(&n_vco, &[hd, keys, 1, groups], &n_vci)?;
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
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pg.0, pg.1)?;
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
        pg.0,
        pg.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_n2, t_wu], &[t_up], pg.0, pg.1)?;
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
        pg.0,
        pg.1,
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
pub(crate) fn build_layer_batched(
    gb: &mut Gb,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'_>,
    sh: &SharedBatched,
    hidden: usize,
    inter: usize,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && hidden % nh == 0 && nkv >= 1 && nh % nkv == 0);
    let hd_us = hidden / nh;
    let hpg_us = nh / nkv;
    let (h, i, hd, hpg, groups, keys, b) = (
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        sh.capacity as u64 + 1,
        sh.batch as u64,
    );
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");
    let wq_scaled: Vec<f32> = w.wq.iter().map(|v| v * w.scale).collect();
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    // With one row per sequence the projections are plain gemms with
    // M = B over the natural `[in, out]` weights: `[hidden, B]` is already
    // `[hd, 1, hpg, groups, B]` in memory (head-major features, sequence
    // outermost), so the head layout is a free reshape. These weights are
    // laid out differently from the wide recipe's per-head blocks, so they
    // get their own names and buffers.
    let t_wq = gb.input(&p("wq2"), &[h, h], &wq_scaled)?;
    let t_wk = gb.input(&p("wk2"), &[hd * groups, h], w.wk)?;
    let t_wv = gb.input(&p("wv2"), &[hd * groups, h], w.wv)?;
    let t_wo = gb.input(&p("wo"), &[h, h], w.wo)?;
    let t_wg = gb.input(&p("wg"), &[i, h], w.wg)?;
    let t_wu = gb.input(&p("wu"), &[i, h], w.wu)?;
    let t_wd = gb.input(&p("wd"), &[h, i], w.wd)?;

    let t_n1 = gb.mid(&p("n1"), &[h, b], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, b], SYN_TYPE_F32)?;
    let t_q2 = gb.mid(&p("q2"), &[h, b], bf)?;
    let t_k2 = gb.mid(&p("k2"), &[hd * groups, b], bf)?;
    let t_v2 = gb.mid(&p("v2"), &[hd * groups, b], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, 1, hpg, groups, b], bf)?;
    let t_qr = gb.mid(&p("qr"), &[hd, 1, hpg, groups, b], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, 1, 1, groups, b], bf)?;
    let t_kr = gb.mid(&p("kr"), &[hd, 1, 1, groups, b], bf)?;
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
    let t_attn = gb.mid(&p("attn"), &[h, b], bf)?;
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
    gb.node("gemm", &p("q_proj"), &[t_n1, t_wq], &[t_q2], pg.0, pg.1)?;
    gb.node("gemm", &p("k_proj"), &[t_n1, t_wk], &[t_k2], pg.0, pg.1)?;
    gb.node("gemm", &p("v_proj"), &[t_n1, t_wv], &[t_v2], pg.0, pg.1)?;
    gb.node("reshape", &p("q_5d"), &[t_q2], &[t_q], none.0, none.1)?;
    gb.node("reshape", &p("k_5d"), &[t_k2], &[t_k], none.0, none.1)?;
    gb.node("reshape", &p("v_5d"), &[t_v2], &[t_v], none.0, none.1)?;
    // Attention biases (when present) broadcast over the sequence batch.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * w.scale).collect();
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
    gb.node(
        "reshape",
        &p("kr_updates"),
        &[t_kr],
        &[t_kru],
        none.0,
        none.1,
    )?;
    gb.node("reshape", &p("v_updates"), &[t_v], &[t_vu], none.0, none.1)?;
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
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pg.0, pg.1)?;
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
        pg.0,
        pg.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_n2, t_wu], &[t_up], pg.0, pg.1)?;
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
        pg.0,
        pg.1,
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
pub(crate) fn build_head(
    gb: &mut Gb,
    cur: synTensor,
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    ids_out: bool,
) -> Result<Out> {
    let (t, h, v) = (tokens as u64, hidden as u64, vocab as u64);
    let bf = SYN_TYPE_BF16;
    let t_gf = gb.input("GF", &[h], m.final_gamma)?;
    let t_lm = gb.input("LM", &[v, h], m.lm_head)?;
    let t_nf = gb.mid("nf", &[h, t], bf)?;
    let t_invf = gb.mid("invf", &[1, t], SYN_TYPE_F32)?;
    let (t_logits, n_logits) = if ids_out {
        let t = gb.scratch("LOGITS", &[v, t])?;
        (t, CString::new("LOGITS").unwrap())
    } else {
        make_tensor(gb.graph, "LOGITS", &[v, t], bf, true)?
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
        bwd_mode: 0,
    };
    let gemm = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    gb.node(
        "rms_norm_fwd_bf16",
        "final_norm",
        &[cur, t_gf],
        &[t_nf, t_invf],
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    )?;
    gb.node(
        "gemm",
        "lm_head",
        &[t_nf, t_lm],
        &[t_lg],
        (&raw const gemm).cast::<c_void>(),
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
    let (t_ids, n_ids) = make_tensor(gb.graph, "IDS", &[1, t], SYN_TYPE_INT32, true)?;
    let red = ReductionParams {
        reduction_dimension: 0,
    };
    gb.node(
        "argmax_fwd_bf16",
        "argmax",
        &[t_logits],
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
pub struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    /// Final RMSNorm gain, length `hidden`.
    pub final_gamma: &'a [f32],
    /// LM head stored `[hidden, vocab]`.
    pub lm_head: &'a [f32],
}

/// Build the shared inputs (activations, RoPE caches, causal mask) and the
/// decoder stack up to and including layer `upto`, returning the graph and the
/// last layer's output tensor. `probe_out` makes layer `upto` write into that
/// persistent tensor.
#[allow(clippy::too_many_arguments)]
fn build_stack(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
    probe_out: Option<synTensor>,
) -> Result<(Gb, synTensor)> {
    let l0 = &m.layers[0];
    let hd = hidden / l0.n_heads;
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
            if k <= q { 0.0 } else { MASK_NEG }
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
pub fn model_forward_bf16(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    assert!(!m.layers.is_empty());
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(m.final_gamma.len(), hidden);
    assert_eq!(m.lm_head.len(), hidden * vocab);
    let last = m.layers.len() - 1;
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, last, None)?;
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
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, upto, None)?;
    let (t_probe, n_probe) = make_tensor(gb.graph, "PROBE", &[h, t], SYN_TYPE_BF16, true)?;
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
    let hd = hidden / nh;
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
    // a[tokens, kin] @ mtx[kin, kout] -> [tokens, kout]
    let matmul = |a: &[f32], mtx: &[f32], kin: usize, kout: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; tokens * kout];
        for tk in 0..tokens {
            for p in 0..kin {
                let av = a[tk * kin + p];
                for c in 0..kout {
                    o[tk * kout + c] += av * mtx[p * kout + c];
                }
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

    let n1 = rmsnorm(x, w.g1);
    let wq_scaled: Vec<f32> = w.wq.iter().map(|v| v * w.scale).collect();
    let bias = |mut m: Vec<f32>, b: &[f32], scale: f32| -> Vec<f32> {
        if !b.is_empty() {
            let cols = b.len();
            for (i, v) in m.iter_mut().enumerate() {
                *v += b[i % cols] * scale;
            }
        }
        m
    };
    let q = bias(matmul(&n1, &wq_scaled, hidden, hidden), w.bq, w.scale);
    let k = bias(matmul(&n1, w.wk, hidden, kvd), w.bk, 1.0);
    let v = bias(matmul(&n1, w.wv, hidden, kvd), w.bv, 1.0);
    let mut qr = vec![0.0f32; tokens * hidden];
    let mut kr = vec![0.0f32; tokens * kvd];
    for head in 0..nh {
        rope_head(&q, hidden, head, &mut qr);
    }
    for g in 0..nkv {
        rope_head(&k, kvd, g, &mut kr);
    }
    let mut attn = vec![0.0f32; tokens * hidden];
    let mut scores = vec![0.0f32; tokens];
    for head in 0..nh {
        let g = head / n_rep;
        let qoff = head * hd;
        let koff = g * hd;
        for qi in 0..tokens {
            let limit = if causal { qi + 1 } else { tokens };
            for (ki, s) in scores.iter_mut().enumerate().take(limit) {
                *s = (0..hd)
                    .map(|d| qr[qi * hidden + qoff + d] * kr[ki * kvd + koff + d])
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
                    attn[qi * hidden + qoff + d] += pr * v[ki * kvd + koff + d];
                }
            }
        }
    }
    let o = matmul(&attn, w.wo, hidden, hidden);
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
                logits[tk * vocab + c] += nv * m.lm_head[p * vocab + c];
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
