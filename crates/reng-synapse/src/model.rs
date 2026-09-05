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
//! `j / (n_heads / n_kv_heads)`, the HF `repeat_kv` convention.
//!
//! With a KV cache (see `cached.rs`) each layer reads per-head cache tensors
//! `[head_dim, capacity + rows]` and writes updated ones: the block's keys
//! (after RoPE) and values are placed at their positions by a gemm with a
//! 0/1 placement matrix and added to the cache read. Attention runs over
//! the whole updated cache with a mask that admits positions up to each
//! query's own. Every step of that data path is MME or TPC work inside the
//! recipe; no DMA node touches the cache.
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
use crate::runtime::{Out, Runtime};
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

#[repr(C)]
struct AxisParams {
    axis: u32,
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
    pub scratch_names: Vec<CString>,
    pub scratch_sizes: Vec<Vec<u64>>,
    /// Whether each scratch tensor is f32 (else bf16); sizes count elements.
    pub scratch_f32: Vec<bool>,
    /// Node ids in creation order (a valid topological order for our graphs).
    node_ids: Vec<synNodeId>,
}

impl Gb {
    pub fn new() -> Result<Self> {
        syn!(synInitialize());
        let mut graph: synGraphHandle = core::ptr::null_mut();
        syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));
        Ok(Self {
            graph,
            names: Vec::new(),
            sizes: Vec::new(),
            data: Vec::new(),
            scratch_names: Vec::new(),
            scratch_sizes: Vec::new(),
            scratch_f32: Vec::new(),
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
        let (t, cname) = make_tensor(self.graph, name, sizes, dtype, true)?;
        self.scratch_names.push(cname);
        self.scratch_sizes.push(sizes.to_vec());
        self.scratch_f32.push(dtype == SYN_TYPE_F32);
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
    /// Additive mask laid out like the score matrix, `[keys, queries]`.
    pub mask: Option<synTensor>,
    /// KV-cache capacity in positions. When set, every layer gets per-head
    /// cache input and output tensors (see [`cache_names`]) of
    /// `capacity + tokens` positions, which is also the key axis of the scores.
    pub cache: Option<usize>,
    /// The placement matrix input (`[tokens, capacity + tokens]` device
    /// sizes: row r of the block goes to the position whose entry is 1).
    pub place: Option<synTensor>,
}

/// Persistent tensor names of layer `li`'s cache state for KV head `g`:
/// `(k_cache_in, v_cache_in, k_cache_out, v_cache_out)`, each
/// `[head_dim, capacity + tokens]` (keys after RoPE).
#[must_use]
pub fn cache_names(li: usize, g: usize) -> (String, String, String, String) {
    (
        format!("l{li}_kci{g}"),
        format!("l{li}_vci{g}"),
        format!("l{li}_kco{g}"),
        format!("l{li}_vco{g}"),
    )
}

/// Append one decoder layer reading `x` and return its output tensor. When
/// `out` is given (a persistent tensor) the layer writes into it instead of a
/// graph-internal tensor.
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
    let n_rep = nh / nkv;
    let (t, h, i, hd, kvd) = (
        tokens as u64,
        hidden as u64,
        inter as u64,
        hd_us as u64,
        (nkv * hd_us) as u64,
    );
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");

    let wq_scaled: Vec<f32> = w.wq.iter().map(|v| v * w.scale).collect();
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    let t_wq = gb.input(&p("wq"), &[h, h], &wq_scaled)?;
    let t_wo = gb.input(&p("wo"), &[h, h], w.wo)?;
    let t_wg = gb.input(&p("wg"), &[i, h], w.wg)?;
    let t_wu = gb.input(&p("wu"), &[i, h], w.wu)?;
    let t_wd = gb.input(&p("wd"), &[h, i], w.wd)?;

    let t_n1 = gb.mid(&p("n1"), &[h, t], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, t], SYN_TYPE_F32)?;
    let t_q = gb.mid(&p("q"), &[h, t], bf)?;
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

    // Key axis of the score matrices: the block alone, or the whole cache
    // (`capacity + rows` positions: a block that starts near the end of the
    // usable capacity spills its padding rows into the slack).
    let keys = sh.cache.map_or(t, |c| c as u64 + t);
    let rms = RmsNormParams {
        epsilon: w.eps,
        fused_gamma_beta: false,
        use_stages: false,
        bwd_mode: 0,
    };
    let rope = RopeParams { offset: 0, mode: 0 };
    let axis0 = AxisParams { axis: 0 };
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
    let pax = (
        (&raw const axis0).cast::<c_void>(),
        core::mem::size_of::<AxisParams>() as u32,
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
    let none = (core::ptr::null::<c_void>(), 0u32);

    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm1"),
        &[x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    gb.node("gemm", &p("q_proj"), &[t_n1, t_wq], &[t_q], pg.0, pg.1)?;
    // Per-head query slices (split only when there is more than one slice).
    let heads = |gb: &mut Gb, pre: &str, count: usize| -> Result<Vec<synTensor>> {
        (0..count)
            .map(|j| gb.mid(&p(&format!("{pre}{j}")), &[hd, t], bf))
            .collect()
    };
    let qs = if nh > 1 {
        let q_h = heads(gb, "q", nh)?;
        gb.node("split", &p("split_q"), &[t_q], &q_h, pax.0, pax.1)?;
        q_h
    } else {
        vec![t_q]
    };
    // Keys and values per KV head. Without a cache they come from one K and
    // one V projection split per head; the rotated keys and the values of the
    // block are graph-internal. With a cache each KV head has its own K and V
    // projection, and the block's rotated keys and values are placed into the
    // cache by compute engines only: `cache_out = cache_in + place(block)`,
    // where `place` is a gemm with the caller's 0/1 placement matrix (block
    // row r goes to position pos + r). The caller binds `cache_in` and
    // `cache_out` to two buffers and swaps them every launch. Attention then
    // reads the whole updated cache. No DMA touches cache data: on this stack
    // a DMA reading freshly compute-written memory can return stale bytes.
    let mut k_full: Vec<synTensor> = Vec::with_capacity(nkv);
    let mut v_full: Vec<synTensor> = Vec::with_capacity(nkv);
    if let Some(place) = sh.place {
        for g in 0..nkv {
            // Column block g of the [hidden, kvd] projections, as [hidden, hd].
            let cols = |m: &[f32]| -> Vec<f32> {
                (0..hidden)
                    .flat_map(|r| {
                        m[r * nkv * hd_us + g * hd_us..r * nkv * hd_us + (g + 1) * hd_us]
                            .iter()
                            .copied()
                    })
                    .collect()
            };
            let t_wk_g = gb.input(&p(&format!("wk{g}")), &[hd, h], &cols(w.wk))?;
            let t_wv_g = gb.input(&p(&format!("wv{g}")), &[hd, h], &cols(w.wv))?;
            let (n_kc_in, n_vc_in, n_kc_out, n_vc_out) = cache_names(li, g);
            let kc_in = gb.scratch(&n_kc_in, &[hd, keys])?;
            let vc_in = gb.scratch(&n_vc_in, &[hd, keys])?;
            let kc_out = gb.scratch(&n_kc_out, &[hd, keys])?;
            let vc_out = gb.scratch(&n_vc_out, &[hd, keys])?;
            let k_g = gb.mid(&p(&format!("k{g}")), &[hd, t], bf)?;
            let kn = gb.mid(&p(&format!("kn{g}")), &[hd, t], bf)?;
            let vn = gb.mid(&p(&format!("vn{g}")), &[hd, t], bf)?;
            let kp = gb.mid(&p(&format!("kp{g}")), &[hd, keys], bf)?;
            let vp = gb.mid(&p(&format!("vp{g}")), &[hd, keys], bf)?;
            gb.node(
                "gemm",
                &p(&format!("k_proj{g}")),
                &[t_n1, t_wk_g],
                &[k_g],
                pg.0,
                pg.1,
            )?;
            gb.node(
                "rope_st2_fwd_bf16",
                &p(&format!("rope_k{g}")),
                &[k_g, sh.sin, sh.cos],
                &[kn],
                pr.0,
                pr.1,
            )?;
            gb.node(
                "gemm",
                &p(&format!("v_proj{g}")),
                &[t_n1, t_wv_g],
                &[vn],
                pg.0,
                pg.1,
            )?;
            // placed[position, hd] = place[position, row] @ block[row, hd]
            gb.node(
                "gemm",
                &p(&format!("k_place{g}")),
                &[place, kn],
                &[kp],
                pg.0,
                pg.1,
            )?;
            gb.node(
                "gemm",
                &p(&format!("v_place{g}")),
                &[place, vn],
                &[vp],
                pg.0,
                pg.1,
            )?;
            gb.node(
                "add_fwd_bf16",
                &p(&format!("k_cache{g}")),
                &[kc_in, kp],
                &[kc_out],
                none.0,
                none.1,
            )?;
            gb.node(
                "add_fwd_bf16",
                &p(&format!("v_cache{g}")),
                &[vc_in, vp],
                &[vc_out],
                none.0,
                none.1,
            )?;
            k_full.push(kc_out);
            v_full.push(vc_out);
        }
    } else {
        let t_wk = gb.input(&p("wk"), &[kvd, h], w.wk)?;
        let t_wv = gb.input(&p("wv"), &[kvd, h], w.wv)?;
        let t_k = gb.mid(&p("k"), &[kvd, t], bf)?;
        let t_v = gb.mid(&p("v"), &[kvd, t], bf)?;
        gb.node("gemm", &p("k_proj"), &[t_n1, t_wk], &[t_k], pg.0, pg.1)?;
        gb.node("gemm", &p("v_proj"), &[t_n1, t_wv], &[t_v], pg.0, pg.1)?;
        let (ks, vs) = if nkv > 1 {
            let (k_h, v_h) = (heads(gb, "k", nkv)?, heads(gb, "v", nkv)?);
            gb.node("split", &p("split_k"), &[t_k], &k_h, pax.0, pax.1)?;
            gb.node("split", &p("split_v"), &[t_v], &v_h, pax.0, pax.1)?;
            (k_h, v_h)
        } else {
            (vec![t_k], vec![t_v])
        };
        for (g, &kg) in ks.iter().enumerate() {
            let kr = gb.mid(&p(&format!("kr{g}")), &[hd, t], bf)?;
            gb.node(
                "rope_st2_fwd_bf16",
                &p(&format!("rope_k{g}")),
                &[kg, sh.sin, sh.cos],
                &[kr],
                pr.0,
                pr.1,
            )?;
            k_full.push(kr);
            v_full.push(vs[g]);
        }
    }
    let mut at_h: Vec<synTensor> = Vec::with_capacity(nh);
    for (j, &qj) in qs.iter().enumerate() {
        let g = j / n_rep;
        let qr = gb.mid(&p(&format!("qr{j}")), &[hd, t], bf)?;
        let sc = gb.mid(&p(&format!("scores{j}")), &[keys, t], bf)?;
        let pr_t = gb.mid(&p(&format!("probs{j}")), &[keys, t], bf)?;
        let at = gb.mid(&p(&format!("attn{j}")), &[hd, t], bf)?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p(&format!("rope_q{j}")),
            &[qj, sh.sin, sh.cos],
            &[qr],
            pr.0,
            pr.1,
        )?;
        // scores[query, key] = qr @ k^T with K in its natural [seq, head_dim] layout.
        gb.node(
            "gemm",
            &p(&format!("qk{j}")),
            &[qr, k_full[g]],
            &[sc],
            pgt.0,
            pgt.1,
        )?;
        let sm_in = if let Some(mask) = sh.mask {
            let masked = gb.mid(&p(&format!("masked{j}")), &[keys, t], bf)?;
            gb.node(
                "add_fwd_bf16",
                &p(&format!("mask{j}")),
                &[sc, mask],
                &[masked],
                none.0,
                none.1,
            )?;
            masked
        } else {
            sc
        };
        gb.node(
            "softmax_fwd_bf16",
            &p(&format!("softmax{j}")),
            &[sm_in],
            &[pr_t],
            ps.0,
            ps.1,
        )?;
        gb.node(
            "gemm",
            &p(&format!("av{j}")),
            &[pr_t, v_full[g]],
            &[at],
            pg.0,
            pg.1,
        )?;
        at_h.push(at);
    }
    let attn_full = if nh > 1 {
        gb.node("concat", &p("merge_heads"), &at_h, &[t_attn], pax.0, pax.1)?;
        t_attn
    } else {
        at_h[0]
    };
    gb.node("gemm", &p("o_proj"), &[attn_full, t_wo], &[t_o], pg.0, pg.1)?;
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

/// Append the final RMSNorm and LM head reading `cur` and return the logits
/// output `[vocab, tokens]` as the graph's read-back tensor.
pub(crate) fn build_head(
    gb: &mut Gb,
    cur: synTensor,
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    vocab: usize,
) -> Result<Out> {
    let (t, h, v) = (tokens as u64, hidden as u64, vocab as u64);
    let bf = SYN_TYPE_BF16;
    let t_gf = gb.input("GF", &[h], m.final_gamma)?;
    let t_lm = gb.input("LM", &[v, h], m.lm_head)?;
    let t_nf = gb.mid("nf", &[h, t], bf)?;
    let t_invf = gb.mid("invf", &[1, t], SYN_TYPE_F32)?;
    let (t_logits, n_logits) = make_tensor(gb.graph, "LOGITS", &[v, t], bf, true)?;
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
    Ok(Out {
        name: n_logits,
        sizes: vec![v, t],
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
        Some(gb.input("MASK", &[t, t], &mask_host)?)
    } else {
        None
    };
    let sh = Shared {
        sin: t_sin,
        cos: t_cos,
        mask: t_mask,
        cache: None,
        place: None,
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
    let out = build_head(&mut gb, cur, m, tokens, hidden, vocab)?;
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
    let q = matmul(&n1, &wq_scaled, hidden, hidden);
    let k = matmul(&n1, w.wk, hidden, kvd);
    let v = matmul(&n1, w.wv, hidden, kvd);
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
