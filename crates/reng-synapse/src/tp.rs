//! Tensor-parallel decoding: one model split over the cards of an HCCL
//! communicator, one process per card (see `hccl.rs`), every rank holding
//! the Megatron shard of each layer (its query and KV heads, its slice of
//! the MLP width) and a replica of the norms, the embedding and the LM
//! head.
//!
//! A layer is two recipes with an all-reduce after each, all on the rank's
//! one stream, in order, without host synchronisation:
//!
//! - recipe A: `X = H + P2` (the residual after the previous layer's
//!   partial MLP sum, `P2` all-reduced already), norm, the local q/k/v
//!   heads, RoPE, the local KV cache update, attention over the local
//!   heads, and the partial `o_proj` over this rank's columns into `P1`
//!   (f32, `[hidden, tokens]`); then `hcclAllReduce(P1)` in place;
//! - recipe B: `H = X + P1`, norm, the local gate/up rows, SiLU, the
//!   partial `down_proj` into `P2` (f32); then `hcclAllReduce(P2)`.
//!
//! The partial sums are produced and reduced in f32 (the MME writes f32),
//! so after the reduction every rank holds the same sum the single-card
//! gemm would have accumulated; the bf16 rounding happens where the
//! single-card graph rounds too (at the residual add). With one rank the
//! collectives are skipped and the graph math is the single-card graph's,
//! recipe boundaries aside.
//!
//! The recipes are compiled once per kind and shape (A and B for decode
//! rows and for prefill rows, an embedding recipe, and the head), not per
//! layer: every layer's weights and KV cache live in their own device
//! buffers and the recipes are bound to layer `l`'s buffers before each
//! launch (`Runtime::rebind_at`). The recipe owns layer 0's buffers; the
//! others are uploaded through a [`Store`]. The residual stream `H`/`X`
//! and the partials `P1`/`P2` are persistent tensors shared by name
//! between the recipes ([`Bindings`]).
//!
//! Decode is the device-loop form of `cached.rs`: an embedding recipe E
//! gathers the token's row, its RoPE rows, mask row and scatter indices
//! from an id and a position bound per launch into an id ring and a
//! position table, the layers follow, and the head recipe (final norm,
//! LM head, argmax on every rank) writes the next id into the ring slot
//! the next token's E reads. `n` tokens are `n x (2 L + 2)` launches and
//! `2 n L` collectives enqueued back to back, then one read of `n` ring
//! slots. Prefill runs the same two recipes at the block width with the
//! per-block inputs (embeddings, RoPE rows, mask, scatter indices)
//! uploaded from the host, as the wide recipe of `cached.rs` does.

use crate::cached::{GatherParams, SLOT};
use crate::ffi::{SYN_TYPE_BF16, SYN_TYPE_F32, synGEMMParams, synSoftmaxParams, synTensor};
use crate::hccl::Rank;
use crate::model::{
    EmbedTable, Gb, MASK_NEG, ModelWeights, RopeTables, build_head, fused_sdpa, lm_head_input,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::{Bindings, Out, OutKind, Runtime, Store};
use crate::{LayerWeights, Stride, f32_to_bf16, to_bf16};
use core::ffi::c_void;
use reng_core::Result;
use std::time::Instant;

/// `ns_LayerNormKernel::ParamsRmsNorm` (see `model.rs`).
#[repr(C)]
struct RmsNormParams {
    eps_valid: u8,
    _pad0: [u8; 3],
    eps: f32,
    norm_axis_bmp: i32,
    param_axis_bmp: i32,
    normalized_shape_dims: u32,
    fast_math: u8,
    _pad1: [u8; 3],
}

impl RmsNormParams {
    const fn new(eps: f32) -> Self {
        Self {
            eps_valid: 1,
            _pad0: [0; 3],
            eps,
            norm_axis_bmp: 1,
            param_axis_bmp: 1,
            normalized_shape_dims: 1,
            fast_math: 0,
            _pad1: [0; 3],
        }
    }
}

#[repr(C)]
struct RopeParams {
    offset: u32,
    mode: i32,
}

#[repr(C)]
struct SdpaParams {
    scale: f32,
    is_causal: u8,
    _pad0: [u8; 3],
    dropout_ratio: f32,
    dropout_seed: u32,
    disable_mask_out: u8,
    _pad1: [u8; 3],
    is_inference: u8,
    _pad2: [u8; 3],
}

#[repr(C)]
struct TransposeParams {
    permutation: [u32; 5],
    tensor_dim: u32,
}

#[repr(C)]
struct ScatterNdUpdateParams {
    mode: i32,
}

/// The per-step attention inputs of recipe A: RoPE rows, the additive
/// mask and the ScatterND indices, as the recipe reads them (derived on
/// the device for decode, uploaded for prefill).
struct Step {
    sin: synTensor,
    cos: synTensor,
    mask: synTensor,
    kidx: synTensor,
}

/// Persistent tensor names of the two layer recipes. The weights carry
/// layer 0's names whatever layer they are bound to.
const N_H: &str = "l0_H";
const N_X: &str = "l0_X";
const N_P1: &str = "l0_P1";
const N_P2: &str = "l0_P2";
const N_KCI: &str = "l0_kci";
const N_VCI: &str = "l0_vci";
const N_KCO: &str = "l0_kco";
const N_VCO: &str = "l0_vco";
/// The wide recipe's separate cache outputs (its ScatterND is not in
/// place; see `cached.rs`).
const N_KCW: &str = "l0_kcw";
const N_VCW: &str = "l0_vcw";
const N_SIN: &str = "l0_SIN";
const N_COS: &str = "l0_COS";
const N_MASK: &str = "l0_MASK";
const N_KIDX: &str = "l0_KIDX";
/// The wide recipe's per-block inputs (uploaded, so named apart from the
/// decode recipes' derived tensors).
const N_SINP: &str = "l0_SINP";
const N_COSP: &str = "l0_COSP";
const N_MASKP: &str = "l0_MASKP";
const N_KIDXP: &str = "l0_KIDXP";

/// Which weights recipe A binds per layer, in the order of
/// [`LayerBufs::a`].
fn a_weight_names(w: &LayerWeights<'_>) -> Vec<&'static str> {
    let mut v = vec!["l0_g1", "l0_wq2", "l0_wk2", "l0_wv2", "l0_wo"];
    if !w.bq.is_empty() {
        v.extend(["l0_bq", "l0_bk", "l0_bv"]);
    }
    if !w.qn.is_empty() {
        v.extend(["l0_qn", "l0_kn"]);
    }
    v
}

/// Which weights recipe B binds per layer, in the order of
/// [`LayerBufs::b`].
const B_WEIGHT_NAMES: [&str; 4] = ["l0_g2", "l0_wg", "l0_wu", "l0_wd"];

/// A layer's device buffers.
struct LayerBufs {
    /// Recipe A's weights, in [`a_weight_names`] order.
    a: Vec<u64>,
    /// Recipe B's weights, in [`B_WEIGHT_NAMES`] order.
    b: Vec<u64>,
    /// The key and value caches: first all `batch` slots (`[hd, keys, 1,
    /// groups]` per slot, the slots contiguous), which decode updates in
    /// place; second the wide recipe's own single-slot buffer, which its
    /// out-of-place ScatterND alternates with a slot (see
    /// [`TpModel::prefill`]).
    k: [u64; 2],
    v: [u64; 2],
}

/// Launch-table indices of the tensors a recipe rebinds per layer.
struct Binds {
    weights: Vec<usize>,
    /// `kci, vci, kco, vco`.
    cache: [usize; 4],
}

impl Binds {
    fn new(rt: &Runtime<'_>, names: &[&str], cache: [&str; 4]) -> Self {
        Self {
            weights: names.iter().map(|n| rt.bind_index(n)).collect(),
            cache: cache.map(|n| rt.bind_index(n)),
        }
    }
}

/// Which parts of a decode step run: the bench's per-layer time split
/// skips the collectives, then recipe B, then the layers altogether, and
/// reads the differences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// The real step.
    Full,
    /// Recipes A and B without the all-reduces (wrong results).
    NoAllReduce,
    /// Recipe A only (wrong results).
    AttnOnly,
    /// The embedding and head recipes only (wrong results).
    NoLayers,
}

/// Times of one decode run, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTimes {
    /// Host time to enqueue everything.
    pub enqueue: f64,
    /// Wall time from the first enqueue to the ids read.
    pub total: f64,
}

/// One rank's share of a tensor-parallel model: its recipes, weights,
/// cache and decode state on one card, for `batch` sequences decoded in
/// lockstep (one, or the batched 5-D form of `batched.rs`).
pub struct TpModel<'a> {
    // Declared before `rank`: the runtimes and the store borrow its device
    // and stream and must drop first.
    a_dec: Runtime<'a>,
    b_dec: Runtime<'a>,
    e_dec: Runtime<'a>,
    head_dec: Runtime<'a>,
    a_pre: Runtime<'a>,
    b_pre: Runtime<'a>,
    head_pre: Runtime<'a>,
    store: Store,
    rank: Rank,
    layers: Vec<LayerBufs>,
    ba_dec: Binds,
    bb_dec: Vec<usize>,
    ba_pre: Binds,
    bb_pre: Vec<usize>,
    /// Device addresses of the decode and prefill partial sums and of the
    /// prefill residual (`H`) and `P2` for the per-block seeding.
    p1_dec: u64,
    p2_dec: u64,
    p1_pre: u64,
    p2_pre: u64,
    h_pre: u64,
    /// Per-block inputs of the wide recipe A.
    ix_sin: usize,
    ix_cos: usize,
    ix_mask: usize,
    ix_kidx: usize,
    /// The id ring and position table of the decode loop (`batched.rs`):
    /// rows of `stride` bytes holding `batch` int32s; ring row `r` holds
    /// the ids the launch at ring step `r` consumes, position table row
    /// `j` every slot's position at launch `j` of the current run.
    ring: u64,
    postab: u64,
    stride: usize,
    /// The ring row holding the ids the next run consumes, and those ids
    /// when the host knows them (the last run's output or the last seed).
    head: usize,
    known: Option<Vec<u32>>,
    hidden: usize,
    vocab: usize,
    rows: usize,
    capacity: usize,
    head_dim: usize,
    n_kv: usize,
    batch: usize,
    /// Position of each sequence.
    pos: Vec<usize>,
    sin: Vec<f32>,
    cos: Vec<f32>,
    /// What [`TpModel::decode`] runs (see [`Mode`]).
    pub mode: Mode,
    /// Seconds spent uploading the layers' weights and the device bytes
    /// they take, for the load report.
    pub upload_s: f64,
    pub device_bytes: u64,
}

impl<'a> TpModel<'a> {
    /// Build this rank's recipes over the shard `m` (every layer with the
    /// rank's local head counts; `inter` the local MLP width), upload the
    /// weights and allocate the KV cache of `capacity` positions for
    /// `batch` sequences. `rows` is the prefill block width. `rope` holds
    /// the tables `[capacity, head_dim]`; `embed` the (replicated) token
    /// table.
    ///
    /// # Errors
    ///
    /// Returns an error if a compile, an allocation or an upload fails.
    ///
    /// # Panics
    ///
    /// Panics on a layer form the tensor-parallel graphs do not build
    /// (post-norm, Gemma's extra norms and softcaps, sliding windows, the
    /// OLMo-2 full-width q/k norm, NoPE layers) or a size mismatch.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        rank: Rank,
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        batch: usize,
        rows: usize,
        capacity: usize,
        rope: &RopeTables<'_>,
        embed: &EmbedTable<'a>,
    ) -> Result<Self> {
        assert!(!m.layers.is_empty() && batch > 0 && rows > 0 && capacity > 0);
        let l0 = &m.layers[0];
        let (hd, n_kv) = (l0.head_dim, l0.n_kv_heads);
        assert_eq!(rope.sin.len(), capacity * hd);
        assert_eq!(rope.cos.len(), capacity * hd);
        assert_eq!(m.final_gamma.len(), hidden);
        assert_eq!(m.lm_head.len(), hidden * vocab);
        assert_eq!(embed.rows.len(), hidden * vocab);
        for l in &m.layers {
            assert!(
                !l.post_norm
                    && l.g_post_attn.is_empty()
                    && l.g_post_mlp.is_empty()
                    && l.attn_softcap.is_none()
                    && l.window.is_none()
                    && !l.local_rope
                    && l.use_rope
                    && (l.qn.is_empty() || l.qn.len() == hd),
                "layer form not supported by the tensor-parallel graphs"
            );
            assert_eq!(l.n_heads, l0.n_heads);
            assert_eq!(l.n_kv_heads, n_kv);
            assert_eq!(l.head_dim, hd);
        }
        let trace = std::env::var_os("RENG_RECIPE_TRACE").is_some();
        let (dev, stream) = (rank.card.device_id(), rank.card.stream_handle());
        let t0 = Instant::now();
        let mut bind = Bindings::new();

        // Decode recipes: one token per sequence.
        let (gb, out) = if batch == 1 {
            build_a(l0, hidden, 1, capacity, true, Derive::Device)?
        } else {
            build_a_batched(l0, hidden, batch, capacity)?
        };
        let a_dec = Runtime::new_on(gb, out, dev, stream)?;
        bind.add(&a_dec);
        let (gb, out) = build_b(l0, hidden, inter, batch)?;
        let b_dec = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        bind.add(&b_dec);
        let (gb, out) = build_head_tp(m, hidden, vocab, batch, None)?;
        let head_dec = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        bind.add(&head_dec);
        let (gb, out) = if batch == 1 {
            build_embed(m, hidden, vocab, capacity, n_kv, hd, rope, embed)?
        } else {
            build_embed_batched(m, hidden, vocab, capacity, n_kv, hd, batch, rope, embed)?
        };
        let e_dec = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        bind.add(&e_dec);
        // Prefill recipes: a block of `rows` tokens of one sequence.
        let (gb, out) = build_a(l0, hidden, rows, capacity, false, Derive::Host)?;
        let a_pre = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        bind.add(&a_pre);
        let (gb, out) = build_b(l0, hidden, inter, rows)?;
        let b_pre = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        bind.add(&b_pre);
        let (gb, out) = build_head_tp(m, hidden, vocab, rows, None)?;
        let mut head_pre = Runtime::new_bound(gb, out, dev, stream, &bind)?;
        let t_recipes = t0.elapsed().as_secs_f64();

        // Layer 0's buffers are the recipes' own; the others come from
        // the store.
        let a_names = a_weight_names(l0);
        let ba_dec = Binds::new(&a_dec, &a_names, [N_KCI, N_VCI, N_KCO, N_VCO]);
        let ba_pre = Binds::new(&a_pre, &a_names, [N_KCI, N_VCI, N_KCW, N_VCW]);
        let bb_dec: Vec<usize> = B_WEIGHT_NAMES.iter().map(|n| b_dec.bind_index(n)).collect();
        let bb_pre: Vec<usize> = B_WEIGHT_NAMES.iter().map(|n| b_pre.bind_index(n)).collect();
        let mut store = Store::new(dev, stream);
        let t1 = Instant::now();
        let slot_bytes = (hd * (capacity + 1) * n_kv * 2) as u64;
        let mut layers = Vec::with_capacity(m.layers.len());
        for (li, w) in m.layers.iter().enumerate() {
            if li == 0 {
                layers.push(LayerBufs {
                    a: a_names.iter().map(|n| a_dec.addr(n)).collect(),
                    b: B_WEIGHT_NAMES.iter().map(|n| b_dec.addr(n)).collect(),
                    k: [a_dec.addr(N_KCI), a_pre.addr(N_KCW)],
                    v: [a_dec.addr(N_VCI), a_pre.addr(N_VCW)],
                });
                continue;
            }
            let mut a = Vec::with_capacity(a_names.len());
            for (bytes, scale, stride) in a_weight_sources(w, hidden) {
                a.push(store.upload(&bytes, scale, stride)?);
            }
            let mut b = Vec::with_capacity(B_WEIGHT_NAMES.len());
            for (bytes, scale, stride) in b_weight_sources(w, hidden, inter) {
                b.push(store.upload(&bytes, scale, stride)?);
            }
            let k = [
                store.alloc_zeroed(slot_bytes * batch as u64)?,
                store.alloc_zeroed(slot_bytes)?,
            ];
            let v = [
                store.alloc_zeroed(slot_bytes * batch as u64)?,
                store.alloc_zeroed(slot_bytes)?,
            ];
            layers.push(LayerBufs { a, b, k, v });
        }
        store.finish()?;
        let upload_s = t1.elapsed().as_secs_f64();
        if trace {
            eprintln!(
                "tp rank {}: recipes {t_recipes:.2} s, layers 1.. uploaded in {upload_s:.2} s ({:.2} GB)",
                rank.rank,
                store.bytes as f64 / 1e9
            );
        }

        // The decode loop's ring and position table: rows of whole cache
        // lines holding `batch` int32s.
        let stride = (4 * batch).div_ceil(SLOT) * SLOT;
        let ring = head_pre.alloc(((capacity + 1) * stride) as u64)?;
        let postab = head_pre.alloc((capacity * stride) as u64)?;

        let (p1_dec, p2_dec) = (a_dec.addr(N_P1), a_dec.addr(N_P2));
        let (p1_pre, p2_pre, h_pre) = (a_pre.addr(N_P1), a_pre.addr(N_P2), a_pre.addr(N_H));
        let ix_sin = a_pre.input_index(N_SINP);
        let ix_cos = a_pre.input_index(N_COSP);
        let ix_mask = a_pre.input_index(N_MASKP);
        let ix_kidx = a_pre.input_index(N_KIDXP);
        let device_bytes = store.bytes;
        Ok(Self {
            a_dec,
            b_dec,
            e_dec,
            head_dec,
            a_pre,
            b_pre,
            head_pre,
            store,
            rank,
            layers,
            ba_dec,
            bb_dec,
            ba_pre,
            bb_pre,
            p1_dec,
            p2_dec,
            p1_pre,
            p2_pre,
            h_pre,
            ix_sin,
            ix_cos,
            ix_mask,
            ix_kidx,
            ring,
            postab,
            stride,
            head: 0,
            known: None,
            hidden,
            vocab,
            rows,
            capacity,
            head_dim: hd,
            n_kv,
            batch,
            pos: vec![0; batch],
            sin: rope.sin.to_vec(),
            cos: rope.cos.to_vec(),
            mode: Mode::Full,
            upload_s,
            device_bytes,
        })
    }

    /// Position of sequence `b`.
    #[must_use]
    pub fn position(&self, b: usize) -> usize {
        self.pos[b]
    }

    /// Prefill block width.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of sequences.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// This rank's index and the world size.
    #[must_use]
    pub fn rank(&self) -> (usize, usize) {
        (self.rank.rank, self.rank.world)
    }

    /// Start sequence `b` afresh (its cache slot needs no clearing: the
    /// mask never admits positions at or beyond its position).
    pub fn reset(&mut self, b: usize) {
        self.pos[b] = 0;
        self.known = None;
    }

    /// Bytes of one sequence's slot in one cache buffer.
    fn slot_bytes(&self) -> u64 {
        (self.head_dim * (self.capacity + 1) * self.n_kv * 2) as u64
    }

    /// Bind layer `li`'s buffers into the decode recipes (the cache of
    /// every slot, in place).
    fn bind_dec(&mut self, li: usize) {
        let l = &self.layers[li];
        for (&idx, &addr) in self.ba_dec.weights.iter().zip(&l.a) {
            self.a_dec.rebind_at(idx, addr);
        }
        let [kci, vci, kco, vco] = self.ba_dec.cache;
        self.a_dec.rebind_at(kci, l.k[0]);
        self.a_dec.rebind_at(vci, l.v[0]);
        self.a_dec.rebind_at(kco, l.k[0]);
        self.a_dec.rebind_at(vco, l.v[0]);
        for (&idx, &addr) in self.bb_dec.iter().zip(&l.b) {
            self.b_dec.rebind_at(idx, addr);
        }
    }

    /// Bind layer `li`'s buffers into the prefill recipes: the cache is
    /// read from the wide recipe's own buffer and written to the slot at
    /// `off` (`to_slot`), or the other way round.
    fn bind_pre(&mut self, li: usize, off: u64, to_slot: bool) {
        let l = &self.layers[li];
        for (&idx, &addr) in self.ba_pre.weights.iter().zip(&l.a) {
            self.a_pre.rebind_at(idx, addr);
        }
        let [kci, vci, kco, vco] = self.ba_pre.cache;
        let (k_in, v_in, k_out, v_out) = if to_slot {
            (l.k[1], l.v[1], l.k[0] + off, l.v[0] + off)
        } else {
            (l.k[0] + off, l.v[0] + off, l.k[1], l.v[1])
        };
        self.a_pre.rebind_at(kci, k_in);
        self.a_pre.rebind_at(vci, v_in);
        self.a_pre.rebind_at(kco, k_out);
        self.a_pre.rebind_at(vco, v_out);
        for (&idx, &addr) in self.bb_pre.iter().zip(&l.b) {
            self.b_pre.rebind_at(idx, addr);
        }
    }

    /// Feed `x` (`[n, hidden]` embeddings, any `n` that fits the cache,
    /// in blocks of at most `rows`) to sequence `b` at its position and
    /// return the greedy id after the last one. The wide recipe's
    /// ScatterND is not in place: the blocks alternate between the
    /// sequence's cache slot and the recipe's own buffer, starting so that
    /// the last block lands in the slot (a stale input is masked out).
    ///
    /// # Errors
    ///
    /// Returns an error if a call fails or the ids never complete.
    ///
    /// # Panics
    ///
    /// Panics if `x` is empty or not whole rows, `b` is not a slot, or the
    /// tokens overflow the cache.
    #[allow(clippy::too_many_lines)]
    pub fn prefill(&mut self, b: usize, x: &[f32]) -> Result<u32> {
        let (h, hd, c, p) = (self.hidden, self.head_dim, self.capacity, self.rows);
        assert_eq!(x.len() % h, 0);
        let n_total = x.len() / h;
        assert!(n_total >= 1 && b < self.batch);
        assert!(
            self.pos[b] + n_total <= c,
            "cache overflow at {}+{n_total} of {c}",
            self.pos[b]
        );
        let off = b as u64 * self.slot_bytes();
        let n_blocks = n_total.div_ceil(p);
        let keys = c + 1;
        let neg = f32_to_bf16(MASK_NEG);
        let count = p * h;
        let mut last = 0u32;
        for (i, chunk) in x.chunks(p * h).enumerate() {
            let to_slot = (n_blocks - 1 - i) % 2 == 0;
            let n = chunk.len() / h;
            let pos = self.pos[b];
            // The residual starts as the embeddings (zero rows after the
            // real ones) with no partial MLP sum to add.
            let mut xb = vec![0u16; p * h];
            for (dst, &v) in xb.iter_mut().zip(chunk) {
                *dst = f32_to_bf16(v);
            }
            let xb_bytes = bf16_bytes(&xb);
            let rope_rows = |table: &[f32]| -> Vec<f32> {
                let mut rows = vec![0.0f32; p * hd];
                for r in 0..p {
                    if pos + r < c {
                        let src = (pos + r) * hd;
                        rows[r * hd..(r + 1) * hd].copy_from_slice(&table[src..src + hd]);
                    }
                }
                rows
            };
            let sb = rope_rows(&self.sin);
            let cb = rope_rows(&self.cos);
            let mut mb = vec![neg; p * keys];
            for q in 0..p {
                let end = (pos + q + 1).min(c);
                mb[q * keys..q * keys + end].fill(0);
            }
            let mut ib: Vec<u8> = Vec::with_capacity(12 * p * self.n_kv);
            for g in 0..self.n_kv {
                for r in 0..p {
                    let target = if r < n { pos + r } else { c };
                    for v in [g as i32, 0i32, target as i32] {
                        ib.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            self.a_pre.memset_d32(self.p2_pre, 0, count)?;
            self.a_pre.upload_at(self.h_pre, &xb_bytes)?;
            self.a_pre.upload(self.ix_sin, &sb)?;
            self.a_pre.upload(self.ix_cos, &cb)?;
            self.a_pre.upload_bf16(self.ix_mask, &mb)?;
            self.a_pre.upload_raw(self.ix_kidx, &ib)?;
            self.a_pre.fence()?;
            for li in 0..self.layers.len() {
                self.bind_pre(li, off, to_slot);
                self.a_pre.launch_only()?;
                self.rank.all_reduce_f32(self.p1_pre, count)?;
                self.b_pre.launch_only()?;
                self.rank.all_reduce_f32(self.p2_pre, count)?;
            }
            let ids = self.head_pre.launch_and_read_i32(n - 1, 1)?;
            last = ids[0] as u32;
            self.pos[b] += n;
        }
        self.known = None;
        Ok(last)
    }

    /// Feed token `seeds[b]` to every sequence `b` at its position and run
    /// `n` decode steps back to back, each consuming the greedy ids the
    /// previous one produced; returns the `n * batch` ids step by step
    /// (`ids[j * batch + b]` is sequence `b`'s id after step `j`; the last
    /// step's are not fed) and the times. One readback for the run.
    ///
    /// # Errors
    ///
    /// Returns an error if a call fails or the ids never complete.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` is not one id per sequence, `n` is 0, or the run
    /// overflows the cache.
    #[allow(clippy::too_many_lines)]
    pub fn decode(&mut self, seeds: &[u32], n: usize) -> Result<(Vec<u32>, StepTimes)> {
        let nb = self.batch;
        assert!(n >= 1);
        assert_eq!(seeds.len(), nb, "one seed id per sequence");
        for &id in seeds {
            assert!((id as usize) < self.vocab, "token id {id} out of range");
        }
        let (c, h) = (self.capacity, self.hidden);
        let furthest = self.pos.iter().copied().max().unwrap_or(0);
        assert!(furthest + n <= c, "cache overflow at {furthest}+{n} of {c}");
        let t0 = Instant::now();
        // A run needs `n + 1` ring rows from `head`: wrap when they are not
        // there (the seed is then uploaded again).
        if self.head + n > c {
            self.head = 0;
            self.known = None;
        }
        let (stride, head) = (self.stride, self.head);
        let resident = self.known.as_deref() == Some(seeds);
        let row = |r: usize| (r * stride) as u64;
        // Position table rows for the run: every slot's position plus the
        // launch number.
        let mut tab = vec![0u8; n * stride];
        for j in 0..n {
            for (b, &p) in self.pos.iter().enumerate() {
                let at = j * stride + b * 4;
                tab[at..at + 4].copy_from_slice(&((p + j) as i32).to_le_bytes());
            }
        }
        let seed_bytes: Vec<u8> = seeds
            .iter()
            .flat_map(|&id| (id as i32).to_le_bytes())
            .collect();
        let mut parts: Vec<(u64, &[u8])> = vec![(self.postab, &tab)];
        if !resident {
            parts.push((self.ring + row(head), &seed_bytes));
        }
        self.head_dec.upload_at_multi(&parts)?;
        self.head_dec.fence()?;
        let first_out = self.ring + row(head + 1);
        self.head_dec.fill_sentinel_d32(first_out, n * stride / 4)?;
        let ids_in = self.e_dec.bind_index("IDS_IN");
        let pos_in = self.e_dec.bind_index("POS");
        let ids_out = self.head_dec.bind_index("IDS");
        let mode = self.mode;
        let count = nb * h;
        let t1 = Instant::now();
        for j in 0..n {
            self.e_dec.rebind_at(ids_in, self.ring + row(head + j));
            self.e_dec.rebind_at(pos_in, self.postab + row(j));
            self.e_dec.launch_only()?;
            if mode != Mode::NoLayers {
                for li in 0..self.layers.len() {
                    self.bind_dec(li);
                    self.a_dec.launch_only()?;
                    if mode == Mode::Full {
                        self.rank.all_reduce_f32(self.p1_dec, count)?;
                    }
                    if mode != Mode::AttnOnly {
                        self.b_dec.launch_only()?;
                        if mode == Mode::Full {
                            self.rank.all_reduce_f32(self.p2_dec, count)?;
                        }
                    }
                }
            }
            self.head_dec
                .rebind_at(ids_out, self.ring + row(head + j + 1));
            self.head_dec.launch_only()?;
        }
        let enqueue = t1.elapsed().as_secs_f64();
        let ids: Vec<u32> = self
            .head_dec
            .read_i32_rows(first_out, stride, n, nb)?
            .into_iter()
            .map(|i| i as u32)
            .collect();
        let total = t0.elapsed().as_secs_f64();
        self.head = head + n;
        self.known = Some(ids[(n - 1) * nb..].to_vec());
        for p in &mut self.pos {
            *p += n;
        }
        Ok((ids, StepTimes { enqueue, total }))
    }

    /// The last decode step's logits `[batch, vocab]` (bf16 on the
    /// device), for checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails.
    pub fn last_logits(&mut self) -> Result<Vec<f32>> {
        self.head_dec.read_bf16_range("LOGITS", 0, self.batch)
    }
}

/// Where recipe A's per-step attention inputs come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Derive {
    /// Persistent tensors the embedding recipe writes (decode).
    Device,
    /// Inputs uploaded per block (prefill).
    Host,
}

/// Recipe A of a layer over `tokens` positions (see the module docs).
/// `inplace` selects the cache update form (`cached.rs`).
#[allow(clippy::too_many_lines)]
fn build_a<'a>(
    w: &LayerWeights<'a>,
    hidden: usize,
    tokens: usize,
    capacity: usize,
    inplace: bool,
    derive: Derive,
) -> Result<(Gb<'a>, Out)> {
    let (nh, nkv, hd_us) = (w.n_heads, w.n_kv_heads, w.head_dim);
    assert!(nh >= 1 && nkv >= 1 && nh % nkv == 0);
    let hpg_us = nh / nkv;
    let qw_us = nh * hd_us;
    let (t, h, hd, hpg, groups, qw) = (
        tokens as u64,
        hidden as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        qw_us as u64,
    );
    let keys = capacity as u64 + 1;
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let mut gb = Gb::new()?;
    // The residual stream and the partial sums.
    let t_h = gb.scratch(N_H, &[h, t])?;
    let t_p2 = gb.scratch_typed(N_P2, &[h, t], SYN_TYPE_F32)?;
    let t_x = gb.scratch(N_X, &[h, t])?;
    let (t_p1, n_p1) = gb.output(N_P1, &[h, t], SYN_TYPE_F32)?;
    // Per-step attention inputs.
    let step = match derive {
        Derive::Device => Step {
            sin: gb.scratch(N_SIN, &[hd, t])?,
            cos: gb.scratch(N_COS, &[hd, t])?,
            mask: gb.scratch(N_MASK, &[keys, t, 1, 1])?,
            kidx: gb.scratch_typed(N_KIDX, &[3, t * groups], SYN_TYPE_INT32)?,
        },
        Derive::Host => Step {
            sin: gb.input(N_SINP, &[hd, t], &vec![0.0; tokens * hd_us])?,
            cos: gb.input(N_COSP, &[hd, t], &vec![0.0; tokens * hd_us])?,
            mask: gb.input(
                N_MASKP,
                &[keys, t, 1, 1],
                &vec![0.0; tokens * (capacity + 1)],
            )?,
            kidx: gb.input_raw(
                N_KIDXP,
                &[3, t * groups],
                SYN_TYPE_INT32,
                &vec![0u8; 12 * tokens * nkv],
            )?,
        },
    };
    // Weights: layer 0's, as the recipe's own inputs (see `a_weight_sources`
    // for the same conversions applied to the other layers).
    let scale = w.scale;
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    let t_g1 = gb.input("l0_g1", &[h], w.g1)?;
    let t_wq = if w.qn.is_empty() {
        gb.input_bf16_scaled("l0_wq2", &[h, qw], w.wq, scale)?
    } else {
        gb.input_bf16("l0_wq2", &[h, qw], std::borrow::Cow::Borrowed(w.wq))?
    };
    let t_wk = gb.input_bf16(
        "l0_wk2",
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wk),
    )?;
    let t_wv = gb.input_bf16(
        "l0_wv2",
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wv),
    )?;
    let t_wo = if w.wo_pitch == 0 {
        gb.input_bf16("l0_wo", &[qw, h], std::borrow::Cow::Borrowed(w.wo))?
    } else {
        let st = Stride {
            rows: hidden,
            cols: qw_us,
            pitch: w.wo_pitch,
        };
        gb.input_bf16_strided("l0_wo", &[qw, h], w.wo, st, None)?
    };

    let rms = RmsNormParams::new(w.eps);
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
    let scatter = ScatterNdUpdateParams { mode: 0 };
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
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let p = |s: &str| format!("l0_{s}");

    // X = H + P2 (the previous layer's reduced MLP partial, cast to bf16
    // where the single-card graph rounds its down_proj output).
    let t_p2b = gb.mid(&p("p2b"), &[h, t], bf)?;
    gb.node(
        "cast_f32_to_bf16",
        &p("p2_cast"),
        &[t_p2],
        &[t_p2b],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("res0"),
        &[t_h, t_p2b],
        &[t_x],
        none.0,
        none.1,
    )?;
    let t_n1 = gb.mid(&p("n1"), &[h, t], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, t], SYN_TYPE_F32)?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm1"),
        &[t_x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    // The flat projections, into the head layout (see `build_layer`).
    let t_q = gb.mid(&p("q"), &[hd, t, hpg, groups], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, t, 1, groups], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, t, 1, groups], bf)?;
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
            ptr_.0,
            ptr_.1,
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
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input("l0_bq", &[hd, 1, hpg, groups], &bq_scaled)?;
        let t_bk = gb.input("l0_bk", &[hd, 1, 1, groups], w.bk)?;
        let t_bv = gb.input("l0_bv", &[hd, 1, 1, groups], w.bv)?;
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
    let (t_q, t_k) = if w.qn.is_empty() {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
        let t_qn = gb.input("l0_qn", &[hd], &qn_scaled)?;
        let t_kn = gb.input("l0_kn", &[hd], w.kn)?;
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
    let t_qr = gb.mid(&p("qr"), &[hd, t, hpg, groups], bf)?;
    let t_kr = gb.mid(&p("kr"), &[hd, t, 1, groups], bf)?;
    gb.node(
        "rope_st2_fwd_bf16",
        &p("rope_q"),
        &[t_q, step.sin, step.cos],
        &[t_qr],
        pr.0,
        pr.1,
    )?;
    gb.node(
        "rope_st2_fwd_bf16",
        &p("rope_k"),
        &[t_k, step.sin, step.cos],
        &[t_kr],
        pr.0,
        pr.1,
    )?;
    // The local KV cache, updated with the block (see `build_layer`).
    let kci = gb.scratch(N_KCI, &[hd, keys, 1, groups])?;
    let vci = gb.scratch(N_VCI, &[hd, keys, 1, groups])?;
    let (kco, vco) = if inplace {
        (
            gb.scratch_alias(N_KCO, &[hd, keys, 1, groups], N_KCI)?,
            gb.scratch_alias(N_VCO, &[hd, keys, 1, groups], N_VCI)?,
        )
    } else {
        (
            gb.scratch(N_KCW, &[hd, keys, 1, groups])?,
            gb.scratch(N_VCW, &[hd, keys, 1, groups])?,
        )
    };
    let kru = gb.mid(&p("kru"), &[hd, t * groups], bf)?;
    let vu = gb.mid(&p("vu"), &[hd, t * groups], bf)?;
    gb.node("reshape", &p("kr_updates"), &[t_kr], &[kru], none.0, none.1)?;
    gb.node("reshape", &p("v_updates"), &[t_v], &[vu], none.0, none.1)?;
    gb.node(
        "scatter_nd_update_fwd_bf16",
        &p("k_scatter"),
        &[kci, step.kidx, kru],
        &[kco],
        psc.0,
        psc.1,
    )?;
    gb.node(
        "scatter_nd_update_fwd_bf16",
        &p("v_scatter"),
        &[vci, step.kidx, vu],
        &[vco],
        psc.0,
        psc.1,
    )?;
    let t_at = gb.mid(&p("at"), &[hd, t, hpg, groups], bf)?;
    if fused_sdpa(tokens == 1) {
        let sdpa = SdpaParams {
            scale: 1.0,
            is_causal: 0,
            _pad0: [0; 3],
            dropout_ratio: 0.0,
            dropout_seed: 0,
            disable_mask_out: 1,
            _pad1: [0; 3],
            is_inference: 1,
            _pad2: [0; 3],
        };
        gb.node(
            "sdpa_recomp_fwd_bf16",
            &p("sdpa"),
            &[t_qr, kco, vco, step.mask],
            &[t_at],
            (&raw const sdpa).cast::<c_void>(),
            core::mem::size_of::<SdpaParams>() as u32,
        )?;
    } else {
        let t_sc = gb.mid(&p("scores"), &[keys, t, hpg, groups], bf)?;
        let t_masked = gb.mid(&p("masked"), &[keys, t, hpg, groups], bf)?;
        let t_pr = gb.mid(&p("probs"), &[keys, t, hpg, groups], bf)?;
        gb.node("batch_gemm", &p("qk"), &[t_qr, kco], &[t_sc], pgt.0, pgt.1)?;
        gb.node(
            "add_fwd_bf16",
            &p("mask"),
            &[t_sc, step.mask],
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
    }
    let t_at3 = gb.mid(&p("at3"), &[hd, t, hpg * groups], bf)?;
    let t_att = gb.mid(&p("att"), &[hd, hpg * groups, t], bf)?;
    let t_attn = gb.mid(&p("attn"), &[qw, t], bf)?;
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
    // The partial output projection over this rank's heads, in f32.
    if std::env::var_os("RENG_TP_BF16_PARTIAL").is_some() {
        let t_o = gb.mid(&p("o"), &[h, t], bf)?;
        gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
        gb.node(
            "cast_bf16_to_f32",
            &p("o_cast"),
            &[t_o],
            &[t_p1],
            none.0,
            none.1,
        )?;
    } else {
        gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_p1], pgt.0, pgt.1)?;
    }
    let out = Out {
        name: n_p1,
        sizes: vec![h, t],
        kind: OutKind::I32,
    };
    Ok((gb, out))
}

/// Recipe B of a layer over `tokens` positions: `H = X + P1`, norm, the
/// local gate/up rows, SiLU, the partial `down_proj` into `P2`.
fn build_b<'a>(
    w: &LayerWeights<'a>,
    hidden: usize,
    inter: usize,
    tokens: usize,
) -> Result<(Gb<'a>, Out)> {
    let (t, h, i) = (tokens as u64, hidden as u64, inter as u64);
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let mut gb = Gb::new()?;
    let t_x = gb.scratch(N_X, &[h, t])?;
    let t_p1 = gb.scratch_typed(N_P1, &[h, t], SYN_TYPE_F32)?;
    let t_h = gb.scratch(N_H, &[h, t])?;
    let (t_p2, n_p2) = gb.output(N_P2, &[h, t], SYN_TYPE_F32)?;
    let t_g2 = gb.input("l0_g2", &[h], w.g2)?;
    let t_wg = gb.input_bf16("l0_wg", &[h, i], std::borrow::Cow::Borrowed(w.wg))?;
    let t_wu = gb.input_bf16("l0_wu", &[h, i], std::borrow::Cow::Borrowed(w.wu))?;
    let t_wd = if w.wd_pitch == 0 {
        gb.input_bf16("l0_wd", &[i, h], std::borrow::Cow::Borrowed(w.wd))?
    } else {
        let st = Stride {
            rows: hidden,
            cols: inter,
            pitch: w.wd_pitch,
        };
        gb.input_bf16_strided("l0_wd", &[i, h], w.wd, st, None)?
    };
    let rms = RmsNormParams::new(w.eps);
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let prm = (
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    );
    let pgt = (
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let p = |s: &str| format!("l0_{s}");
    let t_p1b = gb.mid(&p("p1b"), &[h, t], bf)?;
    gb.node(
        "cast_f32_to_bf16",
        &p("p1_cast"),
        &[t_p1],
        &[t_p1b],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("res1"),
        &[t_x, t_p1b],
        &[t_h],
        none.0,
        none.1,
    )?;
    let t_n2 = gb.mid(&p("n2"), &[h, t], bf)?;
    let t_inv2 = gb.mid(&p("inv2"), &[1, t], SYN_TYPE_F32)?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm2"),
        &[t_h, t_g2],
        &[t_n2, t_inv2],
        prm.0,
        prm.1,
    )?;
    let t_gate = gb.mid(&p("gate"), &[i, t], bf)?;
    let t_up = gb.mid(&p("up"), &[i, t], bf)?;
    let t_sg = gb.mid(&p("sg"), &[i, t], bf)?;
    let t_act = gb.mid(&p("act"), &[i, t], bf)?;
    let t_gated = gb.mid(&p("gated"), &[i, t], bf)?;
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
        &[t_act],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gate_x_up"),
        &[t_act, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    if std::env::var_os("RENG_TP_BF16_PARTIAL").is_some() {
        let t_down = gb.mid(&p("down"), &[h, t], bf)?;
        gb.node(
            "gemm",
            &p("down_proj"),
            &[t_gated, t_wd],
            &[t_down],
            pgt.0,
            pgt.1,
        )?;
        gb.node(
            "cast_bf16_to_f32",
            &p("down_cast"),
            &[t_down],
            &[t_p2],
            none.0,
            none.1,
        )?;
    } else {
        gb.node(
            "gemm",
            &p("down_proj"),
            &[t_gated, t_wd],
            &[t_p2],
            pgt.0,
            pgt.1,
        )?;
    }
    let out = Out {
        name: n_p2,
        sizes: vec![h, t],
        kind: OutKind::I32,
    };
    Ok((gb, out))
}

/// The head recipe over `tokens` positions: `X = H + P2` (the last
/// layer's reduced MLP partial), then the final norm, the LM head and the
/// argmax of `build_head` into `IDS`.
fn build_head_tp<'a>(
    m: &ModelWeights<'a>,
    hidden: usize,
    vocab: usize,
    tokens: usize,
    lm: Option<synTensor>,
) -> Result<(Gb<'a>, Out)> {
    let (t, h) = (tokens as u64, hidden as u64);
    let none = (core::ptr::null::<c_void>(), 0u32);
    let mut gb = Gb::new()?;
    let t_h = gb.scratch(N_H, &[h, t])?;
    let t_p2 = gb.scratch_typed(N_P2, &[h, t], SYN_TYPE_F32)?;
    let t_p2b = gb.mid("p2b_head", &[h, t], SYN_TYPE_BF16)?;
    let t_x = gb.mid("x_head", &[h, t], SYN_TYPE_BF16)?;
    gb.node(
        "cast_f32_to_bf16",
        "p2_cast_head",
        &[t_p2],
        &[t_p2b],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        "res_head",
        &[t_h, t_p2b],
        &[t_x],
        none.0,
        none.1,
    )?;
    let out = build_head(&mut gb, t_x, m, tokens, hidden, vocab, true, lm)?;
    Ok((gb, out))
}

/// The embedding recipe of the decode loop: from the id and the position
/// bound per launch, the token's embedding row into `H`, a zero `P2`, the
/// RoPE rows, the mask row and the ScatterND indices (see `cached.rs`,
/// whose derivations these are).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_embed<'a>(
    m: &ModelWeights<'a>,
    hidden: usize,
    vocab: usize,
    capacity: usize,
    groups: usize,
    hd: usize,
    rope: &RopeTables<'_>,
    embed: &EmbedTable<'a>,
) -> Result<(Gb<'a>, Out)> {
    let (h, v, hd64, cap, keys) = (
        hidden as u64,
        vocab as u64,
        hd as u64,
        capacity as u64,
        capacity as u64 + 1,
    );
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let (rows_p, elems_p) = (GatherParams { axis: 1 }, GatherParams { axis: 0 });
    let size = core::mem::size_of::<GatherParams>() as u32;
    let (pg_rows, pg_elems) = (
        ((&raw const rows_p).cast::<c_void>(), size),
        ((&raw const elems_p).cast::<c_void>(), size),
    );
    let i32s = |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut gb = Gb::new()?;
    let t_ids_in = gb.input_raw("IDS_IN", &[1], SYN_TYPE_INT32, &[0u8; 4])?;
    let t_pos = gb.input_raw("POS", &[1], SYN_TYPE_INT32, &[0u8; 4])?;
    // The residual stream starts as the embedding row, with nothing to add.
    let t_h = gb.scratch(N_H, &[h, 1])?;
    let t_p2 = gb.scratch_typed(N_P2, &[h, 1], SYN_TYPE_F32)?;
    let tied = embed.rows.as_ptr() == m.lm_head.as_ptr() && embed.rows.len() == m.lm_head.len();
    let t_emb = if tied {
        lm_head_input(&mut gb, m, hidden, vocab)?
    } else {
        gb.input_bf16("EMB", &[h, v], std::borrow::Cow::Borrowed(embed.rows))?
    };
    if embed.scale == 1.0 {
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_h],
            pg_rows.0,
            pg_rows.1,
        )?;
    } else {
        let t_row = gb.mid("emb_row", &[h, 1], bf)?;
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_row],
            pg_rows.0,
            pg_rows.1,
        )?;
        let t_scale = gb.input_raw(
            "EMB_SCALE",
            &[1, 1],
            SYN_TYPE_F32,
            &embed.scale.to_le_bytes(),
        )?;
        let t_rf = gb.mid("emb_f32", &[h, 1], SYN_TYPE_F32)?;
        let t_sf = gb.mid("emb_scaled", &[h, 1], SYN_TYPE_F32)?;
        gb.node(
            "cast_bf16_to_f32",
            "emb_cast",
            &[t_row],
            &[t_rf],
            none.0,
            none.1,
        )?;
        gb.node(
            "mult_fwd_f32",
            "emb_scale",
            &[t_rf, t_scale],
            &[t_sf],
            none.0,
            none.1,
        )?;
        gb.node(
            "cast_f32_to_bf16",
            "emb_round",
            &[t_sf],
            &[t_h],
            none.0,
            none.1,
        )?;
    }
    let t_zero = gb.input("ZERO", &[h, 1], &vec![0.0; hidden])?;
    gb.node(
        "cast_bf16_to_f32",
        "p2_zero",
        &[t_zero],
        &[t_p2],
        none.0,
        none.1,
    )?;
    // RoPE rows of the position out of the full tables.
    for (name, table, out_name) in [("SIN", rope.sin, N_SIN), ("COS", rope.cos, N_COS)] {
        let t_tab = gb.input(&format!("{name}T"), &[hd64, cap], table)?;
        let t_row = gb.scratch(out_name, &[hd64, 1])?;
        gb.node(
            "gather_fwd_bf16",
            &format!("rope_{name}"),
            &[t_tab, t_pos],
            &[t_row],
            pg_rows.0,
            pg_rows.1,
        )?;
    }
    // The causal mask row: a window of `keys` elements into `[0; keys] ++
    // [NEG; keys]` at `keys - 1 - position` (see `cached.rs`).
    let neg = f32_to_bf16(MASK_NEG);
    let keys_us = capacity + 1;
    let mut pat = vec![0u16; 2 * keys_us];
    pat[keys_us..].fill(neg);
    let t_pat = gb.input_bf16("MASKP", &[pat.len() as u64], std::borrow::Cow::Owned(pat))?;
    let base: Vec<i32> = (0..keys_us).map(|k| (keys_us - 1 + k) as i32).collect();
    let t_base = gb.input_raw("MASKI", &[keys], SYN_TYPE_INT32, &i32s(&base))?;
    let t_idx = gb.mid("MASK_idx", &[keys], SYN_TYPE_INT32)?;
    let t_mask = gb.scratch(N_MASK, &[keys])?;
    gb.node(
        "sub_fwd_i32",
        "MASK_index",
        &[t_base, t_pos],
        &[t_idx],
        none.0,
        none.1,
    )?;
    gb.node(
        "gather_fwd_bf16",
        "MASK_gather",
        &[t_pat, t_idx],
        &[t_mask],
        pg_elems.0,
        pg_elems.1,
    )?;
    // ScatterND triples (g, 0, position) for the one row of each KV head.
    let (mut kbase, mut ksel) = (
        Vec::with_capacity(3 * groups),
        Vec::with_capacity(3 * groups),
    );
    for g in 0..groups {
        kbase.extend_from_slice(&[g as i32, 0, 0]);
        ksel.extend_from_slice(&[0, 0, 1]);
    }
    let g64 = groups as u64;
    let t_kbase = gb.input_raw("KBASE", &[3, g64], SYN_TYPE_INT32, &i32s(&kbase))?;
    let t_ksel = gb.input_raw("KSEL", &[3, g64], SYN_TYPE_INT32, &i32s(&ksel))?;
    let t_pos2 = gb.mid("pos_2d", &[1, 1], SYN_TYPE_INT32)?;
    let t_kp = gb.mid("kidx_pos", &[3, g64], SYN_TYPE_INT32)?;
    let (t_kidx, n_kidx) = gb.output(N_KIDX, &[3, g64], SYN_TYPE_INT32)?;
    gb.node(
        "reshape",
        "pos_reshape",
        &[t_pos],
        &[t_pos2],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_i32",
        "kidx_scale",
        &[t_ksel, t_pos2],
        &[t_kp],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_i32",
        "kidx_add",
        &[t_kbase, t_kp],
        &[t_kidx],
        none.0,
        none.1,
    )?;
    let out = Out {
        name: n_kidx,
        sizes: vec![3, g64],
        kind: OutKind::I32,
    };
    Ok((gb, out))
}

/// Recipe A of a layer for `batch` sequences of one token each (the 5-D
/// form of `build_layer_batched`: sequences in the outermost dim, one
/// cache slot, RoPE row, mask row and scatter quadruple per sequence),
/// its per-step inputs the persistent tensors the batched embedding
/// recipe writes.
#[allow(clippy::too_many_lines)]
fn build_a_batched<'a>(
    w: &LayerWeights<'a>,
    hidden: usize,
    batch: usize,
    capacity: usize,
) -> Result<(Gb<'a>, Out)> {
    let (nh, nkv, hd_us) = (w.n_heads, w.n_kv_heads, w.head_dim);
    assert!(nh >= 1 && nkv >= 1 && nh % nkv == 0);
    let hpg_us = nh / nkv;
    let qw_us = nh * hd_us;
    let (b, h, hd, hpg, groups, qw) = (
        batch as u64,
        hidden as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        qw_us as u64,
    );
    let keys = capacity as u64 + 1;
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let mut gb = Gb::new()?;
    let t_h = gb.scratch(N_H, &[h, b])?;
    let t_p2 = gb.scratch_typed(N_P2, &[h, b], SYN_TYPE_F32)?;
    let t_x = gb.scratch(N_X, &[h, b])?;
    let (t_p1, n_p1) = gb.output(N_P1, &[h, b], SYN_TYPE_F32)?;
    let t_sin = gb.scratch(N_SIN, &[hd, 1, 1, 1, b])?;
    let t_cos = gb.scratch(N_COS, &[hd, 1, 1, 1, b])?;
    let t_mask = gb.scratch(N_MASK, &[keys, 1, 1, 1, b])?;
    let t_kidx = gb.scratch_typed(N_KIDX, &[4, groups * b], SYN_TYPE_INT32)?;
    let scale = w.scale;
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    let t_g1 = gb.input("l0_g1", &[h], w.g1)?;
    let t_wq = if w.qn.is_empty() {
        gb.input_bf16_scaled("l0_wq2", &[h, qw], w.wq, scale)?
    } else {
        gb.input_bf16("l0_wq2", &[h, qw], std::borrow::Cow::Borrowed(w.wq))?
    };
    let t_wk = gb.input_bf16(
        "l0_wk2",
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wk),
    )?;
    let t_wv = gb.input_bf16(
        "l0_wv2",
        &[h, hd * groups],
        std::borrow::Cow::Borrowed(w.wv),
    )?;
    let t_wo = if w.wo_pitch == 0 {
        gb.input_bf16("l0_wo", &[qw, h], std::borrow::Cow::Borrowed(w.wo))?
    } else {
        let st = Stride {
            rows: hidden,
            cols: qw_us,
            pitch: w.wo_pitch,
        };
        gb.input_bf16_strided("l0_wo", &[qw, h], w.wo, st, None)?
    };
    let rms = RmsNormParams::new(w.eps);
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
    let scatter = ScatterNdUpdateParams { mode: 0 };
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
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let p = |s: &str| format!("l0_{s}");

    let t_p2b = gb.mid(&p("p2b"), &[h, b], bf)?;
    gb.node(
        "cast_f32_to_bf16",
        &p("p2_cast"),
        &[t_p2],
        &[t_p2b],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("res0"),
        &[t_h, t_p2b],
        &[t_x],
        none.0,
        none.1,
    )?;
    let t_n1 = gb.mid(&p("n1"), &[h, b], bf)?;
    let t_inv1 = gb.mid(&p("inv1"), &[1, b], SYN_TYPE_F32)?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p("norm1"),
        &[t_x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    // Plain gemms with M = B; `[hidden, B]` is `[hd, 1, hpg, groups, B]`
    // in memory, so the head layout is a free reshape.
    let t_q2 = gb.mid(&p("q2"), &[qw, b], bf)?;
    let t_k2 = gb.mid(&p("k2"), &[hd * groups, b], bf)?;
    let t_v2 = gb.mid(&p("v2"), &[hd * groups, b], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, 1, hpg, groups, b], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, 1, 1, groups, b], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, 1, 1, groups, b], bf)?;
    gb.node("gemm", &p("q_proj"), &[t_n1, t_wq], &[t_q2], pgt.0, pgt.1)?;
    gb.node("gemm", &p("k_proj"), &[t_n1, t_wk], &[t_k2], pgt.0, pgt.1)?;
    gb.node("gemm", &p("v_proj"), &[t_n1, t_wv], &[t_v2], pgt.0, pgt.1)?;
    gb.node("reshape", &p("q_5d"), &[t_q2], &[t_q], none.0, none.1)?;
    gb.node("reshape", &p("k_5d"), &[t_k2], &[t_k], none.0, none.1)?;
    gb.node("reshape", &p("v_5d"), &[t_v2], &[t_v], none.0, none.1)?;
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input("l0_bq", &[hd, 1, hpg, groups, 1], &bq_scaled)?;
        let t_bk = gb.input("l0_bk", &[hd, 1, 1, groups, 1], w.bk)?;
        let t_bv = gb.input("l0_bv", &[hd, 1, 1, groups, 1], w.bv)?;
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
    let (t_q, t_k) = if w.qn.is_empty() {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
        let t_qn = gb.input("l0_qn", &[hd], &qn_scaled)?;
        let t_kn = gb.input("l0_kn", &[hd], w.kn)?;
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
    let t_qr = gb.mid(&p("qr"), &[hd, 1, hpg, groups, b], bf)?;
    let t_kr = gb.mid(&p("kr"), &[hd, 1, 1, groups, b], bf)?;
    gb.node(
        "rope_st2_fwd_bf16",
        &p("rope_q"),
        &[t_q, t_sin, t_cos],
        &[t_qr],
        pr.0,
        pr.1,
    )?;
    gb.node(
        "rope_st2_fwd_bf16",
        &p("rope_k"),
        &[t_k, t_sin, t_cos],
        &[t_kr],
        pr.0,
        pr.1,
    )?;
    let kci = gb.scratch(N_KCI, &[hd, keys, 1, groups, b])?;
    let vci = gb.scratch(N_VCI, &[hd, keys, 1, groups, b])?;
    let kco = gb.scratch_alias(N_KCO, &[hd, keys, 1, groups, b], N_KCI)?;
    let vco = gb.scratch_alias(N_VCO, &[hd, keys, 1, groups, b], N_VCI)?;
    let t_kru = gb.mid(&p("kru"), &[hd, groups * b], bf)?;
    let t_vu = gb.mid(&p("vu"), &[hd, groups * b], bf)?;
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
        &[kci, t_kidx, t_kru],
        &[kco],
        psc.0,
        psc.1,
    )?;
    gb.node(
        "scatter_nd_update_fwd_bf16",
        &p("v_scatter"),
        &[vci, t_kidx, t_vu],
        &[vco],
        psc.0,
        psc.1,
    )?;
    let t_at = gb.mid(&p("at"), &[hd, 1, hpg, groups, b], bf)?;
    if fused_sdpa(false) {
        let sdpa = SdpaParams {
            scale: 1.0,
            is_causal: 0,
            _pad0: [0; 3],
            dropout_ratio: 0.0,
            dropout_seed: 0,
            disable_mask_out: 1,
            _pad1: [0; 3],
            is_inference: 1,
            _pad2: [0; 3],
        };
        gb.node(
            "sdpa_recomp_fwd_bf16",
            &p("sdpa"),
            &[t_qr, kco, vco, t_mask],
            &[t_at],
            (&raw const sdpa).cast::<c_void>(),
            core::mem::size_of::<SdpaParams>() as u32,
        )?;
    } else {
        let t_sc = gb.mid(&p("scores"), &[keys, 1, hpg, groups, b], bf)?;
        let t_masked = gb.mid(&p("masked"), &[keys, 1, hpg, groups, b], bf)?;
        let t_pr = gb.mid(&p("probs"), &[keys, 1, hpg, groups, b], bf)?;
        gb.node("batch_gemm", &p("qk"), &[t_qr, kco], &[t_sc], pgt.0, pgt.1)?;
        gb.node(
            "add_fwd_bf16",
            &p("mask"),
            &[t_sc, t_mask],
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
    }
    let t_attn = gb.mid(&p("attn"), &[qw, b], bf)?;
    gb.node("reshape", &p("attn_2d"), &[t_at], &[t_attn], none.0, none.1)?;
    if std::env::var_os("RENG_TP_BF16_PARTIAL").is_some() {
        let t_o = gb.mid(&p("o"), &[h, b], bf)?;
        gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
        gb.node(
            "cast_bf16_to_f32",
            &p("o_cast"),
            &[t_o],
            &[t_p1],
            none.0,
            none.1,
        )?;
    } else {
        gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_p1], pgt.0, pgt.1)?;
    }
    let out = Out {
        name: n_p1,
        sizes: vec![h, b],
        kind: OutKind::I32,
    };
    Ok((gb, out))
}

/// The batched embedding recipe: from `batch` ids and positions bound per
/// launch, the embedding rows into `H` (`[hidden, B]`), a zero `P2`, the
/// RoPE rows `[hd, B]`, the mask rows `[keys * B]` and the ScatterND
/// quadruples `[4, groups * B]` (the derivations of `batched.rs`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_embed_batched<'a>(
    m: &ModelWeights<'a>,
    hidden: usize,
    vocab: usize,
    capacity: usize,
    groups: usize,
    hd: usize,
    batch: usize,
    rope: &RopeTables<'_>,
    embed: &EmbedTable<'a>,
) -> Result<(Gb<'a>, Out)> {
    let (h, v, hd64, cap, keys, b, g64) = (
        hidden as u64,
        vocab as u64,
        hd as u64,
        capacity as u64,
        capacity as u64 + 1,
        batch as u64,
        groups as u64,
    );
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let (rows_p, elems_p) = (GatherParams { axis: 1 }, GatherParams { axis: 0 });
    let size = core::mem::size_of::<GatherParams>() as u32;
    let (pg_rows, pg_elems) = (
        ((&raw const rows_p).cast::<c_void>(), size),
        ((&raw const elems_p).cast::<c_void>(), size),
    );
    let i32s = |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut gb = Gb::new()?;
    let t_ids_in = gb.input_raw("IDS_IN", &[b], SYN_TYPE_INT32, &vec![0u8; 4 * batch])?;
    let t_pos = gb.input_raw("POS", &[b], SYN_TYPE_INT32, &vec![0u8; 4 * batch])?;
    let t_h = gb.scratch(N_H, &[h, b])?;
    let t_p2 = gb.scratch_typed(N_P2, &[h, b], SYN_TYPE_F32)?;
    let tied = embed.rows.as_ptr() == m.lm_head.as_ptr() && embed.rows.len() == m.lm_head.len();
    let t_emb = if tied {
        lm_head_input(&mut gb, m, hidden, vocab)?
    } else {
        gb.input_bf16("EMB", &[h, v], std::borrow::Cow::Borrowed(embed.rows))?
    };
    if embed.scale == 1.0 {
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_h],
            pg_rows.0,
            pg_rows.1,
        )?;
    } else {
        let t_rows = gb.mid("emb_rows", &[h, b], bf)?;
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_rows],
            pg_rows.0,
            pg_rows.1,
        )?;
        let t_scale = gb.input_raw(
            "EMB_SCALE",
            &[1, 1],
            SYN_TYPE_F32,
            &embed.scale.to_le_bytes(),
        )?;
        let t_rf = gb.mid("emb_f32", &[h, b], SYN_TYPE_F32)?;
        let t_sf = gb.mid("emb_scaled", &[h, b], SYN_TYPE_F32)?;
        gb.node(
            "cast_bf16_to_f32",
            "emb_cast",
            &[t_rows],
            &[t_rf],
            none.0,
            none.1,
        )?;
        gb.node(
            "mult_fwd_f32",
            "emb_scale",
            &[t_rf, t_scale],
            &[t_sf],
            none.0,
            none.1,
        )?;
        gb.node(
            "cast_f32_to_bf16",
            "emb_round",
            &[t_sf],
            &[t_h],
            none.0,
            none.1,
        )?;
    }
    let t_zero = gb.input("ZERO", &[h, b], &vec![0.0; hidden * batch])?;
    gb.node(
        "cast_bf16_to_f32",
        "p2_zero",
        &[t_zero],
        &[t_p2],
        none.0,
        none.1,
    )?;
    // RoPE rows of the `B` positions, `[hd, B]` (the layers read them as
    // `[hd, 1, 1, 1, B]`).
    for (name, table, out_name) in [("SIN", rope.sin, N_SIN), ("COS", rope.cos, N_COS)] {
        let t_tab = gb.input(&format!("{name}T"), &[hd64, cap], table)?;
        let t_rows = gb.scratch(out_name, &[hd64, b])?;
        gb.node(
            "gather_fwd_bf16",
            &format!("rope_{name}"),
            &[t_tab, t_pos],
            &[t_rows],
            pg_rows.0,
            pg_rows.1,
        )?;
    }
    // A mask row per slot: the `[keys, B]` index tensor is the base
    // vector replicated per slot minus the positions broadcast along the
    // keys; the gather takes it flattened (see `batched.rs`).
    let neg = f32_to_bf16(MASK_NEG);
    let keys_us = capacity + 1;
    let mut pat = vec![0u16; 2 * keys_us];
    pat[keys_us..].fill(neg);
    let t_pat = gb.input_bf16("MASKP", &[pat.len() as u64], std::borrow::Cow::Owned(pat))?;
    let mut base: Vec<i32> = Vec::with_capacity(keys_us * batch);
    for _ in 0..batch {
        base.extend((0..keys_us).map(|k| (keys_us - 1 + k) as i32));
    }
    let t_base = gb.input_raw("MASKI", &[keys, b], SYN_TYPE_INT32, &i32s(&base))?;
    let t_pos2 = gb.mid("pos_2d", &[1, b], SYN_TYPE_INT32)?;
    let t_idx = gb.mid("MASK_idx", &[keys, b], SYN_TYPE_INT32)?;
    let t_flat = gb.mid("MASK_flat", &[keys * b], SYN_TYPE_INT32)?;
    let t_mask = gb.scratch(N_MASK, &[keys * b])?;
    gb.node(
        "reshape",
        "pos_reshape2",
        &[t_pos],
        &[t_pos2],
        none.0,
        none.1,
    )?;
    gb.node(
        "sub_fwd_i32",
        "MASK_index",
        &[t_base, t_pos2],
        &[t_idx],
        none.0,
        none.1,
    )?;
    gb.node(
        "reshape",
        "MASK_flatten",
        &[t_idx],
        &[t_flat],
        none.0,
        none.1,
    )?;
    gb.node(
        "gather_fwd_bf16",
        "MASK_gather",
        &[t_pat, t_flat],
        &[t_mask],
        pg_elems.0,
        pg_elems.1,
    )?;
    // ScatterND quadruples (b, g, 0, position_b) for update g + groups * b:
    // a constant `(b, g, 0, 0)` plus `(0, 0, 0, 1)` times the slot's
    // position, the add writing the flat `[4, groups * B]` the layers take.
    let (mut kbase, mut ksel) = (
        Vec::with_capacity(4 * groups * batch),
        Vec::with_capacity(4 * groups * batch),
    );
    for bi in 0..batch {
        for g in 0..groups {
            kbase.extend_from_slice(&[bi as i32, g as i32, 0, 0]);
            ksel.extend_from_slice(&[0, 0, 0, 1]);
        }
    }
    let t_kbase = gb.input_raw("KBASE", &[4, g64 * b], SYN_TYPE_INT32, &i32s(&kbase))?;
    let t_ksel = gb.input_raw("KSEL", &[4, g64, b], SYN_TYPE_INT32, &i32s(&ksel))?;
    let t_pos3 = gb.mid("pos_3d", &[1, 1, b], SYN_TYPE_INT32)?;
    let t_kp = gb.mid("kidx_pos", &[4, g64, b], SYN_TYPE_INT32)?;
    let t_kp2 = gb.mid("kidx_pos2", &[4, g64 * b], SYN_TYPE_INT32)?;
    let (t_kidx, n_kidx) = gb.output(N_KIDX, &[4, g64 * b], SYN_TYPE_INT32)?;
    gb.node(
        "reshape",
        "pos_reshape3",
        &[t_pos],
        &[t_pos3],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_i32",
        "kidx_scale",
        &[t_ksel, t_pos3],
        &[t_kp],
        none.0,
        none.1,
    )?;
    gb.node("reshape", "kidx_flatten", &[t_kp], &[t_kp2], none.0, none.1)?;
    gb.node(
        "add_fwd_i32",
        "kidx_add",
        &[t_kbase, t_kp2],
        &[t_kidx],
        none.0,
        none.1,
    )?;
    let out = Out {
        name: n_kidx,
        sizes: vec![4, g64 * b],
        kind: OutKind::I32,
    };
    Ok((gb, out))
}

/// An upload source of a layer weight: its host bytes, the scale applied
/// while staging, and the column window when the bytes are a strided view
/// (see `Store::upload`).
type Source<'w> = (std::borrow::Cow<'w, [u8]>, Option<f32>, Option<Stride>);

/// The bytes of a bf16 slice.
fn bf16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The bytes of an f32 vector converted to bf16 (as `Gb::input` does).
fn f32_as_bf16_bytes(v: &[f32]) -> Vec<u8> {
    bf16_bytes(&to_bf16(v))
}

/// Recipe A's weights of a layer as `(bytes, scale, stride)` upload
/// sources, in [`a_weight_names`] order, converted exactly as `build_a`
/// converts layer 0's (the scale applied while staging, the column window
/// gathered while staging).
fn a_weight_sources<'w>(w: &LayerWeights<'w>, hidden: usize) -> Vec<Source<'w>> {
    use std::borrow::Cow;
    let as_bytes = |s: &'w [u16]| -> Cow<'w, [u8]> {
        // SAFETY: a u16 slice is readable as twice as many bytes.
        Cow::Borrowed(unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 2) })
    };
    let scale = w.scale;
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    let mut v: Vec<Source<'w>> = vec![
        (Cow::Owned(f32_as_bf16_bytes(w.g1)), None, None),
        (
            as_bytes(w.wq),
            if w.qn.is_empty() { Some(scale) } else { None },
            None,
        ),
        (as_bytes(w.wk), None, None),
        (as_bytes(w.wv), None, None),
        (
            as_bytes(w.wo),
            None,
            (w.wo_pitch > 0).then_some(Stride {
                rows: hidden,
                cols: w.n_heads * w.head_dim,
                pitch: w.wo_pitch,
            }),
        ),
    ];
    if !w.bq.is_empty() {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|x| x * q_scale).collect();
        v.push((Cow::Owned(f32_as_bf16_bytes(&bq_scaled)), None, None));
        v.push((Cow::Owned(f32_as_bf16_bytes(w.bk)), None, None));
        v.push((Cow::Owned(f32_as_bf16_bytes(w.bv)), None, None));
    }
    if !w.qn.is_empty() {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|x| x * scale).collect();
        v.push((Cow::Owned(f32_as_bf16_bytes(&qn_scaled)), None, None));
        v.push((Cow::Owned(f32_as_bf16_bytes(w.kn)), None, None));
    }
    v
}

/// Recipe B's weights of a layer as upload sources, in
/// [`B_WEIGHT_NAMES`] order (see [`a_weight_sources`]).
fn b_weight_sources<'w>(w: &LayerWeights<'w>, hidden: usize, inter: usize) -> Vec<Source<'w>> {
    use std::borrow::Cow;
    let as_bytes = |s: &'w [u16]| -> Cow<'w, [u8]> {
        // SAFETY: a u16 slice is readable as twice as many bytes.
        Cow::Borrowed(unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 2) })
    };
    vec![
        (Cow::Owned(f32_as_bf16_bytes(w.g2)), None, None),
        (as_bytes(w.wg), None, None),
        (as_bytes(w.wu), None, None),
        (
            as_bytes(w.wd),
            None,
            (w.wd_pitch > 0).then_some(Stride {
                rows: hidden,
                cols: inter,
                pitch: w.wd_pitch,
            }),
        ),
    ]
}

/// The resident-set size of this process from `/proc/self/status`, in
/// bytes, if readable (for the load report).
#[must_use]
pub fn rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

impl std::fmt::Debug for TpModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TpModel(rank {} of {}, {} layers, hidden {}, vocab {}, batch {}, rows {}, capacity {}, kv heads {}, head_dim {}, pos {:?}, ring head {}, store {:.2} GB, mode {:?})",
            self.rank.rank,
            self.rank.world,
            self.layers.len(),
            self.hidden,
            self.vocab,
            self.batch,
            self.rows,
            self.capacity,
            self.n_kv,
            self.head_dim,
            self.pos,
            self.head,
            self.store.bytes as f64 / 1e9,
            self.mode
        )
    }
}
