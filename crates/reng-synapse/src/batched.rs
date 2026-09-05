//! Batched decode: `B` independent sequences advance one token per launch of
//! one recipe, each with its own position, RoPE row, mask, placement column
//! and slot of the KV cache. Prompts are prefilled one sequence at a time
//! with the wide single-sequence recipe of `cached.rs`, bound to that
//! sequence's cache slot.
//!
//! The cache is one 5-D buffer `[hd, cap + 1, 1, groups, B]` per layer for
//! keys and one for values (slot `b` is a contiguous block), updated in
//! place by ScatterND nodes with per-step index tensors. A prefill of
//! sequence `b` binds the wide recipe's cache tensors (input and aliased
//! output) to slot `b`.
//!
//! Attention reads the whole cache every step, so `cap` tracks the longest
//! live sequence rather than the configured capacity: the recipes are
//! compiled for a bucket (256, 512, ... up to the capacity) and when a
//! sequence outgrows it the decode and prefill recipes are recompiled for
//! the next bucket and the used cache rows are copied across. At batch 64
//! on a 24-layer model the cache read is the step, so a 1024-position cache
//! holding 160-token sequences costs four times what it should.

use crate::f32_to_bf16;
use crate::model::{
    Gb, MASK_NEG, ModelWeights, RopeTables, Shared, SharedBatched, build_head, build_layer,
    build_layer_batched, cache_names, common_window, fused_sdpa, uses_full_mask, uses_local_rope,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::{Out, Runtime};
use reng_core::Result;

/// Smallest cache bucket (positions); `RENG_MIN_CAP` overrides it (a huge
/// value disables bucketing, a tiny one exercises growth in tests).
const MIN_BUCKET: usize = 256;

/// Per-step input indices of a recipe; the second RoPE rows and the two
/// masks exist only when some layer reads them.
#[derive(Clone, Copy)]
struct Inputs {
    x: usize,
    sin: usize,
    cos: usize,
    sin_local: Option<usize>,
    cos_local: Option<usize>,
    mask: Option<usize>,
    mask_window: Option<usize>,
    kidx: usize,
}

/// A batched decode recipe with its resident weights, a `B`-slot KV cache,
/// and a prefill recipe sharing the weights.
pub struct BatchedModel<'a> {
    /// Prefill recipe for the current bucket (drops first: it borrows the
    /// decode runtime's device and weights).
    pf: Runtime<'a>,
    pf_ix: Inputs,
    /// Decode recipe for the current bucket when that is not `base`'s.
    cur: Option<Runtime<'a>>,
    ix: Inputs,
    /// The first decode recipe: owns the device and the weights every later
    /// recipe binds to.
    base: Runtime<'a>,
    m: ModelWeights<'a>,
    batch: usize,
    rows: usize,
    /// Configured capacity (positions): the hard limit and the RoPE table
    /// length.
    capacity: usize,
    /// Current bucket (positions the recipes are compiled for).
    cap: usize,
    min_cap: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    head_dim: usize,
    n_kv: usize,
    /// Position of each sequence.
    pos: Vec<usize>,
    /// RoPE tables `[capacity, head_dim]`, and the second pair for the
    /// layers with `local_rope` (empty otherwise).
    sin: Vec<f32>,
    cos: Vec<f32>,
    sin_local: Vec<f32>,
    cos_local: Vec<f32>,
    /// Per layer: the K buffer and the V buffer (5-D, all slots) of the
    /// current bucket.
    slots: Vec<(u64, u64)>,
    /// Sliding window of the windowed layers.
    window: Option<usize>,
}

/// The bucket for `need` positions: the smallest of `min_cap`, doubling,
/// that holds them, clamped to `capacity`.
fn bucket_for(need: usize, min_cap: usize, capacity: usize) -> usize {
    let mut c = min_cap.max(1);
    while c < need {
        c *= 2;
    }
    c.min(capacity)
}

impl<'a> BatchedModel<'a> {
    /// Compile the batched decode recipe for `batch` sequences over caches of
    /// up to `capacity` positions (starting from the smallest bucket), plus a
    /// prefill recipe for blocks of `rows` positions sharing the weights.
    /// `rope` holds RoPE tables `[capacity, head_dim]` (the local pair only
    /// when a layer reads it); the layers' own RoPE slices are unused here.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `layers` is empty, a buffer length disagrees with the sizes,
    /// or `batch`, `rows` or `capacity` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        batch: usize,
        rows: usize,
        capacity: usize,
        rope: &RopeTables<'_>,
    ) -> Result<Self> {
        assert!(!m.layers.is_empty() && batch > 0 && rows > 0 && capacity > 0);
        let l0 = &m.layers[0];
        let hd = l0.head_dim;
        assert_eq!(rope.sin.len(), capacity * hd);
        assert_eq!(rope.cos.len(), capacity * hd);
        if uses_local_rope(&m.layers) {
            assert_eq!(rope.sin_local.len(), capacity * hd);
            assert_eq!(rope.cos_local.len(), capacity * hd);
        }
        assert_eq!(m.final_gamma.len(), hidden);
        assert_eq!(m.lm_head.len(), hidden * vocab);
        let min_cap = std::env::var("RENG_MIN_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MIN_BUCKET);
        let cap = bucket_for(1, min_cap, capacity);

        let (gb, out) = Self::build_decode(&m, hidden, inter, vocab, batch, cap, "")?;
        let base = Runtime::new(gb, out)?;
        let ix = Self::decode_inputs(&base, "");
        let (gb, out) = Self::build_prefill(&m, hidden, inter, vocab, rows, cap)?;
        let pf = Runtime::new_with(gb, out, Some(&base))?;
        let pf_ix = Self::prefill_inputs(&pf);
        let slots = Self::cache_slots(&base, m.layers.len());
        let n_kv = l0.n_kv_heads;
        let window = common_window(&m.layers);
        Ok(Self {
            pf,
            pf_ix,
            cur: None,
            ix,
            base,
            m,
            batch,
            rows,
            capacity,
            cap,
            min_cap,
            hidden,
            inter,
            vocab,
            head_dim: hd,
            n_kv,
            pos: vec![0; batch],
            sin: rope.sin.to_vec(),
            cos: rope.cos.to_vec(),
            sin_local: rope.sin_local.to_vec(),
            cos_local: rope.cos_local.to_vec(),
            slots,
            window,
        })
    }

    /// The decode graph for a `cap`-position bucket. Per-step inputs carry
    /// `tag` in their names so that a later bucket's recipe, a child of
    /// `base`, gets its own (uploadable) buffers rather than binding to
    /// `base`'s by name.
    #[allow(clippy::too_many_arguments)]
    fn build_decode(
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        batch: usize,
        cap: usize,
        tag: &str,
    ) -> Result<(Gb<'a>, Out)> {
        let l0 = &m.layers[0];
        let hd = l0.head_dim;
        let (h, hd64, keys, b) = (hidden as u64, hd as u64, cap as u64 + 1, batch as u64);
        let groups = l0.n_kv_heads as u64;
        let mut gb = Gb::new()?;
        let t_x = gb.input(&format!("XB{tag}"), &[h, b], &vec![0.0; batch * hidden])?;
        let t_sin = gb.input(
            &format!("SINB{tag}"),
            &[hd64, 1, 1, 1, b],
            &vec![0.0; batch * hd],
        )?;
        let t_cos = gb.input(
            &format!("COSB{tag}"),
            &[hd64, 1, 1, 1, b],
            &vec![0.0; batch * hd],
        )?;
        let (t_sin_local, t_cos_local) = if uses_local_rope(&m.layers) {
            (
                Some(gb.input(
                    &format!("SINLB{tag}"),
                    &[hd64, 1, 1, 1, b],
                    &vec![0.0; batch * hd],
                )?),
                Some(gb.input(
                    &format!("COSLB{tag}"),
                    &[hd64, 1, 1, 1, b],
                    &vec![0.0; batch * hd],
                )?),
            )
        } else {
            (None, None)
        };
        let zero_mask = vec![0.0; batch * (cap + 1)];
        let t_mask = if uses_full_mask(&m.layers) {
            Some(gb.input(&format!("MASKB{tag}"), &[keys, 1, 1, 1, b], &zero_mask)?)
        } else {
            None
        };
        let t_mask_window = if common_window(&m.layers).is_some() {
            Some(gb.input(&format!("MASKWB{tag}"), &[keys, 1, 1, 1, b], &zero_mask)?)
        } else {
            None
        };
        let t_kidx = gb.input_raw(
            &format!("KIDXB{tag}"),
            &[4, groups * b],
            SYN_TYPE_INT32,
            &vec![0u8; 16 * l0.n_kv_heads * batch],
        )?;
        let sh = SharedBatched {
            sin: t_sin,
            cos: t_cos,
            sin_local: t_sin_local,
            cos_local: t_cos_local,
            mask: t_mask,
            mask_window: t_mask_window,
            kidx: t_kidx,
            capacity: cap,
            batch,
            sdpa: fused_sdpa(false),
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer_batched(&mut gb, li, cur, lw, &sh, hidden, inter)?;
        }
        let out = build_head(&mut gb, cur, m, batch, hidden, vocab, true, None)?;
        Ok((gb, out))
    }

    fn build_prefill(
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        cap: usize,
    ) -> Result<(Gb<'a>, Out)> {
        let hd = m.layers[0].head_dim;
        let groups = m.layers[0].n_kv_heads;
        let (t, h, hd64, keys) = (rows as u64, hidden as u64, hd as u64, cap as u64 + 1);
        let mut gb = Gb::new()?;
        let t_x = gb.input("X", &[h, t], &vec![0.0; rows * hidden])?;
        let t_sin = gb.input("SIN", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_cos = gb.input("COS", &[hd64, t], &vec![0.0; rows * hd])?;
        let (t_sin_local, t_cos_local) = if uses_local_rope(&m.layers) {
            (
                Some(gb.input("SINL", &[hd64, t], &vec![0.0; rows * hd])?),
                Some(gb.input("COSL", &[hd64, t], &vec![0.0; rows * hd])?),
            )
        } else {
            (None, None)
        };
        let zero_mask = vec![0.0; rows * (cap + 1)];
        let t_mask = if uses_full_mask(&m.layers) {
            Some(gb.input("MASK", &[keys, t, 1, 1], &zero_mask)?)
        } else {
            None
        };
        let t_mask_window = if common_window(&m.layers).is_some() {
            Some(gb.input("MASKW", &[keys, t, 1, 1], &zero_mask)?)
        } else {
            None
        };
        let t_kidx = gb.input_raw(
            "KIDX",
            &[3, t * groups as u64],
            SYN_TYPE_INT32,
            &vec![0u8; 12 * rows * groups],
        )?;
        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            sin_local: t_sin_local,
            cos_local: t_cos_local,
            mask: t_mask,
            mask_window: t_mask_window,
            cache: Some(cap),
            kidx: Some(t_kidx),
            inplace: false,
            sdpa: fused_sdpa(false),
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer(&mut gb, li, cur, lw, &sh, rows, hidden, inter, None)?;
        }
        let out = build_head(&mut gb, cur, m, rows, hidden, vocab, true, None)?;
        Ok((gb, out))
    }

    fn decode_inputs(rt: &Runtime<'_>, tag: &str) -> Inputs {
        Inputs {
            x: rt.input_index(&format!("XB{tag}")),
            sin: rt.input_index(&format!("SINB{tag}")),
            cos: rt.input_index(&format!("COSB{tag}")),
            sin_local: rt.find_input(&format!("SINLB{tag}")),
            cos_local: rt.find_input(&format!("COSLB{tag}")),
            mask: rt.find_input(&format!("MASKB{tag}")),
            mask_window: rt.find_input(&format!("MASKWB{tag}")),
            kidx: rt.input_index(&format!("KIDXB{tag}")),
        }
    }

    fn prefill_inputs(rt: &Runtime<'_>) -> Inputs {
        Inputs {
            x: rt.input_index("X"),
            sin: rt.input_index("SIN"),
            cos: rt.input_index("COS"),
            sin_local: rt.find_input("SINL"),
            cos_local: rt.find_input("COSL"),
            mask: rt.find_input("MASK"),
            mask_window: rt.find_input("MASKW"),
            kidx: rt.input_index("KIDX"),
        }
    }

    fn cache_slots(rt: &Runtime<'_>, layers: usize) -> Vec<(u64, u64)> {
        (0..layers)
            .map(|li| {
                let (kci, vci, _, _) = cache_names(li);
                (rt.addr(&kci), rt.addr(&vci))
            })
            .collect()
    }

    /// The decode runtime of the current bucket.
    fn rt(&mut self) -> &mut Runtime<'a> {
        match self.cur {
            Some(ref mut r) => r,
            None => &mut self.base,
        }
    }

    /// Number of sequences.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Prefill block size.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Position of sequence `b`.
    #[must_use]
    pub fn position(&self, b: usize) -> usize {
        self.pos[b]
    }

    /// Positions the recipes are currently compiled for.
    #[must_use]
    pub fn bucket(&self) -> usize {
        self.cap
    }

    /// Bytes of one sequence's slot in one cache buffer of `cap` positions.
    fn slot_bytes(&self, cap: usize) -> u64 {
        (self.head_dim * (cap + 1) * self.n_kv * 2) as u64
    }

    /// Start sequence `b` afresh. Its cache slot needs no clearing (the mask
    /// never admits positions at or beyond its position), so this is free.
    pub fn reset(&mut self, b: usize) {
        self.pos[b] = 0;
    }

    /// Make the current bucket hold `need` positions: recompile both recipes
    /// for the next bucket and move every sequence's used cache rows.
    fn ensure(&mut self, need: usize) -> Result<()> {
        assert!(need <= self.capacity, "cache overflow: {need} positions");
        if need <= self.cap {
            return Ok(());
        }
        let cap = bucket_for(need, self.min_cap, self.capacity);
        let tag = format!("_{cap}");
        let (gb, out) = Self::build_decode(
            &self.m,
            self.hidden,
            self.inter,
            self.vocab,
            self.batch,
            cap,
            &tag,
        )?;
        let mut rt = Runtime::new_with(gb, out, Some(&self.base))?;
        let slots = Self::cache_slots(&rt, self.m.layers.len());
        // Per layer, buffer, sequence and group: the rows written so far.
        let (hd, old_keys, new_keys) = (self.head_dim, self.cap + 1, cap + 1);
        let (old_slot, new_slot) = (self.slot_bytes(self.cap), self.slot_bytes(cap));
        let mut copies = Vec::new();
        for (old, new) in self.slots.iter().zip(&slots) {
            for (src, dst) in [(old.0, new.0), (old.1, new.1)] {
                for (b, &pos) in self.pos.iter().enumerate() {
                    if pos == 0 {
                        continue;
                    }
                    for g in 0..self.n_kv as u64 {
                        copies.push((
                            src + b as u64 * old_slot + g * (hd * old_keys * 2) as u64,
                            dst + b as u64 * new_slot + g * (hd * new_keys * 2) as u64,
                            (hd * pos * 2) as u64,
                        ));
                    }
                }
            }
        }
        rt.copy_d2d(&copies)?;
        let (gb, out) =
            Self::build_prefill(&self.m, self.hidden, self.inter, self.vocab, self.rows, cap)?;
        let pf = Runtime::new_with(gb, out, Some(&rt))?;
        self.pf_ix = Self::prefill_inputs(&pf);
        self.ix = Self::decode_inputs(&rt, &tag);
        // The old prefill recipe goes before the old decode recipe it binds
        // to; `base` stays for the weights.
        self.pf = pf;
        self.cur = Some(rt);
        self.slots = slots;
        self.cap = cap;
        Ok(())
    }

    /// Feed `x` (`[n, hidden]` embeddings) to sequence `b` at its current
    /// position through the prefill recipe and return the last row's logits.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `x` is empty, not a whole number of rows, or overflows the
    /// cache.
    pub fn prefill(&mut self, b: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.prefill_rows(b, x, true)
    }

    /// Like [`BatchedModel::prefill`] but returns only the argmax id of the
    /// last row, computed on the device.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    pub fn prefill_id(&mut self, b: usize, x: &[f32]) -> Result<u32> {
        let v = self.prefill_rows(b, x, false)?;
        Ok(v[0] as u32)
    }

    fn prefill_rows(&mut self, b: usize, x: &[f32], want_logits: bool) -> Result<Vec<f32>> {
        let (h, hd, p) = (self.hidden, self.head_dim, self.rows);
        assert_eq!(x.len() % h, 0);
        let n_total = x.len() / h;
        assert!(n_total >= 1 && b < self.batch);
        assert!(
            self.pos[b] + n_total <= self.capacity,
            "cache overflow for sequence {b}"
        );
        self.ensure(self.pos[b] + n_total)?;
        let c = self.cap;
        let off = b as u64 * self.slot_bytes(c);
        let neg = f32_to_bf16(MASK_NEG);
        // The wide recipe's ScatterND is not in place: each block reads one
        // buffer and writes another. The blocks alternate between this
        // sequence's slot and the recipe's own buffers, starting so that the
        // last block lands in the slot; the first block's input is stale
        // (its positions are masked).
        let n_blocks = n_total.div_ceil(p);
        let mut last = Vec::new();
        for (i, chunk) in x.chunks(p * h).enumerate() {
            let to_slot = (n_blocks - 1 - i) % 2 == 0;
            for (li, (k, v)) in self.slots.iter().enumerate() {
                let (kci, vci, kco, vco) = cache_names(li);
                let (k_own, v_own) = (self.pf.addr(&kco), self.pf.addr(&vco));
                if to_slot {
                    self.pf.rebind(&kci, k_own);
                    self.pf.rebind(&vci, v_own);
                    self.pf.rebind(&kco, k + off);
                    self.pf.rebind(&vco, v + off);
                } else {
                    self.pf.rebind(&kci, k + off);
                    self.pf.rebind(&vci, v + off);
                    self.pf.rebind(&kco, k_own);
                    self.pf.rebind(&vco, v_own);
                }
            }
            let n = chunk.len() / h;
            let pos = self.pos[b];
            let mut xb = vec![0.0f32; p * h];
            xb[..n * h].copy_from_slice(chunk);
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
            let keys = c + 1;
            let mask_rows = |window: Option<usize>| -> Vec<u16> {
                let mut mb = vec![neg; p * keys];
                for q in 0..p {
                    let end = (pos + q + 1).min(c);
                    let start = window.map_or(0, |w| (pos + q + 1).saturating_sub(w));
                    mb[q * keys + start..q * keys + end].fill(0);
                }
                mb
            };
            let mut ib: Vec<u8> = Vec::with_capacity(12 * p * self.n_kv);
            for g in 0..self.n_kv {
                for r in 0..p {
                    let target = if r < n { pos + r } else { c };
                    for v in [g as i32, 0i32, target as i32] {
                        ib.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let ix = self.pf_ix;
            self.pf.upload(ix.x, &xb)?;
            self.pf.upload(ix.sin, &rope_rows(&self.sin))?;
            self.pf.upload(ix.cos, &rope_rows(&self.cos))?;
            if let (Some(is), Some(ic)) = (ix.sin_local, ix.cos_local) {
                self.pf.upload(is, &rope_rows(&self.sin_local))?;
                self.pf.upload(ic, &rope_rows(&self.cos_local))?;
            }
            if let Some(im) = ix.mask {
                self.pf.upload_bf16(im, &mask_rows(None))?;
            }
            if let Some(im) = ix.mask_window {
                self.pf.upload_bf16(im, &mask_rows(self.window))?;
            }
            self.pf.upload_raw(ix.kidx, &ib)?;
            self.pf.fence()?;
            let ids = self.pf.launch_and_read_i32(n - 1, 1)?;
            last = if want_logits {
                self.pf.read_bf16_range("LOGITS", n - 1, 1)?
            } else {
                vec![ids[0] as f32]
            };
            self.pos[b] += n;
        }
        Ok(last)
    }

    /// Advance every sequence by one token: `x` is `[B, hidden]` (one
    /// embedding per sequence) and the result is logits `[B, vocab]`.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `B` rows or a sequence would overflow the cache.
    pub fn step(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        self.step_rows(x, true)
    }

    /// Like [`BatchedModel::step`] but returns one argmax token id per
    /// sequence, computed on the device.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    pub fn step_ids(&mut self, x: &[f32]) -> Result<Vec<u32>> {
        let v = self.step_rows(x, false)?;
        Ok(v.iter().map(|&i| i as u32).collect())
    }

    fn step_rows(&mut self, x: &[f32], want_logits: bool) -> Result<Vec<f32>> {
        let (h, hd, nb) = (self.hidden, self.head_dim, self.batch);
        assert_eq!(x.len(), nb * h);
        let furthest = self.pos.iter().copied().max().unwrap_or(0);
        assert!(furthest < self.capacity, "cache overflow");
        self.ensure(furthest + 1)?;
        let c = self.cap;
        let neg = f32_to_bf16(MASK_NEG);
        let keys = c + 1;
        // One RoPE row per sequence from a table.
        let rope_rows = |table: &[f32]| -> Vec<f32> {
            let mut rows = vec![0.0f32; nb * hd];
            for (b, &pos) in self.pos.iter().enumerate() {
                rows[b * hd..(b + 1) * hd].copy_from_slice(&table[pos * hd..(pos + 1) * hd]);
            }
            rows
        };
        // One mask row per sequence: its positions up to its own, and with
        // a window none further back than it.
        let mask_rows = |window: Option<usize>| -> Vec<u16> {
            let mut mb = vec![neg; nb * keys];
            for (b, &pos) in self.pos.iter().enumerate() {
                let start = window.map_or(0, |w| (pos + 1).saturating_sub(w));
                mb[b * keys + start..b * keys + pos + 1].fill(0);
            }
            mb
        };
        // Scatter indices, ONNX (b, g, 0, position_b) for update g + groups * b.
        let mut ib: Vec<u8> = Vec::with_capacity(16 * self.n_kv * nb);
        for (b, &pos) in self.pos.iter().enumerate() {
            for g in 0..self.n_kv {
                for v in [b as i32, g as i32, 0i32, pos as i32] {
                    ib.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let ix = self.ix;
        let sb = rope_rows(&self.sin);
        let cb = rope_rows(&self.cos);
        let local = ix
            .sin_local
            .map(|_| (rope_rows(&self.sin_local), rope_rows(&self.cos_local)));
        let mb = ix.mask.map(|_| mask_rows(None));
        let mbw = ix.mask_window.map(|_| mask_rows(self.window));
        let rt = self.rt();
        rt.upload(ix.x, x)?;
        rt.upload(ix.sin, &sb)?;
        rt.upload(ix.cos, &cb)?;
        if let (Some(is), Some(ic), Some((slb, clb))) = (ix.sin_local, ix.cos_local, &local) {
            rt.upload(is, slb)?;
            rt.upload(ic, clb)?;
        }
        if let (Some(im), Some(mb)) = (ix.mask, &mb) {
            rt.upload_bf16(im, mb)?;
        }
        if let (Some(im), Some(mbw)) = (ix.mask_window, &mbw) {
            rt.upload_bf16(im, mbw)?;
        }
        rt.upload_raw(ix.kidx, &ib)?;
        rt.fence()?;
        let ids = rt.launch_and_read_i32(0, nb)?;
        let logits = if want_logits {
            rt.read_bf16_range("LOGITS", 0, nb)?
        } else {
            ids.iter().map(|&i| i as f32).collect()
        };
        for pos in &mut self.pos {
            *pos += 1;
        }
        Ok(logits)
    }
}
