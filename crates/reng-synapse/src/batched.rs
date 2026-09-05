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
    Gb, MASK_NEG, ModelWeights, Shared, SharedBatched, build_head, build_layer,
    build_layer_batched, cache_names,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::{Out, Runtime};
use reng_core::Result;

/// Smallest cache bucket (positions); `RENG_MIN_CAP` overrides it (a huge
/// value disables bucketing, a tiny one exercises growth in tests).
const MIN_BUCKET: usize = 256;

#[derive(Clone, Copy)]
struct Inputs {
    x: usize,
    sin: usize,
    cos: usize,
    mask: usize,
    kidx: usize,
}

/// A batched decode recipe with its resident weights, a `B`-slot KV cache,
/// and a prefill recipe sharing the weights.
pub struct BatchedModel<'a> {
    /// Prefill recipe for the current bucket (drops first: it borrows the
    /// decode runtime's device and weights).
    pf: Runtime,
    pf_ix: Inputs,
    /// Decode recipe for the current bucket when that is not `base`'s.
    cur: Option<Runtime>,
    ix: Inputs,
    /// The first decode recipe: owns the device and the weights every later
    /// recipe binds to.
    base: Runtime,
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
    sin: Vec<f32>,
    cos: Vec<f32>,
    /// Per layer: the K buffer and the V buffer (5-D, all slots) of the
    /// current bucket.
    slots: Vec<(u64, u64)>,
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
    /// `sin`/`cos` are RoPE tables `[capacity, head_dim]`; the layers' own
    /// RoPE slices are unused here.
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
        sin: &[f32],
        cos: &[f32],
    ) -> Result<Self> {
        assert!(!m.layers.is_empty() && batch > 0 && rows > 0 && capacity > 0);
        let l0 = &m.layers[0];
        let hd = hidden / l0.n_heads;
        assert_eq!(sin.len(), capacity * hd);
        assert_eq!(cos.len(), capacity * hd);
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
            sin: sin.to_vec(),
            cos: cos.to_vec(),
            slots,
        })
    }

    /// The decode graph for a `cap`-position bucket. Per-step inputs carry
    /// `tag` in their names so that a later bucket's recipe, a child of
    /// `base`, gets its own (uploadable) buffers rather than binding to
    /// `base`'s by name.
    #[allow(clippy::too_many_arguments)]
    fn build_decode(
        m: &ModelWeights<'_>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        batch: usize,
        cap: usize,
        tag: &str,
    ) -> Result<(Gb, Out)> {
        let l0 = &m.layers[0];
        let hd = hidden / l0.n_heads;
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
        let t_mask = gb.input(
            &format!("MASKB{tag}"),
            &[keys, 1, 1, 1, b],
            &vec![0.0; batch * (cap + 1)],
        )?;
        let t_kidx = gb.input_raw(
            &format!("KIDXB{tag}"),
            &[4, groups * b],
            SYN_TYPE_INT32,
            &vec![0u8; 16 * l0.n_kv_heads * batch],
        )?;
        let sh = SharedBatched {
            sin: t_sin,
            cos: t_cos,
            mask: t_mask,
            kidx: t_kidx,
            capacity: cap,
            batch,
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer_batched(&mut gb, li, cur, lw, &sh, hidden, inter)?;
        }
        let out = build_head(&mut gb, cur, m, batch, hidden, vocab, true)?;
        Ok((gb, out))
    }

    fn build_prefill(
        m: &ModelWeights<'_>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        cap: usize,
    ) -> Result<(Gb, Out)> {
        let hd = hidden / m.layers[0].n_heads;
        let groups = m.layers[0].n_kv_heads;
        let (t, h, hd64, keys) = (rows as u64, hidden as u64, hd as u64, cap as u64 + 1);
        let mut gb = Gb::new()?;
        let t_x = gb.input("X", &[h, t], &vec![0.0; rows * hidden])?;
        let t_sin = gb.input("SIN", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_cos = gb.input("COS", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_mask = gb.input("MASK", &[keys, t, 1, 1], &vec![0.0; rows * (cap + 1)])?;
        let t_kidx = gb.input_raw(
            "KIDX",
            &[3, t * groups as u64],
            SYN_TYPE_INT32,
            &vec![0u8; 12 * rows * groups],
        )?;
        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            mask: Some(t_mask),
            cache: Some(cap),
            kidx: Some(t_kidx),
            inplace: false,
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer(&mut gb, li, cur, lw, &sh, rows, hidden, inter, None)?;
        }
        let out = build_head(&mut gb, cur, m, rows, hidden, vocab, true)?;
        Ok((gb, out))
    }

    fn decode_inputs(rt: &Runtime, tag: &str) -> Inputs {
        Inputs {
            x: rt.input_index(&format!("XB{tag}")),
            sin: rt.input_index(&format!("SINB{tag}")),
            cos: rt.input_index(&format!("COSB{tag}")),
            mask: rt.input_index(&format!("MASKB{tag}")),
            kidx: rt.input_index(&format!("KIDXB{tag}")),
        }
    }

    fn prefill_inputs(rt: &Runtime) -> Inputs {
        Inputs {
            x: rt.input_index("X"),
            sin: rt.input_index("SIN"),
            cos: rt.input_index("COS"),
            mask: rt.input_index("MASK"),
            kidx: rt.input_index("KIDX"),
        }
    }

    fn cache_slots(rt: &Runtime, layers: usize) -> Vec<(u64, u64)> {
        (0..layers)
            .map(|li| {
                let (kci, vci, _, _) = cache_names(li);
                (rt.addr(&kci), rt.addr(&vci))
            })
            .collect()
    }

    /// The decode runtime of the current bucket.
    fn rt(&mut self) -> &mut Runtime {
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
            let mut sb = vec![0.0f32; p * hd];
            let mut cb = vec![1.0f32; p * hd];
            for r in 0..p {
                if pos + r < c {
                    let src = (pos + r) * hd;
                    sb[r * hd..(r + 1) * hd].copy_from_slice(&self.sin[src..src + hd]);
                    cb[r * hd..(r + 1) * hd].copy_from_slice(&self.cos[src..src + hd]);
                }
            }
            let keys = c + 1;
            let mut mb = vec![neg; p * keys];
            for q in 0..p {
                mb[q * keys..q * keys + (pos + q + 1).min(c)].fill(0);
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
            self.pf.upload(self.pf_ix.x, &xb)?;
            self.pf.upload(self.pf_ix.sin, &sb)?;
            self.pf.upload(self.pf_ix.cos, &cb)?;
            self.pf.upload_bf16(self.pf_ix.mask, &mb)?;
            self.pf.upload_raw(self.pf_ix.kidx, &ib)?;
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
        let mut sb = vec![0.0f32; nb * hd];
        let mut cb = vec![0.0f32; nb * hd];
        let mut mb = vec![neg; nb * keys];
        // Scatter indices, ONNX (b, g, 0, position_b) for update g + groups * b.
        let mut ib: Vec<u8> = Vec::with_capacity(16 * self.n_kv * nb);
        for b in 0..nb {
            let pos = self.pos[b];
            sb[b * hd..(b + 1) * hd].copy_from_slice(&self.sin[pos * hd..(pos + 1) * hd]);
            cb[b * hd..(b + 1) * hd].copy_from_slice(&self.cos[pos * hd..(pos + 1) * hd]);
            mb[b * keys..b * keys + pos + 1].fill(0);
            for g in 0..self.n_kv {
                for v in [b as i32, g as i32, 0i32, pos as i32] {
                    ib.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let ix = self.ix;
        let rt = self.rt();
        rt.upload(ix.x, x)?;
        rt.upload(ix.sin, &sb)?;
        rt.upload(ix.cos, &cb)?;
        rt.upload_bf16(ix.mask, &mb)?;
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
