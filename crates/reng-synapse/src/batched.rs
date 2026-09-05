//! Batched decode: `B` independent sequences advance one token per launch of
//! one recipe, each with its own position, RoPE row, mask, placement column
//! and slot of the KV cache. Prompts are prefilled one sequence at a time
//! with the wide single-sequence recipe of `cached.rs`, bound to that
//! sequence's cache slot.
//!
//! The cache is one 5-D buffer `[hd, capacity + 1, 1, groups, B]` per layer
//! for keys and one for values (slot `b` is a contiguous block), updated in
//! place by ScatterND nodes with per-step index tensors. A prefill of
//! sequence `b` binds the wide recipe's cache tensors (input and aliased
//! output) to slot `b`.

use crate::f32_to_bf16;
use crate::model::{
    Gb, MASK_NEG, ModelWeights, Shared, SharedBatched, build_head, build_layer,
    build_layer_batched, cache_names,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::{Out, Runtime};
use reng_core::Result;

struct Inputs {
    x: usize,
    sin: usize,
    cos: usize,
    mask: usize,
    kidx: usize,
}

/// A batched decode recipe with its resident weights, a `B`-slot KV cache,
/// and a prefill recipe sharing the weights.
pub struct BatchedModel {
    /// Prefill recipe (drops first: it borrows `rt`'s device and weights).
    pf: Runtime,
    pf_ix: Inputs,
    rt: Runtime,
    ix: Inputs,
    batch: usize,
    rows: usize,
    capacity: usize,
    hidden: usize,
    head_dim: usize,
    n_kv: usize,
    /// Position of each sequence.
    pos: Vec<usize>,
    sin: Vec<f32>,
    cos: Vec<f32>,
    /// Per layer: the K buffer and the V buffer (5-D, all slots).
    slots: Vec<(u64, u64)>,
}

impl BatchedModel {
    /// Compile the batched decode recipe for `batch` sequences over caches of
    /// `capacity` positions, plus a prefill recipe for blocks of `rows`
    /// positions sharing the weights. `sin`/`cos` are RoPE tables
    /// `[capacity, head_dim]`.
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
        m: &ModelWeights<'_>,
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
        let (h, hd64, keys, b) = (hidden as u64, hd as u64, capacity as u64 + 1, batch as u64);
        let groups = l0.n_kv_heads as u64;

        // The batched decode graph (the parent: it owns the weights and cache).
        let mut gb = Gb::new()?;
        let t_x = gb.input("XB", &[h, b], &vec![0.0; batch * hidden])?;
        let t_sin = gb.input("SINB", &[hd64, 1, 1, 1, b], &vec![0.0; batch * hd])?;
        let t_cos = gb.input("COSB", &[hd64, 1, 1, 1, b], &vec![0.0; batch * hd])?;
        let t_mask = gb.input(
            "MASKB",
            &[keys, 1, 1, 1, b],
            &vec![0.0; batch * (capacity + 1)],
        )?;
        let t_kidx = gb.input_raw(
            "KIDXB",
            &[4, groups * b],
            SYN_TYPE_INT32,
            &vec![0u8; 16 * l0.n_kv_heads * batch],
        )?;
        let sh = SharedBatched {
            sin: t_sin,
            cos: t_cos,
            mask: t_mask,
            kidx: t_kidx,
            capacity,
            batch,
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer_batched(&mut gb, li, cur, lw, &sh, hidden, inter)?;
        }
        let out = build_head(&mut gb, cur, m, batch, hidden, vocab)?;
        let rt = Runtime::new(gb, out)?;
        let ix = Inputs {
            x: rt.input_index("XB"),
            sin: rt.input_index("SINB"),
            cos: rt.input_index("COSB"),
            mask: rt.input_index("MASKB"),
            kidx: rt.input_index("KIDXB"),
        };

        // The prefill graph (single sequence, `rows` per block); its cache
        // tensors are bound into the parent's slots per launch.
        let (gb, out) = Self::build_prefill(m, hidden, inter, vocab, rows, capacity)?;
        let pf = Runtime::new_with(gb, out, Some(&rt))?;
        let pf_ix = Inputs {
            x: pf.input_index("X"),
            sin: pf.input_index("SIN"),
            cos: pf.input_index("COS"),
            mask: pf.input_index("MASK"),
            kidx: pf.input_index("KIDX"),
        };

        let mut slots = Vec::with_capacity(m.layers.len());
        for li in 0..m.layers.len() {
            let (kci, vci, _, _) = cache_names(li);
            slots.push((rt.addr(&kci), rt.addr(&vci)));
        }
        Ok(Self {
            pf,
            pf_ix,
            rt,
            ix,
            batch,
            rows,
            capacity,
            hidden,
            head_dim: hd,
            n_kv: l0.n_kv_heads,
            pos: vec![0; batch],
            sin: sin.to_vec(),
            cos: cos.to_vec(),
            slots,
        })
    }

    fn build_prefill(
        m: &ModelWeights<'_>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        capacity: usize,
    ) -> Result<(Gb, Out)> {
        let hd = hidden / m.layers[0].n_heads;
        let groups = m.layers[0].n_kv_heads;
        let (t, h, hd64, keys) = (rows as u64, hidden as u64, hd as u64, capacity as u64 + 1);
        let mut gb = Gb::new()?;
        let t_x = gb.input("X", &[h, t], &vec![0.0; rows * hidden])?;
        let t_sin = gb.input("SIN", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_cos = gb.input("COS", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_mask = gb.input("MASK", &[keys, t, 1, 1], &vec![0.0; rows * (capacity + 1)])?;
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
            cache: Some(capacity),
            kidx: Some(t_kidx),
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer(&mut gb, li, cur, lw, &sh, rows, hidden, inter, None)?;
        }
        let out = build_head(&mut gb, cur, m, rows, hidden, vocab)?;
        Ok((gb, out))
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

    /// Bytes of one sequence's slot in one cache buffer.
    fn slot_bytes(&self) -> u64 {
        (self.head_dim * (self.capacity + 1) * self.n_kv * 2) as u64
    }

    /// Start sequence `b` afresh. Its cache slot needs no clearing (the mask
    /// never admits positions at or beyond its position), so this is free.
    pub fn reset(&mut self, b: usize) {
        self.pos[b] = 0;
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
        let (h, hd, p, c) = (self.hidden, self.head_dim, self.rows, self.capacity);
        assert_eq!(x.len() % h, 0);
        let n_total = x.len() / h;
        assert!(n_total >= 1 && b < self.batch);
        assert!(
            self.pos[b] + n_total <= c,
            "cache overflow for sequence {b}"
        );
        let off = b as u64 * self.slot_bytes();
        let neg = f32_to_bf16(MASK_NEG);
        // The wide recipe's cache tensors (input and aliased output) both
        // bind to this sequence's slot.
        for (li, (k, v)) in self.slots.iter().enumerate() {
            let (kci, vci, kco, vco) = cache_names(li);
            self.pf.rebind(&kci, k + off);
            self.pf.rebind(&kco, k + off);
            self.pf.rebind(&vci, v + off);
            self.pf.rebind(&vco, v + off);
        }
        let mut last = Vec::new();
        for chunk in x.chunks(p * h) {
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
            last = self.pf.launch_and_read_range(n - 1, 1)?;
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
        let (h, hd, c, nb) = (self.hidden, self.head_dim, self.capacity, self.batch);
        assert_eq!(x.len(), nb * h);
        for b in 0..nb {
            assert!(self.pos[b] < c, "cache overflow for sequence {b}");
        }
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
        self.rt.upload(self.ix.x, x)?;
        self.rt.upload(self.ix.sin, &sb)?;
        self.rt.upload(self.ix.cos, &cb)?;
        self.rt.upload_bf16(self.ix.mask, &mb)?;
        self.rt.upload_raw(self.ix.kidx, &ib)?;
        self.rt.fence()?;
        let logits = self.rt.launch_and_read(nb)?;
        for pos in &mut self.pos {
            *pos += 1;
        }
        Ok(logits)
    }
}
