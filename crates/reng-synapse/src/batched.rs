//! Batched decode: `B` independent sequences advance one token per launch of
//! one recipe, each with its own position, RoPE row, mask, placement column
//! and slot of the KV cache. Prompts are prefilled one sequence at a time
//! with the wide single-sequence recipe of `cached.rs`, bound to that
//! sequence's cache slot.
//!
//! The cache is a pair of 5-D buffers `[hd, capacity, 1, groups, B]` per
//! layer (slot `b` is a contiguous block), swapped every batched launch. A
//! prefill of sequence `b` chooses its first write buffer so that its last
//! launch lands in the buffer the next batched step reads.

use crate::model::{
    Gb, MASK_NEG, ModelWeights, Shared, SharedBatched, build_head, build_layer,
    build_layer_batched, cache_names,
};
use crate::runtime::{Out, Runtime};
use reng_core::Result;

struct Inputs {
    x: usize,
    sin: usize,
    cos: usize,
    mask: usize,
    place: usize,
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
    vocab: usize,
    head_dim: usize,
    n_kv: usize,
    /// Position of each sequence.
    pos: Vec<usize>,
    sin: Vec<f32>,
    cos: Vec<f32>,
    /// Per layer: the two K buffers and the two V buffers (5-D, all slots).
    slots: Vec<([u64; 2], [u64; 2])>,
    /// Which buffer the next batched launch reads.
    parity: usize,
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
        let (h, hd64, keys, b) = (hidden as u64, hd as u64, capacity as u64, batch as u64);

        // The batched decode graph (the parent: it owns the weights and cache).
        let mut gb = Gb::new()?;
        let t_x = gb.input("XB", &[h, b], &vec![0.0; batch * hidden])?;
        let t_sin = gb.input("SINB", &[hd64, 1, 1, 1, b], &vec![0.0; batch * hd])?;
        let t_cos = gb.input("COSB", &[hd64, 1, 1, 1, b], &vec![0.0; batch * hd])?;
        let t_mask = gb.input("MASKB", &[keys, 1, 1, 1, b], &vec![0.0; batch * capacity])?;
        let t_place = gb.input("PLACEB", &[1, keys, 1, 1, b], &vec![0.0; batch * capacity])?;
        let sh = SharedBatched {
            sin: t_sin,
            cos: t_cos,
            mask: t_mask,
            place: t_place,
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
            place: rt.input_index("PLACEB"),
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
            place: pf.input_index("PLACE"),
        };

        let mut slots = Vec::with_capacity(m.layers.len());
        for li in 0..m.layers.len() {
            let (kci, vci, kco, vco) = cache_names(li);
            slots.push((
                [rt.addr(&kci), rt.addr(&kco)],
                [rt.addr(&vci), rt.addr(&vco)],
            ));
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
            vocab,
            head_dim: hd,
            n_kv: l0.n_kv_heads,
            pos: vec![0; batch],
            sin: sin.to_vec(),
            cos: cos.to_vec(),
            slots,
            parity: 0,
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
        let (t, h, hd64, keys) = (rows as u64, hidden as u64, hd as u64, capacity as u64);
        let mut gb = Gb::new()?;
        let t_x = gb.input("X", &[h, t], &vec![0.0; rows * hidden])?;
        let t_sin = gb.input("SIN", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_cos = gb.input("COS", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_mask = gb.input("MASK", &[keys, t, 1, 1], &vec![0.0; rows * capacity])?;
        let t_place = gb.input("PLACE", &[t, keys, 1, 1], &vec![0.0; rows * capacity])?;
        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            mask: Some(t_mask),
            cache: Some(capacity),
            place: Some(t_place),
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
        (self.head_dim * self.capacity * self.n_kv * 2) as u64
    }

    /// Start sequence `b` afresh: zero its cache slots (the placement adds
    /// into them) and reset its position.
    ///
    /// # Errors
    ///
    /// Returns an error if the device memset fails.
    pub fn reset(&mut self, b: usize) -> Result<()> {
        let bytes = self.slot_bytes();
        let off = b as u64 * bytes;
        for (k, v) in &self.slots {
            for &a in k.iter().chain(v.iter()) {
                self.rt.zero(a + off, bytes)?;
            }
        }
        self.rt.settle()?;
        self.pos[b] = 0;
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
        let (h, hd, p, c) = (self.hidden, self.head_dim, self.rows, self.capacity);
        assert_eq!(x.len() % h, 0);
        let n_total = x.len() / h;
        assert!(n_total >= 1 && b < self.batch);
        assert!(
            self.pos[b] + n_total <= c,
            "cache overflow for sequence {b}"
        );
        let launches = n_total.div_ceil(p);
        // The last launch must write the buffer the next batched step reads.
        let mut wr = if launches % 2 == 1 {
            self.parity
        } else {
            1 - self.parity
        };
        let off = b as u64 * self.slot_bytes();
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
            let mut mb = vec![MASK_NEG; p * c];
            for q in 0..p {
                mb[q * c..q * c + (pos + q + 1).min(c)].fill(0.0);
            }
            let mut pb = vec![0.0f32; c * p];
            for r in 0..n {
                pb[(pos + r) * p + r] = 1.0;
            }
            let rd = 1 - wr;
            for (li, (k, v)) in self.slots.iter().enumerate() {
                let (kci, vci, kco, vco) = cache_names(li);
                self.pf.rebind(&kci, k[rd] + off);
                self.pf.rebind(&kco, k[wr] + off);
                self.pf.rebind(&vci, v[rd] + off);
                self.pf.rebind(&vco, v[wr] + off);
            }
            self.pf.upload(self.pf_ix.x, &xb)?;
            self.pf.upload(self.pf_ix.sin, &sb)?;
            self.pf.upload(self.pf_ix.cos, &cb)?;
            self.pf.upload(self.pf_ix.mask, &mb)?;
            self.pf.upload(self.pf_ix.place, &pb)?;
            self.pf.fence_uploads(self.pf_ix.place)?;
            let logits = self.pf.launch_and_read(n)?;
            last = logits[(n - 1) * self.vocab..].to_vec();
            self.pos[b] += n;
            wr = 1 - wr;
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
        let mut sb = vec![0.0f32; nb * hd];
        let mut cb = vec![0.0f32; nb * hd];
        let mut mb = vec![MASK_NEG; nb * c];
        let mut pb = vec![0.0f32; nb * c];
        for b in 0..nb {
            let pos = self.pos[b];
            sb[b * hd..(b + 1) * hd].copy_from_slice(&self.sin[pos * hd..(pos + 1) * hd]);
            cb[b * hd..(b + 1) * hd].copy_from_slice(&self.cos[pos * hd..(pos + 1) * hd]);
            mb[b * c..b * c + pos + 1].fill(0.0);
            pb[b * c + pos] = 1.0;
        }
        let (rd, wr) = (self.parity, 1 - self.parity);
        for (li, (k, v)) in self.slots.iter().enumerate() {
            let (kci, vci, kco, vco) = cache_names(li);
            self.rt.rebind(&kci, k[rd]);
            self.rt.rebind(&kco, k[wr]);
            self.rt.rebind(&vci, v[rd]);
            self.rt.rebind(&vco, v[wr]);
        }
        self.rt.upload(self.ix.x, x)?;
        self.rt.upload(self.ix.sin, &sb)?;
        self.rt.upload(self.ix.cos, &cb)?;
        self.rt.upload(self.ix.mask, &mb)?;
        self.rt.upload(self.ix.place, &pb)?;
        self.rt.fence_uploads(self.ix.place)?;
        let logits = self.rt.launch_and_read(nb)?;
        self.parity = wr;
        for pos in &mut self.pos {
            *pos += 1;
        }
        Ok(logits)
    }
}
