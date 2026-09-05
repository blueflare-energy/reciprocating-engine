//! Decode with a KV cache: one recipe compiled once for a fixed block size and
//! cache capacity, launched once per block of new tokens.
//!
//! Each launch processes a block of `rows` positions (the real tokens first,
//! zero rows after them) starting at the current position. Every layer keeps
//! per-head key and value caches `[head_dim, capacity + rows]` on the device
//! as two buffers: the recipe reads one and writes the other as
//! `cache_out = cache_in + place @ block`, where `place` is a 0/1 matrix
//! uploaded per step that sends block row r to position pos + r; the buffers
//! swap roles every launch. Attention runs over the whole updated cache with
//! an additive mask that admits positions up to each query's own. All cache
//! data movement is done by the MME and TPC inside the recipe: on this stack
//! a DMA reading freshly compute-written memory can return stale bytes, so
//! neither a `memcpy` node nor a copy issued between launches is used.
//! Nothing but the logits crosses the PCIe bus per step.
//!
//! A prompt is fed as one or more full blocks and each generated token as a
//! block of one real row. The block size is a compile-time shape of the
//! recipe; a small block wastes MME columns but keeps the launch count at one
//! per token, which on this stack is what matters.

use crate::model::{Gb, MASK_NEG, ModelWeights, Shared, build_head, build_layer, cache_names};
use crate::runtime::Runtime;
use reng_core::Result;
use std::time::Instant;

/// A compiled decoder recipe with its resident weights and KV cache.
pub struct CachedModel {
    rt: Runtime,
    rows: usize,
    capacity: usize,
    hidden: usize,
    vocab: usize,
    head_dim: usize,
    n_kv: usize,
    pos: usize,
    /// RoPE tables `[capacity, head_dim]`.
    sin: Vec<f32>,
    cos: Vec<f32>,
    ix_x: usize,
    ix_sin: usize,
    ix_cos: usize,
    ix_mask: usize,
    ix_place: usize,
    /// Per layer and KV head: the two K buffers and the two V buffers.
    slots: Vec<([u64; 2], [u64; 2])>,
    /// Which buffer of each pair the next launch reads (the other is written).
    parity: usize,
}

impl CachedModel {
    /// Compile the recipe for blocks of `rows` positions over a cache of
    /// `capacity` positions and upload the weights. `sin`/`cos` are RoPE
    /// tables `[capacity, head_dim]`; the per-layer tables in `m` are unused.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `layers` is empty, a buffer length disagrees with the sizes,
    /// or `rows` or `capacity` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &ModelWeights<'_>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        capacity: usize,
        sin: &[f32],
        cos: &[f32],
    ) -> Result<Self> {
        assert!(!m.layers.is_empty() && rows > 0 && capacity > 0);
        let l0 = &m.layers[0];
        let hd = hidden / l0.n_heads;
        assert_eq!(sin.len(), capacity * hd);
        assert_eq!(cos.len(), capacity * hd);
        assert_eq!(m.final_gamma.len(), hidden);
        assert_eq!(m.lm_head.len(), hidden * vocab);
        let (t, h, hd64, keys) = (
            rows as u64,
            hidden as u64,
            hd as u64,
            (capacity + rows) as u64,
        );
        let mut gb = Gb::new()?;
        // Per-step inputs; their contents are replaced before every launch.
        let t_x = gb.input("X", &[h, t], &vec![0.0; rows * hidden])?;
        let t_sin = gb.input("SIN", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_cos = gb.input("COS", &[hd64, t], &vec![0.0; rows * hd])?;
        let t_mask = gb.input("MASK", &[keys, t], &vec![0.0; rows * (capacity + rows)])?;
        let t_place = gb.input("PLACE", &[t, keys], &vec![0.0; rows * (capacity + rows)])?;
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
        let rt = Runtime::new(gb, out)?;
        let mut slots = Vec::with_capacity(m.layers.len() * l0.n_kv_heads);
        for li in 0..m.layers.len() {
            for g in 0..l0.n_kv_heads {
                let (kci, vci, kco, vco) = cache_names(li, g);
                slots.push((
                    [rt.addr(&kci), rt.addr(&kco)],
                    [rt.addr(&vci), rt.addr(&vco)],
                ));
            }
        }
        Ok(Self {
            ix_x: rt.input_index("X"),
            ix_sin: rt.input_index("SIN"),
            ix_cos: rt.input_index("COS"),
            ix_mask: rt.input_index("MASK"),
            ix_place: rt.input_index("PLACE"),
            rt,
            rows,
            capacity,
            hidden,
            vocab,
            head_dim: hd,
            n_kv: l0.n_kv_heads,
            pos: 0,
            sin: sin.to_vec(),
            cos: cos.to_vec(),
            slots,
            parity: 0,
        })
    }

    /// Positions filled so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Block size (max real rows per [`CachedModel::step`]).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Cache capacity in positions.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forget the cached prefix. Both cache buffers are zeroed, since the
    /// placement adds into them.
    ///
    /// # Errors
    ///
    /// Returns an error if the device memset fails.
    pub fn reset(&mut self) -> Result<()> {
        let bytes = (self.head_dim * (self.capacity + self.rows) * 2) as u64;
        for (k, v) in &self.slots {
            for &a in k.iter().chain(v.iter()) {
                self.rt.zero(a, bytes)?;
            }
        }
        self.rt.settle()?;
        self.pos = 0;
        Ok(())
    }

    /// Process `x` (`[n, hidden]` embeddings for positions `pos..pos+n`,
    /// `1 <= n <= rows`) and return their logits `[n, vocab]`; the block's
    /// keys and values are appended to the cache.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails or the output never completes.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0 or exceeds `rows`, if `x` is not a whole number of
    /// rows, or if the block would overflow the cache.
    pub fn step(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let (h, hd, p, c) = (self.hidden, self.head_dim, self.rows, self.capacity);
        assert_eq!(x.len() % h, 0);
        let n = x.len() / h;
        assert!(n >= 1 && n <= p, "block of {n} rows, recipe has {p}");
        assert!(
            self.pos + n <= c,
            "cache overflow at {}+{n} of {c}",
            self.pos
        );
        let pos = self.pos;

        let mut xb = vec![0.0f32; p * h];
        xb[..n * h].copy_from_slice(x);
        let mut sb = vec![0.0f32; p * hd];
        let mut cb = vec![1.0f32; p * hd];
        for r in 0..p {
            // Rows past the cache end are padding; any finite angle will do.
            if pos + r < c {
                let src = (pos + r) * hd;
                sb[r * hd..(r + 1) * hd].copy_from_slice(&self.sin[src..src + hd]);
                cb[r * hd..(r + 1) * hd].copy_from_slice(&self.cos[src..src + hd]);
            }
        }
        // Mask laid out like the scores, [key (FCD), query]: query row q sits
        // at cache position pos + q and may see every position up to its own.
        let keys = c + p;
        let mut mb = vec![MASK_NEG; p * keys];
        for q in 0..p {
            mb[q * keys..q * keys + pos + q + 1].fill(0.0);
        }
        // Placement, host row-major [position, row] (device sizes [row,
        // position]): real block row r lands at position pos + r. Padding rows
        // must not be placed: their keys and values are zero at layer 0 but
        // not deeper (a padded query's attention is a uniform average of the
        // cached values), and anything placed at a future position would be
        // summed into the real token written there later.
        let mut pb = vec![0.0f32; keys * p];
        for r in 0..n {
            pb[(pos + r) * p + r] = 1.0;
        }
        let trace = std::env::var("RENG_STEP_TRACE").is_ok();
        let t0 = Instant::now();
        // Read the cache buffer written by the previous launch, write the other.
        let (rd, wr) = (self.parity, 1 - self.parity);
        for (li_g, (k, v)) in self.slots.iter().enumerate() {
            let (kci, vci, kco, vco) = cache_names(li_g / self.n_kv, li_g % self.n_kv);
            self.rt.rebind(&kci, k[rd]);
            self.rt.rebind(&kco, k[wr]);
            self.rt.rebind(&vci, v[rd]);
            self.rt.rebind(&vco, v[wr]);
        }
        self.rt.upload(self.ix_x, &xb)?;
        self.rt.upload(self.ix_sin, &sb)?;
        self.rt.upload(self.ix_cos, &cb)?;
        self.rt.upload(self.ix_mask, &mb)?;
        self.rt.upload(self.ix_place, &pb)?;
        self.rt.fence_uploads(self.ix_place)?;
        let t_upload = t0.elapsed();
        let logits = self.rt.launch_and_read(n)?;
        if trace {
            eprintln!(
                "step trace: uploads {:.2} ms, launch+readback {:.2} ms",
                t_upload.as_secs_f64() * 1e3,
                (t0.elapsed() - t_upload).as_secs_f64() * 1e3
            );
        }
        self.parity = wr;
        self.pos += n;
        debug_assert_eq!(logits.len(), n * self.vocab);
        Ok(logits)
    }
}
