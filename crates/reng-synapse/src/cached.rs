//! Decode with a KV cache: one recipe compiled once for a fixed block size and
//! cache capacity, launched once per block of new tokens.
//!
//! Each launch processes a block of `rows` positions (the real tokens first,
//! zero rows after them) starting at the current position. Every layer
//! attends over `cache prefix ++ block` with an additive mask that admits the
//! filled part of the cache and the causal part of the block, and writes the
//! block's per-head rotated keys and values to device-resident outputs. After
//! the launch those rows are copied on the device into the cache at the
//! block's position, so nothing but the logits crosses the PCIe bus per step.
//!
//! A prompt is fed as one or more full blocks and each generated token as a
//! block of one real row. The block size is a compile-time shape of the
//! recipe; a small block wastes MME columns but keeps the launch count at one
//! per token, which on this stack is what matters.

use crate::model::{Gb, MASK_NEG, ModelWeights, Shared, build_head, build_layer, cache_names};
use crate::runtime::Runtime;
use reng_core::Result;

/// A compiled decoder recipe with its resident weights and KV cache.
pub struct CachedModel {
    rt: Runtime,
    rows: usize,
    capacity: usize,
    hidden: usize,
    vocab: usize,
    head_dim: usize,
    pos: usize,
    /// RoPE tables `[capacity, head_dim]`.
    sin: Vec<f32>,
    cos: Vec<f32>,
    ix_x: usize,
    ix_sin: usize,
    ix_cos: usize,
    ix_mask: usize,
    /// Device addresses `(k_cache, v_cache, k_new, v_new)` per layer and KV head.
    slots: Vec<(u64, u64, u64, u64)>,
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
        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            mask: Some(t_mask),
            cache: Some(capacity),
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
                let (kc, vc, kn, vn) = cache_names(li, g);
                slots.push((rt.addr(&kc), rt.addr(&vc), rt.addr(&kn), rt.addr(&vn)));
            }
        }
        Ok(Self {
            ix_x: rt.input_index("X"),
            ix_sin: rt.input_index("SIN"),
            ix_cos: rt.input_index("COS"),
            ix_mask: rt.input_index("MASK"),
            rt,
            rows,
            capacity,
            hidden,
            vocab,
            head_dim: hd,
            pos: 0,
            sin: sin.to_vec(),
            cos: cos.to_vec(),
            slots,
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

    /// Forget the cached prefix. The cache contents need no clearing: the
    /// mask never admits positions at or beyond the current one.
    pub fn reset(&mut self) {
        self.pos = 0;
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
        // Mask laid out like the scores, [key (FCD), query]: the filled cache
        // prefix is open, the rest of the cache closed, the block causal.
        let keys = c + p;
        let mut mb = vec![MASK_NEG; p * keys];
        for q in 0..p {
            let row = &mut mb[q * keys..(q + 1) * keys];
            row[..pos].fill(0.0);
            row[c..=c + q].fill(0.0);
        }
        self.rt.upload(self.ix_x, &xb)?;
        self.rt.upload(self.ix_sin, &sb)?;
        self.rt.upload(self.ix_cos, &cb)?;
        self.rt.upload(self.ix_mask, &mb)?;
        let logits = self.rt.launch_and_read(n)?;

        // Append the block's keys and values: rows are contiguous in the
        // [head_dim (FCD), position] layout, so one copy per tensor.
        let row_bytes = (hd * 2) as u64;
        let (src_bytes, dst_off) = ((n as u64) * row_bytes, (pos as u64) * row_bytes);
        for &(kc, vc, kn, vn) in &self.slots {
            self.rt.copy_d2d(kn, 0, kc, dst_off, src_bytes)?;
            self.rt.copy_d2d(vn, 0, vc, dst_off, src_bytes)?;
        }
        self.rt.sync()?;
        self.pos += n;
        debug_assert_eq!(logits.len(), n * self.vocab);
        Ok(logits)
    }
}
