//! Decode with a KV cache: one recipe compiled once for a fixed block size and
//! cache capacity, launched once per block of new tokens.
//!
//! Each launch processes a block of `rows` positions (the real tokens first,
//! zero rows after them) starting at the current position. Every layer keeps
//! key and value caches `[head_dim, capacity + 1, 1, n_kv_heads]` on the
//! device, updated in place by a ScatterND node: an int32 index tensor
//! uploaded per step sends block row r of KV head g to position pos + r
//! (padded rows to the trash slot at `capacity`), and the node's output
//! tensor aliases its input, so only the written rows move. Attention runs
//! over the whole cache with an additive mask that admits positions up to
//! each query's own. Nothing but the logits crosses the PCIe bus per step.
//!
//! A prompt is fed as one or more full blocks and each generated token as a
//! block of one real row. The block size is a compile-time shape of a recipe,
//! so there are two: a wide one for prompts and a narrow one for decode
//! steps, sharing the weights and the cache buffers (`Runtime::new_with`).

use crate::f32_to_bf16;
use crate::model::{
    Gb, MASK_NEG, ModelWeights, RopeTables, Shared, build_head, build_layer, cache_names,
    common_window, fused_sdpa, uses_full_mask, uses_local_rope,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::Runtime;
use reng_core::Result;
use std::time::Instant;

/// What a step reads back.
#[derive(Clone, Copy)]
enum Read {
    LastId,
    LastLogits,
    AllLogits,
}

/// Per-step input indices of one compiled recipe; the second RoPE rows
/// and the two masks exist only when some layer reads them.
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

/// A compiled decoder recipe with its resident weights and KV cache, plus an
/// optional second recipe for small blocks (decode) that shares them.
pub struct CachedModel<'a> {
    /// The decode recipe, if any; declared first so it drops before `rt`,
    /// whose device and buffers it borrows.
    dec: Option<(Runtime<'a>, Inputs, usize)>,
    rt: Runtime<'a>,
    ix: Inputs,
    rows: usize,
    capacity: usize,
    hidden: usize,
    head_dim: usize,
    n_kv: usize,
    pos: usize,
    /// RoPE tables `[capacity, head_dim]`, and the second pair for the
    /// layers with `local_rope` (empty otherwise).
    sin: Vec<f32>,
    cos: Vec<f32>,
    sin_local: Vec<f32>,
    cos_local: Vec<f32>,
    n_layers: usize,
    /// Sliding window of the windowed layers: such a layer's query attends
    /// only to the last `window` positions (its own included).
    window: Option<usize>,
    /// Which of the two cache buffers per layer holds the cache: the wide
    /// recipe reads one and writes the other (its ScatterND is not in
    /// place), so its launches alternate them; the decode recipe (in place)
    /// is bound to the current one before each launch.
    flipped: bool,
    /// `RENG_ARGMAX_CHECK`: after every id read, also read the logits row
    /// and report when the device argmax disagrees with a host argmax.
    check_argmax: bool,
}

impl<'a> CachedModel<'a> {
    /// Compile the recipe for blocks of `rows` positions over a cache of
    /// `capacity` positions and upload the weights. `rope` holds RoPE
    /// tables `[capacity, head_dim]` (the local pair only when a layer
    /// reads it); the per-layer tables in `m` are unused. With
    /// `decode_rows > 0` (and different from `rows`) a second recipe for
    /// blocks of up to `decode_rows` positions is compiled over the same
    /// weights and cache; [`CachedModel::step`] picks it for small blocks.
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
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        decode_rows: usize,
        capacity: usize,
        rope: &RopeTables<'_>,
    ) -> Result<Self> {
        assert!(!m.layers.is_empty() && rows > 0 && capacity > 0);
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
        let t0 = Instant::now();
        let (gb, out) = Self::build(m, hidden, inter, vocab, rows, capacity, false)?;
        if std::env::var("RENG_RECIPE_TRACE").is_ok() {
            eprintln!("graph build: {:.2} s", t0.elapsed().as_secs_f64());
        }
        let rt = Runtime::new(gb, out)?;
        let ix = Self::inputs(&rt);
        let dec = if decode_rows > 0 && decode_rows != rows {
            let (gb, out) = Self::build(m, hidden, inter, vocab, decode_rows, capacity, true)?;
            let d = Runtime::new_with(gb, out, Some(&rt))?;
            let ix = Self::inputs(&d);
            Some((d, ix, decode_rows))
        } else {
            None
        };
        Ok(Self {
            dec,
            rt,
            ix,
            rows,
            capacity,
            hidden,
            head_dim: hd,
            n_kv: l0.n_kv_heads,
            pos: 0,
            sin: rope.sin.to_vec(),
            cos: rope.cos.to_vec(),
            sin_local: rope.sin_local.to_vec(),
            cos_local: rope.cos_local.to_vec(),
            n_layers: m.layers.len(),
            flipped: false,
            check_argmax: std::env::var_os("RENG_ARGMAX_CHECK").is_some(),
            window: common_window(&m.layers),
        })
    }

    /// The graph for blocks of `rows` positions over a cache of `capacity`;
    /// `inplace` selects the cache update form (see [`Shared::inplace`]).
    #[allow(clippy::too_many_arguments)]
    fn build(
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        rows: usize,
        capacity: usize,
        inplace: bool,
    ) -> Result<(Gb<'a>, crate::runtime::Out)> {
        let hd = m.layers[0].head_dim;
        let (t, h, hd64, keys) = (rows as u64, hidden as u64, hd as u64, capacity as u64 + 1);
        let mut gb = Gb::new()?;
        // Per-step inputs; their contents are replaced before every launch.
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
        let groups = m.layers[0].n_kv_heads as u64;
        let zero_mask = vec![0.0; rows * (capacity + 1)];
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
            &[3, t * groups],
            SYN_TYPE_INT32,
            &vec![0u8; 3 * 4 * rows * groups as usize],
        )?;
        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            sin_local: t_sin_local,
            cos_local: t_cos_local,
            mask: t_mask,
            mask_window: t_mask_window,
            cache: Some(capacity),
            kidx: Some(t_kidx),
            inplace,
            // The in-place recipe is the decode one (small blocks), where
            // the fused attention node is the default.
            sdpa: fused_sdpa(inplace),
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer(&mut gb, li, cur, lw, &sh, rows, hidden, inter, None)?;
        }
        let out = build_head(&mut gb, cur, m, rows, hidden, vocab, true)?;
        Ok((gb, out))
    }

    fn inputs(rt: &Runtime<'_>) -> Inputs {
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

    /// Block size of the decode recipe (`rows` when there is none).
    #[must_use]
    pub fn decode_rows(&self) -> usize {
        self.dec.as_ref().map_or(self.rows, |d| d.2)
    }

    /// Cache capacity in positions.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forget the cached prefix. The cache contents need no clearing (the
    /// mask never admits positions at or beyond the current one), so this
    /// is free.
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
        self.step_rows(x, Read::AllLogits)
    }

    /// Like [`CachedModel::step`] but returns only the last row's logits
    /// (`[1, vocab]`).
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails or the output never completes.
    pub fn step_last(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        self.step_rows(x, Read::LastLogits)
    }

    /// Like [`CachedModel::step`] but returns only the argmax token id of the
    /// last row, computed on the device: the only thing greedy generation
    /// needs, and four bytes over the bus.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails or the output never completes.
    pub fn step_last_id(&mut self, x: &[f32]) -> Result<u32> {
        let v = self.step_rows(x, Read::LastId)?;
        Ok(v[0] as u32)
    }

    fn step_rows(&mut self, x: &[f32], read: Read) -> Result<Vec<f32>> {
        let (h, hd, c) = (self.hidden, self.head_dim, self.capacity);
        assert_eq!(x.len() % h, 0);
        let n = x.len() / h;
        assert!(
            n >= 1 && n <= self.rows,
            "block of {n} rows, recipe has {}",
            self.rows
        );
        assert!(
            self.pos + n <= c,
            "cache overflow at {}+{n} of {c}",
            self.pos
        );
        let pos = self.pos;
        // Small blocks go through the decode recipe when there is one.
        let flipped = self.flipped;
        let check_argmax = self.check_argmax;
        let n_layers = self.n_layers;
        let wide = !matches!(&self.dec, Some((_, _, dr)) if n <= *dr);
        // Cache buffer addresses per layer (the wide runtime owns both).
        let bufs: Vec<[u64; 4]> = (0..n_layers)
            .map(|li| {
                let (kci, vci, kco, vco) = cache_names(li);
                [
                    self.rt.addr(&kci),
                    self.rt.addr(&vci),
                    self.rt.addr(&kco),
                    self.rt.addr(&vco),
                ]
            })
            .collect();
        let (rt, ix, p) = match &mut self.dec {
            Some((d, ix, dr)) if n <= *dr => (d, &*ix, *dr),
            _ => (&mut self.rt, &self.ix, self.rows),
        };
        for (li, b) in bufs.iter().enumerate() {
            let (kci, vci, kco, vco) = cache_names(li);
            let (k_cur, v_cur, k_other, v_other) = if flipped {
                (b[2], b[3], b[0], b[1])
            } else {
                (b[0], b[1], b[2], b[3])
            };
            rt.rebind(&kci, k_cur);
            rt.rebind(&vci, v_cur);
            if wide {
                rt.rebind(&kco, k_other);
                rt.rebind(&vco, v_other);
            } else {
                rt.rebind(&kco, k_cur);
                rt.rebind(&vco, v_cur);
            }
        }

        let mut xb = vec![0.0f32; p * h];
        xb[..n * h].copy_from_slice(x);
        // RoPE rows of the block's positions from a table; rows past the
        // cache end are padding, any finite angle will do.
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
        // Mask laid out like the scores, [key (FCD), query]: query row q sits
        // at cache position pos + q and may see every position up to its own
        // (padding rows past the cache end see everything but the trash slot;
        // they are discarded), and with a window none further back than
        // it. Built directly in bf16: the largest per-step upload.
        let keys = c + 1;
        let neg = f32_to_bf16(MASK_NEG);
        let mask_rows = |window: Option<usize>| -> Vec<u16> {
            let mut mb = vec![neg; p * keys];
            for q in 0..p {
                let end = (pos + q + 1).min(c);
                let start = window.map_or(0, |w| (pos + q + 1).saturating_sub(w));
                mb[q * keys + start..q * keys + end].fill(0);
            }
            mb
        };
        // Scatter indices, ONNX triples (g, 0, position) for update r + p * g
        // (row r of KV head g): real rows go to pos + r (< capacity by the
        // assert above), padded rows to the trash slot. Padded rows must not
        // land on real positions: their keys and values are zero at layer 0
        // but not deeper (a padded query's attention is a uniform average of
        // the cached values).
        let mut ib: Vec<u8> = Vec::with_capacity(12 * p * self.n_kv);
        for g in 0..self.n_kv {
            for r in 0..p {
                let target = if r < n { pos + r } else { c };
                for v in [g as i32, 0i32, target as i32] {
                    ib.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let trace = std::env::var("RENG_STEP_TRACE").is_ok();
        let t0 = Instant::now();
        rt.upload(ix.x, &xb)?;
        rt.upload(ix.sin, &sb)?;
        rt.upload(ix.cos, &cb)?;
        if let (Some(is), Some(ic)) = (ix.sin_local, ix.cos_local) {
            rt.upload(is, &rope_rows(&self.sin_local))?;
            rt.upload(ic, &rope_rows(&self.cos_local))?;
        }
        if let Some(im) = ix.mask {
            rt.upload_bf16(im, &mask_rows(None))?;
        }
        if let Some(im) = ix.mask_window {
            rt.upload_bf16(im, &mask_rows(self.window))?;
        }
        rt.upload_raw(ix.kidx, &ib)?;
        rt.fence()?;
        let t_upload = t0.elapsed();
        // The recipe's read-back tensor is the argmax ids; the logits stay
        // on the device and are read only when asked for.
        let ids = rt.launch_and_read_i32(n - 1, 1)?;
        if check_argmax && matches!(read, Read::LastId) {
            let row = rt.read_bf16_range("LOGITS", n - 1, 1)?;
            let host = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
            let dev = ids[0];
            let at = |i: i32| row.get(usize::try_from(i).unwrap_or(usize::MAX)).copied();
            eprintln!(
                "argmax check: pos {pos} n {n} wide {wide} device {dev} (logit {:?}) host {host} (logit {})",
                at(dev),
                row[host]
            );
        }
        let logits = match read {
            Read::LastId => vec![ids[0] as f32],
            Read::LastLogits => rt.read_bf16_range("LOGITS", n - 1, 1)?,
            Read::AllLogits => rt.read_bf16_range("LOGITS", 0, n)?,
        };
        if wide {
            self.flipped = !flipped;
        }
        if trace {
            eprintln!(
                "step trace: uploads {:.2} ms, launch+readback {:.2} ms",
                t_upload.as_secs_f64() * 1e3,
                (t0.elapsed() - t_upload).as_secs_f64() * 1e3
            );
        }
        self.pos += n;
        Ok(logits)
    }
}
