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
//!
//! Device-resident decode loop (`RENG_DEVICE_LOOP`, on by default when the
//! caller gives an [`EmbedTable`] and `decode_rows` is 1). The narrow
//! recipe then takes no per-step uploads at all: its inputs are one int32
//! token id and one int32 position, and everything the step needs is
//! derived on the device. The embedding row is a `gather_fwd_bf16` over
//! the bf16 table (the LM head's own copy when the embeddings are tied);
//! the RoPE rows are gathers over the full `[head_dim, capacity]` tables;
//! a mask row is a gather over a static pattern (`[0; keys] ++ [NEG; keys]`
//! for the causal mask, with a run of `window` zeros between two runs of
//! `NEG` for the windowed one) at an index vector that an `sub_fwd_i32`
//! shifts by the position; the ScatterND triples are the constant
//! `(g, 0, 0)` plus `(0, 0, 1)` times the position, two int32 nodes. The
//! id and position inputs and the `IDS` output are bound per launch
//! (`Runtime::rebind`) into two device buffers of one cache line per
//! position: a position table (slot `p` holds `p`) and an id ring where
//! the launch at position `p` reads slot `p` and writes slot `p + 1`. So
//! `n` steps are `n` launches enqueued back to back, each consuming the id
//! the previous one produced, followed by one read of `n` slots; the host
//! seeds the ring only when the id at the current position is not the one
//! the last launch left there.

use crate::f32_to_bf16;
use crate::ffi::{SYN_TYPE_BF16, SYN_TYPE_F32, synTensor};
use crate::model::{
    EmbedTable, Gb, MASK_NEG, ModelWeights, RopeTables, Shared, build_head, build_layer,
    cache_names, common_window, fused_sdpa, lm_head_input, uses_full_mask, uses_local_rope,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::Runtime;
use core::ffi::c_void;
use reng_core::Result;
use std::time::Instant;

/// Whether the device decode loop is built: `RENG_DEVICE_LOOP` set to
/// anything but `0` or `off`, or unset (the batched decoder reads the
/// same switch).
pub(crate) fn device_loop_enabled() -> bool {
    match std::env::var("RENG_DEVICE_LOOP") {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("off")),
        Err(_) => true,
    }
}

/// Bytes per slot of the loop's id ring and position table: one int32 on
/// its own cache line, so the tensors of consecutive launches never share
/// a line (the batched loop's rows are whole multiples of it).
pub(crate) const SLOT: usize = 128;

/// `ns_GatherKernel::Params`: the FCD-first axis the indices select along.
#[repr(C)]
pub(crate) struct GatherParams {
    pub(crate) axis: i32,
}

/// The device decode loop's recipe and its per-launch state.
struct Loop<'a> {
    rt: Runtime<'a>,
    /// `capacity + 1` slots of [`SLOT`] bytes: slot `p` holds the id
    /// consumed at position `p` (written by the launch at `p - 1`, or by
    /// the host).
    ring: u64,
    /// `capacity` slots: slot `p` holds `p`.
    postab: u64,
    /// A slot whose id the host knows: `(position, id)`, the last launch's
    /// output or the last seed.
    known: Option<(usize, u32)>,
}

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
    /// The device decode loop, if any (the same borrow).
    lp: Option<Loop<'a>>,
    rt: Runtime<'a>,
    ix: Inputs,
    rows: usize,
    capacity: usize,
    hidden: usize,
    vocab: usize,
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
    /// With an `embed` table, `decode_rows <= 1` and `RENG_DEVICE_LOOP`
    /// not off, the device decode loop (see the module docs) replaces
    /// that second recipe: [`CachedModel::step_ids`] runs steps from token
    /// ids, and one-row blocks of [`CachedModel::step`] go through the
    /// wide recipe.
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
        embed: Option<&EmbedTable<'a>>,
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
        let lp = match embed {
            Some(e) if decode_rows <= 1 && device_loop_enabled() => {
                let t0 = Instant::now();
                let (gb, out) = Self::build_loop(m, hidden, inter, vocab, capacity, rope, e)?;
                if std::env::var("RENG_RECIPE_TRACE").is_ok() {
                    eprintln!("loop graph build: {:.2} s", t0.elapsed().as_secs_f64());
                }
                let mut d = Runtime::new_with(gb, out, Some(&rt))?;
                let ring = d.alloc(((capacity + 1) * SLOT) as u64)?;
                let postab = d.alloc((capacity * SLOT) as u64)?;
                let mut tab = vec![0u8; capacity * SLOT];
                for p in 0..capacity {
                    tab[p * SLOT..p * SLOT + 4].copy_from_slice(&(p as i32).to_le_bytes());
                }
                d.upload_at(postab, &tab)?;
                d.fence()?;
                Some(Loop {
                    rt: d,
                    ring,
                    postab,
                    known: None,
                })
            }
            _ => None,
        };
        let dec = if decode_rows > 0 && decode_rows != rows && lp.is_none() {
            let (gb, out) = Self::build(m, hidden, inter, vocab, decode_rows, capacity, true)?;
            let d = Runtime::new_with(gb, out, Some(&rt))?;
            let ix = Self::inputs(&d);
            Some((d, ix, decode_rows))
        } else {
            None
        };
        Ok(Self {
            dec,
            lp,
            rt,
            ix,
            rows,
            capacity,
            hidden,
            vocab,
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
        let out = build_head(&mut gb, cur, m, rows, hidden, vocab, true, None)?;
        Ok((gb, out))
    }

    /// The graph of the device decode loop: one position per launch from
    /// an id and a position input (see the module docs).
    #[allow(clippy::too_many_lines)]
    fn build_loop(
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        capacity: usize,
        rope: &RopeTables<'_>,
        embed: &EmbedTable<'a>,
    ) -> Result<(Gb<'a>, crate::runtime::Out)> {
        let hd = m.layers[0].head_dim;
        let (h, v, hd64, cap, keys) = (
            hidden as u64,
            vocab as u64,
            hd as u64,
            capacity as u64,
            capacity as u64 + 1,
        );
        let groups = m.layers[0].n_kv_heads;
        let bf = SYN_TYPE_BF16;
        let none = (core::ptr::null::<c_void>(), 0u32);
        let (rows_p, elems_p) = (GatherParams { axis: 1 }, GatherParams { axis: 0 });
        let size = core::mem::size_of::<GatherParams>() as u32;
        let (pg_rows, pg_elems) = (
            ((&raw const rows_p).cast::<c_void>(), size),
            ((&raw const elems_p).cast::<c_void>(), size),
        );
        let i32s =
            |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut gb = Gb::new()?;
        // Per-launch tensors: bound to a ring slot and a position table
        // slot before every launch; their own buffers are never used.
        let t_ids_in = gb.input_raw("IDS_IN", &[1], SYN_TYPE_INT32, &[0u8; 4])?;
        let t_pos = gb.input_raw("POS", &[1], SYN_TYPE_INT32, &[0u8; 4])?;

        // The embedding row of the id: gathered from the head's weights
        // when the embeddings are tied (one device copy), else from the
        // table's own upload; scaled in f32 and rounded like the host
        // gather when the model scales its embeddings.
        assert_eq!(embed.rows.len(), hidden * vocab, "embedding table size");
        let tied = embed.rows.as_ptr() == m.lm_head.as_ptr() && embed.rows.len() == m.lm_head.len();
        let t_lm = if tied {
            Some(lm_head_input(&mut gb, m, hidden, vocab)?)
        } else {
            None
        };
        let t_emb = match t_lm {
            Some(t) => t,
            None => gb.input_bf16("EMB", &[h, v], std::borrow::Cow::Borrowed(embed.rows))?,
        };
        let t_row = gb.mid("emb_row", &[h, 1], bf)?;
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_row],
            pg_rows.0,
            pg_rows.1,
        )?;
        let t_x = if embed.scale == 1.0 {
            t_row
        } else {
            let t_scale = gb.input_raw(
                "EMB_SCALE",
                &[1, 1],
                SYN_TYPE_F32,
                &embed.scale.to_le_bytes(),
            )?;
            let t_rf = gb.mid("emb_f32", &[h, 1], SYN_TYPE_F32)?;
            let t_sf = gb.mid("emb_scaled", &[h, 1], SYN_TYPE_F32)?;
            let t_xs = gb.mid("emb_bf16", &[h, 1], bf)?;
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
                &[t_xs],
                none.0,
                none.1,
            )?;
            t_xs
        };

        // RoPE rows of the position out of the full tables.
        let rope_row = |gb: &mut Gb<'a>, name: &str, table: &[f32]| -> Result<synTensor> {
            let t_tab = gb.input(&format!("{name}T"), &[hd64, cap], table)?;
            let t_row = gb.mid(&format!("{name}_row"), &[hd64, 1], bf)?;
            gb.node(
                "gather_fwd_bf16",
                &format!("rope_{name}"),
                &[t_tab, t_pos],
                &[t_row],
                pg_rows.0,
                pg_rows.1,
            )?;
            Ok(t_row)
        };
        let t_sin = rope_row(&mut gb, "SIN", rope.sin)?;
        let t_cos = rope_row(&mut gb, "COS", rope.cos)?;
        let (t_sin_local, t_cos_local) = if uses_local_rope(&m.layers) {
            (
                Some(rope_row(&mut gb, "SINL", rope.sin_local)?),
                Some(rope_row(&mut gb, "COSL", rope.cos_local)?),
            )
        } else {
            (None, None)
        };

        // A mask row is a window of `keys` elements into a static pattern,
        // gathered at `base - position`: for the causal mask the pattern
        // is `[0; keys] ++ [NEG; keys]` and element k of position p sits
        // at `keys - 1 - p + k`; with a window w the pattern is `[NEG;
        // keys] ++ [0; w] ++ [NEG; keys]` and the element at `keys - 1 + w
        // - p + k`, zero exactly for `p + 1 - w <= k <= p`.
        let neg = f32_to_bf16(MASK_NEG);
        let keys_us = capacity + 1;
        let mask_row =
            |gb: &mut Gb<'a>, name: &str, pattern: Vec<u16>, first: usize| -> Result<synTensor> {
                let t_pat = gb.input_bf16(
                    &format!("{name}P"),
                    &[pattern.len() as u64],
                    std::borrow::Cow::Owned(pattern),
                )?;
                let base: Vec<i32> = (0..keys_us).map(|k| (first + k) as i32).collect();
                let t_base =
                    gb.input_raw(&format!("{name}I"), &[keys], SYN_TYPE_INT32, &i32s(&base))?;
                let t_idx = gb.mid(&format!("{name}_idx"), &[keys], SYN_TYPE_INT32)?;
                let t_row = gb.mid(&format!("{name}_row"), &[keys], bf)?;
                let t_4d = gb.mid(&format!("{name}_4d"), &[keys, 1, 1, 1], bf)?;
                gb.node(
                    "sub_fwd_i32",
                    &format!("{name}_index"),
                    &[t_base, t_pos],
                    &[t_idx],
                    none.0,
                    none.1,
                )?;
                gb.node(
                    "gather_fwd_bf16",
                    &format!("{name}_gather"),
                    &[t_pat, t_idx],
                    &[t_row],
                    pg_elems.0,
                    pg_elems.1,
                )?;
                gb.node(
                    "reshape",
                    &format!("{name}_reshape"),
                    &[t_row],
                    &[t_4d],
                    none.0,
                    none.1,
                )?;
                Ok(t_4d)
            };
        let t_mask = if uses_full_mask(&m.layers) {
            let mut pat = vec![0u16; 2 * keys_us];
            pat[keys_us..].fill(neg);
            Some(mask_row(&mut gb, "MASK", pat, keys_us - 1)?)
        } else {
            None
        };
        let t_mask_window = match common_window(&m.layers) {
            Some(w) => {
                let mut pat = vec![neg; 2 * keys_us + w];
                pat[keys_us..keys_us + w].fill(0);
                Some(mask_row(&mut gb, "MASKW", pat, keys_us - 1 + w)?)
            }
            None => None,
        };

        // ScatterND triples (g, 0, position) for the one row of each KV
        // head: a constant `(g, 0, 0)` plus `(0, 0, 1)` times the position.
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
        let t_kidx = gb.mid("kidx", &[3, g64], SYN_TYPE_INT32)?;
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

        let sh = Shared {
            sin: t_sin,
            cos: t_cos,
            sin_local: t_sin_local,
            cos_local: t_cos_local,
            mask: t_mask,
            mask_window: t_mask_window,
            cache: Some(capacity),
            kidx: Some(t_kidx),
            inplace: true,
            // The loop is the single-sequence decode recipe: fused
            // attention by default (see `fused_sdpa`).
            sdpa: fused_sdpa(true),
        };
        let mut cur = t_x;
        for (li, lw) in m.layers.iter().enumerate() {
            cur = build_layer(&mut gb, li, cur, lw, &sh, 1, hidden, inter, None)?;
        }
        let out = build_head(&mut gb, cur, m, 1, hidden, vocab, true, t_lm)?;
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

    /// Whether the device decode loop was built (see [`CachedModel::new`]).
    #[must_use]
    pub fn has_loop(&self) -> bool {
        self.lp.is_some()
    }

    /// The cache buffer addresses of every layer (the wide runtime owns
    /// both of each layer's pair): `[k_in, v_in, k_out, v_out]`.
    fn cache_bufs(&self) -> Vec<[u64; 4]> {
        (0..self.n_layers)
            .map(|li| {
                let (kci, vci, kco, vco) = cache_names(li);
                [
                    self.rt.addr(&kci),
                    self.rt.addr(&vci),
                    self.rt.addr(&kco),
                    self.rt.addr(&vco),
                ]
            })
            .collect()
    }

    /// Device decode loop: feed token `seed` at the current position and
    /// run `n` steps back to back, each consuming the greedy id the
    /// previous one produced; returns the `n` ids (the last one has not
    /// been fed yet). One readback for the whole run.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails or a launch never
    /// completes.
    ///
    /// # Panics
    ///
    /// Panics if the loop was not built, `n` is 0, or the run would
    /// overflow the cache.
    pub fn step_ids(&mut self, seed: u32, n: usize) -> Result<Vec<u32>> {
        Ok(self.loop_steps(seed, n, false)?.0)
    }

    /// Like [`CachedModel::step_ids`], also returning the last step's
    /// logits `[1, vocab]`.
    ///
    /// # Errors
    ///
    /// As [`CachedModel::step_ids`].
    pub fn step_ids_logits(&mut self, seed: u32, n: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        self.loop_steps(seed, n, true)
    }

    /// Device decode loop: feed token `seed` at the current position and
    /// return its logits `[1, vocab]` (one launch).
    ///
    /// # Errors
    ///
    /// As [`CachedModel::step_ids`].
    pub fn step_id_logits(&mut self, seed: u32) -> Result<Vec<f32>> {
        Ok(self.loop_steps(seed, 1, true)?.1)
    }

    fn loop_steps(
        &mut self,
        seed: u32,
        n: usize,
        want_logits: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        assert!(n >= 1);
        let (pos, c) = (self.pos, self.capacity);
        assert!(pos + n <= c, "cache overflow at {pos}+{n} of {c}");
        assert!((seed as usize) < self.vocab, "token id {seed} out of range");
        let bufs = self.cache_bufs();
        let flipped = self.flipped;
        let check_argmax = self.check_argmax;
        let lp = self.lp.as_mut().expect("device decode loop not built");
        let rt = &mut lp.rt;
        // In place: both sides of every cache bound to the current buffer.
        for (li, b) in bufs.iter().enumerate() {
            let (kci, vci, kco, vco) = cache_names(li);
            let (k_cur, v_cur) = if flipped { (b[2], b[3]) } else { (b[0], b[1]) };
            rt.rebind(&kci, k_cur);
            rt.rebind(&vci, v_cur);
            rt.rebind(&kco, k_cur);
            rt.rebind(&vco, v_cur);
        }
        let trace = std::env::var("RENG_STEP_TRACE").is_ok();
        let t0 = Instant::now();
        let slot = |p: usize| (p * SLOT) as u64;
        // The seed goes into its position's ring slot unless the last
        // launch left that very id there.
        let resident = lp.known == Some((pos, seed));
        if !resident {
            rt.upload_at(lp.ring + slot(pos), &(seed as i32).to_le_bytes())?;
            rt.fence()?;
        }
        let t_seed = t0.elapsed();
        let first_out = lp.ring + slot(pos + 1);
        rt.fill_sentinel_d32(first_out, n * SLOT / 4)?;
        let t_fill = t0.elapsed() - t_seed;
        for j in 0..n {
            rt.rebind("IDS_IN", lp.ring + slot(pos + j));
            rt.rebind("IDS", lp.ring + slot(pos + j + 1));
            rt.rebind("POS", lp.postab + slot(pos + j));
            rt.launch_only()?;
        }
        let t_launch = t0.elapsed() - t_seed - t_fill;
        let ids: Vec<u32> = rt
            .read_i32_strided(first_out, SLOT, n)?
            .into_iter()
            .map(|i| i as u32)
            .collect();
        let t_read = t0.elapsed() - t_seed - t_fill - t_launch;
        let logits = if want_logits {
            rt.read_bf16_range("LOGITS", 0, 1)?
        } else {
            Vec::new()
        };
        if check_argmax && n == 1 {
            let row = if want_logits {
                logits.clone()
            } else {
                rt.read_bf16_range("LOGITS", 0, 1)?
            };
            let host = (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b });
            let dev = ids[0] as usize;
            eprintln!(
                "argmax check: pos {pos} loop device {dev} (logit {:?}) host {host} (logit {})",
                row.get(dev),
                row[host]
            );
        }
        lp.known = Some((pos + n, ids[n - 1]));
        if trace {
            eprintln!(
                "loop trace: {n} steps from position {pos}: seed {} {:.2} ms, sentinel fill {:.2} ms, launches {:.2} ms, wait+readback {:.2} ms",
                if resident { "resident" } else { "uploaded" },
                t_seed.as_secs_f64() * 1e3,
                t_fill.as_secs_f64() * 1e3,
                t_launch.as_secs_f64() * 1e3,
                t_read.as_secs_f64() * 1e3
            );
        }
        self.pos += n;
        Ok((ids, logits))
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
        let wide = !matches!(&self.dec, Some((_, _, dr)) if n <= *dr);
        let bufs = self.cache_bufs();
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
