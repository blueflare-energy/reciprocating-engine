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
//!
//! Device-resident decode loop (`RENG_DEVICE_LOOP`, on by default when the
//! caller gives an [`EmbedTable`]): the decode recipe then takes no
//! per-step uploads. Its inputs are `B` int32 token ids and `B` int32
//! positions, one per slot, and everything else is derived on the device
//! as in the batch-1 loop of `cached.rs`, one column per slot: the `B`
//! embedding rows are one `gather_fwd_bf16` over the bf16 table (the LM
//! head's device copy when tied), the RoPE rows are gathers over the full
//! `[head_dim, capacity]` tables at the `B` positions, a mask row per slot
//! is a gather over the static pattern at `[keys, B]` indices that a
//! `sub_fwd_i32` shifts by each slot's position (the windowed and
//! per-layer masks keep their own patterns), and the ScatterND quadruples
//! `(b, g, 0, position_b)` are the constant `(b, g, 0, 0)` plus
//! `(0, 0, 0, 1)` times the position. The id and position inputs and the
//! `IDS` output are rebound per launch (`Runtime::rebind`) into an id ring
//! and a position table of one row of `B` int32s (whole cache lines) per
//! launch: launch `j` of a run reads ring row `head + j` and writes row
//! `head + j + 1`, and reads position row `j`, which the host fills with
//! every slot's position plus `j` before the run (one small upload per run,
//! together with the seed ids unless the previous run left them in row
//! `head`). So [`BatchedModel::run_ids`] enqueues `n` launches back to
//! back and reads the `n * B` ids once. The loop runs a fixed `n` for every
//! slot: a sequence that finishes early (EOS) keeps advancing on its own
//! output, and the caller drops its ids or restarts the slot with
//! [`BatchedModel::reset`] and a new prefill.

use crate::cached::{GatherParams, SLOT, device_loop_enabled};
use crate::f32_to_bf16;
use crate::ffi::{SYN_TYPE_BF16, SYN_TYPE_F32, synTensor};
use crate::model::{
    EmbedTable, Gb, MASK_NEG, ModelWeights, RopeTables, Shared, SharedBatched, build_head,
    build_layer, build_layer_batched, cache_names, common_window, fused_sdpa, lm_head_input,
    uses_full_mask, uses_local_rope,
};
use crate::probe::SYN_TYPE_INT32;
use crate::runtime::{Out, Runtime};
use core::ffi::c_void;
use reng_core::Result;
use std::time::Instant;

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

/// The device decode loop's per-run state; its recipe is the current
/// bucket's decode runtime, its buffers belong to `base`.
struct Loop {
    /// `capacity + 1` rows of `stride` bytes: row `r` holds the `B` ids the
    /// launch at ring step `r` consumes (written by the launch at `r - 1`,
    /// or by the host).
    ring: u64,
    /// `capacity` rows of `stride` bytes: row `j` holds every slot's
    /// position at launch `j` of the current run.
    postab: u64,
    /// Bytes per row: `B` int32s rounded up to whole cache lines.
    stride: usize,
    /// The ring row holding the ids the next run consumes.
    head: usize,
    /// The ids in row `head`, when the host knows them (the last run's
    /// output or the last seed).
    known: Option<Vec<u32>>,
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
    /// The per-step inputs of the decode recipe; none with the device
    /// loop, whose inputs are rebound per launch.
    ix: Option<Inputs>,
    /// The first decode recipe: owns the device and the weights every later
    /// recipe binds to.
    base: Runtime<'a>,
    m: ModelWeights<'a>,
    /// The embedding table of the device loop, kept for the recompiles.
    embed: Option<EmbedTable<'a>>,
    lp: Option<Loop>,
    /// Name suffix of the current bucket's per-step or per-launch inputs.
    tag: String,
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
    /// With an `embed` table and `RENG_DEVICE_LOOP` not off, the decode
    /// recipe is the device decode loop (see the module docs): steps then
    /// go through [`BatchedModel::run_ids`] from token ids, and
    /// [`BatchedModel::step`] from embeddings is unavailable.
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
        embed: Option<&EmbedTable<'a>>,
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
        let embed = match embed {
            Some(e) if device_loop_enabled() => Some(*e),
            _ => None,
        };

        let t0 = Instant::now();
        let (gb, out) = match &embed {
            Some(e) => Self::build_decode_loop(&m, hidden, inter, vocab, batch, cap, "", rope, e)?,
            None => Self::build_decode(&m, hidden, inter, vocab, batch, cap, "")?,
        };
        if std::env::var("RENG_RECIPE_TRACE").is_ok() {
            eprintln!(
                "{}graph build: {:.2} s",
                if embed.is_some() { "loop " } else { "" },
                t0.elapsed().as_secs_f64()
            );
        }
        let mut base = Runtime::new(gb, out)?;
        let ix = if embed.is_some() {
            None
        } else {
            Some(Self::decode_inputs(&base, ""))
        };
        let lp = if embed.is_some() {
            let stride = (batch * 4).div_ceil(SLOT) * SLOT;
            let ring = base.alloc(((capacity + 1) * stride) as u64)?;
            let postab = base.alloc((capacity * stride) as u64)?;
            Some(Loop {
                ring,
                postab,
                stride,
                head: 0,
                known: None,
            })
        } else {
            None
        };
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
            embed,
            lp,
            tag: String::new(),
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

    /// The decode graph of the device loop for a `cap`-position bucket:
    /// one position per slot per launch from `B` ids and `B` positions
    /// (see the module docs). The per-launch tensors and the bucket-sized
    /// constants carry `tag`; the tables (embeddings, RoPE, the scatter
    /// constants) keep their names and are shared across buckets.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn build_decode_loop(
        m: &ModelWeights<'a>,
        hidden: usize,
        inter: usize,
        vocab: usize,
        batch: usize,
        cap: usize,
        tag: &str,
        rope: &RopeTables<'_>,
        embed: &EmbedTable<'a>,
    ) -> Result<(Gb<'a>, Out)> {
        let l0 = &m.layers[0];
        let hd = l0.head_dim;
        let groups = l0.n_kv_heads;
        let (h, v, hd64, keys, b, g64) = (
            hidden as u64,
            vocab as u64,
            hd as u64,
            cap as u64 + 1,
            batch as u64,
            groups as u64,
        );
        // The RoPE tables span the configured capacity, not the bucket.
        let table_len = (rope.sin.len() / hd) as u64;
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
        // Per-launch tensors: bound to a ring row and a position table row
        // before every launch; their own buffers are never used.
        let t_ids_in = gb.input_raw(
            &format!("IDS_IN{tag}"),
            &[b],
            SYN_TYPE_INT32,
            &vec![0u8; 4 * batch],
        )?;
        let t_pos = gb.input_raw(
            &format!("POS{tag}"),
            &[b],
            SYN_TYPE_INT32,
            &vec![0u8; 4 * batch],
        )?;

        // The embedding rows of the ids, `[hidden, B]`: gathered from the
        // head's weights when the embeddings are tied (one device copy),
        // else from the table's own upload; scaled in f32 and rounded like
        // the host gather when the model scales its embeddings.
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
        let t_rows = gb.mid("emb_rows", &[h, b], bf)?;
        gb.node(
            "gather_fwd_bf16",
            "embed",
            &[t_emb, t_ids_in],
            &[t_rows],
            pg_rows.0,
            pg_rows.1,
        )?;
        let t_x = if embed.scale == 1.0 {
            t_rows
        } else {
            let t_scale = gb.input_raw(
                "EMB_SCALE",
                &[1, 1],
                SYN_TYPE_F32,
                &embed.scale.to_le_bytes(),
            )?;
            let t_rf = gb.mid("emb_f32", &[h, b], SYN_TYPE_F32)?;
            let t_sf = gb.mid("emb_scaled", &[h, b], SYN_TYPE_F32)?;
            let t_xs = gb.mid("emb_bf16", &[h, b], bf)?;
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
                &[t_xs],
                none.0,
                none.1,
            )?;
            t_xs
        };

        // RoPE rows of the `B` positions out of the full tables, as the
        // batched layers read them: `[hd, 1, 1, 1, B]`.
        let rope_rows = |gb: &mut Gb<'a>, name: &str, table: &[f32]| -> Result<synTensor> {
            let t_tab = gb.input(&format!("{name}T"), &[hd64, table_len], table)?;
            let t_2d = gb.mid(&format!("{name}_rows"), &[hd64, b], bf)?;
            let t_5d = gb.mid(&format!("{name}_5d"), &[hd64, 1, 1, 1, b], bf)?;
            gb.node(
                "gather_fwd_bf16",
                &format!("rope_{name}"),
                &[t_tab, t_pos],
                &[t_2d],
                pg_rows.0,
                pg_rows.1,
            )?;
            gb.node(
                "reshape",
                &format!("{name}_reshape"),
                &[t_2d],
                &[t_5d],
                none.0,
                none.1,
            )?;
            Ok(t_5d)
        };
        let t_sin = rope_rows(&mut gb, "SIN", rope.sin)?;
        let t_cos = rope_rows(&mut gb, "COS", rope.cos)?;
        let (t_sin_local, t_cos_local) = if uses_local_rope(&m.layers) {
            (
                Some(rope_rows(&mut gb, "SINL", rope.sin_local)?),
                Some(rope_rows(&mut gb, "COSL", rope.cos_local)?),
            )
        } else {
            (None, None)
        };

        // A mask row per slot is a window of `keys` elements into a static
        // pattern, gathered at `base - position_b` (see `cached.rs`: the
        // causal pattern is `[0; keys] ++ [NEG; keys]` read from `keys - 1
        // - p`, the windowed one `[NEG; keys] ++ [0; w] ++ [NEG; keys]` from
        // `keys - 1 + w - p`). The `[keys, B]` index tensor is the base
        // vector, replicated per slot, minus the positions broadcast along
        // the keys; the gather takes it flattened.
        let neg = f32_to_bf16(MASK_NEG);
        let keys_us = cap + 1;
        let t_pos2 = gb.mid("pos_2d", &[1, b], SYN_TYPE_INT32)?;
        gb.node(
            "reshape",
            "pos_reshape2",
            &[t_pos],
            &[t_pos2],
            none.0,
            none.1,
        )?;
        let mask_rows =
            |gb: &mut Gb<'a>, name: &str, pattern: Vec<u16>, first: usize| -> Result<synTensor> {
                let t_pat = gb.input_bf16(
                    &format!("{name}P{tag}"),
                    &[pattern.len() as u64],
                    std::borrow::Cow::Owned(pattern),
                )?;
                let mut base: Vec<i32> = Vec::with_capacity(keys_us * batch);
                for _ in 0..batch {
                    base.extend((0..keys_us).map(|k| (first + k) as i32));
                }
                let t_base = gb.input_raw(
                    &format!("{name}I{tag}"),
                    &[keys, b],
                    SYN_TYPE_INT32,
                    &i32s(&base),
                )?;
                let t_idx = gb.mid(&format!("{name}_idx"), &[keys, b], SYN_TYPE_INT32)?;
                let t_flat = gb.mid(&format!("{name}_flat"), &[keys * b], SYN_TYPE_INT32)?;
                let t_rows = gb.mid(&format!("{name}_rows"), &[keys * b], bf)?;
                let t_5d = gb.mid(&format!("{name}_5d"), &[keys, 1, 1, 1, b], bf)?;
                gb.node(
                    "sub_fwd_i32",
                    &format!("{name}_index"),
                    &[t_base, t_pos2],
                    &[t_idx],
                    none.0,
                    none.1,
                )?;
                gb.node(
                    "reshape",
                    &format!("{name}_flatten"),
                    &[t_idx],
                    &[t_flat],
                    none.0,
                    none.1,
                )?;
                gb.node(
                    "gather_fwd_bf16",
                    &format!("{name}_gather"),
                    &[t_pat, t_flat],
                    &[t_rows],
                    pg_elems.0,
                    pg_elems.1,
                )?;
                gb.node(
                    "reshape",
                    &format!("{name}_reshape"),
                    &[t_rows],
                    &[t_5d],
                    none.0,
                    none.1,
                )?;
                Ok(t_5d)
            };
        let t_mask = if uses_full_mask(&m.layers) {
            let mut pat = vec![0u16; 2 * keys_us];
            pat[keys_us..].fill(neg);
            Some(mask_rows(&mut gb, "MASK", pat, keys_us - 1)?)
        } else {
            None
        };
        let t_mask_window = match common_window(&m.layers) {
            Some(w) => {
                let mut pat = vec![neg; 2 * keys_us + w];
                pat[keys_us..keys_us + w].fill(0);
                Some(mask_rows(&mut gb, "MASKW", pat, keys_us - 1 + w)?)
            }
            None => None,
        };

        // ScatterND quadruples (b, g, 0, position_b) for update g + groups
        // * b: a constant `(b, g, 0, 0)` plus `(0, 0, 0, 1)` times the
        // slot's position, then flattened to the `[4, groups * B]` the
        // layers take.
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
        let t_kbase = gb.input_raw("KBASE", &[4, g64, b], SYN_TYPE_INT32, &i32s(&kbase))?;
        let t_ksel = gb.input_raw("KSEL", &[4, g64, b], SYN_TYPE_INT32, &i32s(&ksel))?;
        let t_pos3 = gb.mid("pos_3d", &[1, 1, b], SYN_TYPE_INT32)?;
        let t_kp = gb.mid("kidx_pos", &[4, g64, b], SYN_TYPE_INT32)?;
        let t_k3 = gb.mid("kidx_3d", &[4, g64, b], SYN_TYPE_INT32)?;
        let t_kidx = gb.mid("kidx", &[4, g64 * b], SYN_TYPE_INT32)?;
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
        gb.node(
            "add_fwd_i32",
            "kidx_add",
            &[t_kbase, t_kp],
            &[t_k3],
            none.0,
            none.1,
        )?;
        gb.node(
            "reshape",
            "kidx_reshape",
            &[t_k3],
            &[t_kidx],
            none.0,
            none.1,
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
        let out = build_head(&mut gb, cur, m, batch, hidden, vocab, true, t_lm)?;
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

    /// Whether the decode recipe is the device decode loop (see
    /// [`BatchedModel::new`]).
    #[must_use]
    pub fn has_loop(&self) -> bool {
        self.lp.is_some()
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
        let (gb, out) = match &self.embed {
            Some(e) => {
                let rope = RopeTables {
                    sin: &self.sin,
                    cos: &self.cos,
                    sin_local: &self.sin_local,
                    cos_local: &self.cos_local,
                };
                Self::build_decode_loop(
                    &self.m,
                    self.hidden,
                    self.inter,
                    self.vocab,
                    self.batch,
                    cap,
                    &tag,
                    &rope,
                    e,
                )?
            }
            None => Self::build_decode(
                &self.m,
                self.hidden,
                self.inter,
                self.vocab,
                self.batch,
                cap,
                &tag,
            )?,
        };
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
        self.ix = if self.embed.is_some() {
            None
        } else {
            Some(Self::decode_inputs(&rt, &tag))
        };
        // The old prefill recipe goes before the old decode recipe it binds
        // to; `base` stays for the weights.
        self.pf = pf;
        self.cur = Some(rt);
        self.slots = slots;
        self.cap = cap;
        self.tag = tag;
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

    /// Device decode loop: feed token `ids[b]` to every sequence `b` at its
    /// position and run `n` steps back to back, each consuming the greedy
    /// ids the previous one produced; returns the `n * B` ids step by step
    /// (`ids[j * B + b]` is sequence `b`'s id after step `j`; the last
    /// step's ids have not been fed yet). One readback for the whole run.
    /// Every sequence advances `n` positions: a finished one keeps going on
    /// its own output until the caller resets it.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails or a launch never
    /// completes.
    ///
    /// # Panics
    ///
    /// Panics if the loop was not built, `ids` is not one id per sequence,
    /// `n` is 0, or the run would overflow the cache.
    pub fn run_ids(&mut self, ids: &[u32], n: usize) -> Result<Vec<u32>> {
        Ok(self.loop_steps(ids, n, false)?.0)
    }

    /// Like [`BatchedModel::run_ids`], also returning the last step's
    /// logits `[B, vocab]`.
    ///
    /// # Errors
    ///
    /// As [`BatchedModel::run_ids`].
    pub fn run_ids_logits(&mut self, ids: &[u32], n: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        self.loop_steps(ids, n, true)
    }

    fn loop_steps(
        &mut self,
        seeds: &[u32],
        n: usize,
        want_logits: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let nb = self.batch;
        assert!(n >= 1);
        assert_eq!(seeds.len(), nb, "one seed id per sequence");
        for &id in seeds {
            assert!((id as usize) < self.vocab, "token id {id} out of range");
        }
        let furthest = self.pos.iter().copied().max().unwrap_or(0);
        assert!(
            furthest + n <= self.capacity,
            "cache overflow at {furthest}+{n} of {}",
            self.capacity
        );
        self.ensure(furthest + n)?;
        let trace = std::env::var("RENG_STEP_TRACE").is_ok();
        let t0 = Instant::now();
        let (ring, postab, stride, head, resident) = {
            let lp = self.lp.as_mut().expect("device decode loop not built");
            // A run needs `n + 1` ring rows from `head`: wrap when they are
            // not there (the seed is then uploaded again).
            if lp.head + n > self.capacity {
                lp.head = 0;
                lp.known = None;
            }
            let resident = lp.known.as_deref() == Some(seeds);
            (lp.ring, lp.postab, lp.stride, lp.head, resident)
        };
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
        let row = |r: usize| (r * stride) as u64;
        let (n_in, n_pos) = (format!("IDS_IN{}", self.tag), format!("POS{}", self.tag));
        let rt = match self.cur {
            Some(ref mut r) => r,
            None => &mut self.base,
        };
        // The seed goes into the head row unless the last run left those
        // very ids there; the positions go in either way.
        let mut parts: Vec<(u64, &[u8])> = vec![(postab, &tab)];
        if !resident {
            parts.push((ring + row(head), &seed_bytes));
        }
        rt.upload_at_multi(&parts)?;
        rt.fence()?;
        let t_seed = t0.elapsed();
        let first_out = ring + row(head + 1);
        rt.fill_sentinel_d32(first_out, n * stride / 4)?;
        let t_fill = t0.elapsed() - t_seed;
        for j in 0..n {
            rt.rebind(&n_in, ring + row(head + j));
            rt.rebind("IDS", ring + row(head + j + 1));
            rt.rebind(&n_pos, postab + row(j));
            rt.launch_only()?;
        }
        let t_launch = t0.elapsed() - t_seed - t_fill;
        let ids: Vec<u32> = rt
            .read_i32_rows(first_out, stride, n, nb)?
            .into_iter()
            .map(|i| i as u32)
            .collect();
        let t_read = t0.elapsed() - t_seed - t_fill - t_launch;
        let logits = if want_logits {
            rt.read_bf16_range("LOGITS", 0, nb)?
        } else {
            Vec::new()
        };
        let lp = self.lp.as_mut().expect("checked above");
        lp.head = head + n;
        lp.known = Some(ids[(n - 1) * nb..].to_vec());
        if trace {
            eprintln!(
                "loop trace: {n} steps x {nb} sequences from ring row {head}: {} {:.2} ms, sentinel fill {:.2} ms, launches {:.2} ms, wait+readback {:.2} ms",
                if resident {
                    "positions uploaded, seed resident"
                } else {
                    "positions and seed uploaded"
                },
                t_seed.as_secs_f64() * 1e3,
                t_fill.as_secs_f64() * 1e3,
                t_launch.as_secs_f64() * 1e3,
                t_read.as_secs_f64() * 1e3
            );
        }
        for p in &mut self.pos {
            *p += n;
        }
        Ok((ids, logits))
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
    /// Panics if `x` is not `B` rows, a sequence would overflow the cache,
    /// or the decode recipe is the device decode loop (which takes ids:
    /// [`BatchedModel::run_ids`]).
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
        let ix = self
            .ix
            .expect("the device decode loop takes token ids: use run_ids");
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
