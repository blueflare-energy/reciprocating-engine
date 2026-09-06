//! Parallelism strategies over N cards, and the ceiling each one admits.
//!
//! [`decode_ceiling`](crate::decode_ceiling) answers "how fast can one card
//! go". This module answers the two questions that follow: how fast can N
//! cards go, and which way of using the N cards is the best one for *this*
//! model. Different models want different splits, so the answer is per model,
//! per card count and per objective.
//!
//! Two objectives, because they do not have the same winner:
//!
//! - [`Objective::SingleStream`]: tokens per second for one sequence, batch 1.
//!   The latency a single user sees.
//! - [`Objective::Aggregate`]: tokens per second summed over the whole machine
//!   at a given batch per replica. The throughput a server sees.
//!
//! Five strategies: data, tensor, pipeline and expert parallelism, and the
//! data-by-tensor hybrids. Every one is a bytes-per-card-per-token count
//! divided by one card's HBM bandwidth, plus the communication that split
//! forces. The bandwidth term is the *physical* ceiling; the physical ceiling
//! plus the measured collective floor is the *practical* one. Neither is a
//! prediction: they are what the hardware admits.

use std::fmt;

use reng_core::{Error, Result};

use crate::{Bottleneck, HardwareSpec, ModelShape, Precision};

/// What a plan is being judged on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// One sequence at batch 1: the latency of a single stream.
    SingleStream,
    /// Every card summed, at the scenario's batch per replica.
    Aggregate,
}

impl Objective {
    /// Short name for a table header.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Objective::SingleStream => "single-stream",
            Objective::Aggregate => "aggregate",
        }
    }
}

/// How the N cards are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// N independent replicas, each holding the whole model.
    Data,
    /// One model split across N cards, Megatron style.
    Tensor,
    /// Consecutive layers split into N stages.
    Pipeline,
    /// Routed experts spread over N cards (mixture-of-experts models only).
    Expert,
    /// `replicas` data-parallel groups of `world` tensor-parallel cards.
    Hybrid { replicas: u32, world: u32 },
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Strategy::Data => write!(f, "data"),
            Strategy::Tensor => write!(f, "tensor"),
            Strategy::Pipeline => write!(f, "pipeline"),
            Strategy::Expert => write!(f, "expert"),
            Strategy::Hybrid { replicas, world } => write!(f, "dp{replicas} x tp{world}"),
        }
    }
}

/// The measured all-reduce latency floor, per world and message size.
///
/// The default table is the median of three repeats of `reng-hccl-test` on
/// this box on 2026-09-06 (the `us/op wall` column of a 1000-call in-place f32
/// all-reduce, enqueued back to back on one stream). World 2 is the pair with
/// no port-9 link (modules 5 and 6), which is the shape `reng-tp --modules
/// 0,1` runs in; world 4 is modules 0 to 3; world 8 is all eight. Between two
/// tabulated sizes the latency is interpolated linearly in the log of the
/// message bytes, and it is clamped at both ends of the table.
///
/// The table is a property of one machine on one day, so it is overridable:
/// [`CollectiveFloor::from_json`] reads the same shape from a file.
#[derive(Debug, Clone)]
pub struct CollectiveFloor {
    /// Where the numbers came from, for the CLI to print.
    pub source: String,
    /// One entry per world, each a size-ordered list of (message bytes,
    /// seconds per all-reduce).
    pub worlds: Vec<(u32, Vec<(u64, f64)>)>,
}

/// The 8 KB to 64 MB all-reduce sweep measured on this box on 2026-09-06,
/// in microseconds per call.
const MEASURED_2026_09_06: &[(u32, &[(u64, f64)])] = &[
    (
        2,
        &[
            (8 * 1024, 18.0),
            (32 * 1024, 19.6),
            (128 * 1024, 23.1),
            (1024 * 1024, 50.8),
            (16 * 1024 * 1024, 491.7),
            (64 * 1024 * 1024, 1918.2),
        ],
    ),
    (
        4,
        &[
            (8 * 1024, 19.7),
            (32 * 1024, 19.7),
            (128 * 1024, 22.1),
            (1024 * 1024, 37.0),
            (16 * 1024 * 1024, 253.6),
            (64 * 1024 * 1024, 1013.1),
        ],
    ),
    (
        8,
        &[
            (8 * 1024, 24.7),
            (32 * 1024, 25.0),
            (128 * 1024, 25.1),
            (1024 * 1024, 30.7),
            (16 * 1024 * 1024, 142.4),
            (64 * 1024 * 1024, 558.6),
        ],
    ),
];

impl Default for CollectiveFloor {
    fn default() -> Self {
        Self::measured()
    }
}

impl CollectiveFloor {
    /// The documented constant table: measured on this box on 2026-09-06.
    #[must_use]
    pub fn measured() -> Self {
        Self {
            source: "measured on this box on 2026-09-06 (reng-hccl-test, \
                     median of three repeats)"
                .to_string(),
            worlds: MEASURED_2026_09_06
                .iter()
                .map(|(w, pts)| {
                    (
                        *w,
                        pts.iter()
                            .map(|(b, us)| (*b, us * 1e-6))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        }
    }

    /// Read a table from JSON: `{"source": "...", "worlds": {"2": [[8192,
    /// 18.0], ...], ...}}`, with the message size in bytes and the latency in
    /// microseconds per all-reduce.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON does not parse or does not have that
    /// shape.
    pub fn from_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("collective table parse: {e}")))?;
        let src = v
            .get("source")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("from JSON")
            .to_string();
        let worlds = v
            .get("worlds")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| Error::Other("collective table: no \"worlds\" object".into()))?;
        let mut out = Vec::new();
        for (k, points) in worlds {
            let world: u32 = k.parse().map_err(|_| {
                Error::Other(format!("collective table: world {k:?} is not a number"))
            })?;
            let arr = points.as_array().ok_or_else(|| {
                Error::Other(format!("collective table: world {k} is not a list"))
            })?;
            let mut pts = Vec::new();
            for p in arr {
                let pair = p.as_array().filter(|a| a.len() == 2).ok_or_else(|| {
                    Error::Other(format!(
                        "collective table: world {k} wants [bytes, us] pairs"
                    ))
                })?;
                let bytes = pair[0].as_u64().ok_or_else(|| {
                    Error::Other(format!(
                        "collective table: world {k} has a non-integer size"
                    ))
                })?;
                let us = pair[1].as_f64().ok_or_else(|| {
                    Error::Other(format!(
                        "collective table: world {k} has a non-number latency"
                    ))
                })?;
                pts.push((bytes, us * 1e-6));
            }
            pts.sort_by_key(|(b, _)| *b);
            if pts.is_empty() {
                return Err(Error::Other(format!(
                    "collective table: world {k} is empty"
                )));
            }
            out.push((world, pts));
        }
        if out.is_empty() {
            return Err(Error::Other("collective table: no worlds".into()));
        }
        out.sort_by_key(|(w, _)| *w);
        Ok(Self {
            source: src,
            worlds: out,
        })
    }

    /// Seconds for one all-reduce of `bytes` over `world` cards.
    ///
    /// World 1 has no communicator and costs nothing. A world the table does
    /// not carry takes the largest tabulated world below it, or the smallest
    /// one there is.
    #[must_use]
    pub fn all_reduce_s(&self, world: u32, bytes: u64) -> f64 {
        if world <= 1 {
            return 0.0;
        }
        let pts = match self.worlds.iter().find(|(w, _)| *w == world) {
            Some((_, p)) => p,
            None => {
                let below = self.worlds.iter().rev().find(|(w, _)| *w < world);
                match below.or_else(|| self.worlds.first()) {
                    Some((_, p)) => p,
                    None => return 0.0,
                }
            }
        };
        interpolate_log(pts, bytes)
    }
}

/// Linear interpolation in the log of the message size, clamped at both ends.
fn interpolate_log(points: &[(u64, f64)], bytes: u64) -> f64 {
    let Some(&(first_b, first_t)) = points.first() else {
        return 0.0;
    };
    if bytes <= first_b {
        return first_t;
    }
    let &(last_b, last_t) = points.last().unwrap_or(&(first_b, first_t));
    if bytes >= last_b {
        return last_t;
    }
    for w in points.windows(2) {
        let ((b0, t0), (b1, t1)) = (w[0], w[1]);
        if bytes <= b1 {
            let f =
                ((bytes as f64).ln() - (b0 as f64).ln()) / ((b1 as f64).ln() - (b0 as f64).ln());
            return t0 + f * (t1 - t0);
        }
    }
    last_t
}

/// A ceiling for one objective: what the bandwidth alone admits, and what it
/// admits once the split's own communication is charged.
#[derive(Debug, Clone)]
pub struct Rate {
    /// Seconds per decode step with no communication charged.
    pub physical_s: f64,
    /// Seconds per decode step with the measured collective floor charged.
    pub practical_s: f64,
    /// Tokens per second at `practical_s`.
    pub tokens_per_s: f64,
    /// Tokens per second at `physical_s`.
    pub physical_tokens_per_s: f64,
    /// Which resource sets `physical_s`.
    pub bottleneck: Bottleneck,
}

impl Rate {
    fn new(tokens: f64, physical_s: f64, comm_s: f64, bottleneck: Bottleneck) -> Self {
        let practical_s = physical_s + comm_s;
        Self {
            physical_s,
            practical_s,
            tokens_per_s: tokens / practical_s,
            physical_tokens_per_s: tokens / physical_s,
            bottleneck,
        }
    }
}

/// One way of using N cards for one model, and what it admits.
#[derive(Debug, Clone)]
pub struct Plan {
    pub strategy: Strategy,
    pub cards: u32,
    /// Batch per replica the aggregate rate was computed at.
    pub batch: u32,
    /// `None` when the split is admissible; why it is not, otherwise.
    pub rejected: Option<String>,
    /// One stream at batch 1.
    pub single_stream: Rate,
    /// Every card summed, at `batch` per replica.
    pub aggregate: Rate,
    /// Resident bytes on the busiest card: its share of the weights plus its
    /// share of the KV cache.
    pub resident_bytes_per_card: u64,
    /// Communication charged to one single-stream token.
    pub collective_s: f64,
    /// True where the engine has no such layer yet, so the row is a
    /// projection from the config rather than a ceiling for code that exists.
    pub projected: bool,
    /// What decided this plan's numbers, one short statement each.
    pub notes: Vec<String>,
}

impl Plan {
    /// Whether the split is admissible at all.
    #[must_use]
    pub fn admissible(&self) -> bool {
        self.rejected.is_none()
    }

    /// The rate for one objective.
    #[must_use]
    pub fn rate(&self, obj: Objective) -> &Rate {
        match obj {
            Objective::SingleStream => &self.single_stream,
            Objective::Aggregate => &self.aggregate,
        }
    }
}

/// The model, the precision and the decode shape every strategy is measured
/// at.
#[derive(Debug, Clone)]
pub struct Scenario<'a> {
    /// One card. Strategies scale it themselves; do not pass a scaled spec.
    pub hw: &'a HardwareSpec,
    pub model: &'a ModelShape,
    pub prec: Precision,
    pub kv: Precision,
    /// Batch per replica for the aggregate objective.
    pub batch: u32,
    /// Context length the KV cache is charged at.
    pub ctx: u32,
    pub floor: &'a CollectiveFloor,
}

impl<'a> Scenario<'a> {
    /// A scenario at bf16 weights and bf16 KV cache.
    #[must_use]
    pub fn new(
        hw: &'a HardwareSpec,
        model: &'a ModelShape,
        batch: u32,
        ctx: u32,
        floor: &'a CollectiveFloor,
    ) -> Self {
        Self {
            hw,
            model,
            prec: Precision::Bf16,
            kv: Precision::Bf16,
            batch,
            ctx,
            floor,
        }
    }

    /// Weight bytes of the layer stack that a token reads (the active MLP for
    /// a mixture of experts). The embedding table is a lookup, not a matmul,
    /// so it is not here; the LM head is counted separately because tensor
    /// parallelism replicates it.
    fn body_bytes(&self) -> f64 {
        let m = self.model;
        f64::from(m.layers)
            * (m.attn_params_per_layer() + m.active_mlp_params_per_layer()) as f64
            * self.prec.weight_bytes()
    }

    fn head_bytes(&self) -> f64 {
        self.model.lm_head_params() as f64 * self.prec.weight_bytes()
    }

    fn embedding_bytes(&self) -> f64 {
        (self.model.hidden * self.model.vocab) as f64 * self.prec.weight_bytes()
    }

    /// Resident weight bytes of the layer stack: every expert, not just the
    /// active ones.
    fn resident_body_bytes(&self) -> f64 {
        let m = self.model;
        f64::from(m.layers)
            * (m.attn_params_per_layer() + m.total_mlp_params_per_layer()) as f64
            * self.prec.weight_bytes()
    }

    fn kv_bytes(&self, batch: u32) -> f64 {
        f64::from(batch) * f64::from(self.ctx) * self.model.kv_bytes_per_token(self.kv)
    }

    /// Matmul FLOPs for the layer stack and the attention scores.
    fn body_flops(&self, batch: u32) -> f64 {
        let m = self.model;
        let b = f64::from(batch);
        let params = f64::from(m.layers)
            * (m.attn_params_per_layer() + m.active_mlp_params_per_layer()) as f64;
        let attn = 4.0
            * f64::from(m.layers)
            * b
            * f64::from(m.n_heads)
            * f64::from(self.ctx)
            * m.head_dim as f64;
        2.0 * params * b + attn
    }

    fn head_flops(&self, batch: u32) -> f64 {
        2.0 * self.model.lm_head_params() as f64 * f64::from(batch)
    }

    /// The all-reduce message a tensor-parallel layer sends: the hidden vector
    /// of every sequence in the batch, in f32.
    fn all_reduce_bytes(&self, batch: u32) -> u64 {
        self.model.hidden * u64::from(batch) * 4
    }

    fn card_step(&self, bytes: f64, flops: f64) -> (f64, Bottleneck) {
        let mem = bytes / self.hw.hbm_bw;
        let compute = flops / self.prec.compute_flops(self.hw);
        if compute >= mem {
            (compute, Bottleneck::Compute)
        } else {
            (mem, Bottleneck::HbmBandwidth)
        }
    }
}

/// (a) Data parallel: N replicas, each holding the whole model and decoding
/// its own batch. Nothing crosses the interconnect, so a stream is exactly as
/// fast as it is on one card and the machine does N times the work. The one
/// requirement is that the model and its cache fit one card.
#[must_use]
pub fn data_parallel(s: &Scenario, cards: u32) -> Plan {
    let n = f64::from(cards.max(1));
    let bytes1 = s.body_bytes() + s.head_bytes() + s.kv_bytes(1);
    let (step1, bn1) = s.card_step(bytes1, s.body_flops(1) + s.head_flops(1));
    let bytes_b = s.body_bytes() + s.head_bytes() + s.kv_bytes(s.batch);
    let (step_b, bn_b) = s.card_step(bytes_b, s.body_flops(s.batch) + s.head_flops(s.batch));

    let resident =
        (s.resident_body_bytes() + s.embedding_bytes() + s.head_bytes() + s.kv_bytes(s.batch))
            as u64;
    let rejected = (resident > s.hw.hbm_bytes).then(|| {
        format!(
            "does not fit one card ({:.1} GB resident against {:.1} GB of HBM)",
            resident as f64 / 1e9,
            s.hw.hbm_bytes as f64 / 1e9
        )
    });

    Plan {
        strategy: Strategy::Data,
        cards,
        batch: s.batch,
        rejected,
        single_stream: Rate::new(1.0, step1, 0.0, bn1),
        aggregate: Rate::new(n * f64::from(s.batch), step_b, 0.0, bn_b),
        resident_bytes_per_card: resident,
        collective_s: 0.0,
        projected: false,
        notes: vec![
            "no collective on the token path: a replica never talks to another".to_string(),
            format!("aggregate is {cards} x the single-card ceiling"),
        ],
    }
}

/// Whether the Megatron split divides this shape `world` ways, and why not.
fn tensor_split_reason(model: &ModelShape, world: u32) -> Option<String> {
    if world <= 1 {
        return None;
    }
    let mut bad = Vec::new();
    if model.n_heads % world != 0 {
        bad.push(format!("{} attention heads", model.n_heads));
    }
    if model.n_kv_heads % world != 0 {
        bad.push(format!("{} kv heads", model.n_kv_heads));
    }
    if model.ff != 0 && model.ff % u64::from(world) != 0 {
        bad.push(format!("intermediate size {}", model.ff));
    }
    (!bad.is_empty()).then(|| format!("{} not divisible by {world}", bad.join(" and ")))
}

/// (b) Tensor parallel: one model over N cards, Megatron style. Each card
/// streams `1/N` of the layer weights and `1/N` of the KV cache per token; the
/// LM head and the norms are replicated, so they cost what they cost on one
/// card, and the embedding stays the lookup the single-card formula treats it
/// as. The split's own cost is two all-reduces per layer of `hidden x batch x
/// 4` bytes, at the measured floor for this world.
#[must_use]
pub fn tensor_parallel(s: &Scenario, cards: u32) -> Plan {
    let n = f64::from(cards.max(1));
    let layers = f64::from(s.model.layers);

    let card_bytes = |batch: u32| s.body_bytes() / n + s.head_bytes() + s.kv_bytes(batch) / n;
    let card_flops = |batch: u32| s.body_flops(batch) / n + s.head_flops(batch);
    let (step1, bn1) = s.card_step(card_bytes(1), card_flops(1));
    let (step_b, bn_b) = s.card_step(card_bytes(s.batch), card_flops(s.batch));

    let coll1 = 2.0 * layers * s.floor.all_reduce_s(cards, s.all_reduce_bytes(1));
    let coll_b = 2.0 * layers * s.floor.all_reduce_s(cards, s.all_reduce_bytes(s.batch));

    let resident = (s.resident_body_bytes() / n
        + s.embedding_bytes()
        + s.head_bytes()
        + s.kv_bytes(s.batch) / n) as u64;
    let rejected = tensor_split_reason(s.model, cards).or_else(|| {
        (resident > s.hw.hbm_bytes).then(|| {
            format!(
                "does not fit {cards} cards ({:.1} GB per card against {:.1} GB of HBM)",
                resident as f64 / 1e9,
                s.hw.hbm_bytes as f64 / 1e9
            )
        })
    });

    let per_layer_weights = s.body_bytes() / n / layers / s.hw.hbm_bw;
    Plan {
        strategy: Strategy::Tensor,
        cards,
        batch: s.batch,
        rejected,
        single_stream: Rate::new(1.0, step1, coll1, bn1),
        aggregate: Rate::new(f64::from(s.batch), step_b, coll_b, bn_b),
        resident_bytes_per_card: resident,
        collective_s: coll1,
        projected: false,
        notes: vec![
            format!(
                "two all-reduces of {} bytes per layer: {:.1} us each, {:.2} ms per token",
                s.all_reduce_bytes(1),
                s.floor.all_reduce_s(cards, s.all_reduce_bytes(1)) * 1e6,
                coll1 * 1e3
            ),
            format!(
                "collective floor {:.1} us per layer against {:.1} us of weight time per layer",
                2.0 * s.floor.all_reduce_s(cards, s.all_reduce_bytes(1)) * 1e6,
                per_layer_weights * 1e6
            ),
        ],
    }
}

/// Layer counts of `n` consecutive stages, the longer ones first.
fn stage_layers(layers: u32, n: u32) -> Vec<u32> {
    let n = n.max(1);
    let base = layers / n;
    let rem = layers % n;
    (0..n)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect()
}

/// (c) Pipeline parallel: consecutive layers cut into N stages. One token
/// crosses every stage in turn, so it still reads the whole model once and a
/// single stream is no faster than on one card -- only the hand-off per stage
/// boundary is added. With N micro-batches in flight the stages all work at
/// once and the machine reaches up to N times the throughput. Of the splits
/// that let a model larger than one card run, this is the one that
/// communicates least: one activation per boundary against two all-reduces per
/// layer.
#[must_use]
pub fn pipeline_parallel(s: &Scenario, cards: u32) -> Plan {
    let n = cards.max(1);
    let sizes = stage_layers(s.model.layers, n);
    let layers = f64::from(s.model.layers);
    let per_layer_bytes = s.body_bytes() / layers;
    let per_layer_flops = |batch: u32| s.body_flops(batch) / layers;

    // The hand-off is one activation, hidden x batch, in the weight dtype. No
    // point-to-point send is measured on this box, so it is charged at the
    // world-2 all-reduce floor for the same message: an upper bound, since an
    // all-reduce moves the message twice and a send moves it once.
    let handoff = |batch: u32| {
        s.floor.all_reduce_s(
            2,
            s.model.hidden * u64::from(batch) * (s.prec.weight_bytes() as u64).max(1),
        )
    };

    let stage_step = |batch: u32| {
        let kv_per_layer = s.kv_bytes(batch) / layers;
        let mut worst = 0.0f64;
        let mut total = 0.0;
        let mut bn = Bottleneck::HbmBandwidth;
        for (i, sz) in sizes.iter().enumerate() {
            let l = f64::from(*sz);
            let head = if i + 1 == sizes.len() {
                s.head_bytes()
            } else {
                0.0
            };
            let head_f = if i + 1 == sizes.len() {
                s.head_flops(batch)
            } else {
                0.0
            };
            let (t, b) = s.card_step(
                l * (per_layer_bytes + kv_per_layer) + head,
                l * per_layer_flops(batch) + head_f,
            );
            total += t;
            if t > worst {
                worst = t;
                bn = b;
            }
        }
        (total, worst, bn)
    };

    let (total1, worst1, bn1) = stage_step(1);
    let (_, worst_b, bn_b) = stage_step(s.batch);
    let boundaries = f64::from(n - 1);

    // Resident: the busiest stage's weights, plus the embedding on the first
    // stage and the LM head on the last.
    let biggest = f64::from(*sizes.iter().max().unwrap_or(&0));
    let resident = (biggest / layers * (s.resident_body_bytes() + s.kv_bytes(s.batch))
        + s.embedding_bytes().max(s.head_bytes())) as u64;
    let rejected = (resident > s.hw.hbm_bytes).then(|| {
        format!(
            "does not fit {cards} stages ({:.1} GB on the busiest against {:.1} GB of HBM)",
            resident as f64 / 1e9,
            s.hw.hbm_bytes as f64 / 1e9
        )
    });

    Plan {
        strategy: Strategy::Pipeline,
        cards,
        batch: s.batch,
        rejected,
        single_stream: Rate::new(1.0, total1, boundaries * handoff(1), bn1),
        // In steady state with N micro-batches in flight the pipeline retires
        // one micro-batch per bottleneck stage, so the machine's rate is that
        // stage's, not one token's.
        aggregate: Rate::new(f64::from(s.batch), worst_b, handoff(s.batch), bn_b),
        resident_bytes_per_card: resident,
        collective_s: boundaries * handoff(1),
        projected: false,
        notes: vec![
            format!(
                "stages of {} layers; a token still reads every layer, so one stream is not faster",
                sizes
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("+")
            ),
            format!(
                "{} hand-offs of {} bytes at {:.1} us each",
                n - 1,
                s.model.hidden * (s.prec.weight_bytes() as u64).max(1),
                handoff(1) * 1e6
            ),
            format!(
                "aggregate needs {n} micro-batches in flight; the bottleneck stage is {:.2} ms",
                (worst1 + handoff(1)) * 1e3
            ),
        ],
    }
}

/// (d) A hybrid: `replicas` data-parallel groups, each a `world`-card
/// tensor-parallel model. A stream sees the tensor-parallel latency; the
/// machine sees `replicas` times that group's throughput.
#[must_use]
pub fn hybrid(s: &Scenario, replicas: u32, world: u32) -> Plan {
    let inner = tensor_parallel(s, world);
    let r = f64::from(replicas.max(1));
    Plan {
        strategy: Strategy::Hybrid { replicas, world },
        cards: replicas * world,
        batch: s.batch,
        rejected: inner.rejected.clone(),
        single_stream: inner.single_stream.clone(),
        aggregate: Rate {
            tokens_per_s: inner.aggregate.tokens_per_s * r,
            physical_tokens_per_s: inner.aggregate.physical_tokens_per_s * r,
            ..inner.aggregate.clone()
        },
        resident_bytes_per_card: inner.resident_bytes_per_card,
        collective_s: inner.collective_s,
        projected: false,
        notes: vec![
            format!("{replicas} replicas of a {world}-card tensor-parallel group"),
            format!(
                "one stream is the tensor-parallel one ({:.1} tok/s); the machine is {replicas} x the group",
                inner.single_stream.tokens_per_s
            ),
        ],
    }
}

/// (e) Expert parallel, for a mixture of experts: the E routed experts are
/// spread over the N cards and a token reads its k of them. On average k/N of
/// the activated experts live on any one card, so a card streams the shared
/// weights (attention, router, shared experts) plus `k/N` of the expert bytes
/// per token. Attention is replicated, so the KV cache is not divided.
///
/// The dispatch and combine all-to-alls are an **assumption**: no all-to-all
/// is measured on this box, so they are charged at the measured all-reduce
/// floor for the same message and world, two per layer. That is a stand-in to
/// be replaced by a measurement. The engine has no mixture-of-experts layer
/// either, so plans from this function are marked [`Plan::projected`].
#[must_use]
pub fn expert_parallel(s: &Scenario, cards: u32) -> Plan {
    let n = f64::from(cards.max(1));
    let m = s.model;
    let layers = f64::from(m.layers);
    let wb = s.prec.weight_bytes();

    let Some(moe) = m.moe.as_ref() else {
        let dense = data_parallel(s, cards);
        return Plan {
            strategy: Strategy::Expert,
            rejected: Some("not a mixture of experts: no routed experts to spread".to_string()),
            projected: true,
            notes: vec!["dense model: every token reads every MLP weight".to_string()],
            ..dense
        };
    };

    let one_expert = 3.0 * (m.hidden * moe.expert_ff) as f64 * wb;
    let shared_per_layer = (m.hidden * u64::from(moe.n_experts)) as f64 * wb // router
        + f64::from(moe.shared_experts) * one_expert
        + m.attn_params_per_layer() as f64 * wb;
    let experts_per_card = f64::from(moe.top_k) / n * one_expert;

    let card_bytes = |batch: u32| {
        layers * (shared_per_layer + experts_per_card) + s.head_bytes() + s.kv_bytes(batch)
    };
    // Compute divides with the experts; attention and the head do not.
    let card_flops = |batch: u32| {
        let expert_flops = 2.0
            * layers
            * f64::from(moe.top_k)
            * (3 * m.hidden * moe.expert_ff) as f64
            * f64::from(batch)
            / n;
        let dense_flops = s.body_flops(batch)
            - 2.0
                * layers
                * f64::from(moe.top_k)
                * (3 * m.hidden * moe.expert_ff) as f64
                * f64::from(batch);
        dense_flops + expert_flops + s.head_flops(batch)
    };
    let (step1, bn1) = s.card_step(card_bytes(1), card_flops(1));
    let (step_b, bn_b) = s.card_step(card_bytes(s.batch), card_flops(s.batch));

    let a2a = |batch: u32| 2.0 * layers * s.floor.all_reduce_s(cards, s.all_reduce_bytes(batch));

    let resident = (layers * (shared_per_layer + f64::from(moe.n_experts) / n * one_expert)
        + s.embedding_bytes()
        + s.head_bytes()
        + s.kv_bytes(s.batch)) as u64;
    let rejected = (resident > s.hw.hbm_bytes).then(|| {
        format!(
            "does not fit {cards} cards ({:.1} GB per card against {:.1} GB of HBM)",
            resident as f64 / 1e9,
            s.hw.hbm_bytes as f64 / 1e9
        )
    });

    Plan {
        strategy: Strategy::Expert,
        cards,
        batch: s.batch,
        rejected,
        single_stream: Rate::new(1.0, step1, a2a(1), bn1),
        aggregate: Rate::new(f64::from(s.batch), step_b, a2a(s.batch), bn_b),
        resident_bytes_per_card: resident,
        collective_s: a2a(1),
        projected: true,
        notes: vec![
            format!(
                "{} of {} experts per token; a card averages {:.2} of them",
                moe.top_k,
                moe.n_experts,
                f64::from(moe.top_k) / n
            ),
            format!(
                "bytes per card per token: shared {:.2} GB + experts {:.2} GB",
                layers * shared_per_layer / 1e9,
                layers * experts_per_card / 1e9
            ),
            "the all-to-all floor is an assumption: charged at the measured all-reduce \
             latency, two per layer, not yet measured on this box"
                .to_string(),
            "projected: the engine has no mixture-of-experts layer yet".to_string(),
        ],
    }
}

/// Every strategy that could use `cards` cards, admissible or not.
///
/// The hybrids are every `replicas x world` factorisation with both sides
/// above one. Expert parallel is only included for a mixture of experts.
#[must_use]
pub fn plans(s: &Scenario, cards: u32) -> Vec<Plan> {
    let cards = cards.max(1);
    let mut out = vec![
        data_parallel(s, cards),
        tensor_parallel(s, cards),
        pipeline_parallel(s, cards),
    ];
    for world in 2..cards {
        if cards % world == 0 && cards / world > 1 {
            out.push(hybrid(s, cards / world, world));
        }
    }
    if s.model.moe.is_some() {
        out.push(expert_parallel(s, cards));
    }
    out
}

/// The best plan for one objective, and why it won.
#[derive(Debug, Clone)]
pub struct Choice {
    pub objective: Objective,
    pub plan: Plan,
    /// Every plan considered, best first.
    pub ranked: Vec<Plan>,
    /// The facts that decided it.
    pub reasons: Vec<String>,
}

/// (f) The chooser: the best strategy for this model on this many cards for
/// this objective, with the reasons it won.
///
/// Ties go to the simpler strategy in the order data, tensor, pipeline,
/// hybrid, expert, and a projected plan never beats a measured-path one on an
/// equal number.
#[must_use]
pub fn choose(s: &Scenario, cards: u32, obj: Objective) -> Choice {
    let mut ranked: Vec<Plan> = plans(s, cards)
        .into_iter()
        .filter(Plan::admissible)
        .collect();
    ranked.sort_by(|a, b| {
        b.rate(obj)
            .tokens_per_s
            .partial_cmp(&a.rate(obj).tokens_per_s)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.projected.cmp(&b.projected))
            .then(order(a.strategy).cmp(&order(b.strategy)))
    });

    let one_card = data_parallel(s, 1);
    let tp = tensor_parallel(s, cards);
    let mut reasons = Vec::new();
    reasons.push(match &one_card.rejected {
        None => format!(
            "fits one card: {:.1} GB resident, so data parallel is admissible",
            one_card.resident_bytes_per_card as f64 / 1e9
        ),
        Some(why) => format!("{why}, so data parallel is out"),
    });
    reasons.push(match tensor_split_reason(s.model, cards) {
        None => format!(
            "heads divide {cards} ways: {} q heads, {} kv heads, intermediate {}",
            s.model.n_heads, s.model.n_kv_heads, s.model.ff
        ),
        Some(why) => format!("no tensor split at {cards}: {why}"),
    });
    let per_layer_weights =
        s.body_bytes() / f64::from(cards.max(1)) / f64::from(s.model.layers) / s.hw.hbm_bw;
    let per_layer_coll = 2.0 * s.floor.all_reduce_s(cards, s.all_reduce_bytes(1));
    reasons.push(format!(
        "collective floor {:.1} us per layer against {:.1} us of weight time per layer at {cards} \
         cards: the split {} the token",
        per_layer_coll * 1e6,
        per_layer_weights * 1e6,
        if per_layer_coll > per_layer_weights {
            "costs more than it saves on"
        } else {
            "pays for itself on"
        }
    ));
    if let Some(best) = ranked.first() {
        reasons.push(format!(
            "{} wins {} at {:.1} tok/s; {}",
            best.strategy,
            obj.label(),
            best.rate(obj).tokens_per_s,
            match ranked.get(1) {
                Some(second) => format!(
                    "next is {} at {:.1}",
                    second.strategy,
                    second.rate(obj).tokens_per_s
                ),
                None => "no other split is admissible".to_string(),
            }
        ));
    }
    let plan = ranked.first().cloned().unwrap_or_else(|| {
        let mut p = tp;
        p.notes
            .push("no admissible split on this many cards".to_string());
        p
    });
    Choice {
        objective: obj,
        plan,
        ranked,
        reasons,
    }
}

fn order(s: Strategy) -> u8 {
    match s {
        Strategy::Data => 0,
        Strategy::Tensor => 1,
        Strategy::Pipeline => 2,
        Strategy::Hybrid { .. } => 3,
        Strategy::Expert => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_from_hf_config;

    /// DeepSeek-R1-Distill-Llama-70B, from its config.json.
    fn llama_70b() -> ModelShape {
        model_from_hf_config(
            r#"{"_name_or_path":"DeepSeek-R1-Distill-Llama-70B","hidden_size":8192,
                "num_hidden_layers":80,"num_attention_heads":64,"num_key_value_heads":8,
                "head_dim":128,"intermediate_size":28672,"vocab_size":128256,
                "tie_word_embeddings":false}"#,
        )
        .expect("70B config")
    }

    /// Qwen2.5-32B, from its config.json.
    fn qwen_32b() -> ModelShape {
        model_from_hf_config(
            r#"{"_name_or_path":"Qwen2.5-32B","hidden_size":5120,"num_hidden_layers":64,
                "num_attention_heads":40,"num_key_value_heads":8,"intermediate_size":27648,
                "vocab_size":152064,"tie_word_embeddings":false}"#,
        )
        .expect("32B config")
    }

    /// Llama-3.2-1B, from its config.json.
    fn llama_1b() -> ModelShape {
        model_from_hf_config(
            r#"{"_name_or_path":"Llama-3.2-1B","hidden_size":2048,"num_hidden_layers":16,
                "num_attention_heads":32,"num_key_value_heads":8,"head_dim":64,
                "intermediate_size":8192,"vocab_size":128256,"tie_word_embeddings":true}"#,
        )
        .expect("1B config")
    }

    /// Mixtral-8x7B-v0.1, from its config.json.
    fn mixtral() -> ModelShape {
        model_from_hf_config(
            r#"{"_name_or_path":"Mixtral-8x7B-v0.1","hidden_size":4096,"num_hidden_layers":32,
                "num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,
                "num_local_experts":8,"num_experts_per_tok":2,"vocab_size":32000,
                "tie_word_embeddings":false}"#,
        )
        .expect("mixtral config")
    }

    #[test]
    fn collective_floor_interpolates_the_measured_sweep() {
        let f = CollectiveFloor::measured();
        // Tabulated points come back exactly.
        assert!((f.all_reduce_s(2, 8 * 1024) - 18.0e-6).abs() < 1e-12);
        assert!((f.all_reduce_s(8, 128 * 1024) - 25.1e-6).abs() < 1e-12);
        // Below the smallest message the floor is flat.
        assert!((f.all_reduce_s(2, 4 * 1024) - 18.0e-6).abs() < 1e-12);
        // World 1 has no communicator.
        assert!(f.all_reduce_s(1, 1 << 20).abs() < 1e-15);
        // 10 KB at world 2 is 36.5 us for two collectives, as the
        // measurement report's per-layer table has it.
        assert!((2.0 * f.all_reduce_s(2, 10 * 1024) * 1e6 - 36.5).abs() < 0.05);
        // 28 KB at world 2: 38.9 us for the pair.
        assert!((2.0 * f.all_reduce_s(2, 28 * 1024) * 1e6 - 38.9).abs() < 0.05);
        // Bigger worlds cost more at the floor.
        assert!(f.all_reduce_s(8, 8 * 1024) > f.all_reduce_s(4, 8 * 1024));
        assert!(f.all_reduce_s(4, 8 * 1024) > f.all_reduce_s(2, 8 * 1024));
    }

    #[test]
    fn collective_floor_from_json_overrides_the_constant() {
        let f = CollectiveFloor::from_json(
            r#"{"source":"a faster fabric","worlds":{"2":[[8192,9.0],[1048576,25.0]],
                "8":[[8192,12.0],[1048576,15.0]]}}"#,
        )
        .expect("parses");
        assert!((f.all_reduce_s(2, 8 * 1024) - 9.0e-6).abs() < 1e-12);
        assert!((f.all_reduce_s(8, 1024 * 1024) - 15.0e-6).abs() < 1e-12);
        // A world the table does not carry falls back to the one below it.
        assert!((f.all_reduce_s(4, 8 * 1024) - 9.0e-6).abs() < 1e-12);
        assert!(CollectiveFloor::from_json("{}").is_err());
        assert!(CollectiveFloor::from_json(r#"{"worlds":{"2":[[1]]}}"#).is_err());
    }

    #[test]
    fn tensor_parallel_70b_at_world_2_is_the_readme_figure() {
        // The README carries 28.8 ms per token for the 70B on two cards:
        // 136.90 GB of layer weights halved, plus the 2.10 GB LM head every
        // rank keeps, over 2.45 TB/s. The physical ceiling reproduces it.
        let hw = HardwareSpec::gaudi2();
        let m = llama_70b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 1, 192, &floor);
        let p = tensor_parallel(&s, 2);
        assert!(p.admissible(), "{:?}", p.rejected);
        assert!(
            (p.single_stream.physical_s * 1e3 - 28.8).abs() < 0.05,
            "physical {:.3} ms",
            p.single_stream.physical_s * 1e3
        );
        assert_eq!(p.single_stream.bottleneck, Bottleneck::HbmBandwidth);
        // Two all-reduces of 32 KB per layer, 80 layers: 3.136 ms on top.
        assert!(
            (p.collective_s * 1e3 - 3.136).abs() < 0.01,
            "collective {:.3} ms",
            p.collective_s * 1e3
        );
        assert!(p.single_stream.practical_s > p.single_stream.physical_s);
        // Eight cards halve it twice more, minus the bigger floor.
        let p8 = tensor_parallel(&s, 8);
        assert!(p8.single_stream.physical_s < p.single_stream.physical_s / 3.0);
    }

    #[test]
    fn data_parallel_32b_at_8_replicas_is_8x_one_card() {
        // The README's single-card figure for Qwen2.5-32B is 37.7 tok/s (65 GB
        // per token at 2.45 TB/s), at the CLI's default 4096-token context.
        // Eight replicas is exactly eight times that, and a stream is no
        // faster than on one card.
        let hw = HardwareSpec::gaudi2();
        let m = qwen_32b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 1, 4096, &floor);
        let one = data_parallel(&s, 1);
        assert!(
            (one.single_stream.tokens_per_s - 37.7).abs() < 0.05,
            "one card {:.2} tok/s",
            one.single_stream.tokens_per_s
        );
        let eight = data_parallel(&s, 8);
        assert!(eight.admissible(), "{:?}", eight.rejected);
        assert!(
            (eight.aggregate.tokens_per_s - 8.0 * one.aggregate.tokens_per_s).abs() < 1e-9,
            "aggregate {:.3}",
            eight.aggregate.tokens_per_s
        );
        assert!((eight.aggregate.tokens_per_s - 8.0 * 37.7).abs() < 0.4);
        assert!(
            (eight.single_stream.tokens_per_s - one.single_stream.tokens_per_s).abs() < 1e-9,
            "data parallel does not move single-stream"
        );
        assert!(eight.collective_s.abs() < 1e-15);
    }

    #[test]
    fn data_parallel_rejects_a_model_that_does_not_fit_one_card() {
        let hw = HardwareSpec::gaudi2();
        let m = llama_70b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 1, 192, &floor);
        let dp = data_parallel(&s, 8);
        assert!(!dp.admissible());
        assert!(dp.rejected.unwrap().contains("does not fit one card"));
        // The 32B does fit.
        let m32 = qwen_32b();
        let s32 = Scenario::new(&hw, &m32, 1, 192, &floor);
        assert!(data_parallel(&s32, 8).admissible());
    }

    #[test]
    fn tensor_parallel_rejects_an_indivisible_shape() {
        let hw = HardwareSpec::gaudi2();
        let floor = CollectiveFloor::measured();
        // SmolLM2-135M: 9 attention heads, so no world in {2, 4, 8} divides.
        let m = model_from_hf_config(
            r#"{"hidden_size":576,"num_hidden_layers":30,"num_attention_heads":9,
                "num_key_value_heads":3,"intermediate_size":1536,"vocab_size":49152}"#,
        )
        .expect("config");
        let s = Scenario::new(&hw, &m, 1, 192, &floor);
        for n in [2u32, 4, 8] {
            let p = tensor_parallel(&s, n);
            assert!(!p.admissible(), "world {n}");
            assert!(p.rejected.as_deref().unwrap().contains("9 attention heads"));
        }
        // Qwen2.5-7B: 28 heads and 4 kv heads split four ways, not eight.
        let q7 = model_from_hf_config(
            r#"{"hidden_size":3584,"num_hidden_layers":28,"num_attention_heads":28,
                "num_key_value_heads":4,"intermediate_size":18944,"vocab_size":152064}"#,
        )
        .expect("config");
        let s7 = Scenario::new(&hw, &q7, 1, 192, &floor);
        assert!(tensor_parallel(&s7, 4).admissible());
        let w8 = tensor_parallel(&s7, 8);
        assert!(!w8.admissible());
        let why = w8.rejected.unwrap();
        assert!(
            why.contains("28 attention heads") && why.contains("4 kv heads"),
            "{why}"
        );
    }

    #[test]
    fn pipeline_keeps_single_stream_and_multiplies_throughput() {
        let hw = HardwareSpec::gaudi2();
        let m = llama_70b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        let one = data_parallel(&s, 1);
        let pp = pipeline_parallel(&s, 8);
        assert!(pp.admissible(), "{:?}", pp.rejected);
        // A token still reads every layer: the physical single-stream time is
        // the single-card one, and only the hand-offs are added.
        assert!(
            (pp.single_stream.physical_s - one.single_stream.physical_s).abs()
                / one.single_stream.physical_s
                < 1e-9
        );
        assert!(pp.single_stream.practical_s > pp.single_stream.physical_s);
        assert!(pp.collective_s * 1e3 < 0.2, "seven hand-offs are cheap");
        // Eight stages with eight micro-batches in flight: up to 8x.
        let ratio = pp.aggregate.tokens_per_s / (8.0 * one.aggregate.physical_tokens_per_s);
        assert!(ratio > 0.9 && ratio <= 1.0, "ratio {ratio}");
        // 80 layers over 8 stages is 10 each.
        assert_eq!(stage_layers(80, 8), vec![10; 8]);
        // 30 layers over 8 stages: six of four, two of three.
        assert_eq!(stage_layers(30, 8), vec![4, 4, 4, 4, 4, 4, 3, 3]);
    }

    #[test]
    fn pipeline_communicates_less_than_tensor_for_a_model_over_one_card() {
        let hw = HardwareSpec::gaudi2();
        let m = llama_70b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 1, 192, &floor);
        let pp = pipeline_parallel(&s, 8);
        let tp = tensor_parallel(&s, 8);
        assert!(pp.collective_s < tp.collective_s / 10.0);
        // But tensor parallelism is the one that makes a single stream fast.
        assert!(tp.single_stream.tokens_per_s > pp.single_stream.tokens_per_s * 4.0);
    }

    #[test]
    fn hybrid_composes_the_arithmetic() {
        let hw = HardwareSpec::gaudi2();
        let m = qwen_32b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        let tp4 = tensor_parallel(&s, 4);
        let h = hybrid(&s, 2, 4);
        assert_eq!(h.cards, 8);
        // A stream sees the four-card latency.
        assert!((h.single_stream.tokens_per_s - tp4.single_stream.tokens_per_s).abs() < 1e-9);
        // The machine sees two of those groups.
        assert!((h.aggregate.tokens_per_s - 2.0 * tp4.aggregate.tokens_per_s).abs() < 1e-9);
        let h2 = hybrid(&s, 4, 2);
        let tp2 = tensor_parallel(&s, 2);
        assert!((h2.aggregate.tokens_per_s - 4.0 * tp2.aggregate.tokens_per_s).abs() < 1e-9);
        // And a hybrid inherits the inner world's admissibility.
        let q7 = model_from_hf_config(
            r#"{"hidden_size":3584,"num_hidden_layers":28,"num_attention_heads":28,
                "num_key_value_heads":4,"intermediate_size":18944,"vocab_size":152064}"#,
        )
        .expect("config");
        let s7 = Scenario::new(&hw, &q7, 8, 192, &floor);
        assert!(hybrid(&s7, 2, 4).admissible());
        assert!(!tensor_parallel(&s7, 8).admissible());
    }

    #[test]
    fn expert_parallel_spreads_the_experts() {
        let hw = HardwareSpec::gaudi2();
        let m = mixtral();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 1, 192, &floor);
        // At world 1 every activated expert is on the one card, so expert
        // parallel is exactly the single-card decode ceiling.
        let ep1 = expert_parallel(&s, 1);
        let dp1 = data_parallel(&s, 1);
        assert!(
            (ep1.single_stream.physical_s - dp1.single_stream.physical_s).abs()
                / dp1.single_stream.physical_s
                < 1e-9
        );
        // Spreading 8 experts over 8 cards leaves k/N = 0.25 of an expert's
        // bytes per card, so the physical ceiling rises but not eightfold: the
        // shared weights do not divide.
        let ep8 = expert_parallel(&s, 8);
        let gain =
            ep8.single_stream.physical_tokens_per_s / ep1.single_stream.physical_tokens_per_s;
        assert!(gain > 3.0 && gain < 8.0, "gain {gain}");
        assert!(ep8.projected, "no MoE layer in the engine yet");
        assert!(ep8.notes.iter().any(|n| n.contains("assumption")));
        // A dense model has no experts to spread.
        let dense = qwen_32b();
        let sd = Scenario::new(&hw, &dense, 1, 192, &floor);
        let e = expert_parallel(&sd, 8);
        assert!(!e.admissible());
        assert!(e.rejected.unwrap().contains("not a mixture of experts"));
    }

    #[test]
    fn chooser_picks_data_parallel_for_a_small_model() {
        let hw = HardwareSpec::gaudi2();
        let m = llama_1b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        // 16 layers at 8 cards: the collective floor is far larger than the
        // per-layer weight time, so splitting the model costs throughput.
        let single = choose(&s, 8, Objective::SingleStream);
        assert_eq!(single.plan.strategy, Strategy::Data);
        let agg = choose(&s, 8, Objective::Aggregate);
        assert_eq!(agg.plan.strategy, Strategy::Data);
        assert!(
            single
                .reasons
                .iter()
                .any(|r| r.contains("costs more than it saves on"))
        );
        assert!(single.reasons.iter().any(|r| r.contains("fits one card")));
    }

    #[test]
    fn chooser_picks_tensor_parallel_for_a_32b_single_stream() {
        let hw = HardwareSpec::gaudi2();
        let m = qwen_32b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        let single = choose(&s, 8, Objective::SingleStream);
        assert_eq!(single.plan.strategy, Strategy::Tensor);
        assert!(single.plan.single_stream.tokens_per_s > 3.0 * 38.0);
        // The same model at the aggregate objective wants replicas instead.
        let agg = choose(&s, 8, Objective::Aggregate);
        assert_eq!(agg.plan.strategy, Strategy::Data);
        assert!(
            single
                .reasons
                .iter()
                .any(|r| r.contains("collective floor"))
        );
        // Two cards is where the split plainly pays: 199 us of weight time a
        // layer against 38 us of collective. By eight cards the two are level
        // (49.8 us against 49.8), and it is the seven-eighths of the weights
        // already saved that keeps tensor parallel ahead, not the last split.
        assert!(
            choose(&s, 2, Objective::SingleStream)
                .reasons
                .iter()
                .any(|r| r.contains("pays for itself"))
        );
    }

    #[test]
    fn chooser_falls_back_when_the_model_does_not_fit_one_card() {
        let hw = HardwareSpec::gaudi2();
        let m = llama_70b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        let single = choose(&s, 8, Objective::SingleStream);
        assert_eq!(single.plan.strategy, Strategy::Tensor);
        assert!(single.ranked.iter().all(|p| p.strategy != Strategy::Data));
        // With eight micro-batches in flight the pipeline moves the most
        // tokens: it reads the same bytes as tensor parallel without the two
        // all-reduces per layer.
        let agg = choose(&s, 8, Objective::Aggregate);
        assert_eq!(agg.plan.strategy, Strategy::Pipeline);
        assert!(
            single
                .reasons
                .iter()
                .any(|r| r.contains("does not fit one card"))
        );
    }

    #[test]
    fn every_strategy_at_every_admissible_world() {
        let hw = HardwareSpec::gaudi2();
        let m = qwen_32b();
        let floor = CollectiveFloor::measured();
        let s = Scenario::new(&hw, &m, 8, 192, &floor);
        for n in [1u32, 2, 4, 8] {
            for p in plans(&s, n) {
                assert_eq!(p.cards, n, "{} at {n}", p.strategy);
                assert!(p.single_stream.tokens_per_s.is_finite());
                assert!(p.aggregate.tokens_per_s.is_finite());
                assert!(p.single_stream.practical_s >= p.single_stream.physical_s);
                assert!(p.aggregate.tokens_per_s >= p.single_stream.tokens_per_s);
            }
            // More cards never lower the tensor-parallel physical ceiling.
            let tp = tensor_parallel(&s, n);
            let tp1 = tensor_parallel(&s, 1);
            assert!(tp.single_stream.physical_s <= tp1.single_stream.physical_s + 1e-12);
        }
        // Only 1, 2, 4 and 8 are wired into the hybrids at eight cards.
        let names: Vec<String> = plans(&s, 8)
            .iter()
            .map(|p| p.strategy.to_string())
            .collect();
        assert!(names.contains(&"dp2 x tp4".to_string()));
        assert!(names.contains(&"dp4 x tp2".to_string()));
    }
}
