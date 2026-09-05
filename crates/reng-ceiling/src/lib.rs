//! First-principles performance ceilings for LLM inference on Gaudi2.
//!
//! A roofline model. Prefill is compute-bound: matmul FLOPs measured against
//! the MME peak. Decode is memory-bound: weight and KV-cache bytes measured
//! against HBM bandwidth. The hardware constants are Gaudi2 (HL-225) figures
//! from the Intel whitepaper and HL-225B datasheet; the project wiki carries
//! the citations. These are hard ceilings (100% utilization), not predictions.

use reng_core::{Error, Result};

/// Peak throughput and bandwidth figures for one accelerator.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Human-readable name.
    pub name: &'static str,
    /// Peak MME throughput in FLOP/s, by compute dtype.
    pub flops_bf16: f64,
    pub flops_fp16: f64,
    pub flops_fp8: f64,
    pub flops_fp32: f64,
    /// HBM capacity in bytes.
    pub hbm_bytes: u64,
    /// HBM peak bandwidth in bytes/s.
    pub hbm_bw: f64,
    /// On-die SRAM in bytes.
    pub sram_bytes: u64,
    /// PCIe host-link bandwidth per direction, in bytes/s.
    pub pcie_bw: f64,
}

impl HardwareSpec {
    /// Intel Gaudi2 (HL-225), one card. Figures from the Gaudi2 whitepaper and
    /// the HL-225B datasheet (see wiki pages Specifications and Memory
    /// Hierarchy). FP32 MME throughput is not firmly published (sources give
    /// 27-45 TFLOPS); the upper bound is used.
    #[must_use]
    pub fn gaudi2() -> Self {
        Self {
            name: "Gaudi2 (HL-225)",
            flops_bf16: 432.0e12,
            flops_fp16: 432.0e12,
            flops_fp8: 865.0e12,
            flops_fp32: 45.0e12,
            hbm_bytes: 98_304 * 1024 * 1024, // 96 GiB, matches hl-smi (98304 MiB)
            hbm_bw: 2.45e12,                 // 2.45 TB/s
            sram_bytes: 48 * 1024 * 1024,    // 48 MB
            pcie_bw: 32.0e9,                 // PCIe Gen4 x16, ~32 GB/s per direction
        }
    }
}

/// Storage and compute format of the model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Bf16,
    Fp16,
    Fp8,
    Fp32,
    Int8,
    Int4,
}

impl Precision {
    /// Bytes per weight element.
    #[must_use]
    pub fn weight_bytes(self) -> f64 {
        match self {
            Precision::Fp32 => 4.0,
            Precision::Bf16 | Precision::Fp16 => 2.0,
            Precision::Fp8 | Precision::Int8 => 1.0,
            Precision::Int4 => 0.5,
        }
    }

    /// Peak MME compute rate for this format's matmuls. Integer weight-only
    /// formats (Int8, Int4) dequantize to bf16 for the MME, so they take the
    /// bf16 rate; only FP8 uses the FP8 compute path.
    #[must_use]
    pub fn compute_flops(self, hw: &HardwareSpec) -> f64 {
        match self {
            Precision::Fp8 => hw.flops_fp8,
            Precision::Fp32 => hw.flops_fp32,
            Precision::Fp16 => hw.flops_fp16,
            Precision::Bf16 | Precision::Int8 | Precision::Int4 => hw.flops_bf16,
        }
    }

    /// Parse a precision name such as `bf16`, `fp8`, or `int4`.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is not recognized.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Precision::Bf16,
            "fp16" | "float16" | "half" => Precision::Fp16,
            "fp8" | "float8" => Precision::Fp8,
            "fp32" | "float32" => Precision::Fp32,
            "int8" | "i8" => Precision::Int8,
            "int4" | "i4" => Precision::Int4,
            other => return Err(Error::Other(format!("unknown precision {other:?}"))),
        })
    }
}

/// Mixture-of-experts parameters.
#[derive(Debug, Clone)]
pub struct Moe {
    /// Total routed experts.
    pub n_experts: u32,
    /// Experts activated per token.
    pub top_k: u32,
    /// Intermediate size of one expert.
    pub expert_ff: u64,
    /// Always-on shared experts.
    pub shared_experts: u32,
}

/// Transformer shape needed for the roofline.
#[derive(Debug, Clone)]
pub struct ModelShape {
    pub name: String,
    pub layers: u32,
    pub hidden: u64,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u64,
    /// Dense MLP intermediate size (also the fallback for expert size).
    pub ff: u64,
    pub vocab: u64,
    pub tied_embeddings: bool,
    pub moe: Option<Moe>,
}

impl ModelShape {
    fn q_dim(&self) -> u64 {
        u64::from(self.n_heads) * self.head_dim
    }
    fn kv_dim(&self) -> u64 {
        u64::from(self.n_kv_heads) * self.head_dim
    }

    /// Attention projection params per layer (Q, K, V, O; biases ignored).
    fn attn_params_per_layer(&self) -> u64 {
        self.hidden * self.q_dim()          // Q
            + 2 * self.hidden * self.kv_dim() // K, V
            + self.q_dim() * self.hidden // O
    }

    /// MLP params activated per token, per layer (SwiGLU: gate, up, down).
    fn active_mlp_params_per_layer(&self) -> u64 {
        match &self.moe {
            None => 3 * self.hidden * self.ff,
            Some(m) => {
                let router = self.hidden * u64::from(m.n_experts);
                let active = u64::from(m.top_k + m.shared_experts) * 3 * self.hidden * m.expert_ff;
                router + active
            }
        }
    }

    /// MLP params resident in memory (all experts), per layer.
    fn total_mlp_params_per_layer(&self) -> u64 {
        match &self.moe {
            None => 3 * self.hidden * self.ff,
            Some(m) => {
                let router = self.hidden * u64::from(m.n_experts);
                let all = u64::from(m.n_experts + m.shared_experts) * 3 * self.hidden * m.expert_ff;
                router + all
            }
        }
    }

    fn lm_head_params(&self) -> u64 {
        self.hidden * self.vocab
    }

    /// Params touched to produce one token: attention and the active MLP for
    /// every layer, plus the LM head. Excludes the embedding table (a lookup,
    /// not a matmul).
    #[must_use]
    pub fn active_params(&self) -> u64 {
        u64::from(self.layers) * (self.attn_params_per_layer() + self.active_mlp_params_per_layer())
            + self.lm_head_params()
    }

    /// Total resident params (all experts, embedding, and LM head).
    #[must_use]
    pub fn total_params(&self) -> u64 {
        let body = u64::from(self.layers)
            * (self.attn_params_per_layer() + self.total_mlp_params_per_layer());
        let embedding = self.hidden * self.vocab;
        let head = if self.tied_embeddings {
            embedding
        } else {
            embedding + self.lm_head_params()
        };
        body + head
    }

    /// KV-cache bytes for one token of context (K and V, all layers).
    #[must_use]
    pub fn kv_bytes_per_token(&self, kv: Precision) -> f64 {
        2.0 * self.kv_dim() as f64 * f64::from(self.layers) * kv.weight_bytes()
    }
}

/// Which resource sets the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// MME compute throughput.
    Compute,
    /// HBM bandwidth.
    HbmBandwidth,
}

/// A computed ceiling for one scenario.
#[derive(Debug, Clone)]
pub struct Ceiling {
    pub latency_s: f64,
    pub tokens_per_s: f64,
    pub bottleneck: Bottleneck,
    pub compute_time_s: f64,
    pub memory_time_s: f64,
    pub arithmetic_intensity: f64,
}

fn ceiling_from(flops: f64, bytes: f64, peak_flops: f64, hbm_bw: f64, tokens: f64) -> Ceiling {
    let compute_time = flops / peak_flops;
    let memory_time = bytes / hbm_bw;
    let latency = compute_time.max(memory_time);
    let bottleneck = if compute_time >= memory_time {
        Bottleneck::Compute
    } else {
        Bottleneck::HbmBandwidth
    };
    Ceiling {
        latency_s: latency,
        tokens_per_s: tokens / latency,
        bottleneck,
        compute_time_s: compute_time,
        memory_time_s: memory_time,
        arithmetic_intensity: flops / bytes,
    }
}

/// Prefill ceiling: process `seq_len` prompt tokens for `batch` sequences at a
/// 0% cache hit.
#[must_use]
pub fn prefill_ceiling(
    hw: &HardwareSpec,
    model: &ModelShape,
    prec: Precision,
    kv: Precision,
    batch: u32,
    seq_len: u32,
) -> Ceiling {
    let b = f64::from(batch);
    let s = f64::from(seq_len);
    let active = model.active_params() as f64;
    let linear_flops = 2.0 * active * b * s;
    let attn_flops = 4.0
        * f64::from(model.layers)
        * b
        * f64::from(model.n_heads)
        * s
        * s
        * model.head_dim as f64;
    let flops = linear_flops + attn_flops;
    // Weights read once; KV cache written for the whole prompt.
    let bytes = active * prec.weight_bytes() + b * s * model.kv_bytes_per_token(kv);
    ceiling_from(flops, bytes, prec.compute_flops(hw), hw.hbm_bw, b * s)
}

/// Decode ceiling: generate one token for `batch` sequences each at context
/// length `ctx_len`.
#[must_use]
pub fn decode_ceiling(
    hw: &HardwareSpec,
    model: &ModelShape,
    prec: Precision,
    kv: Precision,
    batch: u32,
    ctx_len: u32,
) -> Ceiling {
    let b = f64::from(batch);
    let ctx = f64::from(ctx_len);
    let active = model.active_params() as f64;
    let attn_flops =
        4.0 * f64::from(model.layers) * b * f64::from(model.n_heads) * ctx * model.head_dim as f64;
    let flops = 2.0 * active * b + attn_flops;
    // Active weights read once; the full KV cache read every step.
    let bytes = active * prec.weight_bytes() + b * ctx * model.kv_bytes_per_token(kv);
    ceiling_from(flops, bytes, prec.compute_flops(hw), hw.hbm_bw, b)
}

/// VRAM bytes for resident weights plus the KV cache of `batch` x `ctx_len`
/// tokens.
#[must_use]
pub fn vram_bytes(
    model: &ModelShape,
    prec: Precision,
    kv: Precision,
    batch: u32,
    ctx_len: u32,
) -> u64 {
    let weights = model.total_params() as f64 * prec.weight_bytes();
    let kv_cache = f64::from(batch) * f64::from(ctx_len) * model.kv_bytes_per_token(kv);
    (weights + kv_cache) as u64
}

/// Whether the scenario fits in the aggregate HBM of `n_cards`.
#[must_use]
pub fn fits(
    hw: &HardwareSpec,
    model: &ModelShape,
    prec: Precision,
    kv: Precision,
    batch: u32,
    ctx_len: u32,
    n_cards: u32,
) -> bool {
    vram_bytes(model, prec, kv, batch, ctx_len) <= hw.hbm_bytes * u64::from(n_cards)
}

impl HardwareSpec {
    /// Idealized linear scaling to `n` cards in tensor parallel: peak compute,
    /// HBM bandwidth, and HBM capacity all multiply by `n`. This ignores
    /// interconnect and synchronization overhead, so the result is an upper
    /// bound on aggregate throughput.
    #[must_use]
    pub fn scaled(&self, n: u32) -> HardwareSpec {
        let n = f64::from(n.max(1));
        HardwareSpec {
            name: self.name,
            flops_bf16: self.flops_bf16 * n,
            flops_fp16: self.flops_fp16 * n,
            flops_fp8: self.flops_fp8 * n,
            flops_fp32: self.flops_fp32 * n,
            hbm_bytes: (self.hbm_bytes as f64 * n) as u64,
            hbm_bw: self.hbm_bw * n,
            sram_bytes: (self.sram_bytes as f64 * n) as u64,
            pcie_bw: self.pcie_bw,
        }
    }
}

/// One cell of a ceiling grid: a (batch, context) scenario and its ceilings.
#[derive(Debug, Clone)]
pub struct GridCell {
    pub batch: u32,
    pub context: u32,
    pub prefill: Ceiling,
    pub decode: Ceiling,
    pub vram_bytes: u64,
}

/// Compute a context-by-batch grid of ceilings on `n_cards` (tensor-parallel,
/// idealized linear scaling).
///
/// Context runs as a descending power-of-two series from `max_context` down to
/// 256; batch runs 1, 2, 4, ... and stops for a given context once the scenario
/// no longer fits in aggregate HBM. Only fitting cells are returned. Prefill
/// uses a prompt of `context` tokens at a 0% cache hit; decode uses a KV cache
/// of `context` tokens.
#[must_use]
pub fn ceiling_grid(
    hw: &HardwareSpec,
    model: &ModelShape,
    prec: Precision,
    kv: Precision,
    max_context: u32,
    n_cards: u32,
) -> Vec<GridCell> {
    let agg = hw.scaled(n_cards);
    let mut cells = Vec::new();
    let mut ctx = max_context;
    while ctx >= 256 {
        let mut batch = 1u32;
        loop {
            let vram = vram_bytes(model, prec, kv, batch, ctx);
            if vram > agg.hbm_bytes {
                break;
            }
            cells.push(GridCell {
                batch,
                context: ctx,
                prefill: prefill_ceiling(&agg, model, prec, kv, batch, ctx),
                decode: decode_ceiling(&agg, model, prec, kv, batch, ctx),
                vram_bytes: vram,
            });
            if batch >= 1024 {
                break;
            }
            batch *= 2;
        }
        ctx /= 2;
    }
    cells
}

/// Build a [`ModelShape`] from a HuggingFace `config.json` string.
///
/// # Errors
///
/// Returns an error if the JSON cannot be parsed or is missing the core
/// dimensions (`hidden_size`, `num_hidden_layers`, `num_attention_heads`).
pub fn model_from_hf_config(json: &str) -> Result<ModelShape> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Error::Other(format!("config.json parse: {e}")))?;
    let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
    let miss = |k: &str| Error::Other(format!("config.json missing {k}"));

    let hidden = u("hidden_size").ok_or_else(|| miss("hidden_size"))?;
    let layers = u("num_hidden_layers").ok_or_else(|| miss("num_hidden_layers"))? as u32;
    let n_heads = u("num_attention_heads").ok_or_else(|| miss("num_attention_heads"))? as u32;
    let n_kv_heads = u("num_key_value_heads").unwrap_or(u64::from(n_heads)) as u32;
    let head_dim = u("head_dim").unwrap_or(hidden / u64::from(n_heads));
    let vocab = u("vocab_size").unwrap_or(0);
    let ff = u("intermediate_size").unwrap_or(0);
    // Gemma configs omit the key; their HF config classes default it to
    // true (no `lm_head` tensor in the checkpoints).
    let gemma = matches!(
        v.get("model_type").and_then(serde_json::Value::as_str),
        Some("gemma" | "gemma2" | "gemma3_text" | "gemma3")
    );
    let tied = v
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(gemma);
    let name = v
        .get("_name_or_path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("model")
        .to_string();

    let n_experts = u("num_local_experts")
        .or_else(|| u("n_routed_experts"))
        .or_else(|| u("num_experts"));
    let moe = n_experts.map(|ne| {
        let top_k = u("num_experts_per_tok")
            .or_else(|| u("moe_topk"))
            .or_else(|| u("n_activated_experts"))
            .unwrap_or(2) as u32;
        let expert_ff = u("moe_intermediate_size").unwrap_or(ff);
        let shared = u("n_shared_experts").unwrap_or(0) as u32;
        Moe {
            n_experts: ne as u32,
            top_k,
            expert_ff,
            shared_experts: shared,
        }
    });

    Ok(ModelShape {
        name,
        layers,
        hidden,
        n_heads,
        n_kv_heads,
        head_dim,
        ff,
        vocab,
        tied_embeddings: tied,
        moe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_7b() -> ModelShape {
        // Llama-2-7B-ish dimensions.
        ModelShape {
            name: "dense-7b".into(),
            layers: 32,
            hidden: 4096,
            n_heads: 32,
            n_kv_heads: 32,
            head_dim: 128,
            ff: 11008,
            vocab: 32000,
            tied_embeddings: false,
            moe: None,
        }
    }

    #[test]
    fn decode_is_memory_bound_at_batch_one() {
        let hw = HardwareSpec::gaudi2();
        let m = dense_7b();
        let c = decode_ceiling(&hw, &m, Precision::Bf16, Precision::Bf16, 1, 2048);
        assert_eq!(c.bottleneck, Bottleneck::HbmBandwidth);
        assert!(c.tokens_per_s.is_finite() && c.tokens_per_s > 0.0);
        // ~7B params x 2 bytes / 2.45 TB/s is a few ms, so a few hundred tok/s.
        assert!(
            c.tokens_per_s > 50.0 && c.tokens_per_s < 1000.0,
            "tok/s = {}",
            c.tokens_per_s
        );
    }

    #[test]
    fn prefill_is_compute_bound_at_long_context() {
        let hw = HardwareSpec::gaudi2();
        let m = dense_7b();
        let c = prefill_ceiling(&hw, &m, Precision::Bf16, Precision::Bf16, 1, 8192);
        assert_eq!(c.bottleneck, Bottleneck::Compute);
        assert!(c.arithmetic_intensity > 100.0);
    }

    #[test]
    fn int4_speeds_up_decode_but_not_prefill() {
        let hw = HardwareSpec::gaudi2();
        let m = dense_7b();
        let d_bf16 = decode_ceiling(&hw, &m, Precision::Bf16, Precision::Bf16, 1, 512);
        let d_int4 = decode_ceiling(&hw, &m, Precision::Int4, Precision::Bf16, 1, 512);
        // Weight-only int4 quarters the weight bytes, so decode is faster.
        assert!(d_int4.tokens_per_s > d_bf16.tokens_per_s * 2.0);
        // Prefill compute is still bf16-rate, so no speedup there.
        let p_bf16 = prefill_ceiling(&hw, &m, Precision::Bf16, Precision::Bf16, 1, 8192);
        let p_int4 = prefill_ceiling(&hw, &m, Precision::Int4, Precision::Bf16, 1, 8192);
        assert!((p_int4.latency_s - p_bf16.latency_s).abs() / p_bf16.latency_s < 0.05);
    }

    #[test]
    fn moe_active_params_below_total() {
        let m = ModelShape {
            name: "moe".into(),
            layers: 24,
            hidden: 2048,
            n_heads: 16,
            n_kv_heads: 16,
            head_dim: 128,
            ff: 0,
            vocab: 32000,
            tied_embeddings: false,
            moe: Some(Moe {
                n_experts: 64,
                top_k: 6,
                expert_ff: 1408,
                shared_experts: 2,
            }),
        };
        assert!(m.active_params() < m.total_params());
    }

    #[test]
    fn parses_hf_config() {
        let json = r#"{
            "_name_or_path": "test/model",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000
        }"#;
        let m = model_from_hf_config(json).unwrap();
        assert_eq!(m.layers, 32);
        assert_eq!(m.hidden, 4096);
        assert_eq!(m.n_kv_heads, 8);
        assert_eq!(m.head_dim, 128); // derived hidden/heads
        assert!(m.moe.is_none());
    }

    #[test]
    fn parses_moe_config() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408,
            "n_shared_experts": 2,
            "vocab_size": 32000
        }"#;
        let m = model_from_hf_config(json).unwrap();
        let moe = m.moe.expect("moe detected");
        assert_eq!(moe.n_experts, 64);
        assert_eq!(moe.top_k, 6);
        assert_eq!(moe.shared_experts, 2);
    }

    #[test]
    fn grid_fits_and_scales() {
        let hw = HardwareSpec::gaudi2();
        let m = dense_7b();
        let cells = ceiling_grid(&hw, &m, Precision::Bf16, Precision::Bf16, 8192, 1);
        assert!(!cells.is_empty());
        // Every returned cell fits in one card's HBM.
        assert!(cells.iter().all(|c| c.vram_bytes <= hw.hbm_bytes));
        // Eight cards admit at least as many fitting cells as one.
        let cells8 = ceiling_grid(&hw, &m, Precision::Bf16, Precision::Bf16, 8192, 8);
        assert!(cells8.len() >= cells.len());
        // Aggregate decode ceiling on 8 cards beats a single card.
        let d1 = decode_ceiling(&hw, &m, Precision::Bf16, Precision::Bf16, 1, 2048);
        let d8 = decode_ceiling(&hw.scaled(8), &m, Precision::Bf16, Precision::Bf16, 1, 2048);
        assert!(d8.tokens_per_s > d1.tokens_per_s * 5.0);
    }
}
