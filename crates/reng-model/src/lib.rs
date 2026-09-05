//! Load a Llama-family model from a HuggingFace directory (`config.json` +
//! `model.safetensors`, single file or shards) and run it on Gaudi2 through the fused-graph engine
//! in `reng-synapse`: one-shot prefill, or KV-cached generation through
//! [`Generator`].
//!
//! Layout conventions: HF stores every `nn.Linear` weight as `[out, in]` and
//! computes `y = x @ W^T`; the engine wants `[in, out]`, so projections are
//! transposed once at load time. The embedding table stays `[vocab, hidden]`
//! for the host-side gather; a tied LM head is its transpose, `[hidden, vocab]`.
//! Qwen2-style attention biases and Qwen3-style per-head q/k norm gains are
//! loaded when present (Llama has neither). Granite's four scalar
//! multipliers need no graph change: the embedding multiplier is applied
//! by the host gather, the attention multiplier is the per-layer attention
//! scale, and the residual multiplier and `1/logits_scaling` are folded
//! into `o_proj`/`down_proj` and the LM head at load.
//! loaded when present (Llama has neither). OLMo-2 (`model_type: olmo2`)
//! keeps its two layer norms on the branch outputs (`post_attention_layernorm`
//! and `post_feedforward_layernorm`; no `input_layernorm`) and its q/k norms
//! span the whole projection; they map onto `g1`/`g2` with `post_norm` set
//! and the full-width `qn`/`kn` form of `LayerWeights`.
//!
//! Gemma (`gemma2`, `gemma3_text`): every RMSNorm gain is stored
//! zero-centred and applied as `1 + w`, so the loader adds 1 to each; the
//! two extra norms per layer (on the attention and MLP branch outputs,
//! on top of the pre-norms) and the GELU-tanh activation are layer
//! flags; the embeddings are scaled by `sqrt(hidden)` on the host; the
//! sliding layers get the sliding window and, for Gemma-3, the local RoPE
//! table (`rope_local_base_freq`) while the full layers get `rope_theta`;
//! and `tie_word_embeddings` defaults to true (the key is absent from the
//! configs).

use reng_core::{Error, Result};
use reng_synapse::{Activation, bf16_to_f32, f32_to_bf16, scale_bf16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::path::Path;

fn default_theta() -> f32 {
    10000.0
}

/// The subset of a HF Llama-style `config.json` the engine needs.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// HF `model_type`; `olmo2` selects the post-norm layer layout (see
    /// [`LlamaConfig::post_norm`]). Absent from some older configs.
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Explicit per-head width (Qwen3); `hidden_size / num_attention_heads`
    /// when absent. `num_attention_heads * head_dim` may differ from
    /// `hidden_size` (Qwen3-0.6B: 16 x 128 over a hidden size of 1024).
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    /// HF `rope_scaling`; only the `llama3` type is applied (the others are
    /// reported as unsupported by [`LlamaConfig::load`]).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub vocab_size: usize,
    /// Absent in every Gemma config, whose HF config classes default it to
    /// true; see [`LlamaConfig::tied`].
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// SmolLM3: one entry per layer, 1 where the layer uses RoPE and 0 for
    /// a NoPE layer; absent means every layer uses RoPE.
    #[serde(default)]
    pub no_rope_layers: Option<Vec<u8>>,
    /// Sliding window in positions (Phi-3, Mistral; Qwen2 only with
    /// `use_sliding_window`).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// Granite: the embedding rows are multiplied by this before the first
    /// layer (`GraniteModel.forward`); absent means unscaled.
    #[serde(default)]
    pub embedding_multiplier: Option<f32>,
    /// Granite: the attention scale, in place of `1/sqrt(head_dim)`
    /// (`GraniteAttention.scaling`).
    #[serde(default)]
    pub attention_multiplier: Option<f32>,
    /// Granite: every residual branch (attention and MLP) is multiplied by
    /// this before the add (`GraniteDecoderLayer.forward`); folded into
    /// `o_proj` and `down_proj` by [`load_weights`].
    #[serde(default)]
    pub residual_multiplier: Option<f32>,
    /// Granite: the logits are divided by this
    /// (`GraniteForCausalLM.forward`); folded into the LM head by
    /// [`load_weights`], which leaves tied embeddings unscaled.
    #[serde(default)]
    pub logits_scaling: Option<f32>,
    /// Whether the MLP projections carry biases; the engine has none, so
    /// [`load_weights`] refuses such a checkpoint.
    #[serde(default)]
    pub mlp_bias: bool,
    /// The MLP gate activation, `hidden_act` (Llama) or `hidden_activation`
    /// (Gemma): `silu` (the default) or `gelu_pytorch_tanh`.
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    /// Gemma: the attention scale is `query_pre_attn_scalar ** -0.5` in
    /// place of `1/sqrt(head_dim)`.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Gemma-3: RoPE theta of the sliding layers (the full layers use
    /// `rope_theta`); 1e4 when absent.
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    /// Gemma-3: every layer whose `(index + 1)` is a multiple of this is a
    /// full-attention layer, the others slide; 6 when absent.
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// Per-layer `"sliding_attention"` / `"full_attention"`; takes
    /// precedence over the pattern (Gemma only).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Gemma-2: `tanh(logits / cap) * cap` on the final logits.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    /// Gemma-2: the same on the attention scores before the mask (Gemma-3
    /// configs carry `null`, and `Gemma3Attention` ignores the value).
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
}

/// The `rope_scaling` object of a HF config. Llama 3.1 style scaling
/// rescales the low-frequency rotary dims (see [`rope_caches_scaled`]).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// `rope_type` (newer configs) or `type` (older ones).
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

/// RoPE caches of a model for `positions` positions: the tables every
/// layer reads and, for Gemma-3, the local ones its sliding layers read
/// (empty otherwise).
pub struct RopeCaches {
    pub sin: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin_local: Vec<f32>,
    pub cos_local: Vec<f32>,
}

impl RopeCaches {
    /// Borrow as the engine's table set.
    #[cfg(feature = "link-synapse")]
    fn tables(&self) -> reng_synapse::RopeTables<'_> {
        reng_synapse::RopeTables {
            sin: &self.sin,
            cos: &self.cos,
            sin_local: &self.sin_local,
            cos_local: &self.cos_local,
        }
    }
}

impl LlamaConfig {
    /// The sliding window attention uses model-wide, if the architecture
    /// applies one (a query sees the last `window` positions, its own
    /// included). Gemma applies it per layer: see
    /// [`LlamaConfig::layer_window`]. Diagnostic `RENG_NO_WINDOW` disables
    /// it.
    #[must_use]
    pub fn window(&self) -> Option<usize> {
        if std::env::var("RENG_NO_WINDOW").is_ok() {
            return None;
        }
        let w = self.sliding_window.filter(|&w| w > 0);
        match self.model_type.as_deref() {
            Some("phi3" | "mistral") => w,
            Some("qwen2" | "qwen3") if self.use_sliding_window == Some(true) => w,
            _ => None,
        }
    }

    /// Whether this is a Gemma text model (`gemma2` or `gemma3_text`; the
    /// multimodal `gemma3` nests its text config and is not loaded).
    #[must_use]
    pub fn is_gemma(&self) -> bool {
        matches!(self.model_type.as_deref(), Some("gemma2" | "gemma3_text"))
    }

    /// Whether the LM head is the embedding table: the config's value, or
    /// (with the key absent) true for Gemma and false otherwise.
    #[must_use]
    pub fn tied(&self) -> bool {
        self.tie_word_embeddings.unwrap_or_else(|| self.is_gemma())
    }

    /// Factor on the gathered embedding rows: Gemma's `sqrt(hidden_size)`
    /// (the f32 value, as the f32 oracle uses) times Granite's
    /// `embedding_multiplier`; 1 for everything else.
    #[must_use]
    pub fn embed_scale(&self) -> f32 {
        let gemma = if self.is_gemma() {
            (self.hidden_size as f32).sqrt()
        } else {
            1.0
        };
        gemma * self.embedding_multiplier.unwrap_or(1.0)
    }

    /// The MLP gate activation.
    ///
    /// # Errors
    ///
    /// Returns an error for an activation the engine has no node for.
    pub fn activation(&self) -> Result<Activation> {
        let name = self
            .hidden_activation
            .as_deref()
            .or(self.hidden_act.as_deref());
        match name {
            None | Some("silu") => Ok(Activation::Silu),
            Some("gelu_pytorch_tanh") => Ok(Activation::GeluTanh),
            Some(other) => Err(Error::Other(format!(
                "config.json: activation {other} is not supported"
            ))),
        }
    }

    /// Whether layer `li` is a sliding-attention layer of a Gemma model:
    /// the config's `layer_types` entry when present, else every layer
    /// but each `sliding_window_pattern`-th (6 for Gemma-3) or each second
    /// (Gemma-2). False for every other model type.
    #[must_use]
    pub fn sliding(&self, li: usize) -> bool {
        if !self.is_gemma() {
            return false;
        }
        if let Some(types) = &self.layer_types {
            return types.get(li).is_some_and(|t| t == "sliding_attention");
        }
        let pattern = match self.model_type.as_deref() {
            Some("gemma2") => 2,
            _ => self.sliding_window_pattern.unwrap_or(6),
        };
        (li + 1) % pattern != 0
    }

    /// The sliding window of layer `li`: for Gemma the config's
    /// `sliding_window` on its sliding layers and none on its full layers,
    /// otherwise the model-wide [`LlamaConfig::window`].
    #[must_use]
    pub fn layer_window(&self, li: usize) -> Option<usize> {
        if !self.is_gemma() {
            return self.window();
        }
        if std::env::var("RENG_NO_WINDOW").is_ok() || !self.sliding(li) {
            return None;
        }
        self.sliding_window.filter(|&w| w > 0)
    }

    /// The attention softcap: Gemma-2's `attn_logit_softcapping`
    /// (`Gemma2Attention` passes it to the attention; `Gemma3Attention`
    /// does not, whatever the config says).
    #[must_use]
    pub fn attn_softcap(&self) -> Option<f32> {
        if self.model_type.as_deref() == Some("gemma2") {
            self.attn_logit_softcapping.filter(|&c| c > 0.0)
        } else {
            None
        }
    }

    /// The final logit softcap, `final_logit_softcapping` when set.
    #[must_use]
    pub fn final_softcap(&self) -> Option<f32> {
        self.final_logit_softcapping.filter(|&c| c > 0.0)
    }

    /// Whether layer `li` reads the local RoPE table: Gemma-3's sliding
    /// layers (Gemma-2 has one theta for every layer).
    #[must_use]
    pub fn local_rope(&self, li: usize) -> bool {
        self.model_type.as_deref() == Some("gemma3_text") && self.sliding(li)
    }

    /// RoPE caches for `positions` positions: the global table from
    /// `rope_theta` (with `rope_scaling`, which applies to the full layers
    /// only) and, for Gemma-3, the local one from `rope_local_base_freq`.
    #[must_use]
    pub fn rope_caches(&self, positions: usize) -> RopeCaches {
        let hd = self.head_dim();
        let (sin, cos) =
            rope_caches_scaled(positions, hd, self.rope_theta, self.rope_scaling.as_ref());
        let (sin_local, cos_local) = if (0..self.num_hidden_layers).any(|li| self.local_rope(li)) {
            rope_caches_scaled(
                positions,
                hd,
                self.rope_local_base_freq.unwrap_or(1e4),
                None,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        RopeCaches {
            sin,
            cos,
            sin_local,
            cos_local,
        }
    }

    /// Read `config.json` from a model directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Error::Other(format!("config.json: {e}")))?;
        let cfg: Self =
            serde_json::from_str(&text).map_err(|e| Error::Other(format!("config.json: {e}")))?;
        if let Some(t) = cfg
            .rope_scaling
            .as_ref()
            .and_then(|s| s.rope_type.as_deref())
        {
            if t != "llama3" && t != "default" {
                eprintln!("config.json: rope_scaling type {t} is not applied");
            }
        }
        cfg.activation()?;
        Ok(cfg)
    }

    #[must_use]
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Width of the query projection, `num_attention_heads * head_dim`.
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    /// The softmax scale on `q . k`: `attention_multiplier` when the config
    /// has one (Granite), else `query_pre_attn_scalar ** -0.5` when it has
    /// that (Gemma), else `1/sqrt(head_dim)`.
    #[must_use]
    pub fn attention_scale(&self) -> f32 {
        self.attention_multiplier
            .or_else(|| self.query_pre_attn_scalar.map(|s| 1.0 / s.sqrt()))
            .unwrap_or_else(|| 1.0 / (self.head_dim() as f32).sqrt())
    }

    /// Whether the layers normalise their branch outputs instead of their
    /// inputs (OLMo-2: `h = x + norm(attn(x))`, `y = h + norm(mlp(h))`).
    #[must_use]
    pub fn post_norm(&self) -> bool {
        self.model_type.as_deref() == Some("olmo2")
    }
}

/// One layer's weights in engine layout (`[in, out]` projections), owned.
pub struct LayerTensors {
    /// The layer's two RMSNorm gains: `input_layernorm` and
    /// `post_attention_layernorm` (pre-norm), `post_attention_layernorm`
    /// and `post_feedforward_layernorm` (OLMo-2 post-norm), or `1 + w` of
    /// `input_layernorm` and `pre_feedforward_layernorm` (Gemma).
    pub g1: Vec<f32>,
    pub g2: Vec<f32>,
    /// Gemma's post norms on the attention and MLP branch outputs (`1 + w`
    /// of `post_attention_layernorm` and `post_feedforward_layernorm`);
    /// empty for every other model.
    pub g_post_attn: Vec<f32>,
    pub g_post_mlp: Vec<f32>,
    /// Projections as stored, bf16 `[out, in]`; `wo` (and `wd`) carry the
    /// config's `residual_multiplier` when it has one.
    pub wq: Vec<u16>,
    pub wk: Vec<u16>,
    pub wv: Vec<u16>,
    pub wo: Vec<u16>,
    /// Attention biases; empty when the checkpoint has none.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    /// q/k norm gains: each `head_dim` (Qwen3, per head) or the full
    /// projection widths (OLMo-2: `n_heads * head_dim` and `n_kv_heads *
    /// head_dim`); empty when the checkpoint has none.
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    pub wg: Vec<u16>,
    pub wu: Vec<u16>,
    pub wd: Vec<u16>,
}

/// A whole model's weights on the host: the matrices bf16 in the
/// checkpoint's `[out, in]` layout (the device format, uploaded as is),
/// the norm gains and biases f32.
pub struct LlamaWeights {
    /// `[vocab, hidden]`, bf16, row per token id (never scaled).
    pub embed: Vec<u16>,
    pub layers: Vec<LayerTensors>,
    pub final_gamma: Vec<f32>,
    /// `[vocab, hidden]`, bf16 (a copy of the tied embeddings when the
    /// checkpoint has no head), divided by the config's `logits_scaling`
    /// when it has one.
    pub lm_head: Vec<u16>,
}

/// A tensor as bf16 with its shape, converted from f32 or f16 checkpoints
/// and copied verbatim from bf16 ones.
fn tensor_bf16(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<u16>, Vec<usize>)> {
    let view = st
        .tensor(name)
        .map_err(|e| Error::Other(format!("tensor {name}: {e}")))?;
    let data = view.data();
    let out = match view.dtype() {
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect(),
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f32_to_bf16(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
            .collect(),
        other => {
            return Err(Error::Other(format!(
                "tensor {name}: unsupported dtype {other:?}"
            )));
        }
    };
    Ok((out, view.shape().to_vec()))
}

fn tensor_f32(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = st
        .tensor(name)
        .map_err(|e| Error::Other(format!("tensor {name}: {e}")))?;
    let data = view.data();
    let out = match view.dtype() {
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => {
            return Err(Error::Other(format!(
                "tensor {name}: unsupported dtype {other:?}"
            )));
        }
    };
    Ok((out, view.shape().to_vec()))
}

/// IEEE half to f32 (for models stored in f16).
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// A 1-D tensor that a checkpoint may omit (attention biases, q/k norm
/// gains): empty when absent, checked against the accepted lengths when
/// present.
fn optional_vec(st: &SafeTensors<'_>, name: &str, lens: &[usize]) -> Result<Vec<f32>> {
    if st.tensor(name).is_err() {
        return Ok(Vec::new());
    }
    let (v, shape) = tensor_f32(st, name)?;
    if shape.len() != 1 || !lens.contains(&shape[0]) {
        return Err(Error::Other(format!(
            "tensor {name}: shape {shape:?}, expected one of {lens:?}"
        )));
    }
    Ok(v)
}

/// A `[out, in]` HF linear weight, shape-checked and kept in that layout.
fn linear(st: &SafeTensors<'_>, name: &str, out_dim: usize, in_dim: usize) -> Result<Vec<u16>> {
    let (v, shape) = tensor_bf16(st, name)?;
    if shape != [out_dim, in_dim] {
        return Err(Error::Other(format!(
            "tensor {name}: shape {shape:?}, expected [{out_dim}, {in_dim}]"
        )));
    }
    Ok(v)
}

/// The checkpoint's safetensors files, one or several (sharded checkpoints
/// list their tensors in `model.safetensors.index.json`), all loaded.
struct Shards {
    files: Vec<Vec<u8>>,
    /// Tensor name to shard index; empty for a single-file checkpoint.
    index: std::collections::HashMap<String, usize>,
}

impl Shards {
    fn open(dir: &Path) -> Result<Self> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            let bytes = std::fs::read(&single)
                .map_err(|e| Error::Other(format!("model.safetensors: {e}")))?;
            return Ok(Self {
                files: vec![bytes],
                index: std::collections::HashMap::new(),
            });
        }
        let idx_path = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&idx_path)
            .map_err(|e| Error::Other(format!("neither model.safetensors nor its index: {e}")))?;
        let idx: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Other(format!("safetensors index: {e}")))?;
        let map = idx
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| Error::Other("safetensors index has no weight_map".into()))?;
        let mut names: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        names.sort();
        names.dedup();
        let mut files = Vec::with_capacity(names.len());
        for name in &names {
            files.push(
                std::fs::read(dir.join(name)).map_err(|e| Error::Other(format!("{name}: {e}")))?,
            );
        }
        let index = map
            .iter()
            .filter_map(|(k, v)| {
                let file = v.as_str()?;
                Some((k.clone(), names.iter().position(|n| n == file)?))
            })
            .collect();
        Ok(Self { files, index })
    }

    /// Parse every shard (borrowing the bytes).
    fn parse(&self) -> Result<Vec<SafeTensors<'_>>> {
        self.files
            .iter()
            .map(|b| {
                SafeTensors::deserialize(b).map_err(|e| Error::Other(format!("safetensors: {e}")))
            })
            .collect()
    }

    /// The parsed shard holding `name` (the only shard when unsharded).
    fn shard<'a>(&self, parsed: &'a [SafeTensors<'a>], name: &str) -> &'a SafeTensors<'a> {
        let i = self.index.get(name).copied().unwrap_or(0);
        &parsed[i]
    }
}

/// Load the checkpoint (`model.safetensors`, or its shards) from a model
/// directory into engine layout. Granite's `residual_multiplier` is folded
/// into `o_proj` and `down_proj` and `1/logits_scaling` into the LM head
/// (scaled bf16 copies: one extra rounding per element, none for
/// power-of-two scales). Gemma's norm gains get their `1 + w` offset here
/// (exact in f32).
///
/// # Errors
///
/// Returns an error if the files are missing, a tensor is absent or has an
/// unexpected shape or dtype, the LM head is untied and absent, or the
/// config asks for MLP biases.
pub fn load_weights(dir: &Path, cfg: &LlamaConfig) -> Result<LlamaWeights> {
    if cfg.mlp_bias {
        return Err(Error::Other("mlp_bias is not supported".into()));
    }
    let shards = Shards::open(dir)?;
    let parsed = shards.parse()?;
    let st = |name: &str| shards.shard(&parsed, name);
    let (h, i, v) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size);
    let (hd, qd) = (cfg.head_dim(), cfg.q_dim());
    let kvd = cfg.n_kv_heads() * hd;

    let (embed, eshape) =
        tensor_bf16(st("model.embed_tokens.weight"), "model.embed_tokens.weight")?;
    if eshape != [v, h] {
        return Err(Error::Other(format!(
            "embed_tokens shape {eshape:?}, expected [{v}, {h}]"
        )));
    }
    let post_norm = cfg.post_norm();
    let gemma = cfg.is_gemma();
    // Gemma applies every RMSNorm gain as `1 + w`.
    let plus_one = |mut v: Vec<f32>| -> Vec<f32> {
        if gemma {
            for x in &mut v {
                *x += 1.0;
            }
        }
        v
    };
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        let g = |name: &str| -> Result<Vec<f32>> {
            let n = p(name);
            Ok(plus_one(tensor_f32(st(&n), &n)?.0))
        };
        let lin = |name: &str, o: usize, inp: usize| -> Result<Vec<u16>> {
            let n = p(name);
            linear(st(&n), &n, o, inp)
        };
        // Phi-3 stores q/k/v as one [q + k + v, hidden] matrix and gate/up
        // as one [2 * inter, hidden] matrix; in the [out, in] layout the
        // parts are contiguous row blocks.
        let has = |name: &str| -> bool {
            let n = p(name);
            shards.index.contains_key(&n)
                || (shards.index.is_empty() && parsed[0].tensor(&n).is_ok())
        };
        let split = |name: &str, rows: &[usize], inp: usize| -> Result<Vec<Vec<u16>>> {
            let n = p(name);
            let total: usize = rows.iter().sum();
            let v = linear(st(&n), &n, total, inp)?;
            let mut out = Vec::with_capacity(rows.len());
            let mut at = 0;
            for &r in rows {
                out.push(v[at * inp..(at + r) * inp].to_vec());
                at += r;
            }
            Ok(out)
        };
        let (wq, wk, wv) = if has("self_attn.qkv_proj.weight") {
            let mut parts = split("self_attn.qkv_proj.weight", &[qd, kvd, kvd], h)?;
            let wv = parts.pop().unwrap();
            let wk = parts.pop().unwrap();
            let wq = parts.pop().unwrap();
            (wq, wk, wv)
        } else {
            (
                lin("self_attn.q_proj.weight", qd, h)?,
                lin("self_attn.k_proj.weight", kvd, h)?,
                lin("self_attn.v_proj.weight", kvd, h)?,
            )
        };
        let (wg, wu) = if has("mlp.gate_up_proj.weight") {
            let mut parts = split("mlp.gate_up_proj.weight", &[i, i], h)?;
            let wu = parts.pop().unwrap();
            let wg = parts.pop().unwrap();
            (wg, wu)
        } else {
            (
                lin("mlp.gate_proj.weight", i, h)?,
                lin("mlp.up_proj.weight", i, h)?,
            )
        };
        let opt = |name: &str, lens: &[usize]| -> Result<Vec<f32>> {
            let n = p(name);
            optional_vec(st(&n), &n, lens)
        };
        // The norm gains by layout. Gemma has all four: the pre-norms under
        // `input_layernorm` and `pre_feedforward_layernorm`, and the post
        // norms on the branch outputs (its `post_attention_layernorm` is
        // the attention branch's post norm). OLMo-2 has no input norm: its
        // two gains normalise the attention and MLP outputs (see
        // `LayerWeights::post_norm`). Llama has the two pre-norms.
        let (g1, g2, g_post_attn, g_post_mlp) = if gemma {
            (
                g("input_layernorm.weight")?,
                g("pre_feedforward_layernorm.weight")?,
                g("post_attention_layernorm.weight")?,
                g("post_feedforward_layernorm.weight")?,
            )
        } else if post_norm {
            (
                g("post_attention_layernorm.weight")?,
                g("post_feedforward_layernorm.weight")?,
                Vec::new(),
                Vec::new(),
            )
        } else {
            (
                g("input_layernorm.weight")?,
                g("post_attention_layernorm.weight")?,
                Vec::new(),
                Vec::new(),
            )
        };
        // Granite: `x + branch * residual_multiplier` for both branches;
        // the scalar rides on the branches' output projections.
        let residual = |w: Vec<u16>| -> Vec<u16> {
            match cfg.residual_multiplier {
                Some(r) => scale_bf16(&w, r),
                None => w,
            }
        };
        layers.push(LayerTensors {
            g1,
            g2,
            g_post_attn,
            g_post_mlp,
            wq,
            wk,
            wv,
            wo: residual(lin("self_attn.o_proj.weight", h, qd)?),
            bq: opt("self_attn.q_proj.bias", &[qd])?,
            bk: opt("self_attn.k_proj.bias", &[kvd])?,
            bv: opt("self_attn.v_proj.bias", &[kvd])?,
            // Per head (Qwen3, Gemma-3) or over the whole projection
            // (OLMo-2).
            qn: plus_one(opt("self_attn.q_norm.weight", &[hd, qd])?),
            kn: plus_one(opt("self_attn.k_norm.weight", &[hd, kvd])?),
            wg,
            wu,
            wd: residual(lin("mlp.down_proj.weight", h, i)?),
        });
    }
    let mut final_gamma = plus_one(tensor_f32(st("model.norm.weight"), "model.norm.weight")?.0);
    // Gemma-2: the head applies `tanh(logits / cap) * cap`; the division
    // rides on the final norm gain.
    if let Some(cap) = cfg.final_softcap() {
        for g in &mut final_gamma {
            *g /= cap;
        }
    }
    let has_head = shards.index.contains_key("lm_head.weight")
        || (shards.index.is_empty() && parsed[0].tensor("lm_head.weight").is_ok());
    let lm_head = if has_head {
        linear(st("lm_head.weight"), "lm_head.weight", v, h)?
    } else if cfg.tied() {
        embed.clone()
    } else {
        return Err(Error::Other(
            "lm_head.weight missing and embeddings are not tied".into(),
        ));
    };
    // Granite: `logits / logits_scaling`, folded into the head's own copy
    // so the embedding table stays as stored.
    let lm_head = match cfg.logits_scaling {
        Some(s) => scale_bf16(&lm_head, 1.0 / s),
        None => lm_head,
    };
    Ok(LlamaWeights {
        embed,
        layers,
        final_gamma,
        lm_head,
    })
}

/// Rotate-half RoPE caches `[tokens, head_dim]` for positions `0..tokens`.
#[must_use]
pub fn rope_caches(tokens: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    rope_caches_scaled(tokens, head_dim, theta, None)
}

/// Rotate-half RoPE caches `[tokens, head_dim]` with the config's
/// `rope_scaling` applied: for the `llama3` type each inverse frequency
/// whose wavelength exceeds `original_max_position_embeddings /
/// low_freq_factor` is divided by `factor`, those below `original /
/// high_freq_factor` are kept, and the band between is blended linearly
/// (`transformers` `_compute_llama3_parameters`). Other types are
/// ignored. Diagnostic `RENG_NO_ROPE_SCALING` ignores every type.
///
/// # Panics
///
/// Panics if a `llama3` scaling lacks one of its four parameters.
#[must_use]
pub fn rope_caches_scaled(
    tokens: usize,
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScaling>,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut inv: Vec<f32> = (0..half)
        .map(|i| theta.powf(-2.0 * (i as f32) / head_dim as f32))
        .collect();
    let llama3 = scaling.filter(|s| {
        s.rope_type.as_deref() == Some("llama3") && std::env::var("RENG_NO_ROPE_SCALING").is_err()
    });
    if let Some(s) = llama3 {
        let factor = s.factor.expect("rope_scaling.factor");
        let low = s.low_freq_factor.expect("rope_scaling.low_freq_factor");
        let high = s.high_freq_factor.expect("rope_scaling.high_freq_factor");
        let orig = s
            .original_max_position_embeddings
            .expect("rope_scaling.original_max_position_embeddings") as f32;
        let (low_wavelen, high_wavelen) = (orig / low, orig / high);
        for f in &mut inv {
            let wavelen = 2.0 * std::f32::consts::PI / *f;
            if wavelen > low_wavelen {
                *f /= factor;
            } else if wavelen >= high_wavelen {
                let smooth = (orig / wavelen - low) / (high - low);
                *f = (1.0 - smooth) * *f / factor + smooth * *f;
            }
        }
    }
    let mut sin = vec![0.0f32; tokens * head_dim];
    let mut cos = vec![0.0f32; tokens * head_dim];
    for p in 0..tokens {
        for d in 0..head_dim {
            let ang = p as f32 * inv[d % half];
            sin[p * head_dim + d] = ang.sin();
            cos[p * head_dim + d] = ang.cos();
        }
    }
    (sin, cos)
}

/// Host-side embedding gather: `[tokens, hidden]` for the given ids, times
/// the config's [`LlamaConfig::embed_scale`] when that is not 1 (Granite's
/// `embedding_multiplier`, Gemma's `sqrt(hidden)`).
///
/// # Panics
///
/// Panics if an id is outside the vocabulary.
#[must_use]
pub fn embed_tokens(w: &LlamaWeights, cfg: &LlamaConfig, ids: &[u32]) -> Vec<f32> {
    let h = cfg.hidden_size;
    let mut x = Vec::with_capacity(ids.len() * h);
    for &id in ids {
        let id = id as usize;
        assert!(id < cfg.vocab_size, "token id {id} out of range");
        x.extend(
            w.embed[id * h..(id + 1) * h]
                .iter()
                .map(|&b| bf16_to_f32(b)),
        );
    }
    let m = cfg.embed_scale();
    if m != 1.0 {
        for v in &mut x {
            *v *= m;
        }
    }
    x
}

/// Per-row argmax of `[rows, vocab]` logits.
#[must_use]
pub fn argmax_rows(logits: &[f32], vocab: usize) -> Vec<usize> {
    logits
        .chunks_exact(vocab)
        .map(|row| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                    if x > b.1 { (i, x) } else { b }
                })
                .0
        })
        .collect()
}

/// Smallest token count a prefill recipe is launched with. The final kernel
/// (the LM head gemm, whose M dimension is the token count) must run long
/// enough for its HBM writeback to be visible to the readback DMA on this
/// stack; below about 256 rows the first rows read back as zeros. Because
/// attention is causal, right-padding the prompt is exact for every real
/// position (padded queries sit after all real keys and are never attended).
pub const MIN_PREFILL_TOKENS: usize = 256;

/// Borrow `w` as the engine's per-layer weight views, each layer with its
/// RoPE tables (`rope`'s global or local pair, empty tables for the cached
/// decoders, which take rows per step), window, activation and the
/// attention scale folded in.
#[cfg(feature = "link-synapse")]
fn layer_views<'a>(
    w: &'a LlamaWeights,
    cfg: &LlamaConfig,
    rope: &reng_synapse::RopeTables<'a>,
) -> reng_synapse::ModelWeights<'a> {
    use reng_synapse::{LayerWeights, ModelWeights};
    let hd = cfg.head_dim();
    let act = cfg.activation().expect("checked by LlamaConfig::load");
    let layers: Vec<LayerWeights<'a>> = w
        .layers
        .iter()
        .enumerate()
        .map(|(li, l)| LayerWeights {
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.n_kv_heads(),
            head_dim: hd,
            g1: &l.g1,
            g2: &l.g2,
            post_norm: cfg.post_norm(),
            g_post_attn: &l.g_post_attn,
            g_post_mlp: &l.g_post_mlp,
            wq: &l.wq,
            wk: &l.wk,
            wv: &l.wv,
            wo: &l.wo,
            bq: &l.bq,
            bk: &l.bk,
            bv: &l.bv,
            qn: &l.qn,
            kn: &l.kn,
            wg: &l.wg,
            wu: &l.wu,
            wd: &l.wd,
            sin: if cfg.local_rope(li) {
                rope.sin_local
            } else {
                rope.sin
            },
            cos: if cfg.local_rope(li) {
                rope.cos_local
            } else {
                rope.cos
            },
            scale: cfg.attention_scale(),
            use_rope: cfg
                .no_rope_layers
                .as_ref()
                .is_none_or(|v| v.get(li).copied().unwrap_or(1) != 0),
            local_rope: cfg.local_rope(li),
            window: cfg.layer_window(li),
            act,
            attn_softcap: cfg.attn_softcap(),
            eps: cfg.rms_norm_eps,
        })
        .collect();
    ModelWeights {
        layers,
        final_gamma: &w.final_gamma,
        lm_head: &w.lm_head,
        final_softcap: cfg.final_softcap(),
    }
}

/// Run causal prefill on `ids` through the fused engine and return logits
/// `[ids.len(), vocab]`. The prompt is right-padded to
/// [`MIN_PREFILL_TOKENS`] internally; only the real positions are returned.
/// Each call compiles a fresh recipe; for generation use [`Generator`].
///
/// # Errors
///
/// Returns an error if the device run fails.
#[cfg(feature = "link-synapse")]
pub fn prefill_logits(w: &LlamaWeights, cfg: &LlamaConfig, ids: &[u32]) -> Result<Vec<f32>> {
    let real = ids.len();
    let tokens = real.max(MIN_PREFILL_TOKENS);
    let mut padded: Vec<u32> = ids.to_vec();
    padded.resize(tokens, 0);
    let rope = cfg.rope_caches(tokens);
    let x = embed_tokens(w, cfg, &padded);
    let m = layer_views(w, cfg, &rope.tables());
    let mut logits = reng_synapse::model_forward_bf16(
        &x,
        &m,
        tokens,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.vocab_size,
        true,
    )?;
    logits.truncate(real * cfg.vocab_size);
    Ok(logits)
}

/// A model compiled once with a KV cache, fed token ids block by block.
#[cfg(feature = "link-synapse")]
pub struct Generator<'a> {
    model: reng_synapse::CachedModel<'a>,
    w: &'a LlamaWeights,
    cfg: &'a LlamaConfig,
}

#[cfg(feature = "link-synapse")]
impl<'a> Generator<'a> {
    /// Compile for prompt blocks of `rows` tokens and decode blocks of
    /// `decode_rows` tokens (0: one recipe for both) over a cache of
    /// `capacity` positions, and upload the weights.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or the upload fails.
    pub fn new(
        w: &'a LlamaWeights,
        cfg: &'a LlamaConfig,
        rows: usize,
        decode_rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let rope = cfg.rope_caches(capacity);
        // The cached recipes take RoPE rows as per-step inputs, so the layer
        // views carry no tables (they would have to outlive `rope`).
        let m = layer_views(w, cfg, &reng_synapse::RopeTables::single(&[], &[]));
        let model = reng_synapse::CachedModel::new(
            &m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            rows,
            decode_rows,
            capacity,
            &rope.tables(),
        )?;
        Ok(Self { model, w, cfg })
    }

    /// Positions in the cache so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.model.position()
    }

    /// Forget the cached prefix.
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Append `ids` to the sequence (any length that fits the cache; fed in
    /// blocks of at most `rows`) and return the logits of the last one.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is empty or would overflow the cache.
    pub fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        assert!(!ids.is_empty());
        let mut last = Vec::new();
        for block in ids.chunks(self.model.rows()) {
            let x = embed_tokens(self.w, self.cfg, block);
            last = self.model.step_last(&x)?;
        }
        Ok(last)
    }

    /// Append `ids` and return the greedy next token (argmax on the device).
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is empty or would overflow the cache.
    pub fn feed_id(&mut self, ids: &[u32]) -> Result<u32> {
        assert!(!ids.is_empty());
        if std::env::var_os("RENG_HOST_ARGMAX").is_some_and(|v| v != "0") {
            let logits = self.feed(ids)?;
            return Ok(argmax_rows(&logits, self.cfg.vocab_size)[0] as u32);
        }
        let mut last = 0;
        for block in ids.chunks(self.model.rows()) {
            let x = embed_tokens(self.w, self.cfg, block);
            last = self.model.step_last_id(&x)?;
        }
        Ok(last)
    }
}

/// `B` sequences decoded in lockstep with a `B`-slot KV cache; prompts are
/// prefilled one sequence at a time.
#[cfg(feature = "link-synapse")]
pub struct BatchedGenerator<'a> {
    model: reng_synapse::BatchedModel<'a>,
    w: &'a LlamaWeights,
    cfg: &'a LlamaConfig,
}

#[cfg(feature = "link-synapse")]
impl<'a> BatchedGenerator<'a> {
    /// Compile for `batch` sequences over a cache of `capacity` positions,
    /// with prefill blocks of `rows` tokens, and upload the weights.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or the upload fails.
    pub fn new(
        w: &'a LlamaWeights,
        cfg: &'a LlamaConfig,
        batch: usize,
        rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let rope = cfg.rope_caches(capacity);
        // The batched recipes take RoPE rows as per-step inputs, so the
        // layer views carry no tables (they would have to outlive `rope`).
        let m = layer_views(w, cfg, &reng_synapse::RopeTables::single(&[], &[]));
        let model = reng_synapse::BatchedModel::new(
            m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            batch,
            rows,
            capacity,
            &rope.tables(),
        )?;
        Ok(Self { model, w, cfg })
    }

    /// Number of sequences.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.model.batch()
    }

    /// Start sequence `b` afresh and feed it `ids`; returns the logits of
    /// the last id.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is empty or would overflow the cache.
    pub fn prefill(&mut self, b: usize, ids: &[u32]) -> Result<Vec<f32>> {
        assert!(!ids.is_empty());
        self.model.reset(b);
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.prefill(b, &x)
    }

    /// Advance every sequence by one token (`ids.len() == batch`) and return
    /// the logits `[batch, vocab]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is not one id per sequence.
    pub fn step(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        assert_eq!(ids.len(), self.model.batch());
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.step(&x)
    }

    /// Start sequence `b` afresh, feed it `ids`, and return the greedy next
    /// token (argmax on the device).
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    pub fn prefill_id(&mut self, b: usize, ids: &[u32]) -> Result<u32> {
        assert!(!ids.is_empty());
        self.model.reset(b);
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.prefill_id(b, &x)
    }

    /// Advance every sequence by one token and return the greedy next token
    /// of each (argmax on the device).
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    pub fn step_ids(&mut self, ids: &[u32]) -> Result<Vec<u32>> {
        assert_eq!(ids.len(), self.model.batch());
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.step_ids(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_scaling_moves_only_low_frequencies() {
        let s = RopeScaling {
            rope_type: Some("llama3".into()),
            factor: Some(8.0),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            original_max_position_embeddings: Some(8192),
        };
        let (sin_a, _) = rope_caches_scaled(2, 128, 500000.0, None);
        let (sin_b, _) = rope_caches_scaled(2, 128, 500000.0, Some(&s));
        // Dim 0 (highest frequency) is untouched, the last pair is divided by 8.
        assert_eq!(sin_a[128], sin_b[128]);
        let ang_a = sin_a[128 + 63].asin();
        let ang_b = sin_b[128 + 63].asin();
        assert!((ang_a / ang_b - 8.0).abs() < 1e-3, "{ang_a} {ang_b}");
    }

    #[test]
    fn granite_multipliers_parse_and_default() {
        let base = r#"{"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 4, "rms_norm_eps": 1e-5, "vocab_size": 16"#;
        let llama: LlamaConfig = serde_json::from_str(&format!("{base}}}")).unwrap();
        assert_eq!(llama.embedding_multiplier, None);
        assert_eq!(llama.attention_multiplier, None);
        assert_eq!(llama.residual_multiplier, None);
        assert_eq!(llama.logits_scaling, None);
        assert!(!llama.mlp_bias);
        assert_eq!(llama.attention_scale(), 0.25);
        let granite: LlamaConfig = serde_json::from_str(&format!(
            r#"{base}, "embedding_multiplier": 12.0, "attention_multiplier": 0.015625,
            "residual_multiplier": 0.22, "logits_scaling": 8.0, "mlp_bias": false,
            "attention_bias": false, "rope_scaling": null}}"#
        ))
        .unwrap();
        assert_eq!(granite.embedding_multiplier, Some(12.0));
        assert_eq!(granite.attention_scale(), 0.015625);
        assert_eq!(granite.residual_multiplier, Some(0.22));
        assert_eq!(granite.logits_scaling, Some(8.0));
        assert!(granite.rope_scaling.is_none());
    }

    #[test]
    fn gemma_layer_kinds_tables_and_defaults() {
        let base = r#"{"hidden_size": 640, "intermediate_size": 2048,
            "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
            "rms_norm_eps": 1e-6, "vocab_size": 262144, "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0, "sliding_window": 512,
            "query_pre_attn_scalar": 256, "hidden_activation": "gelu_pytorch_tanh""#;
        // Gemma-3-270m: an explicit layer_types list (full at 5, 11, 17)
        // and `_sliding_window_pattern`, which is ignored.
        let types: Vec<String> = (0..18)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention".to_owned()
                } else {
                    "sliding_attention".to_owned()
                }
            })
            .collect();
        let g270: LlamaConfig = serde_json::from_str(&format!(
            r#"{base}, "model_type": "gemma3_text", "num_hidden_layers": 18,
            "_sliding_window_pattern": 6, "layer_types": {}}}"#,
            serde_json::to_string(&types).unwrap()
        ))
        .unwrap();
        assert!(g270.is_gemma() && g270.tied());
        assert_eq!(g270.activation().unwrap(), Activation::GeluTanh);
        assert_eq!(g270.attention_scale(), 0.0625);
        assert!((g270.embed_scale() - 640f32.sqrt()).abs() < 1e-6);
        let full: Vec<usize> = (0..18).filter(|&li| !g270.sliding(li)).collect();
        assert_eq!(full, vec![5, 11, 17]);
        assert_eq!(g270.layer_window(0), Some(512));
        assert_eq!(g270.layer_window(5), None);
        assert!(g270.local_rope(0) && !g270.local_rope(5));
        // Gemma-3-1B: the pattern alone, 26 layers.
        let g1b: LlamaConfig = serde_json::from_str(&format!(
            r#"{base}, "model_type": "gemma3_text", "num_hidden_layers": 26,
            "sliding_window_pattern": 6}}"#
        ))
        .unwrap();
        let full: Vec<usize> = (0..26).filter(|&li| !g1b.sliding(li)).collect();
        assert_eq!(full, vec![5, 11, 17, 23]);
        let rope = g1b.rope_caches(16);
        assert_eq!(rope.sin_local.len(), 16 * 256);
        // Position 13, rotary dim 1: theta 1e4 and 1e6 differ by 0.43 rad.
        let ang = |t: &[f32]| t[13 * 256 + 1].asin();
        assert!((ang(&rope.sin_local) - ang(&rope.sin) - 0.43).abs() < 0.02);
        // Gemma-2: every second layer slides, one table.
        let g2: LlamaConfig = serde_json::from_str(&format!(
            r#"{base}, "model_type": "gemma2", "num_hidden_layers": 26}}"#
        ))
        .unwrap();
        let sliding: Vec<usize> = (0..6).filter(|&li| g2.sliding(li)).collect();
        assert_eq!(sliding, vec![0, 2, 4]);
        assert!(!g2.local_rope(0));
        assert!(g2.rope_caches(4).sin_local.is_empty());
        assert_eq!(g2.attn_softcap(), None);
        let g2: LlamaConfig = serde_json::from_str(&format!(
            r#"{base}, "model_type": "gemma2", "num_hidden_layers": 26,
            "final_logit_softcapping": 30.0, "attn_logit_softcapping": 50.0}}"#
        ))
        .unwrap();
        assert_eq!(g2.attn_softcap(), Some(50.0));
        assert_eq!(g2.final_softcap(), Some(30.0));
        // Gemma-3 ignores an attention softcap.
        assert_eq!(g270.attn_softcap(), None);
        assert_eq!(g270.final_softcap(), None);
        // Llama: untied by default, SiLU, no per-layer window.
        let llama: LlamaConfig = serde_json::from_str(
            r#"{"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "rms_norm_eps": 1e-5, "vocab_size": 16,
            "hidden_act": "silu"}"#,
        )
        .unwrap();
        assert!(!llama.is_gemma() && !llama.tied() && !llama.sliding(0));
        assert_eq!(llama.activation().unwrap(), Activation::Silu);
        assert_eq!(llama.embed_scale(), 1.0);
        assert_eq!(llama.layer_window(0), None);
        assert!(llama.rope_caches(4).sin_local.is_empty());
        let gelu: LlamaConfig = serde_json::from_str(
            r#"{"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "rms_norm_eps": 1e-5, "vocab_size": 16,
            "hidden_act": "gelu"}"#,
        )
        .unwrap();
        assert!(gelu.activation().is_err());
    }

    #[test]
    fn bf16_roundtrip() {
        // Small integers and powers of two are exact in bf16.
        let v: Vec<f32> = vec![0.0, 1.0, -2.0, 0.5, 256.0, -0.125];
        let b = reng_synapse::to_bf16(&v);
        let back: Vec<f32> = b.iter().map(|&x| bf16_to_f32(x)).collect();
        assert_eq!(back, v);
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_25).abs() < 1e-4);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let (sin, cos) = rope_caches(2, 4, 10000.0);
        assert!(sin[..4].iter().all(|&s| s == 0.0));
        assert!(cos[..4].iter().all(|&c| c == 1.0));
        assert!(sin[4..].iter().any(|&s| s != 0.0));
    }

    #[test]
    fn argmax_rows_picks_max() {
        let l = [0.1, 0.9, 0.3, 2.0, -1.0, 0.0];
        assert_eq!(argmax_rows(&l, 3), vec![1, 0]);
    }
}
