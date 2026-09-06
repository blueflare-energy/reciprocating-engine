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
//! configs). A multimodal Gemma-3 config (`model_type: gemma3`, the 4B
//! and up) is flattened to its `text_config` by [`LlamaConfig::from_json`]
//! and its weights are read under `language_model.model.`; the vision
//! tower and projector are never touched.
//!
//! RoPE scaling (`rope_scaling`): the `llama3`, `linear`, `yarn` and
//! `longrope` types are host-side table recipes ([`rope_spec`]: an
//! inverse-frequency vector and an attention factor on the tables).
//! longrope picks its short or long factor list from the length of the
//! sequence the tables serve, so a prefill passes its prompt length
//! ([`LlamaConfig::rope_caches_for`]) and the cached decoders their
//! capacity. A partial rotation (`partial_rotary_factor`; Phi-4-mini
//! rotates 96 of its 128 head dims, HF pairing `i, i + 48` and passing
//! the rest through) needs no graph change: the loader permutes each
//! head's q and k rows so that HF's rotary pairs sit on the kernel's
//! `j, j + head_dim / 2` pairs ([`rope_head_perm`]) and the tables give
//! the pass-through pairs cos 1 / sin 0. `q . k` does not depend on the
//! order of the head dims, and `v` and `o_proj` are untouched.
//!
//! The safetensors files are memory-mapped and never read into heap
//! buffers: a bf16 tensor is a [`Bf16Slice`] viewing the checkpoint bytes
//! in place (the maps stay alive for as long as any view does), so loading
//! costs no copy and the file pages are shared with the page cache and
//! reclaimable. Only derived data is owned: f32 and f16 checkpoints
//! converted to bf16, the scaled Granite copies, and a tensor whose data
//! offset is not 2-aligned (94 of the 723 tensors of the 70B distill,
//! 19.2 GB: safetensors pads its header to 8 bytes but not the tensors
//! inside it) or on a big-endian host. An unaligned tensor is not copied
//! when it is read from the file but when it is first used, and a shard
//! narrows it first, so a rank copies its own rows and column blocks and
//! never the whole tensor (see [`Bf16Slice::sub`] and
//! [`Bf16Slice::column_block`]).

use memmap2::{Advice, Mmap};
use reng_core::{Error, Result};
use reng_synapse::{Activation, bf16_to_f32, f32_to_bf16, scale_bf16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

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
    /// HF `rope_scaling`: the `llama3`, `linear`, `yarn` and `longrope`
    /// types are applied (see [`rope_spec`]); any other is reported by
    /// [`LlamaConfig::load`] and ignored.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// Phi-3: the pretraining length, kept at the top level of the config
    /// (it takes priority over a copy inside `rope_scaling`).
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// The fraction of each head that RoPE rotates (Phi-4-mini 0.75: dims
    /// 0..96 of 128, paired `i, i + 48`; the rest pass through); see
    /// [`LlamaConfig::rotary_dim`].
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// Where the language model's tensors live in the checkpoint:
    /// `language_model.model.` for a multimodal `gemma3` config (set by
    /// [`LlamaConfig::from_json`]), `model.` otherwise.
    #[serde(skip)]
    pub weight_prefix: Option<String>,
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

/// The `rope_scaling` object of a HF config (`rope_parameters` in
/// transformers 5). [`rope_spec`] applies the `llama3`, `linear`, `yarn`
/// and `longrope` types; `dynamic` (a sequence-length dependent base) is
/// reported by [`LlamaConfig::load`] and ignored.
#[derive(Debug, Clone, Default, Deserialize)]
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
    /// The pretraining length. Phi-3 keeps it at the top level of the
    /// config, which takes priority; [`LlamaConfig::load`] copies it here.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// longrope: one divisor per rotary pair for sequences up to
    /// `original_max_position_embeddings` (`short_factor`) and beyond it
    /// (`long_factor`).
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
    /// longrope, yarn: the multiplier on the sin and cos tables; derived
    /// from `factor` (and yarn's `mscale`s) when absent.
    #[serde(default)]
    pub attention_factor: Option<f32>,
    /// yarn: the rotation counts bounding the interpolation ramp (32 and
    /// 1 when absent), the `mscale` pair and whether the ramp bounds are
    /// rounded to whole dims (true when absent).
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub mscale: Option<f32>,
    #[serde(default)]
    pub mscale_all_dim: Option<f32>,
    #[serde(default)]
    pub truncate: Option<bool>,
    /// The config's top-level `max_position_embeddings`, copied here by
    /// [`LlamaConfig::load`]: longrope and yarn derive `factor` from its
    /// ratio to the pretraining length when the key is absent.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
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

    /// RoPE caches for `positions` positions serving a sequence of
    /// `positions` tokens; see [`LlamaConfig::rope_caches_for`].
    #[must_use]
    pub fn rope_caches(&self, positions: usize) -> RopeCaches {
        self.rope_caches_for(positions, positions)
    }

    /// RoPE caches for `positions` positions: the global table from
    /// `rope_theta` (with `rope_scaling`, which applies to the full layers
    /// only) and, for Gemma-3, the local one from `rope_local_base_freq`.
    /// `seq_len` is the length of the sequence the tables serve (the
    /// prompt length of a prefill; a padded table has more positions than
    /// that): `longrope` selects its long factors from it (see
    /// [`rope_spec`]).
    #[must_use]
    pub fn rope_caches_for(&self, positions: usize, seq_len: usize) -> RopeCaches {
        let (hd, rd) = (self.head_dim(), self.rotary_dim());
        let (sin, cos) = rope_caches_partial(
            positions,
            hd,
            rd,
            self.rope_theta,
            self.rope_scaling.as_ref(),
            seq_len,
        );
        let (sin_local, cos_local) = if (0..self.num_hidden_layers).any(|li| self.local_rope(li)) {
            rope_caches_partial(
                positions,
                hd,
                rd,
                self.rope_local_base_freq.unwrap_or(1e4),
                None,
                positions,
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

    /// Read `config.json` from a model directory (see
    /// [`LlamaConfig::from_json`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Error::Other(format!("config.json: {e}")))?;
        Self::from_json(&text)
    }

    /// Parse the text of a `config.json`. A multimodal Gemma-3 config
    /// (`model_type: gemma3`, weights under `language_model.model.`) is
    /// flattened to its `text_config`, completed with `Gemma3TextConfig`'s
    /// defaults for the keys the file leaves out (the 4B and 12B files
    /// carry six). The `rope_scaling` gets the top-level lengths it
    /// derives its factors from (`standardize_rope_params`), and Phi-3's
    /// legacy `su` / `yarn` type names mean `longrope` (`Phi3Config`). A
    /// `dynamic` or unknown rope type is reported on stderr and ignored.
    ///
    /// # Errors
    ///
    /// Returns an error if the text cannot be parsed, the activation is
    /// unknown, `partial_rotary_factor` does not give an even rotary width
    /// between 2 and `head_dim`, or a longrope factor list does not hold
    /// one entry per rotary pair.
    pub fn from_json(text: &str) -> Result<Self> {
        let err = |e: serde_json::Error| Error::Other(format!("config.json: {e}"));
        let mut v: serde_json::Value = serde_json::from_str(text).map_err(err)?;
        let mut prefix = None;
        if v.get("model_type").and_then(serde_json::Value::as_str) == Some("gemma3") {
            let Some(mut text_cfg) = v.get("text_config").cloned() else {
                return Err(Error::Other(
                    "config.json: gemma3 without text_config".into(),
                ));
            };
            let defaults = serde_json::json!({
                "model_type": "gemma3_text", "vocab_size": 262_208, "hidden_size": 2304,
                "intermediate_size": 9216, "num_hidden_layers": 26, "num_attention_heads": 8,
                "num_key_value_heads": 4, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 131_072, "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
                "query_pre_attn_scalar": 256, "sliding_window": 4096, "sliding_window_pattern": 6,
                "rope_theta": 1_000_000.0, "rope_local_base_freq": 10_000.0
            });
            if let (Some(obj), Some(d)) = (text_cfg.as_object_mut(), defaults.as_object()) {
                for (k, dv) in d {
                    obj.entry(k.clone()).or_insert_with(|| dv.clone());
                }
            }
            v = text_cfg;
            prefix = Some("language_model.model.".to_owned());
        }
        let mut cfg: Self = serde_json::from_value(v).map_err(err)?;
        cfg.weight_prefix = prefix;
        let phi3 = cfg.model_type.as_deref() == Some("phi3");
        let (max_pos, orig) = (
            cfg.max_position_embeddings,
            cfg.original_max_position_embeddings,
        );
        if let Some(s) = cfg.rope_scaling.as_mut() {
            if phi3 && matches!(s.rope_type.as_deref(), Some("su" | "yarn")) {
                s.rope_type = Some("longrope".to_owned());
            }
            s.max_position_embeddings = max_pos;
            if orig.is_some() {
                s.original_max_position_embeddings = orig;
            } else if s.original_max_position_embeddings.is_none() {
                s.original_max_position_embeddings = max_pos;
            }
            let t = s.rope_type.as_deref().unwrap_or("default");
            if !matches!(t, "default" | "llama3" | "linear" | "yarn" | "longrope") {
                eprintln!("config.json: rope_scaling type {t} is not applied");
            }
        }
        if let Some(f) = cfg.partial_rotary_factor {
            let rd = cfg.rotary_dim();
            if !(f > 0.0 && f <= 1.0) || rd == 0 || rd % 2 != 0 {
                return Err(Error::Other(format!(
                    "config.json: partial_rotary_factor {f} rotates {rd} of {} head dims",
                    cfg.head_dim()
                )));
            }
        }
        // longrope divides one rotary pair per factor-list entry
        // (transformers `validate_rope`); a list of another length would
        // panic in [`rope_spec`].
        if let Some(s) = cfg
            .rope_scaling
            .as_ref()
            .filter(|s| s.rope_type.as_deref() == Some("longrope"))
        {
            let half = cfg.rotary_dim() / 2;
            for (name, list) in [
                ("short_factor", s.short_factor.as_ref()),
                ("long_factor", s.long_factor.as_ref()),
            ] {
                let n = list.map_or(0, Vec::len);
                if n != half {
                    return Err(Error::Other(format!(
                        "config.json: rope_scaling.{name} has {n} entries for {half} rotary pairs"
                    )));
                }
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

    /// The head dims RoPE rotates: `int(head_dim * partial_rotary_factor)`
    /// (transformers), the whole head without the key.
    #[must_use]
    pub fn rotary_dim(&self) -> usize {
        let hd = self.head_dim();
        match self.partial_rotary_factor {
            Some(f) => (hd as f32 * f) as usize,
            None => hd,
        }
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

    /// The config of one tensor-parallel shard: `world` cards each hold
    /// `num_attention_heads / world` query heads, `n_kv_heads / world` KV
    /// heads (whole GQA groups) and `intermediate_size / world` MLP
    /// columns; the hidden size, the norms, the embedding and the LM head
    /// are replicated.
    ///
    /// # Panics
    ///
    /// Panics if `world` does not divide the head counts and the
    /// intermediate size, or if `rank >= world`.
    #[must_use]
    pub fn shard(&self, rank: usize, world: usize) -> Self {
        assert!(world >= 1 && rank < world, "rank {rank} of {world}");
        let n_kv = self.n_kv_heads();
        assert!(
            self.num_attention_heads % world == 0
                && n_kv % world == 0
                && self.intermediate_size % world == 0,
            "heads {} / kv heads {n_kv} / intermediate {} are not divisible by {world}",
            self.num_attention_heads,
            self.intermediate_size
        );
        let mut c = self.clone();
        c.head_dim = Some(self.head_dim());
        c.num_attention_heads = self.num_attention_heads / world;
        c.num_key_value_heads = Some(n_kv / world);
        c.intermediate_size = self.intermediate_size / world;
        c
    }
}

/// The bf16 elements of a weight tensor: a view into a memory-mapped
/// safetensors file (a bf16 checkpoint, the common case: nothing is copied
/// and the pages belong to the page cache) or an owned buffer (an f32 or
/// f16 checkpoint converted at load, a scaled or otherwise derived copy,
/// or the fallback for a view that cannot be taken). Dereferences to
/// `[u16]`; cloning is cheap (the tied LM head shares the embedding view).
#[derive(Clone)]
pub enum Bf16Slice {
    /// Heap data (shared, so that clones cost nothing).
    Owned(Arc<Vec<u16>>),
    /// `len` elements at byte `offset` of a mapped file. The offset is
    /// 2-aligned (checked by [`Bf16Slice::mapped`]).
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
    /// `len` elements at an odd byte `offset` of a mapped file (or on a
    /// big-endian host): they cannot be viewed as `[u16]`, so the first
    /// read copies them into the aligned buffer in `cell` and every read
    /// after that returns it. [`Bf16Slice::sub`] and
    /// [`Bf16Slice::column_block`] narrow the range before that happens,
    /// so a tensor-parallel shard copies its own rows once instead of the
    /// whole tensor.
    Unaligned {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
        /// Shared with the clones of this slice, so they copy once
        /// between them; a sub-view gets a cell of its own.
        cell: Arc<std::sync::OnceLock<Vec<u16>>>,
    },
}

impl Bf16Slice {
    /// A view of `len` bf16 elements at byte `offset` of `map`: read in
    /// place when the elements can be (the offset is 2-aligned and the
    /// host is little-endian, as the file format is), otherwise a
    /// [`Bf16Slice::Unaligned`] view that copies when it is read.
    ///
    /// # Panics
    ///
    /// Panics if the range lies outside the map.
    #[must_use]
    pub fn mapped(map: Arc<Mmap>, offset: usize, len: usize) -> Self {
        assert!(
            offset <= map.len() && len * 2 <= map.len() - offset,
            "mapped range outside the map"
        );
        if cfg!(target_endian = "little") && offset % align_of::<u16>() == 0 {
            return Self::Mapped { map, offset, len };
        }
        Self::Unaligned {
            map,
            offset,
            len,
            cell: Arc::new(std::sync::OnceLock::new()),
        }
    }

    /// The number of bf16 elements, without reading (and so without
    /// copying) them.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Mapped { len, .. } | Self::Unaligned { len, .. } => *len,
        }
    }

    /// Whether the slice has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements `start..start + len` as a slice of their own: a sub-view
    /// of a mapped slice, aligned or not (no copy: an unaligned sub-view
    /// copies its own elements only, and only when it is read), a copy of
    /// an owned one.
    ///
    /// # Panics
    ///
    /// Panics if the range lies outside the slice.
    #[must_use]
    pub fn sub(&self, start: usize, len: usize) -> Self {
        assert!(start + len <= self.len(), "sub-slice out of range");
        match self {
            Self::Owned(v) => Self::from(v[start..start + len].to_vec()),
            Self::Mapped { map, offset, .. } => Self::Mapped {
                map: Arc::clone(map),
                offset: offset + start * 2,
                len,
            },
            Self::Unaligned { map, offset, .. } => Self::Unaligned {
                map: Arc::clone(map),
                offset: offset + start * 2,
                len,
                cell: Arc::new(std::sync::OnceLock::new()),
            },
        }
    }

    /// The column window `[col0, col0 + cols)` of every row of this slice
    /// read as a `[rows, pitch]` row-major matrix, gathered into a
    /// contiguous owned `[rows, cols]` copy of its own. Reads the window
    /// straight out of the map, so an unaligned tensor is copied once and
    /// only where the window falls (`rows * cols` elements, not the
    /// `(rows - 1) * pitch + cols` a [`Bf16Slice::sub`] view spans).
    ///
    /// # Panics
    ///
    /// Panics if the window lies outside the slice.
    #[must_use]
    pub fn column_block(&self, rows: usize, pitch: usize, col0: usize, cols: usize) -> Self {
        assert!(
            cols <= pitch && rows * pitch <= self.len() && col0 + cols <= pitch,
            "column block [{rows}, {cols}] at {col0} of a [{rows}, {pitch}] matrix of {} elements",
            self.len()
        );
        let mut v: Vec<u16> = vec![0; rows * cols];
        match self {
            Self::Owned(_) | Self::Mapped { .. } => {
                let src: &[u16] = self;
                for r in 0..rows {
                    let at = r * pitch + col0;
                    v[r * cols..(r + 1) * cols].copy_from_slice(&src[at..at + cols]);
                }
            }
            Self::Unaligned { map, offset, .. } => {
                for r in 0..rows {
                    let at = offset + (r * pitch + col0) * 2;
                    copy_bf16_bytes(&map[at..at + cols * 2], &mut v[r * cols..(r + 1) * cols]);
                }
            }
        }
        Self::from(v)
    }

    /// Whether the elements are read from a mapped file in place. False
    /// for an owned buffer and for an unaligned view, which is copied into
    /// one when it is read.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }

    /// The bytes this slice holds, or will hold once it is read, in a
    /// buffer of its own: zero for a view read in place.
    #[must_use]
    pub fn owned_bytes(&self) -> usize {
        match self {
            Self::Mapped { .. } => 0,
            Self::Owned(_) | Self::Unaligned { .. } => self.len() * 2,
        }
    }

    /// What identifies the elements this slice reads, so that
    /// [`LlamaWeights::footprint`] can count a shared buffer (the tied LM
    /// head, a clone) once without reading it.
    fn source_key(&self) -> (usize, usize) {
        match self {
            Self::Owned(v) => (Arc::as_ptr(v) as usize, 0),
            Self::Mapped { map, offset, .. } | Self::Unaligned { map, offset, .. } => {
                (Arc::as_ptr(map) as usize, *offset)
            }
        }
    }

    /// Whether an unaligned view has already been copied into its aligned
    /// buffer (always true for the other forms).
    #[cfg(test)]
    fn is_materialised(&self) -> bool {
        match self {
            Self::Owned(_) | Self::Mapped { .. } => true,
            Self::Unaligned { cell, .. } => cell.get().is_some(),
        }
    }
}

/// Copy `src` (little-endian bf16 bytes, any alignment) into `dst`.
///
/// # Panics
///
/// Panics unless `src` is exactly `dst` twice over: the copy below writes
/// `src.len()` bytes through a raw pointer, so a wrong length is a write
/// past the end of `dst` rather than a panic, and the check costs nothing
/// next to the memcpy it guards.
fn copy_bf16_bytes(src: &[u8], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len() * 2, "copy_bf16_bytes: length mismatch");
    // SAFETY: `dst` holds `src.len()` bytes and the ranges do not overlap
    // (`dst` is a fresh buffer); a u16 accepts every bit pattern.
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<u8>(), src.len());
    }
    if cfg!(target_endian = "big") {
        for x in dst {
            *x = x.swap_bytes();
        }
    }
}

impl From<Vec<u16>> for Bf16Slice {
    fn from(v: Vec<u16>) -> Self {
        Self::Owned(Arc::new(v))
    }
}

impl std::ops::Deref for Bf16Slice {
    type Target = [u16];

    fn deref(&self) -> &[u16] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped { map, offset, len } => {
                let bytes = &map[*offset..*offset + *len * 2];
                // SAFETY: u16 has no invalid bit patterns and the map is
                // read-only; `mapped` checked the 2-byte alignment, so the
                // whole range is in the middle part.
                let (head, mid, tail) = unsafe { bytes.align_to::<u16>() };
                assert!(head.is_empty() && tail.is_empty());
                mid
            }
            // The one copy an unaligned view makes, of exactly the
            // elements it spans, kept for every read after this one.
            Self::Unaligned {
                map,
                offset,
                len,
                cell,
            } => cell.get_or_init(|| {
                let mut v: Vec<u16> = vec![0; *len];
                copy_bf16_bytes(&map[*offset..*offset + *len * 2], &mut v);
                v
            }),
        }
    }
}

impl std::fmt::Debug for Bf16Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(v) => write!(f, "Bf16Slice::Owned({} elements)", v.len()),
            Self::Mapped { offset, len, .. } => {
                write!(f, "Bf16Slice::Mapped({len} elements at byte {offset})")
            }
            Self::Unaligned { offset, len, .. } => {
                write!(f, "Bf16Slice::Unaligned({len} elements at byte {offset})")
            }
        }
    }
}

/// One layer's weights: the projections bf16 in the checkpoint's
/// `[out, in]` layout (views into the mapped files, or owned where
/// derived), the norm gains and biases f32.
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
    /// config's `residual_multiplier` when it has one. Under a partial
    /// rotation the rows of `wq` and `wk` (and `bq`, `bk`, `qn`, `kn`)
    /// are in [`rope_head_perm`] order within each head.
    pub wq: Bf16Slice,
    pub wk: Bf16Slice,
    pub wv: Bf16Slice,
    pub wo: Bf16Slice,
    /// Row pitch of `wo` in elements when it is the column window of a
    /// tensor-parallel shard (a view of the whole checkpoint matrix from
    /// the shard's first column on; see [`LlamaWeights::shard`]); 0 for
    /// the contiguous matrix.
    pub wo_pitch: usize,
    /// Attention biases; empty when the checkpoint has none.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    /// q/k norm gains: each `head_dim` (Qwen3, per head) or the full
    /// projection widths (OLMo-2: `n_heads * head_dim` and `n_kv_heads *
    /// head_dim`); empty when the checkpoint has none.
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    pub wg: Bf16Slice,
    pub wu: Bf16Slice,
    pub wd: Bf16Slice,
    /// Row pitch of `wd`, as `wo_pitch`.
    pub wd_pitch: usize,
}

/// A whole model's weights on the host: the matrices bf16 in the
/// checkpoint's `[out, in]` layout (the device format, uploaded as is;
/// views into the memory-mapped checkpoint unless derived), the norm
/// gains and biases f32. The mapped files live as long as their views.
pub struct LlamaWeights {
    /// `[vocab, hidden]`, bf16, row per token id (never scaled).
    pub embed: Bf16Slice,
    pub layers: Vec<LayerTensors>,
    pub final_gamma: Vec<f32>,
    /// `[vocab, hidden]`, bf16 (the tied embeddings' own view when the
    /// checkpoint has no head), divided by the config's `logits_scaling`
    /// when it has one.
    pub lm_head: Bf16Slice,
}

impl LlamaWeights {
    /// Every bf16 matrix of the model: the embedding table, the layers'
    /// projections and the LM head.
    fn matrices(&self) -> impl Iterator<Item = &Bf16Slice> {
        let per_layer = self
            .layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.wg, &l.wu, &l.wd]);
        [&self.embed, &self.lm_head].into_iter().chain(per_layer)
    }

    /// Bytes of bf16 weights viewed in place in the mapped checkpoint
    /// files and bytes held in buffers of their own (converted, scaled or
    /// split copies, and unaligned views, which are copied when they are
    /// read; a tied head counts once). Reads no weights, so asking does
    /// not itself make an unaligned view copy.
    #[must_use]
    pub fn footprint(&self) -> (usize, usize) {
        let (mut mapped, mut owned) = (0, 0);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for m in self.matrices() {
            let k = m.source_key();
            if seen.contains(&k) {
                continue;
            }
            seen.push(k);
            if m.is_mapped() {
                mapped += m.len() * 2;
            } else {
                owned += m.owned_bytes();
            }
        }
        (mapped, owned)
    }

    /// The weights of tensor-parallel shard `rank` of `world` (Megatron
    /// split): the q/k/v and gate/up projections keep the rows of this
    /// rank's heads and MLP columns (views into the mapped checkpoint,
    /// since `[out, in]` rows are contiguous), the o and down projections
    /// keep the matching columns as strided views (the sub-view from the
    /// shard's first column to the end of its last row, with the full
    /// matrix width as the row pitch in `wo_pitch` / `wd_pitch`, gathered
    /// row by row while they are uploaded), the biases and the OLMo-2
    /// full-width q/k gains are sliced the same way, and the norms, the
    /// embedding and the LM head are shared. Nothing is copied for a bf16
    /// checkpoint whose tensors are 2-aligned. `cfg` is the unsharded
    /// config; the shard's config is [`LlamaConfig::shard`].
    ///
    /// A tensor at an odd data offset cannot be viewed in place
    /// ([`Bf16Slice::Unaligned`]): its row block is copied when it is
    /// read, and its column block is gathered here (`wo_pitch` /
    /// `wd_pitch` 0, as with `RENG_SHARD_GATHER`) rather than left as a
    /// strided view whose read would copy the whole row range. Either way
    /// a rank copies `1 / world` of such a tensor, once.
    ///
    /// # Panics
    ///
    /// As [`LlamaConfig::shard`], or if a matrix does not have the shape
    /// the config implies.
    #[must_use]
    pub fn shard(&self, cfg: &LlamaConfig, rank: usize, world: usize) -> Self {
        let sc = cfg.shard(rank, world);
        let (h, hd) = (cfg.hidden_size, cfg.head_dim());
        let (q_rows, kv_rows, i_rows) = (sc.q_dim(), sc.n_kv_heads() * hd, sc.intermediate_size);
        let (q_all, kv_all, i_all) = (cfg.q_dim(), cfg.n_kv_heads() * hd, cfg.intermediate_size);
        let rows = |m: &Bf16Slice, all_rows: usize, part: usize, cols: usize| -> Bf16Slice {
            assert_eq!(m.len(), all_rows * cols, "matrix shape");
            m.sub(rank * part * cols, part * cols)
        };
        // The column window `rank * part ..` of every row and the row
        // pitch that goes with it: the view from the first row's window to
        // the end of the last row's, with the full width as its pitch,
        // where the elements can be read in place; or the window gathered
        // into an owned contiguous copy with pitch 0, where they cannot
        // (an odd data offset, an owned buffer) or with diagnostic
        // `RENG_SHARD_GATHER` (the loader's earlier form, kept to measure
        // the strided view against).
        let gather = std::env::var_os("RENG_SHARD_GATHER").is_some();
        let cols =
            |m: &Bf16Slice, out_rows: usize, all_cols: usize, part: usize| -> (Bf16Slice, usize) {
                assert_eq!(m.len(), out_rows * all_cols, "matrix shape");
                if gather || !m.is_mapped() {
                    (m.column_block(out_rows, all_cols, rank * part, part), 0)
                } else {
                    (
                        m.sub(rank * part, (out_rows - 1) * all_cols + part),
                        all_cols,
                    )
                }
            };
        let vec_part = |v: &[f32], all: usize, part: usize| -> Vec<f32> {
            if v.is_empty() {
                return Vec::new();
            }
            assert_eq!(v.len(), all, "vector length");
            v[rank * part..(rank + 1) * part].to_vec()
        };
        let layers = self
            .layers
            .iter()
            .map(|l| {
                let (wo, wo_pitch) = cols(&l.wo, h, q_all, q_rows);
                let (wd, wd_pitch) = cols(&l.wd, h, i_all, i_rows);
                LayerTensors {
                    g1: l.g1.clone(),
                    g2: l.g2.clone(),
                    g_post_attn: l.g_post_attn.clone(),
                    g_post_mlp: l.g_post_mlp.clone(),
                    wq: rows(&l.wq, q_all, q_rows, h),
                    wk: rows(&l.wk, kv_all, kv_rows, h),
                    wv: rows(&l.wv, kv_all, kv_rows, h),
                    wo,
                    wo_pitch,
                    bq: vec_part(&l.bq, q_all, q_rows),
                    bk: vec_part(&l.bk, kv_all, kv_rows),
                    bv: vec_part(&l.bv, kv_all, kv_rows),
                    // Per-head gains (length head_dim) are the same for
                    // every head; full-width gains follow the projection
                    // rows.
                    qn: if l.qn.len() == q_all && q_all != hd {
                        vec_part(&l.qn, q_all, q_rows)
                    } else {
                        l.qn.clone()
                    },
                    kn: if l.kn.len() == kv_all && kv_all != hd {
                        vec_part(&l.kn, kv_all, kv_rows)
                    } else {
                        l.kn.clone()
                    },
                    wg: rows(&l.wg, i_all, i_rows, h),
                    wu: rows(&l.wu, i_all, i_rows, h),
                    wd,
                    wd_pitch,
                }
            })
            .collect();
        Self {
            embed: self.embed.clone(),
            layers,
            final_gamma: self.final_gamma.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}

/// One shard: its map (for the in-place bf16 views) and its parsed header.
#[derive(Clone, Copy)]
struct Src<'a> {
    map: &'a Arc<Mmap>,
    st: &'a SafeTensors<'a>,
}

/// A tensor as bf16 with its shape: a view into the mapped file for bf16
/// checkpoints, a conversion for f32 and f16 ones.
fn tensor_bf16(src: Src<'_>, name: &str) -> Result<(Bf16Slice, Vec<usize>)> {
    let view = src
        .st
        .tensor(name)
        .map_err(|e| Error::Other(format!("tensor {name}: {e}")))?;
    let data = view.data();
    let out = match view.dtype() {
        Dtype::BF16 => {
            // `data` is a sub-slice of the map (`SafeTensors` borrows it).
            let base = src.map.as_ptr() as usize;
            let offset = data.as_ptr() as usize - base;
            assert!(
                offset <= src.map.len() && data.len() <= src.map.len() - offset,
                "tensor {name}: data outside its shard"
            );
            Bf16Slice::mapped(Arc::clone(src.map), offset, data.len() / 2)
        }
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect::<Vec<u16>>()
            .into(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f32_to_bf16(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
            .collect::<Vec<u16>>()
            .into(),
        other => {
            return Err(Error::Other(format!(
                "tensor {name}: unsupported dtype {other:?}"
            )));
        }
    };
    Ok((out, view.shape().to_vec()))
}

fn tensor_f32(src: Src<'_>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = src
        .st
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
fn optional_vec(src: Src<'_>, name: &str, lens: &[usize]) -> Result<Vec<f32>> {
    if src.st.tensor(name).is_err() {
        return Ok(Vec::new());
    }
    let (v, shape) = tensor_f32(src, name)?;
    if shape.len() != 1 || !lens.contains(&shape[0]) {
        return Err(Error::Other(format!(
            "tensor {name}: shape {shape:?}, expected one of {lens:?}"
        )));
    }
    Ok(v)
}

/// A `[out, in]` HF linear weight, shape-checked and kept in that layout.
fn linear(src: Src<'_>, name: &str, out_dim: usize, in_dim: usize) -> Result<Bf16Slice> {
    let (v, shape) = tensor_bf16(src, name)?;
    if shape != [out_dim, in_dim] {
        return Err(Error::Other(format!(
            "tensor {name}: shape {shape:?}, expected [{out_dim}, {in_dim}]"
        )));
    }
    Ok(v)
}

/// Memory-map a checkpoint file read-only. A background thread asks the
/// kernel to populate the whole mapping at once (`MADV_POPULATE_READ`;
/// readahead from a cold page cache, page-table entries from a warm one),
/// so that by the time the weights are copied to the device most of their
/// pages are already mapped; where the kernel lacks the call the pages
/// fault in on first touch instead.
fn map_file(path: &Path) -> Result<Arc<Mmap>> {
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let file = std::fs::File::open(path).map_err(|e| Error::Other(format!("{name}: {e}")))?;
    // SAFETY: the map is read-only and private; a checkpoint file is not
    // modified while a model is loaded from it (as with any loader that
    // maps its files, a concurrent writer would corrupt the weights).
    let map = unsafe { Mmap::map(&file) }.map_err(|e| Error::Other(format!("{name}: {e}")))?;
    let map = Arc::new(map);
    let prefault = Arc::clone(&map);
    // A hint only: the result is ignored, the pages are read either way.
    std::thread::spawn(move || {
        let _ = prefault.advise(Advice::PopulateRead);
    });
    Ok(map)
}

/// The checkpoint's safetensors files, one or several (sharded checkpoints
/// list their tensors in `model.safetensors.index.json`), all mapped.
struct Shards {
    files: Vec<Arc<Mmap>>,
    /// Tensor name to shard index; empty for a single-file checkpoint.
    index: std::collections::HashMap<String, usize>,
}

impl Shards {
    fn open(dir: &Path) -> Result<Self> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Ok(Self {
                files: vec![map_file(&single)?],
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
            files.push(map_file(&dir.join(name))?);
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

    /// Parse every shard's header (the tensor data stays in the maps).
    fn parse(&self) -> Result<Vec<SafeTensors<'_>>> {
        self.files
            .iter()
            .map(|m| {
                SafeTensors::deserialize(m).map_err(|e| Error::Other(format!("safetensors: {e}")))
            })
            .collect()
    }

    /// The shard holding `name` (the only shard when unsharded).
    fn shard<'a>(&'a self, parsed: &'a [SafeTensors<'a>], name: &str) -> Src<'a> {
        let i = self.index.get(name).copied().unwrap_or(0);
        Src {
            map: &self.files[i],
            st: &parsed[i],
        }
    }
}

/// Load the checkpoint (`model.safetensors`, or its shards) from a model
/// directory: the files are memory-mapped and every bf16 matrix is a view
/// of its file (see [`Bf16Slice`]), so the call returns at once and the
/// weight bytes are first read when they are uploaded. Granite's
/// `residual_multiplier` is folded into `o_proj` and `down_proj` and
/// `1/logits_scaling` into the LM head (scaled bf16 copies: one extra
/// rounding per element, none for power-of-two scales). Gemma's norm
/// gains get their `1 + w` offset here (exact in f32).
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

    let pre = cfg.weight_prefix.as_deref().unwrap_or("model.");
    let embed_name = format!("{pre}embed_tokens.weight");
    let (embed, eshape) = tensor_bf16(st(&embed_name), &embed_name)?;
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
    // A partial rotation: q and k head dims go into the kernel's pair order.
    let rotary_perm = (cfg.rotary_dim() < hd).then(|| rope_head_perm(hd, cfg.rotary_dim()));
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = |s: &str| format!("{pre}layers.{l}.{s}");
        let g = |name: &str| -> Result<Vec<f32>> {
            let n = p(name);
            Ok(plus_one(tensor_f32(st(&n), &n)?.0))
        };
        let lin = |name: &str, o: usize, inp: usize| -> Result<Bf16Slice> {
            let n = p(name);
            linear(st(&n), &n, o, inp)
        };
        // Phi-3 stores q/k/v as one [q + k + v, hidden] matrix and gate/up
        // as one [2 * inter, hidden] matrix; in the [out, in] layout the
        // parts are contiguous row blocks (sub-views of the one tensor).
        let has = |name: &str| -> bool {
            let n = p(name);
            shards.index.contains_key(&n)
                || (shards.index.is_empty() && parsed[0].tensor(&n).is_ok())
        };
        let split = |name: &str, rows: &[usize], inp: usize| -> Result<Vec<Bf16Slice>> {
            let n = p(name);
            let total: usize = rows.iter().sum();
            let v = linear(st(&n), &n, total, inp)?;
            let mut out = Vec::with_capacity(rows.len());
            let mut at = 0;
            for &r in rows {
                out.push(v.sub(at * inp, r * inp));
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
        let residual = |w: Bf16Slice| -> Bf16Slice {
            match cfg.residual_multiplier {
                Some(r) => scale_bf16(&w, r).into(),
                None => w,
            }
        };
        let (mut wq, mut wk) = (wq, wk);
        let mut bq = opt("self_attn.q_proj.bias", &[qd])?;
        let mut bk = opt("self_attn.k_proj.bias", &[kvd])?;
        // Per head (Qwen3, Gemma-3) or over the whole projection (OLMo-2).
        let mut qn = plus_one(opt("self_attn.q_norm.weight", &[hd, qd])?);
        let mut kn = plus_one(opt("self_attn.k_norm.weight", &[hd, kvd])?);
        if let Some(perm) = rotary_perm.as_deref() {
            wq = permute_head_rows(&wq, h, hd, perm);
            wk = permute_head_rows(&wk, h, hd, perm);
            bq = permute_head_vec(&bq, hd, perm);
            bk = permute_head_vec(&bk, hd, perm);
            qn = permute_head_vec(&qn, hd, perm);
            kn = permute_head_vec(&kn, hd, perm);
        }
        layers.push(LayerTensors {
            g1,
            g2,
            g_post_attn,
            g_post_mlp,
            wq,
            wk,
            wv,
            wo: residual(lin("self_attn.o_proj.weight", h, qd)?),
            wo_pitch: 0,
            bq,
            bk,
            bv: opt("self_attn.v_proj.bias", &[kvd])?,
            qn,
            kn,
            wg,
            wu,
            wd: residual(lin("mlp.down_proj.weight", h, i)?),
            wd_pitch: 0,
        });
    }
    let norm_name = format!("{pre}norm.weight");
    let mut final_gamma = plus_one(tensor_f32(st(&norm_name), &norm_name)?.0);
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
        Some(s) => scale_bf16(&lm_head, 1.0 / s).into(),
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

/// HF's `(inv_freq, attention_factor)` pair for one rotary table: the
/// inverse frequency of each of the `head_dim / 2` rotary pairs from
/// `theta`, with the config's `rope_scaling` applied, and the factor that
/// multiplies the sin and cos tables (transformers 5.16
/// `modeling_rope_utils.py`, `ROPE_INIT_FUNCTIONS`).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    pub inv_freq: Vec<f32>,
    pub attention_factor: f32,
}

/// The [`RopeSpec`] of a `rotary_dim`-wide rotation (the head, or the
/// rotated part of it under a partial rotation) with base `theta` under
/// `scaling`, for a sequence of `seq_len` tokens (its largest position
/// plus one):
///
/// - `llama3`: each inverse frequency whose wavelength exceeds
///   `original_max_position_embeddings / low_freq_factor` is divided by
///   `factor`, those below `original / high_freq_factor` are kept, and the
///   band between is blended linearly (`_compute_llama3_parameters`);
/// - `linear`: every inverse frequency is divided by `factor`
///   (`_compute_linear_scaling_rope_parameters`);
/// - `yarn`: the pairs below the dim that completes `beta_fast` rotations
///   over the pretraining length are kept, those above the `beta_slow`
///   dim are divided by `factor`, the ramp between is blended, and the
///   attention factor is `0.1 * ln(factor) + 1` (or the ratio of the two
///   `mscale` forms of it) (`_compute_yarn_parameters`);
/// - `longrope`: each inverse frequency is divided by the matching
///   `short_factor` entry, or the `long_factor` one when `seq_len` exceeds
///   `original_max_position_embeddings`, and the attention factor is
///   `sqrt(1 + ln(factor) / ln(original))` (`_compute_longrope_parameters`;
///   `factor` defaults to `max_position_embeddings / original`).
///
/// Other types leave the default table. Diagnostic `RENG_NO_ROPE_SCALING`
/// ignores every type.
///
/// # Panics
///
/// Panics if a scaling lacks a parameter its type requires, or a longrope
/// factor list has the wrong length.
#[must_use]
pub fn rope_spec(
    rotary_dim: usize,
    theta: f32,
    scaling: Option<&RopeScaling>,
    seq_len: usize,
) -> RopeSpec {
    let half = rotary_dim / 2;
    let mut inv: Vec<f32> = (0..half)
        .map(|i| theta.powf(-2.0 * (i as f32) / rotary_dim as f32))
        .collect();
    let mut attention_factor = 1.0f32;
    let scaling = scaling.filter(|_| std::env::var("RENG_NO_ROPE_SCALING").is_err());
    let Some(s) = scaling else {
        return RopeSpec {
            inv_freq: inv,
            attention_factor,
        };
    };
    let orig = || {
        s.original_max_position_embeddings
            .expect("rope_scaling.original_max_position_embeddings") as f32
    };
    // longrope and yarn: `max_position_embeddings / original` when the key is absent.
    let factor_or_ratio = || {
        s.factor.unwrap_or_else(|| {
            s.max_position_embeddings.expect("max_position_embeddings") as f32 / orig()
        })
    };
    match s.rope_type.as_deref().unwrap_or("default") {
        "llama3" => {
            let factor = s.factor.expect("rope_scaling.factor");
            let low = s.low_freq_factor.expect("rope_scaling.low_freq_factor");
            let high = s.high_freq_factor.expect("rope_scaling.high_freq_factor");
            let orig = orig();
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
        "linear" => {
            let factor = s.factor.expect("rope_scaling.factor");
            for f in &mut inv {
                *f /= factor;
            }
        }
        "yarn" => {
            let orig = orig();
            let factor = factor_or_ratio();
            let mscale = |scale: f32, m: f32| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * m * scale.ln() + 1.0
                }
            };
            attention_factor =
                s.attention_factor
                    .unwrap_or_else(|| match (s.mscale, s.mscale_all_dim) {
                        (Some(m), Some(a)) if m != 0.0 && a != 0.0 => {
                            mscale(factor, m) / mscale(factor, a)
                        }
                        _ => mscale(factor, 1.0),
                    });
            let beta_fast = s.beta_fast.filter(|&b| b != 0.0).unwrap_or(32.0);
            let beta_slow = s.beta_slow.filter(|&b| b != 0.0).unwrap_or(1.0);
            let dim = rotary_dim as f32;
            // The dim whose wavelength is `orig / rotations`.
            let corr = |rot: f32| {
                dim * (orig / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * theta.ln())
            };
            let (mut low, mut high) = (corr(beta_fast), corr(beta_slow));
            if s.truncate.unwrap_or(true) {
                low = low.floor();
                high = high.ceil();
            }
            let low = low.max(0.0);
            let mut high = high.min(dim - 1.0);
            if low == high {
                high += 0.001;
            }
            for (i, f) in inv.iter_mut().enumerate() {
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                *f = *f / factor * ramp + *f * (1.0 - ramp);
            }
        }
        "longrope" => {
            let orig_len = s
                .original_max_position_embeddings
                .expect("rope_scaling.original_max_position_embeddings");
            let factor = factor_or_ratio();
            attention_factor = s.attention_factor.unwrap_or_else(|| {
                if factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / (orig_len as f32).ln()).sqrt()
                }
            });
            let ext = if seq_len > orig_len {
                s.long_factor.as_ref().expect("rope_scaling.long_factor")
            } else {
                s.short_factor.as_ref().expect("rope_scaling.short_factor")
            };
            assert_eq!(
                ext.len(),
                half,
                "rope_scaling factors: {} entries for {half} rotary pairs",
                ext.len()
            );
            for (f, e) in inv.iter_mut().zip(ext) {
                *f /= e;
            }
        }
        _ => {}
    }
    RopeSpec {
        inv_freq: inv,
        attention_factor,
    }
}

/// Rotate-half RoPE caches `[tokens, head_dim]` for positions `0..tokens`
/// with the config's `rope_scaling` applied for a sequence of `tokens`
/// tokens; see [`rope_caches_len`].
#[must_use]
pub fn rope_caches_scaled(
    tokens: usize,
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScaling>,
) -> (Vec<f32>, Vec<f32>) {
    rope_caches_len(tokens, head_dim, theta, scaling, tokens)
}

/// Rotate-half RoPE caches `[tokens, head_dim]` for positions `0..tokens`
/// from the [`rope_spec`] of `scaling` at `seq_len`: `sin(p * inv_freq[d %
/// half]) * attention_factor` and the cosine likewise (every dim of a
/// rotary pair gets the same angle). The factor on both tables scales
/// every `q . k` score by its square, which is what HF's rotary module
/// does (`cos = cos * attention_scaling`).
///
/// # Panics
///
/// Panics where [`rope_spec`] does.
#[must_use]
pub fn rope_caches_len(
    tokens: usize,
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScaling>,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    rope_caches_partial(tokens, head_dim, head_dim, theta, scaling, seq_len)
}

/// [`rope_caches_len`] for a rotation of `rotary_dim` of the `head_dim`
/// dims: the tables are `[tokens, head_dim]` in the kernel's pair order
/// (see [`rope_head_perm`]), the first `rotary_dim / 2` pairs carry the
/// angles of a `rotary_dim`-wide [`rope_spec`] with the attention factor,
/// and the remaining pairs read cos 1 / sin 0 (the pass-through dims, not
/// scaled by the factor, as HF leaves `q_pass` alone). With `rotary_dim ==
/// head_dim` the tables are those of [`rope_caches_len`].
///
/// # Panics
///
/// Panics if `rotary_dim` is odd or exceeds `head_dim`, or where
/// [`rope_spec`] does.
#[must_use]
pub fn rope_caches_partial(
    tokens: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    scaling: Option<&RopeScaling>,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "rotary width {rotary_dim} of {head_dim}"
    );
    let (half, rot_half) = (head_dim / 2, rotary_dim / 2);
    let spec = rope_spec(rotary_dim, theta, scaling, seq_len);
    let af = spec.attention_factor;
    let mut sin = vec![0.0f32; tokens * head_dim];
    let mut cos = vec![1.0f32; tokens * head_dim];
    for p in 0..tokens {
        for d in 0..head_dim {
            let j = d % half;
            if j < rot_half {
                let ang = p as f32 * spec.inv_freq[j];
                sin[p * head_dim + d] = ang.sin() * af;
                cos[p * head_dim + d] = ang.cos() * af;
            }
        }
    }
    (sin, cos)
}

/// The head-dim order the loader gives q and k under a partial rotation:
/// entry `e` is the HF dim that engine dim `e` holds. HF rotates dims
/// `0..rotary_dim` in rotate-half pairs `(i, i + rotary_dim / 2)` and
/// passes `rotary_dim..head_dim` through; the kernel rotates every pair
/// `(j, j + head_dim / 2)`. So the rotary pairs go to `j < rotary_dim /
/// 2` and the pass-through dims fill the pairs after them, whose table
/// entries are the identity (see [`rope_caches_partial`]). The identity
/// permutation when `rotary_dim == head_dim`.
///
/// # Panics
///
/// Panics if `rotary_dim` is odd or exceeds `head_dim`.
#[must_use]
pub fn rope_head_perm(head_dim: usize, rotary_dim: usize) -> Vec<usize> {
    assert!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "rotary width {rotary_dim} of {head_dim}"
    );
    let (half, rot_half) = (head_dim / 2, rotary_dim / 2);
    let pass = half - rot_half;
    (0..head_dim)
        .map(|e| {
            let (j, second) = (e % half, e >= half);
            if j < rot_half {
                // Rotary pair j: HF dims j and j + rotary_dim / 2.
                j + if second { rot_half } else { 0 }
            } else {
                // Pass-through dims, in order: rotary_dim + (j - rot_half)
                // for the first half, the same plus `pass` for the second.
                rotary_dim + (j - rot_half) + if second { pass } else { 0 }
            }
        })
        .collect()
}

/// `m` (`[rows, cols]`, `rows` a multiple of `hd`) with the rows of every
/// `hd`-row head block reordered so that row `e` of the block is the
/// block's row `perm[e]`: an owned copy.
fn permute_head_rows(m: &Bf16Slice, cols: usize, hd: usize, perm: &[usize]) -> Bf16Slice {
    let rows = m.len() / cols;
    assert!(rows * cols == m.len() && rows % hd == 0, "matrix shape");
    let mut v = Vec::with_capacity(m.len());
    for head in 0..rows / hd {
        for &src in perm {
            let r = head * hd + src;
            v.extend_from_slice(&m[r * cols..(r + 1) * cols]);
        }
    }
    Bf16Slice::from(v)
}

/// A per-row vector (a bias, or a per-head or full-width gain; a multiple
/// of `hd` long, or empty) reordered like [`permute_head_rows`].
fn permute_head_vec(v: &[f32], hd: usize, perm: &[usize]) -> Vec<f32> {
    assert!(v.len() % hd == 0, "vector length");
    v.chunks_exact(hd)
        .flat_map(|head| perm.iter().map(|&src| head[src]))
        .collect()
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
            wo_pitch: l.wo_pitch,
            bq: &l.bq,
            bk: &l.bk,
            bv: &l.bv,
            qn: &l.qn,
            kn: &l.kn,
            wg: &l.wg,
            wu: &l.wu,
            wd: &l.wd,
            wd_pitch: l.wd_pitch,
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
    let rope = cfg.rope_caches_for(tokens, real);
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
/// Prompts go through the wide recipe from host-gathered embeddings; single
/// tokens go through the device decode loop when it was built (see
/// `reng_synapse::CachedModel`), which gathers the embedding on the device
/// and can run many greedy steps per readback ([`Generator::generate`]).
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
    /// `capacity` positions, and upload the weights. The RoPE tables serve
    /// a sequence of `capacity` tokens: a longrope model reads its long
    /// factors when that exceeds its pretraining length (HF switches at
    /// the first forward past it and recomputes the cache), so size the
    /// cache for the run's total length.
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
        let embed = reng_synapse::EmbedTable {
            rows: &w.embed,
            scale: cfg.embed_scale(),
        };
        let model = reng_synapse::CachedModel::new(
            &m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            rows,
            decode_rows,
            capacity,
            &rope.tables(),
            Some(&embed),
        )?;
        Ok(Self { model, w, cfg })
    }

    /// Positions in the cache so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.model.position()
    }

    /// Whether single tokens run through the device decode loop.
    #[must_use]
    pub fn device_loop(&self) -> bool {
        self.model.has_loop()
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
            last = if block.len() == 1 && self.model.has_loop() {
                self.model.step_id_logits(block[0])?
            } else {
                let x = embed_tokens(self.w, self.cfg, block);
                self.model.step_last(&x)?
            };
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
            last = if block.len() == 1 && self.model.has_loop() {
                self.model.step_ids(block[0], 1)?[0]
            } else {
                let x = embed_tokens(self.w, self.cfg, block);
                self.model.step_last_id(&x)?
            };
        }
        Ok(last)
    }

    /// Append `seed` and continue greedily for `n` tokens in all: the
    /// returned `n` ids are the argmax after `seed` and after each of the
    /// first `n - 1` of them (the last one is not appended). With the
    /// device decode loop this is `n` launches and one readback; without
    /// it, `n` calls of [`Generator::feed_id`].
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0 or the run would overflow the cache.
    pub fn generate(&mut self, seed: u32, n: usize) -> Result<Vec<u32>> {
        assert!(n >= 1);
        if self.model.has_loop() {
            return self.model.step_ids(seed, n);
        }
        let mut out = Vec::with_capacity(n);
        let mut next = seed;
        for _ in 0..n {
            next = self.feed_id(&[next])?;
            out.push(next);
        }
        Ok(out)
    }
}

/// `B` sequences decoded in lockstep with a `B`-slot KV cache; prompts are
/// prefilled one sequence at a time. Steps go through the device decode
/// loop when it was built (see `reng_synapse::BatchedModel`), which
/// gathers the `B` embeddings on the device and can run many greedy steps
/// per readback ([`BatchedGenerator::generate`]).
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
        let embed = reng_synapse::EmbedTable {
            rows: &w.embed,
            scale: cfg.embed_scale(),
        };
        let model = reng_synapse::BatchedModel::new(
            m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            batch,
            rows,
            capacity,
            &rope.tables(),
            Some(&embed),
        )?;
        Ok(Self { model, w, cfg })
    }

    /// Number of sequences.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.model.batch()
    }

    /// Whether steps run through the device decode loop.
    #[must_use]
    pub fn device_loop(&self) -> bool {
        self.model.has_loop()
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
        if self.model.has_loop() {
            return Ok(self.model.run_ids_logits(ids, 1)?.1);
        }
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
        if self.model.has_loop() {
            return self.model.run_ids(ids, 1);
        }
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.step_ids(&x)
    }

    /// Feed `seeds` (one id per sequence) and continue every sequence
    /// greedily for `n` tokens in all: the returned `n * batch` ids are
    /// step by step (`out[j * batch + b]` is sequence `b`'s argmax after
    /// step `j`; the last step's ids are not appended). With the device
    /// decode loop this is `n` launches and one readback; without it, `n`
    /// calls of [`BatchedGenerator::step_ids`]. Every sequence advances
    /// `n` positions whether or not it has finished: a caller that wants
    /// to stop a sequence at EOS drops its later ids, or runs shorter runs
    /// and restarts the slot with a new prefill.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` is not one id per sequence, `n` is 0 or the run
    /// would overflow the cache.
    pub fn generate(&mut self, seeds: &[u32], n: usize) -> Result<Vec<u32>> {
        assert!(n >= 1);
        assert_eq!(seeds.len(), self.model.batch());
        if self.model.has_loop() {
            return self.model.run_ids(seeds, n);
        }
        let mut out = Vec::with_capacity(n * seeds.len());
        let mut next = seeds.to_vec();
        for _ in 0..n {
            next = self.step_ids(&next)?;
            out.extend_from_slice(&next);
        }
        Ok(out)
    }
}

/// One rank of a tensor-parallel model (see `reng_synapse::tp`): the
/// shard's recipes and cache on this process's card, fed token ids, for
/// `batch` sequences decoded in lockstep. Prompts go through the wide
/// recipes from host-gathered embeddings one sequence at a time,
/// generated tokens through the device decode loop.
#[cfg(feature = "link-synapse")]
pub struct TpGenerator<'a> {
    model: reng_synapse::tp::TpModel<'a>,
    w: &'a LlamaWeights,
    cfg: &'a LlamaConfig,
}

#[cfg(feature = "link-synapse")]
impl<'a> TpGenerator<'a> {
    /// Compile this rank's recipes for `batch` sequences, prompt blocks
    /// of `rows` tokens and a cache of `capacity` positions, and upload
    /// its shard. `w` is the shard ([`LlamaWeights::shard`]) and `cfg` its
    /// config ([`LlamaConfig::shard`]); `rank` the joined card and
    /// communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or an upload fails.
    pub fn new(
        rank: reng_synapse::hccl::Rank,
        w: &'a LlamaWeights,
        cfg: &'a LlamaConfig,
        batch: usize,
        rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let rope = cfg.rope_caches(capacity);
        let m = layer_views(w, cfg, &reng_synapse::RopeTables::single(&[], &[]));
        let embed = reng_synapse::EmbedTable {
            rows: &w.embed,
            scale: cfg.embed_scale(),
        };
        let model = reng_synapse::tp::TpModel::new(
            rank,
            &m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            batch,
            rows,
            capacity,
            &rope.tables(),
            &embed,
        )?;
        Ok(Self { model, w, cfg })
    }

    /// The model, for its mode switch and load report.
    pub fn model(&mut self) -> &mut reng_synapse::tp::TpModel<'a> {
        &mut self.model
    }

    /// Number of sequences.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.model.batch()
    }

    /// Positions of sequence `b` in the cache so far.
    #[must_use]
    pub fn position(&self, b: usize) -> usize {
        self.model.position(b)
    }

    /// Start sequence `b` afresh.
    pub fn reset(&mut self, b: usize) {
        self.model.reset(b);
    }

    /// Prefill sequence `b` with `ids` (fed in blocks of at most `rows`)
    /// and return the greedy next token. A sequence is prefilled once,
    /// from position 0: the wide recipe's out-of-place ScatterND makes the
    /// blocks alternate between the sequence's slot and a shared scratch
    /// buffer, so a second prefill onto a non-empty sequence is rejected
    /// unless its block count is even (`TpModel::prefill`). Continue a
    /// sequence with [`TpGenerator::generate`], or [`TpGenerator::reset`]
    /// it first.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is empty, would overflow the cache, or is a second
    /// prefill of an odd number of blocks onto a non-empty sequence.
    pub fn prefill(&mut self, b: usize, ids: &[u32]) -> Result<u32> {
        assert!(!ids.is_empty());
        let x = embed_tokens(self.w, self.cfg, ids);
        self.model.prefill(b, &x)
    }

    /// Feed `seeds` (one id per sequence) and continue every sequence
    /// greedily for `n` tokens in all (see `TpModel::decode`), with the
    /// run's times.
    ///
    /// # Errors
    ///
    /// Returns an error if a device run fails.
    pub fn generate(
        &mut self,
        seeds: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, reng_synapse::tp::StepTimes)> {
        self.model.decode(seeds, n)
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
            ..RopeScaling::default()
        };
        let (sin_a, _) = rope_caches_scaled(2, 128, 500000.0, None);
        let (sin_b, _) = rope_caches_scaled(2, 128, 500000.0, Some(&s));
        // Dim 0 (highest frequency) is untouched, the last pair is divided by 8.
        assert_eq!(sin_a[128], sin_b[128]);
        let ang_a = sin_a[128 + 63].asin();
        let ang_b = sin_b[128 + 63].asin();
        assert!((ang_a / ang_b - 8.0).abs() < 1e-3, "{ang_a} {ang_b}");
    }

    fn scaling(json: &str) -> RopeScaling {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn linear_scaling_divides_every_frequency() {
        // Gemma-3-4B global layers: head_dim 256, theta 1e6, factor 8.
        let s = scaling(r#"{"factor": 8.0, "rope_type": "linear"}"#);
        let a = rope_spec(256, 1e6, None, 16);
        let b = rope_spec(256, 1e6, Some(&s), 16);
        assert_eq!(b.attention_factor, 1.0);
        assert_eq!(b.inv_freq[0], 0.125);
        for (x, y) in a.inv_freq.iter().zip(&b.inv_freq) {
            assert!((x / y - 8.0).abs() < 1e-4, "{x} {y}");
        }
    }

    #[test]
    fn yarn_ramps_between_the_correction_dims() {
        // DeepSeek-V2-Lite: rotary dim 64, base 1e4, factor 40, pretrained
        // at 4096, beta 32/1, mscale 0.707 / 0.707. transformers 5.16
        // gives the correction range (10, 23) and an attention factor of
        // 1 (the two mscale forms cancel).
        let s = scaling(
            r#"{"rope_type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
            "beta_fast": 32, "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}"#,
        );
        let a = rope_spec(64, 1e4, None, 16);
        let b = rope_spec(64, 1e4, Some(&s), 16);
        assert_eq!(b.attention_factor, 1.0);
        for i in 0..=10 {
            assert_eq!(a.inv_freq[i], b.inv_freq[i], "dim {i} is extrapolated");
        }
        for i in 23..32 {
            assert!(
                (a.inv_freq[i] / b.inv_freq[i] - 40.0).abs() < 1e-3,
                "dim {i}"
            );
        }
        // Dim 13 sits 3/13 of the way into the ramp: 0.01838 (analytic).
        assert!(
            (b.inv_freq[13] - 0.01838).abs() < 2e-5,
            "{}",
            b.inv_freq[13]
        );
        // Without the mscale pair the factor is 0.1 ln(40) + 1.
        let s = scaling(
            r#"{"rope_type": "yarn", "factor": 40, "original_max_position_embeddings": 4096}"#,
        );
        let c = rope_spec(64, 1e4, Some(&s), 16);
        assert!((c.attention_factor - 1.368_888_5).abs() < 1e-5);
        assert_eq!(c.inv_freq, b.inv_freq);
    }

    #[test]
    fn longrope_picks_the_factors_by_length_and_scales_the_tables() {
        // Phi-3.5-mini shape: rotary dim 96, base 1e4, 4096 pretrained,
        // 131072 max, so factor 32 and attention factor sqrt(1 + 5/12).
        let short: Vec<f32> = (0..48).map(|i| 1.0 + i as f32 / 48.0).collect();
        let long: Vec<f32> = vec![2.0; 48];
        let mut s = scaling(&format!(
            r#"{{"type": "longrope", "short_factor": {}, "long_factor": {}}}"#,
            serde_json::to_string(&short).unwrap(),
            serde_json::to_string(&long).unwrap()
        ));
        assert_eq!(s.rope_type.as_deref(), Some("longrope"));
        s.original_max_position_embeddings = Some(4096);
        s.max_position_embeddings = Some(131_072);
        let base = rope_spec(96, 1e4, None, 4096);
        let at = |n: usize| rope_spec(96, 1e4, Some(&s), n);
        let af = (17.0f32 / 12.0).sqrt();
        for n in [1, 300, 4096] {
            let r = at(n);
            assert!((r.attention_factor - af).abs() < 1e-6);
            for (i, (x, y)) in base.inv_freq.iter().zip(&r.inv_freq).enumerate() {
                assert!((x / short[i] - y).abs() < 1e-7, "short dim {i} at {n}");
            }
        }
        for n in [4097, 4500] {
            let r = at(n);
            assert!((r.attention_factor - af).abs() < 1e-6);
            for (x, y) in base.inv_freq.iter().zip(&r.inv_freq) {
                assert!((x / 2.0 - y).abs() < 1e-7, "long at {n}");
            }
        }
        // The tables carry the attention factor: position 0 reads cos = af.
        let (sin, cos) = rope_caches_len(2, 96, 1e4, Some(&s), 4097);
        assert!((cos[0] - af).abs() < 1e-6 && sin[0] == 0.0);
        assert!((sin[96] - (0.5f32).sin() * af).abs() < 1e-6, "{}", sin[96]);
        // An explicit attention_factor wins.
        let mut explicit = s.clone();
        explicit.attention_factor = Some(1.0);
        assert_eq!(rope_spec(96, 1e4, Some(&explicit), 8).attention_factor, 1.0);
    }

    /// A `[1.0, ...]` list of `n` entries, one per rotary pair.
    fn ones(n: usize) -> String {
        serde_json::to_string(&vec![1.0f32; n]).unwrap()
    }

    #[test]
    fn phi3_config_lengths_and_legacy_type_names() {
        // Phi-3.5-mini shape: 96 head dims, all rotated, 48 factors.
        let phi35 = format!(
            r#"{{"model_type": "phi3", "hidden_size": 3072, "intermediate_size": 8192,
            "num_hidden_layers": 32, "num_attention_heads": 32, "rms_norm_eps": 1e-5,
            "vocab_size": 32064, "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096, "partial_rotary_factor": 1.0,
            "rope_scaling": {{"type": "su", "short_factor": {f}, "long_factor": {f}}}}}"#,
            f = ones(48)
        );
        let cfg = LlamaConfig::from_json(&phi35).unwrap();
        let s = cfg.rope_scaling.as_ref().unwrap();
        assert_eq!(s.rope_type.as_deref(), Some("longrope"));
        assert_eq!(s.original_max_position_embeddings, Some(4096));
        assert_eq!(s.max_position_embeddings, Some(131_072));
        assert!(cfg.weight_prefix.is_none());
        assert_eq!(cfg.rotary_dim(), 96);
        // Phi-4-mini rotates 96 of its 128 dims, and its lists are 48 long too.
        let partial = format!(
            r#"{{"model_type": "phi3", "hidden_size": 3072, "intermediate_size": 8192,
            "num_hidden_layers": 32, "num_attention_heads": 24, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5, "vocab_size": 200064, "partial_rotary_factor": 0.75,
            "max_position_embeddings": 131072, "original_max_position_embeddings": 4096,
            "rope_scaling": {{"type": "longrope", "short_factor": {f}, "long_factor": {f}}}}}"#,
            f = ones(48)
        );
        let cfg = LlamaConfig::from_json(&partial).unwrap();
        assert_eq!((cfg.head_dim(), cfg.rotary_dim()), (128, 96));
        // 0.2 of 128 rotates 25 dims: no whole rotary pair count.
        assert!(
            LlamaConfig::from_json(&partial.replace("0.75", "0.2")).is_err(),
            "odd rotary width"
        );
        // The factor lists must match the rotary pair count (64 here).
        assert!(
            LlamaConfig::from_json(&partial.replace("0.75", "1.0")).is_err(),
            "48 factors for 64 rotary pairs"
        );
    }

    /// The `llama3` recipe as first committed (322641a), kept verbatim so
    /// the tables can be checked bit for bit.
    fn legacy_llama3_tables(
        tokens: usize,
        head_dim: usize,
        theta: f32,
        s: Option<&RopeScaling>,
    ) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let mut inv: Vec<f32> = (0..half)
            .map(|i| theta.powf(-2.0 * (i as f32) / head_dim as f32))
            .collect();
        if let Some(s) = s {
            let factor = s.factor.unwrap();
            let low = s.low_freq_factor.unwrap();
            let high = s.high_freq_factor.unwrap();
            let orig = s.original_max_position_embeddings.unwrap() as f32;
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

    #[test]
    fn llama3_and_default_tables_are_bit_identical_to_the_first_recipe() {
        // Llama-3.1: head_dim 128, theta 500000, factor 8, low 1, high 4,
        // 8192 pretrained; 1000 positions.
        let s = scaling(
            r#"{"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}"#,
        );
        for sc in [None, Some(&s)] {
            let (sin_a, cos_a) = legacy_llama3_tables(1000, 128, 500_000.0, sc);
            let (sin_b, cos_b) = rope_caches_scaled(1000, 128, 500_000.0, sc);
            assert!(
                sin_a
                    .iter()
                    .zip(&sin_b)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
            assert!(
                cos_a
                    .iter()
                    .zip(&cos_b)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
    }

    #[test]
    fn partial_rotation_permutes_heads_and_matches_hf() {
        // Phi-4-mini: 128 dims, 96 rotated; theta 1e4, no scaling here.
        let (hd, rd) = (128, 96);
        let perm = rope_head_perm(hd, rd);
        assert_eq!(rope_head_perm(hd, hd), (0..hd).collect::<Vec<_>>());
        // Rotary pair j < 48 holds HF dims j and j + 48; the pairs after
        // hold the pass-through dims 96..128 in order.
        for j in 0..48 {
            assert_eq!((perm[j], perm[j + 64]), (j, j + 48));
        }
        assert_eq!(perm[48..64], (96..112).collect::<Vec<_>>()[..]);
        assert_eq!(perm[112..128], (112..128).collect::<Vec<_>>()[..]);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..hd).collect::<Vec<_>>());
        // Pass-through pairs read the identity at every position, rotary
        // pairs the 96-wide spec's angles.
        let (sin, cos) = rope_caches_partial(20, hd, rd, 1e4, None, 20);
        let spec = rope_spec(rd, 1e4, None, 20);
        for p in 0..20 {
            for d in 0..hd {
                let j = d % 64;
                let (s, c) = (sin[p * hd + d], cos[p * hd + d]);
                if j < 48 {
                    let ang = p as f32 * spec.inv_freq[j];
                    assert_eq!((s, c), (ang.sin(), ang.cos()));
                } else {
                    assert_eq!((s, c), (0.0, 1.0));
                }
            }
        }
        // HF's partial rotation of a head at position p (rotate-half over
        // the first 96 dims, the rest kept), scaled by `af`.
        let hf = |x: &[f32], p: usize, af: f32| -> Vec<f32> {
            let mut out = x.to_vec();
            for i in 0..48 {
                let ang = p as f32 * spec.inv_freq[i];
                let (s, c) = (ang.sin() * af, ang.cos() * af);
                out[i] = x[i] * c - x[i + 48] * s;
                out[i + 48] = x[i + 48] * c + x[i] * s;
            }
            out
        };
        // The engine: the permuted head through a whole-head rotate-half
        // with the partial tables.
        let engine = |x: &[f32], p: usize, sin: &[f32], cos: &[f32]| -> Vec<f32> {
            let xp: Vec<f32> = perm.iter().map(|&src| x[src]).collect();
            (0..hd)
                .map(|d| {
                    let rot = if d < 64 { -xp[d + 64] } else { xp[d - 64] };
                    xp[d] * cos[p * hd + d] + rot * sin[p * hd + d]
                })
                .collect()
        };
        let vec = |seed: u32| -> Vec<f32> {
            (0..hd)
                .map(|i| {
                    ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000) as f32
                        / 500.0
                        - 1.0
                })
                .collect()
        };
        let (q, k) = (vec(1), vec(7));
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        // Without a factor and with Phi-4-mini's sqrt(17/12) on the tables.
        for af in [1.0f32, (17.0f32 / 12.0).sqrt()] {
            let s = scaling(
                r#"{"rope_type": "longrope", "short_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "long_factor": [1.0], "original_max_position_embeddings": 4096, "max_position_embeddings": 131072}"#,
            );
            let sc = (af != 1.0).then_some(&s);
            let (sin, cos) = rope_caches_partial(20, hd, rd, 1e4, sc, 20);
            let want = dot(&hf(&q, 17, af), &hf(&k, 5, af));
            let got = dot(&engine(&q, 17, &sin, &cos), &engine(&k, 5, &sin, &cos));
            assert!(
                (want - got).abs() < 1e-3 * want.abs().max(1.0),
                "{want} {got} af {af}"
            );
            // Only the pass-through dims escape the factor: their part of
            // the score is unscaled in both.
            let pass_hf: f32 = (96..128).map(|i| q[i] * k[i]).sum();
            let eq = engine(&q, 17, &sin, &cos);
            let ek = engine(&k, 5, &sin, &cos);
            let pass_engine: f32 = (48..64).chain(112..128).map(|d| eq[d] * ek[d]).sum();
            assert!(
                (pass_hf - pass_engine).abs() < 1e-5,
                "{pass_hf} {pass_engine}"
            );
        }
        // The loader's row permutation: head blocks of a [rows, cols]
        // matrix and a per-row vector follow `perm`.
        let m = Bf16Slice::from((0..2 * hd * 3).map(|i| i as u16).collect::<Vec<u16>>());
        let pm = permute_head_rows(&m, 3, hd, &perm);
        for head in 0..2 {
            for e in 0..hd {
                let src = head * hd + perm[e];
                assert_eq!(
                    &pm[(head * hd + e) * 3..(head * hd + e + 1) * 3],
                    &m[src * 3..(src + 1) * 3]
                );
            }
        }
        let v: Vec<f32> = (0..hd).map(|i| i as f32).collect();
        let pv = permute_head_vec(&v, hd, &perm);
        assert_eq!(pv[48], 96.0);
        assert_eq!(pv[64], 48.0);
        assert!(permute_head_vec(&[], hd, &perm).is_empty());
    }

    /// One case of `testdata/rope_reference.json`
    /// (`tools/oracle/rope_reference.py`): a checkpoint's `config.json`,
    /// the inverse frequencies and attention factor transformers derives
    /// from it, and the `cos` / `sin` rows its rotary module returns at a
    /// few positions on both sides of the pretraining length.
    #[derive(Deserialize)]
    struct RopeCase {
        name: String,
        config: serde_json::Value,
        /// Which of the engine's tables the case is: `global`, or `local`
        /// for the table Gemma-3's sliding layers read.
        table: String,
        head_dim: usize,
        rotary_dim: usize,
        /// The length of the sequence the tables serve (longrope reads its
        /// long factors past the pretraining length) and the number of
        /// positions the reference covers, which need not be the same.
        seq_len: usize,
        table_len: usize,
        attention_factor: f32,
        inv_freq: Vec<f32>,
        positions: Vec<usize>,
        /// `[positions, rotary_dim]`, in HF's dim order.
        cos: Vec<Vec<f32>>,
        sin: Vec<Vec<f32>>,
    }

    #[derive(Deserialize)]
    struct RopeFixture {
        transformers: String,
        cases: Vec<RopeCase>,
    }

    /// Every table the engine builds for Phi-3.5-mini-instruct (longrope),
    /// Phi-4-mini-instruct (longrope over a partial rotation),
    /// google/gemma-3-4b-pt (a multimodal config whose text half scales
    /// `linear`, plus its unscaled local table) and three yarn
    /// configurations, against transformers 5.16's own tables.
    #[test]
    fn rope_tables_match_the_transformers_reference() {
        let fx: RopeFixture =
            serde_json::from_str(include_str!("../testdata/rope_reference.json")).unwrap();
        assert!(fx.transformers.starts_with('5'), "{}", fx.transformers);
        assert_eq!(fx.cases.len(), 10);
        // A relative gap, against the reference value.
        let rel = |a: f32, b: f32| {
            let (a, b) = (f64::from(a), f64::from(b));
            (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
        };
        for c in &fx.cases {
            let cfg = LlamaConfig::from_json(&c.config.to_string())
                .unwrap_or_else(|e| panic!("{}: {e}", c.name));
            let (hd, rd, local) = (cfg.head_dim(), cfg.rotary_dim(), c.table == "local");
            assert_eq!((hd, rd), (c.head_dim, c.rotary_dim), "{}", c.name);
            // A multimodal config parses through its text half.
            if c.config
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                == Some("gemma3")
            {
                assert_eq!(cfg.model_type.as_deref(), Some("gemma3_text"), "{}", c.name);
                assert_eq!(
                    cfg.weight_prefix.as_deref(),
                    Some("language_model.model."),
                    "{}",
                    c.name
                );
            }
            let spec = if local {
                rope_spec(
                    rd,
                    cfg.rope_local_base_freq.unwrap_or(1e4),
                    None,
                    c.table_len,
                )
            } else {
                rope_spec(rd, cfg.rope_theta, cfg.rope_scaling.as_ref(), c.seq_len)
            };
            assert!(
                rel(spec.attention_factor, c.attention_factor) < 1e-6,
                "{}: attention factor {} against {}",
                c.name,
                spec.attention_factor,
                c.attention_factor
            );
            assert_eq!(spec.inv_freq.len(), c.inv_freq.len(), "{}", c.name);
            for (i, (&e, &r)) in spec.inv_freq.iter().zip(&c.inv_freq).enumerate() {
                assert!(
                    rel(e, r) < 1e-6,
                    "{}: inv_freq[{i}] {e} against {r}",
                    c.name
                );
            }
            let caches = cfg.rope_caches_for(c.table_len, c.seq_len);
            let (sin, cos) = if local {
                (&caches.sin_local, &caches.cos_local)
            } else {
                (&caches.sin, &caches.cos)
            };
            assert_eq!(cos.len(), c.table_len * hd, "{}", c.name);
            // Engine dim `d` of a head holds HF dim `perm[d]` (the identity
            // without a partial rotation); the dims HF does not rotate read
            // cos 1 / sin 0 here and are compared as such.
            let perm = rope_head_perm(hd, rd);
            for (row, &p) in c.positions.iter().enumerate() {
                for d in 0..hd {
                    let src = perm[d];
                    let (rc, rs, slack) = if src < rd {
                        let j = src % (rd / 2);
                        // The two angles are the same f32 product of the
                        // same position and inverse frequency, up to the
                        // last ulp of `inv_freq`; no table can be closer
                        // than that (`|cos a - cos b| <= |a - b|`).
                        let (ae, ar) = (p as f32 * spec.inv_freq[j], p as f32 * c.inv_freq[j]);
                        let da = (f64::from(ae) - f64::from(ar)).abs();
                        (
                            c.cos[row][src],
                            c.sin[row][src],
                            da * f64::from(c.attention_factor),
                        )
                    } else {
                        (1.0, 0.0, 0.0)
                    };
                    // Plus the factor's own gap, which multiplies both tables.
                    let daf =
                        (f64::from(spec.attention_factor) - f64::from(c.attention_factor)).abs();
                    let tol = 1e-6 + slack + daf;
                    let (gc, gs) = (cos[p * hd + d], sin[p * hd + d]);
                    assert!(
                        (f64::from(gc) - f64::from(rc)).abs() < tol,
                        "{}: cos at position {p} dim {d} (HF dim {src}): {gc} against {rc}",
                        c.name
                    );
                    assert!(
                        (f64::from(gs) - f64::from(rs)).abs() < tol,
                        "{}: sin at position {p} dim {d} (HF dim {src}): {gs} against {rs}",
                        c.name
                    );
                }
            }
        }
    }

    #[test]
    fn multimodal_gemma3_config_flattens_to_its_text_config() {
        // google/gemma-3-4b-pt: six keys under text_config, the rest are
        // Gemma3TextConfig defaults; weights under language_model.model.
        let cfg = LlamaConfig::from_json(
            r#"{"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
            "text_config": {"hidden_size": 2560, "intermediate_size": 10240,
            "model_type": "gemma3_text", "num_hidden_layers": 34,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"}, "sliding_window": 1024},
            "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"}}"#,
        )
        .unwrap();
        assert_eq!(cfg.weight_prefix.as_deref(), Some("language_model.model."));
        assert_eq!(cfg.model_type.as_deref(), Some("gemma3_text"));
        assert!(cfg.is_gemma() && cfg.tied());
        assert_eq!((cfg.hidden_size, cfg.num_hidden_layers), (2560, 34));
        assert_eq!(
            (cfg.num_attention_heads, cfg.n_kv_heads(), cfg.head_dim()),
            (8, 4, 256)
        );
        assert_eq!((cfg.vocab_size, cfg.rope_theta), (262_208, 1e6));
        assert_eq!(cfg.rope_local_base_freq, Some(1e4));
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.activation().unwrap(), Activation::GeluTanh);
        assert_eq!(cfg.attention_scale(), 0.0625);
        assert_eq!(cfg.max_position_embeddings, Some(131_072));
        let full: Vec<usize> = (0..34).filter(|&li| !cfg.sliding(li)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29]);
        assert_eq!(cfg.layer_window(0), Some(1024));
        assert_eq!(cfg.layer_window(5), None);
        // The linear factor lands on the global table only.
        // Position 1, dim 0: angle 1/8 on the global table, 1 on the local.
        let r = cfg.rope_caches(4);
        assert!(
            (r.cos[256] - (0.125f32).cos()).abs() < 1e-6,
            "{}",
            r.cos[256]
        );
        assert!((r.cos_local[256] - (1.0f32).cos()).abs() < 1e-6);
        assert!(LlamaConfig::from_json(r#"{"model_type": "gemma3"}"#).is_err());
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

    /// A hand-built safetensors file: `a` (bf16 [2, 3]) at the 8-aligned
    /// data start, then one u8 byte, then `c` (bf16 [2]) at an odd offset,
    /// then `d` (f32 [2]).
    fn write_checkpoint(path: &Path, a: &[u16], c: &[u16], d: &[f32]) {
        let mut data = Vec::new();
        for v in a {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.push(7);
        for v in c {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let d_start = data.len();
        for v in d {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut header = format!(
            concat!(
                "{{\"a\":{{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,{a_end}]}},",
                "\"b\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[{a_end},{c_start}]}},",
                "\"c\":{{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[{c_start},{c_end}]}},",
                "\"d\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[{d_start},{d_end}]}}}}"
            ),
            a_end = a.len() * 2,
            c_start = a.len() * 2 + 1,
            c_end = d_start,
            d_start = d_start,
            d_end = data.len(),
        );
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn mapped_views_sub_views_and_fallbacks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let a: Vec<u16> = (0..6).map(|i| f32_to_bf16(i as f32 * 0.5)).collect();
        let c = [f32_to_bf16(-1.0), f32_to_bf16(2.0)];
        let d = [0.25f32, -8.0];
        write_checkpoint(&path, &a, &c, &d);
        let map = map_file(&path).unwrap();
        let st = SafeTensors::deserialize(&map).unwrap();
        let src = Src { map: &map, st: &st };

        // An aligned bf16 tensor is a view of the map: same bytes, no copy.
        let (va, shape) = tensor_bf16(src, "a").unwrap();
        assert_eq!(shape, [2, 3]);
        assert!(va.is_mapped());
        assert_eq!(&va[..], &a[..]);
        let in_map = map.as_ptr_range();
        assert!(in_map.contains(&va.as_ptr().cast::<u8>()));
        // Row blocks of a mapped tensor are views too; of an owned one, copies.
        let row1 = va.sub(3, 3);
        assert!(row1.is_mapped());
        assert_eq!(&row1[..], &a[3..]);
        assert_eq!(row1.as_ptr(), va[3..].as_ptr());
        let owned = Bf16Slice::from(a.clone());
        let part = owned.sub(1, 2);
        assert!(!part.is_mapped());
        assert_eq!(&part[..], &a[1..3]);
        // A clone shares the data (a tied LM head costs nothing).
        assert_eq!(va.clone().as_ptr(), va.as_ptr());
        // An odd data offset cannot be viewed as u16: an unaligned view
        // that copies its elements the first time it is read, and not
        // before.
        let (vc, shape) = tensor_bf16(src, "c").unwrap();
        assert_eq!(shape, [2]);
        assert!(!vc.is_mapped());
        assert!(!vc.is_materialised());
        assert_eq!(vc.len(), 2);
        assert_eq!(vc.owned_bytes(), 4);
        assert_eq!(&vc[..], &c[..]);
        assert!(vc.is_materialised());
        // f32 tensors are converted (and readable as f32 too).
        let (vd, _) = tensor_bf16(src, "d").unwrap();
        assert!(!vd.is_mapped());
        assert_eq!(&vd[..], &[f32_to_bf16(0.25), f32_to_bf16(-8.0)]);
        assert_eq!(tensor_f32(src, "d").unwrap().0, d);
        assert_eq!(tensor_f32(src, "a").unwrap().0[3], 1.5);
        // The views keep the map alive after the loader's handles go.
        drop(st);
        drop(map);
        assert_eq!(&row1[..], &a[3..]);
        assert_eq!(
            format!("{va:?}"),
            "Bf16Slice::Mapped(6 elements at byte 232)"
        );
        assert_eq!(
            format!("{vc:?}"),
            "Bf16Slice::Unaligned(2 elements at byte 245)"
        );
    }

    /// A safetensors file holding `mats` in the order given, each a bf16
    /// matrix; a one-byte `U8` tensor is inserted before one whose flag
    /// asks for an odd data offset (the header is padded to eight bytes
    /// but the tensors inside it are not padded at all, which is how 94
    /// tensors of the 70B distill come to start on an odd byte).
    fn write_matrices(path: &Path, mats: &[(&str, &[usize], &[u16], bool)]) {
        let mut data: Vec<u8> = Vec::new();
        let mut parts: Vec<String> = Vec::new();
        for (name, shape, v, odd) in mats {
            if *odd != (data.len() % 2 == 1) {
                let at = data.len();
                data.push(0);
                parts.push(format!(
                    "\"pad.{name}\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[{at},{}]}}",
                    data.len()
                ));
            }
            let at = data.len();
            for x in *v {
                data.extend_from_slice(&x.to_le_bytes());
            }
            let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
            parts.push(format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{at},{}]}}",
                dims.join(","),
                data.len()
            ));
        }
        let mut header = format!("{{{}}}", parts.join(","));
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes).unwrap();
    }

    /// A shard of a checkpoint whose tensors start on an odd byte copies
    /// its own rows and column block, once, and never the whole tensor.
    #[test]
    fn shard_of_odd_offset_tensors_copies_only_its_own_block() {
        // hidden 4, 2 heads of head_dim 2, 2 kv heads, intermediate 4.
        let cfg: LlamaConfig = serde_json::from_str(
            r#"{"hidden_size": 4, "intermediate_size": 4, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 8, "tie_word_embeddings": true}"#,
        )
        .unwrap();
        // Every matrix is [4, 4] and the embedding [8, 4]; the flag says
        // whether the tensor starts on an odd byte.
        let m: Vec<u16> = (0..16).collect();
        let e: Vec<u16> = (100..132).collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_matrices(
            &path,
            &[
                ("embed", &[8, 4], &e, false),
                ("wq", &[4, 4], &m, true),
                ("wk", &[4, 4], &m, false),
                ("wv", &[4, 4], &m, true),
                ("wo", &[4, 4], &m, true),
                ("wg", &[4, 4], &m, false),
                ("wu", &[4, 4], &m, true),
                ("wd", &[4, 4], &m, false),
            ],
        );
        let map = map_file(&path).unwrap();
        let st = SafeTensors::deserialize(&map).unwrap();
        let src = Src { map: &map, st: &st };
        let t = |n: &str| tensor_bf16(src, n).unwrap().0;
        let embed = t("embed");
        let full = LlamaWeights {
            embed: embed.clone(),
            layers: vec![LayerTensors {
                g1: vec![1.0; 4],
                g2: vec![1.0; 4],
                g_post_attn: Vec::new(),
                g_post_mlp: Vec::new(),
                wq: t("wq"),
                wk: t("wk"),
                wv: t("wv"),
                wo: t("wo"),
                wo_pitch: 0,
                bq: Vec::new(),
                bk: Vec::new(),
                bv: Vec::new(),
                qn: Vec::new(),
                kn: Vec::new(),
                wg: t("wg"),
                wu: t("wu"),
                wd: t("wd"),
                wd_pitch: 0,
            }],
            final_gamma: vec![1.0; 4],
            // Tied: the same view, counted once by `footprint`.
            lm_head: embed,
        };
        let l0 = &full.layers[0];
        assert!(!l0.wq.is_mapped() && !l0.wo.is_mapped());
        assert!(l0.wk.is_mapped() && l0.wd.is_mapped());
        let s1 = full.shard(&cfg, 1, 2);
        let l = &s1.layers[0];

        // The unaligned tensors of the full model are still not copied:
        // the shard narrowed them first.
        assert!(!l0.wq.is_materialised() && !l0.wo.is_materialised());
        // Rank 1's rows of q: elements 8..16, copied when they are read
        // and only then, and only those eight.
        assert!(!l.wq.is_materialised());
        assert_eq!(l.wq.len(), 8);
        assert_eq!(&l.wq[..], &m[8..]);
        assert!(l.wq.is_materialised());
        assert_eq!(l.wq.owned_bytes(), 16);
        // Rank 1's columns of o, gathered into its own block rather than
        // left as a strided view over the whole row range.
        assert_eq!((l.wo_pitch, l.wo.len()), (0, 8));
        assert_eq!(&l.wo[..], &[2u16, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!(l.wo.owned_bytes(), 16);
        // The 2-aligned tensors are unchanged: a row view and a strided
        // column view, neither of them copied.
        assert!(l.wk.is_mapped() && l.wd.is_mapped());
        assert_eq!((l.wk.owned_bytes(), l.wd.owned_bytes()), (0, 0));
        assert_eq!(&l.wk[..], &m[8..]);
        assert_eq!((l.wd_pitch, l.wd.len()), (4, 14));
        assert_eq!(
            reng_synapse::gather_columns(
                &l.wd,
                reng_synapse::Stride {
                    rows: 4,
                    cols: 2,
                    pitch: 4,
                }
            ),
            [2u16, 3, 6, 7, 10, 11, 14, 15]
        );
        // What the rank holds in buffers of its own: the four unaligned
        // matrices' halves, 8 elements each, and nothing more. Viewed in
        // place: the embedding (once, the head is tied), the two row
        // views of 8 elements and the strided column view of 14.
        assert_eq!(s1.footprint(), (64 + 16 + 16 + 28, 4 * 16));
        // Rank 0 is the other half of the same tensors.
        let s0 = full.shard(&cfg, 0, 2);
        assert_eq!(&s0.layers[0].wq[..], &m[..8]);
        assert_eq!(&s0.layers[0].wo[..], &[0u16, 1, 4, 5, 8, 9, 12, 13]);
        assert_eq!(s0.footprint(), s1.footprint());
    }

    /// A column block reads the window and nothing else, whatever the
    /// slice it comes from.
    #[test]
    fn column_block_gathers_from_every_form() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let m: Vec<u16> = (0..12).collect();
        write_matrices(
            &path,
            &[("odd", &[3, 4], &m, true), ("even", &[3, 4], &m, false)],
        );
        let map = map_file(&path).unwrap();
        let st = SafeTensors::deserialize(&map).unwrap();
        let src = Src { map: &map, st: &st };
        let want = [1u16, 2, 5, 6, 9, 10];
        for name in ["odd", "even"] {
            let v = tensor_bf16(src, name).unwrap().0;
            let b = v.column_block(3, 4, 1, 2);
            assert_eq!(&b[..], &want, "{name}");
            assert_eq!(b.owned_bytes(), 12);
        }
        let owned = Bf16Slice::from(m.clone());
        assert_eq!(&owned.column_block(3, 4, 1, 2)[..], &want);
        // A sub-view of an unaligned tensor is itself unaligned, and a
        // column block of it reads the right window.
        let odd = tensor_bf16(src, "odd").unwrap().0;
        let tail = odd.sub(4, 8);
        assert!(!tail.is_materialised());
        assert_eq!(&tail.column_block(2, 4, 2, 2)[..], &[6u16, 7, 10, 11]);
        assert!(!tail.is_materialised());
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
    #[test]
    fn tensor_parallel_shard_takes_rows_and_columns() {
        // hidden 4, 2 heads of head_dim 2, 2 kv heads, intermediate 4; two shards.
        let cfg: LlamaConfig = serde_json::from_str(
            r#"{"hidden_size": 4, "intermediate_size": 4, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 8, "tie_word_embeddings": true}"#,
        )
        .unwrap();
        let mat = |rows: usize, cols: usize| -> Bf16Slice {
            Bf16Slice::from((0..rows * cols).map(|i| i as u16).collect::<Vec<u16>>())
        };
        let layer = LayerTensors {
            g1: vec![1.0; 4],
            g2: vec![1.0; 4],
            g_post_attn: Vec::new(),
            g_post_mlp: Vec::new(),
            wq: mat(4, 4),
            wk: mat(4, 4),
            wv: mat(4, 4),
            wo: mat(4, 4),
            wo_pitch: 0,
            bq: (0..4).map(|i| i as f32).collect(),
            bk: Vec::new(),
            bv: Vec::new(),
            qn: Vec::new(),
            kn: Vec::new(),
            wg: mat(4, 4),
            wu: mat(4, 4),
            wd: mat(4, 4),
            wd_pitch: 0,
        };
        let w = LlamaWeights {
            embed: mat(8, 4),
            layers: vec![layer],
            final_gamma: vec![1.0; 4],
            lm_head: mat(8, 4),
        };
        let s1 = w.shard(&cfg, 1, 2);
        let c1 = cfg.shard(1, 2);
        assert_eq!(
            (
                c1.num_attention_heads,
                c1.n_kv_heads(),
                c1.intermediate_size
            ),
            (1, 1, 2)
        );
        let l = &s1.layers[0];
        // Rank 1 keeps rows 2..4 of q (its head): elements 8..16.
        assert_eq!(
            &l.wq[..],
            &(8..16).map(|i| i as u16).collect::<Vec<u16>>()[..]
        );
        assert_eq!(l.bq, vec![2.0, 3.0]);
        // o keeps columns 2..4 of every row: 2,3, 6,7, 10,11, 14,15.
        // These matrices are owned buffers, not views of a mapped file,
        // so the block is gathered (pitch 0) rather than left strided;
        // `shard_of_odd_offset_tensors_copies_only_its_own_block` covers
        // both forms.
        assert_eq!((l.wo_pitch, l.wo.len()), (0, 8));
        assert_eq!(&l.wo[..], &[2u16, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!((l.wd_pitch, l.wd.len()), (0, 8));
        assert_eq!(&l.wd[..], &[2u16, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!(l.wg.len(), 8);
        assert_eq!(
            &l.wg[..],
            &(8..16).map(|i| i as u16).collect::<Vec<u16>>()[..]
        );
        assert_eq!(s1.embed.len(), 32);
    }
}
