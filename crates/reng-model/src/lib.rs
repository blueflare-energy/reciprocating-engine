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
//!
//! The safetensors files are memory-mapped and never read into heap
//! buffers: a bf16 tensor is a [`Bf16Slice`] viewing the checkpoint bytes
//! in place (the maps stay alive for as long as any view does), so loading
//! costs no copy and the file pages are shared with the page cache and
//! reclaimable. Only derived data is owned: f32 and f16 checkpoints
//! converted to bf16, the scaled Granite copies, and a tensor whose data
//! offset is not 2-aligned (never seen; safetensors pads its header to 8
//! bytes) or on a big-endian host.

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
}

impl Bf16Slice {
    /// A view of `len` bf16 elements at byte `offset` of `map` when the
    /// elements can be read in place (the offset is 2-aligned and the host
    /// is little-endian, as the file format is); otherwise a converted
    /// copy.
    ///
    /// # Panics
    ///
    /// Panics if the range lies outside the map.
    #[must_use]
    pub fn mapped(map: Arc<Mmap>, offset: usize, len: usize) -> Self {
        let bytes = &map[offset..offset + len * 2];
        if cfg!(target_endian = "little") && bytes.as_ptr().align_offset(align_of::<u16>()) == 0 {
            return Self::Mapped { map, offset, len };
        }
        let v: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Self::from(v)
    }

    /// Elements `start..start + len` as a slice of their own: a sub-view
    /// of a mapped slice (no copy), a copy of an owned one.
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
        }
    }

    /// Whether the elements are read from a mapped file.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
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
    /// config's `residual_multiplier` when it has one.
    pub wq: Bf16Slice,
    pub wk: Bf16Slice,
    pub wv: Bf16Slice,
    pub wo: Bf16Slice,
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
    /// files and bytes held in owned buffers (converted, scaled or split
    /// copies; a tied head counts once).
    #[must_use]
    pub fn footprint(&self) -> (usize, usize) {
        let (mut mapped, mut owned) = (0, 0);
        let mut seen: Vec<*const u16> = Vec::new();
        for m in self.matrices() {
            let p = m.as_ptr();
            if seen.contains(&p) {
                continue;
            }
            seen.push(p);
            if m.is_mapped() {
                mapped += m.len() * 2;
            } else {
                owned += m.len() * 2;
            }
        }
        (mapped, owned)
    }

    /// The weights of tensor-parallel shard `rank` of `world` (Megatron
    /// split): the q/k/v and gate/up projections keep the rows of this
    /// rank's heads and MLP columns (views into the mapped checkpoint,
    /// since `[out, in]` rows are contiguous), the o and down projections
    /// keep the matching columns (gathered into owned copies), the biases
    /// and the OLMo-2 full-width q/k gains are sliced the same way, and
    /// the norms, the embedding and the LM head are shared. `cfg` is the
    /// unsharded config; the shard's config is [`LlamaConfig::shard`].
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
        let cols = |m: &Bf16Slice, out_rows: usize, all_cols: usize, part: usize| -> Bf16Slice {
            assert_eq!(m.len(), out_rows * all_cols, "matrix shape");
            let mut v = Vec::with_capacity(out_rows * part);
            for r in 0..out_rows {
                let base = r * all_cols + rank * part;
                v.extend_from_slice(&m[base..base + part]);
            }
            Bf16Slice::from(v)
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
            .map(|l| LayerTensors {
                g1: l.g1.clone(),
                g2: l.g2.clone(),
                g_post_attn: l.g_post_attn.clone(),
                g_post_mlp: l.g_post_mlp.clone(),
                wq: rows(&l.wq, q_all, q_rows, h),
                wk: rows(&l.wk, kv_all, kv_rows, h),
                wv: rows(&l.wv, kv_all, kv_rows, h),
                wo: cols(&l.wo, h, q_all, q_rows),
                bq: vec_part(&l.bq, q_all, q_rows),
                bk: vec_part(&l.bk, kv_all, kv_rows),
                bv: vec_part(&l.bv, kv_all, kv_rows),
                // Per-head gains (length head_dim) are the same for every
                // head; full-width gains follow the projection rows.
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
                wd: cols(&l.wd, h, i_all, i_rows),
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
        // An odd data offset cannot be viewed as u16: converted copy.
        let (vc, shape) = tensor_bf16(src, "c").unwrap();
        assert_eq!(shape, [2]);
        assert!(!vc.is_mapped());
        assert_eq!(&vc[..], &c[..]);
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
            bq: (0..4).map(|i| i as f32).collect(),
            bk: Vec::new(),
            bv: Vec::new(),
            qn: Vec::new(),
            kn: Vec::new(),
            wg: mat(4, 4),
            wu: mat(4, 4),
            wd: mat(4, 4),
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
        assert_eq!(&l.wo[..], &[2u16, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!(&l.wd[..], &[2u16, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!(l.wg.len(), 8);
        assert_eq!(
            &l.wg[..],
            &(8..16).map(|i| i as u16).collect::<Vec<u16>>()[..]
        );
        assert_eq!(s1.embed.len(), 32);
    }
}
