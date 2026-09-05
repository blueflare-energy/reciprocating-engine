//! Load a Llama-family model from a HuggingFace directory (`config.json` +
//! `model.safetensors`, single file or shards) and run it on Gaudi2 through the fused-graph engine
//! in `reng-synapse`: one-shot prefill, or KV-cached generation through
//! [`Generator`].
//!
//! Layout conventions: HF stores every `nn.Linear` weight as `[out, in]` and
//! computes `y = x @ W^T`; the engine wants `[in, out]`, so projections are
//! transposed once at load time. The embedding table stays `[vocab, hidden]`
//! for the host-side gather; a tied LM head is its transpose, `[hidden, vocab]`.
//! Qwen2-style attention biases are loaded when present (Llama has none).

use reng_core::{Error, Result};
use reng_synapse::{bf16_to_f32, f32_to_bf16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::path::Path;

fn default_theta() -> f32 {
    10000.0
}

/// The subset of a HF Llama-style `config.json` the engine needs.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    /// HF `rope_scaling`; only the `llama3` type is applied (the others are
    /// reported as unsupported by [`LlamaConfig::load`]).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
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

impl LlamaConfig {
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
        Ok(cfg)
    }

    #[must_use]
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// One layer's weights in engine layout (`[in, out]` projections), owned.
pub struct LayerTensors {
    pub g1: Vec<f32>,
    pub g2: Vec<f32>,
    /// Projections as stored, bf16 `[out, in]`.
    pub wq: Vec<u16>,
    pub wk: Vec<u16>,
    pub wv: Vec<u16>,
    pub wo: Vec<u16>,
    /// Attention biases; empty when the checkpoint has none.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub wg: Vec<u16>,
    pub wu: Vec<u16>,
    pub wd: Vec<u16>,
}

/// A whole model's weights on the host: the matrices bf16 in the
/// checkpoint's `[out, in]` layout (the device format, uploaded as is),
/// the norm gains and biases f32.
pub struct LlamaWeights {
    /// `[vocab, hidden]`, bf16, row per token id.
    pub embed: Vec<u16>,
    pub layers: Vec<LayerTensors>,
    pub final_gamma: Vec<f32>,
    /// `[vocab, hidden]`, bf16 (the tied embeddings themselves when the
    /// checkpoint has no head).
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

/// `[rows, cols]` row-major to `[cols, rows]`.
/// A 1-D tensor that a checkpoint may omit (attention biases): empty when
/// absent, checked for length when present.
fn optional_vec(st: &SafeTensors<'_>, name: &str, len: usize) -> Result<Vec<f32>> {
    if st.tensor(name).is_err() {
        return Ok(Vec::new());
    }
    let (v, shape) = tensor_f32(st, name)?;
    if shape != [len] {
        return Err(Error::Other(format!(
            "tensor {name}: shape {shape:?}, expected [{len}]"
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
/// directory into engine layout.
///
/// # Errors
///
/// Returns an error if the files are missing, a tensor is absent or has an
/// unexpected shape or dtype, or the LM head is untied and absent.
pub fn load_weights(dir: &Path, cfg: &LlamaConfig) -> Result<LlamaWeights> {
    let shards = Shards::open(dir)?;
    let parsed = shards.parse()?;
    let st = |name: &str| shards.shard(&parsed, name);
    let (h, i, v) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size);
    let kvd = cfg.n_kv_heads() * cfg.head_dim();

    let (embed, eshape) =
        tensor_bf16(st("model.embed_tokens.weight"), "model.embed_tokens.weight")?;
    if eshape != [v, h] {
        return Err(Error::Other(format!(
            "embed_tokens shape {eshape:?}, expected [{v}, {h}]"
        )));
    }
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        let g = |name: &str| -> Result<Vec<f32>> {
            let n = p(name);
            Ok(tensor_f32(st(&n), &n)?.0)
        };
        let lin = |name: &str, o: usize, inp: usize| -> Result<Vec<u16>> {
            let n = p(name);
            linear(st(&n), &n, o, inp)
        };
        let opt = |name: &str, len: usize| -> Result<Vec<f32>> {
            let n = p(name);
            optional_vec(st(&n), &n, len)
        };
        layers.push(LayerTensors {
            g1: g("input_layernorm.weight")?,
            g2: g("post_attention_layernorm.weight")?,
            wq: lin("self_attn.q_proj.weight", h, h)?,
            wk: lin("self_attn.k_proj.weight", kvd, h)?,
            wv: lin("self_attn.v_proj.weight", kvd, h)?,
            wo: lin("self_attn.o_proj.weight", h, h)?,
            bq: opt("self_attn.q_proj.bias", h)?,
            bk: opt("self_attn.k_proj.bias", kvd)?,
            bv: opt("self_attn.v_proj.bias", kvd)?,
            wg: lin("mlp.gate_proj.weight", i, h)?,
            wu: lin("mlp.up_proj.weight", i, h)?,
            wd: lin("mlp.down_proj.weight", h, i)?,
        });
    }
    let final_gamma = tensor_f32(st("model.norm.weight"), "model.norm.weight")?.0;
    let has_head = shards.index.contains_key("lm_head.weight")
        || (shards.index.is_empty() && parsed[0].tensor("lm_head.weight").is_ok());
    let lm_head = if has_head {
        linear(st("lm_head.weight"), "lm_head.weight", v, h)?
    } else if cfg.tie_word_embeddings {
        embed.clone()
    } else {
        return Err(Error::Other(
            "lm_head.weight missing and embeddings are not tied".into(),
        ));
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

/// Host-side embedding gather: `[tokens, hidden]` for the given ids.
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

/// Borrow `w` as the engine's per-layer weight views, with `sin`/`cos`
/// RoPE tables and the attention scale folded in.
#[cfg(feature = "link-synapse")]
fn layer_views<'a>(
    w: &'a LlamaWeights,
    cfg: &LlamaConfig,
    sin: &'a [f32],
    cos: &'a [f32],
) -> reng_synapse::ModelWeights<'a> {
    use reng_synapse::{LayerWeights, ModelWeights};
    let hd = cfg.head_dim();
    let layers: Vec<LayerWeights<'a>> = w
        .layers
        .iter()
        .map(|l| LayerWeights {
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.n_kv_heads(),
            g1: &l.g1,
            g2: &l.g2,
            wq: &l.wq,
            wk: &l.wk,
            wv: &l.wv,
            wo: &l.wo,
            bq: &l.bq,
            bk: &l.bk,
            bv: &l.bv,
            wg: &l.wg,
            wu: &l.wu,
            wd: &l.wd,
            sin,
            cos,
            scale: 1.0 / (hd as f32).sqrt(),
            eps: cfg.rms_norm_eps,
        })
        .collect();
    ModelWeights {
        layers,
        final_gamma: &w.final_gamma,
        lm_head: &w.lm_head,
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
    let (sin, cos) = rope_caches_scaled(
        tokens,
        cfg.head_dim(),
        cfg.rope_theta,
        cfg.rope_scaling.as_ref(),
    );
    let x = embed_tokens(w, cfg, &padded);
    let m = layer_views(w, cfg, &sin, &cos);
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
        let (sin, cos) = rope_caches_scaled(
            capacity,
            cfg.head_dim(),
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        );
        // The cached recipes take RoPE rows as per-step inputs, so the layer
        // views carry no tables (they would have to outlive `sin`).
        let m = layer_views(w, cfg, &[], &[]);
        let model = reng_synapse::CachedModel::new(
            &m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            rows,
            decode_rows,
            capacity,
            &sin,
            &cos,
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
        let (sin, cos) = rope_caches_scaled(
            capacity,
            cfg.head_dim(),
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
        );
        // The batched recipes take RoPE rows as per-step inputs, so the
        // layer views carry no tables (they would have to outlive `sin`).
        let m = layer_views(w, cfg, &[], &[]);
        let model = reng_synapse::BatchedModel::new(
            m,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
            batch,
            rows,
            capacity,
            &sin,
            &cos,
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
