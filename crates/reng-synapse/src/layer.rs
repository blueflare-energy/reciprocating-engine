//! One transformer decoder layer (pre-norm, OLMo-2 post-norm or Gemma's
//! pre-and-post norms, RoPE, grouped-query attention, gated MLP) as a fused
//! SynapseAI recipe. The graph construction and the
//! launch/readback protocol live in `model.rs`; this module holds the layer's
//! weight description and a single-layer convenience entry point built on the
//! model runner so that every device path shares one readback implementation.

use crate::Activation;
use reng_core::Result;

/// Weights and inputs of one decoder layer. The projections are bf16 in
/// the checkpoint's own `[out, in]` row-major layout (a HF `Linear`
/// weight as stored), so a loaded checkpoint is borrowed, never copied
/// or transposed; the small vectors are f32.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA); `n_heads % n_kv_heads == 0`.
    pub n_kv_heads: usize,
    /// Per-head width. Normally `hidden / n_heads`; Qwen3 sets it in its
    /// config, and `n_heads * head_dim` (the q width) may differ from
    /// `hidden`.
    pub head_dim: usize,
    /// RMSNorm gains, each length `hidden`. With `post_norm` off, `g1`
    /// normalises the block input before attention and `g2` the residual
    /// before the MLP (Llama); with it on, `g1` normalises the attention
    /// output and `g2` the MLP output, each before its residual add
    /// (OLMo-2's `post_attention_layernorm` and
    /// `post_feedforward_layernorm`).
    pub g1: &'a [f32],
    pub g2: &'a [f32],
    /// Norm placement: false for pre-norm (`n = rms(x); h = x + attn(n)`),
    /// true for OLMo-2 post-norm (`h = x + rms(attn(x))`), as described at
    /// `g1`.
    pub post_norm: bool,
    /// Gemma's additional post norms: RMSNorm gains applied to the
    /// attention branch output and to the MLP branch output, each before
    /// its residual add, on top of the pre-norms `g1`/`g2`
    /// (`post_attention_layernorm` and `post_feedforward_layernorm` with
    /// `input_layernorm` and `pre_feedforward_layernorm` as `g1`/`g2`);
    /// empty when the model has none.
    pub g_post_attn: &'a [f32],
    pub g_post_mlp: &'a [f32],
    /// Projections stored `[out, in]`, bf16: `wq` is `(n_heads * head_dim)
    /// x hidden` and `wo` is `hidden x (n_heads * head_dim)`; `wk`, `wv`
    /// are `(n_kv_heads * head_dim) x hidden`.
    pub wq: &'a [u16],
    pub wk: &'a [u16],
    pub wv: &'a [u16],
    pub wo: &'a [u16],
    /// Attention biases (Qwen2-style), `n_heads * head_dim` for `bq` and
    /// `n_kv_heads * head_dim` for `bk`/`bv`; empty when the model has none.
    pub bq: &'a [f32],
    pub bk: &'a [f32],
    pub bv: &'a [f32],
    /// q/k RMSNorm gains applied after the projection and before RoPE;
    /// empty when the model has none. Length `head_dim`: the Qwen3 form,
    /// one RMS per query head (`qn`) and key head (`kn`), applied after
    /// the bias when there is one. Length `n_heads * head_dim` for `qn`
    /// and `n_kv_heads * head_dim` for `kn`: the OLMo-2 form, one RMS
    /// over the whole projected width before the head reshape (both gains
    /// take the same form, which excludes attention biases). With `qn`
    /// present the attention scale is folded into it, not into `wq`.
    pub qn: &'a [f32],
    pub kn: &'a [f32],
    /// Whether this layer applies RoPE to q and k (false for the NoPE
    /// layers of SmolLM3).
    pub use_rope: bool,
    /// Whether this layer's RoPE reads the model's second ("local") table
    /// (Gemma-3 sliding layers) instead of the first.
    pub local_rope: bool,
    /// Sliding window in positions: a query sees only the last `window`
    /// positions, its own included (Gemma sliding layers, Phi-3, Mistral);
    /// `None` is full causal attention.
    pub window: Option<usize>,
    /// The MLP gate activation.
    pub act: Activation,
    /// Gemma-2 attention softcap: the scores become `tanh(scores / cap) *
    /// cap` before the mask (the `1 / cap` is folded into the attention
    /// scale, then a `tanh` node and a multiply by `cap`); `None` for no
    /// softcap.
    pub attn_softcap: Option<f32>,
    /// MLP, bf16 `[out, in]`: `wg`, `wu` are `[inter, hidden]`; `wd` is
    /// `[hidden, inter]`.
    pub wg: &'a [u16],
    pub wu: &'a [u16],
    pub wd: &'a [u16],
    /// RoPE caches `[tokens, head_dim]` (head_dim contiguous), shared by heads.
    pub sin: &'a [f32],
    pub cos: &'a [f32],
    /// Attention scale (normally `1/sqrt(head_dim)`), folded into `wq` (or
    /// into `qn` when present).
    pub scale: f32,
    pub eps: f32,
}

/// Run one fused decoder layer (non-causal attention) on `x`
/// (`[tokens, hidden]`, row-major) and return `out` (`[tokens, hidden]`) as
/// f32. Built as a one-layer model probe, so it uses the same graph builder and
/// complete-readback protocol as the full model.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails or the output never completes.
///
/// # Panics
///
/// Panics if any buffer length disagrees with the sizes or if `n_heads` is
/// not a multiple of `n_kv_heads`.
pub fn decoder_layer_bf16(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let m = crate::ModelWeights {
        layers: vec![*w],
        final_gamma: &[],
        lm_head: &[],
        final_softcap: None,
    };
    crate::model_probe_bf16(x, &m, tokens, hidden, inter, false, 0)
}

/// CPU reference for [`decoder_layer_bf16`] (f32, same layouts, non-causal).
#[must_use]
pub fn decoder_layer_cpu(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    crate::layer_cpu(x, w, tokens, hidden, inter, false)
}
