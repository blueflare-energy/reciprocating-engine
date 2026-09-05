//! One transformer decoder layer (pre-norm, RoPE, grouped-query attention,
//! SwiGLU MLP) as a fused SynapseAI recipe. The graph construction and the
//! launch/readback protocol live in `model.rs`; this module holds the layer's
//! weight description and a single-layer convenience entry point built on the
//! model runner so that every device path shares one readback implementation.

use reng_core::Result;

/// Weights and inputs of one decoder layer. The projections are bf16 in
/// the checkpoint's own `[out, in]` row-major layout (a HF `Linear`
/// weight as stored), so a loaded checkpoint is borrowed, never copied
/// or transposed; the small vectors are f32.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    /// Number of query heads; `hidden % n_heads == 0`.
    pub n_heads: usize,
    /// Number of key/value heads (GQA); `n_heads % n_kv_heads == 0`.
    pub n_kv_heads: usize,
    /// RMSNorm gains, each length `hidden`.
    pub g1: &'a [f32],
    pub g2: &'a [f32],
    /// Projections stored `[out, in]`, bf16: `wq`, `wo` are `hidden x
    /// hidden`; `wk`, `wv` are `(n_kv_heads * head_dim) x hidden`.
    pub wq: &'a [u16],
    pub wk: &'a [u16],
    pub wv: &'a [u16],
    pub wo: &'a [u16],
    /// Attention biases (Qwen2-style), `hidden` for `bq` and `n_kv_heads *
    /// head_dim` for `bk`/`bv`; empty when the model has none.
    pub bq: &'a [f32],
    pub bk: &'a [f32],
    pub bv: &'a [f32],
    /// MLP, bf16 `[out, in]`: `wg`, `wu` are `[inter, hidden]`; `wd` is
    /// `[hidden, inter]`.
    pub wg: &'a [u16],
    pub wu: &'a [u16],
    pub wd: &'a [u16],
    /// RoPE caches `[tokens, head_dim]` (head_dim contiguous), shared by heads.
    pub sin: &'a [f32],
    pub cos: &'a [f32],
    /// Attention scale (normally `1/sqrt(head_dim)`), folded into `wq`.
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
/// Panics if any buffer length disagrees with the sizes, if `hidden` is not a
/// multiple of `n_heads`, or if `n_heads` is not a multiple of `n_kv_heads`.
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
