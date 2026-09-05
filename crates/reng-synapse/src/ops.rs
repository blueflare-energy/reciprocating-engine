//! Elementwise / normalization ops on the Gaudi2, plus CPU references. The
//! device work lives on [`crate::Device`]; these are one-shot convenience
//! wrappers that acquire a device, run one op, and release it.

use crate::Device;
use reng_core::Result;

/// Row-wise softmax of a `rows x cols` matrix in bf16 on the HPU (softmax over
/// the `cols` axis), via `softmax_fwd_bf16`. Returns f32. Acquires and releases
/// a device for the single call; for several ops reuse one [`Device`] instead.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `input.len() != rows*cols`.
pub fn softmax_bf16(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    Device::acquire()?.softmax(input, rows, cols)
}

/// RMSNorm over the feature axis via `rms_norm_fwd_bf16`. `x` is row-major
/// `[features, tokens]`, `gamma` is length `features`. Acquires and releases a
/// device for the single call.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `x.len() != features*tokens` or `gamma.len() != features`.
pub fn rms_norm_bf16(
    x: &[f32],
    gamma: &[f32],
    features: usize,
    tokens: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    Device::acquire()?.rms_norm(x, gamma, features, tokens, eps)
}

/// CPU reference RMSNorm over the feature axis. `x` is laid out with `features`
/// as the contiguous (FCD) dimension and `tokens` as the outer dimension, i.e.
/// row-major `[tokens, features]` (`x[t*features + f]`); `gamma` is length
/// `features`. Matches the on-device tensor layout used by [`crate::Device`].
#[must_use]
pub fn rms_norm_cpu(
    x: &[f32],
    gamma: &[f32],
    features: usize,
    tokens: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; features * tokens];
    for t in 0..tokens {
        let base = t * features;
        let mut ms = 0.0f32;
        for f in 0..features {
            let v = x[base + f];
            ms += v * v;
        }
        ms /= features as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for f in 0..features {
            out[base + f] = x[base + f] * inv * gamma[f];
        }
    }
    out
}

/// Elementwise SiLU (swish) `y = x * sigmoid(x)` in bf16 on the HPU. `x` has
/// `rows*cols` elements. Acquires and releases a device for the single call.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `x.len() != rows*cols`.
pub fn silu_bf16(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    Device::acquire()?.silu(x, rows, cols)
}

/// CPU reference elementwise SiLU `y = x * sigmoid(x)`, f32.
#[must_use]
pub fn silu_cpu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// CPU reference row-wise softmax over `cols`, f32.
#[must_use]
pub fn softmax_cpu(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..r * cols + cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &v) in out[r * cols..r * cols + cols].iter_mut().zip(row) {
            let e = (v - m).exp();
            *o = e;
            sum += e;
        }
        for o in &mut out[r * cols..r * cols + cols] {
            *o /= sum;
        }
    }
    out
}
