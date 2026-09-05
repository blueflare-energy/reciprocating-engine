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
