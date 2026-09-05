//! General (non-square) bf16 matmul on the Gaudi2 MME, plus a CPU reference.
//! The device work lives on [`crate::Device`]; this is the one-shot convenience
//! wrapper that acquires a device, runs a single gemm, and releases it.

use crate::Device;
use reng_core::Result;

/// Compute `C = A @ B` in bf16 on the MME: `a` row-major `[m,k]`, `b` row-major
/// `[k,n]`, result row-major `[m,n]` as f32. Acquires and releases a device for
/// the single call; for several ops reuse one [`Device`] instead.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if `a.len() != m*k` or `b.len() != k*n`.
pub fn gemm_bf16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    Device::acquire()?.gemm(a, b, m, k, n)
}

/// CPU reference `C = A @ B`, row-major f32: `a` is `[m,k]`, `b` is `[k,n]`.
#[must_use]
pub fn gemm_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}
