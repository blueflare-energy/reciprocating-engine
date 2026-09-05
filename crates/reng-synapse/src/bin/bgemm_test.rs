//! Pin the contract of the `batch_gemm` guid: 3-D operands with the batch
//! outermost, the `gemm` transpose flags, and whether a batch-1 operand
//! broadcasts against a batched one (what grouped-query attention needs: one
//! K/V head serving several query heads in one node).
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-bgemm-test -- [case]`
//! where `case` is 0: plain batch, 1: transpose_b batch, 2: B broadcast,
//! 3: B broadcast with transpose_b (default: all).

use reng_synapse::{NodeInput, run_node, synGEMMParams};

/// `C[b][m,n] = op(A[b]) @ op(B[b or 0])`, row-major per batch.
#[allow(clippy::too_many_arguments)]
fn cpu(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    b_batch: usize,
    ta: bool,
    tb: bool,
) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let ab = &a[bi * m * k..(bi + 1) * m * k];
        let bb_i = if b_batch == 1 { 0 } else { bi };
        let bb = &b[bb_i * k * n..(bb_i + 1) * k * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    let av = if ta { ab[p * m + i] } else { ab[i * k + p] };
                    let bv = if tb { bb[j * k + p] } else { bb[p * n + j] };
                    s += av * bv;
                }
                c[bi * m * n + i * n + j] = s;
            }
        }
    }
    c
}

fn rel(h: &[f32], c: &[f32]) -> f32 {
    let num: f64 = h
        .iter()
        .zip(c)
        .map(|(x, y)| f64::from(*x - *y).powi(2))
        .sum();
    let den: f64 = c.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn main() -> reng_core::Result<()> {
    // Attention-like shapes: 3 heads, 32 query rows, head_dim 64, 512 keys.
    let (batch, m, k, n) = (3usize, 32usize, 64usize, 512usize);
    let a: Vec<f32> = (0..batch * m * k)
        .map(|i| (((i * 5 + 1) % 17) as f32 - 8.0) / 8.0)
        .collect();
    let b_full: Vec<f32> = (0..batch * k * n)
        .map(|i| (((i * 3 + 2) % 19) as f32 - 9.0) / 9.0)
        .collect();
    let cases = [(false, false), (false, true), (true, false), (true, true)];
    let selected: Vec<usize> = match std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(i) if i < cases.len() => vec![i],
        _ => (0..cases.len()).collect(),
    };
    let mut worst = 0.0f32;
    for ci in selected {
        let (broadcast, tb) = cases[ci];
        let b_batch = if broadcast { 1 } else { batch };
        let b = &b_full[..b_batch * k * n];
        let (mm, kk, nn) = (m as u64, k as u64, n as u64);
        // Device sizes are FCD-first with the batch outermost.
        let a_sizes = [kk, mm, batch as u64];
        let b_sizes = if tb {
            [kk, nn, b_batch as u64]
        } else {
            [nn, kk, b_batch as u64]
        };
        let params = synGEMMParams {
            transpose_a: false,
            transpose_b: tb,
        };
        let res = run_node(
            "batch_gemm",
            &[
                NodeInput {
                    name: "A",
                    sizes: &a_sizes,
                    data: &a,
                    raw: None,
                },
                NodeInput {
                    name: "B",
                    sizes: &b_sizes,
                    data: b,
                    raw: None,
                },
            ],
            &[nn, mm, batch as u64],
            (&raw const params).cast::<core::ffi::c_void>(),
            core::mem::size_of::<synGEMMParams>() as u32,
        );
        match res {
            Ok(hpu) => {
                let ref_c = cpu(&a, b, m, k, n, batch, b_batch, false, tb);
                let r = rel(&hpu, &ref_c);
                worst = worst.max(r);
                println!(
                    "case {ci}: B batch {b_batch}, transpose_b={tb}: rel_L2={r:.4} {}",
                    if r < 0.02 { "ok" } else { "MISMATCH" }
                );
            }
            Err(e) => {
                worst = 1.0;
                println!("case {ci}: B batch {b_batch}, transpose_b={tb}: FAILED ({e})");
            }
        }
    }
    if worst < 0.02 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "batch_gemm contract check failed (worst rel_L2 {worst})"
        )))
    }
}
