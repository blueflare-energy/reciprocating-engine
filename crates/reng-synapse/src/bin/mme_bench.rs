//! Steady-state bf16 gemm throughput on the Gaudi2 MME against the roofline:
//! one `gemm` node compiled once and launched many times, at the shape given.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-mme-bench -- [m] [k] [n] [iters] [transpose_b]`
//! (default 4096 x 4096 x 4096, 100 launches, weights `[n, k]` as the engine
//! stores them; `transpose_b` 1 stores them `[k, n]`). The prefill gemms are
//! `1024 2048 8192` and friends.

use core::ffi::c_void;
use reng_synapse::{NodeInput, bench_node, synGEMMParams};

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (m, k, n) = (arg(1, 4096usize), arg(2, 4096usize), arg(3, 4096usize));
    let iters = arg(4, 100usize);
    let tb = arg(5, 0usize) == 1;
    const CEILING_TFLOPS: f64 = 432.0; // Gaudi2 MME peak, bf16

    // A is `[k, m]` (FCD k), B is `[n, k]` or, transposed, `[k, n]`; C is
    // `[n, m]`; all bf16-exact values so the spot check is tight.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b_sizes: [u64; 2] = if tb {
        [k as u64, n as u64]
    } else {
        [n as u64, k as u64]
    };
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b: tb,
    };
    let ins = [
        NodeInput {
            name: "A",
            sizes: &[k as u64, m as u64],
            data: &a,
            raw: None,
        },
        NodeInput {
            name: "B",
            sizes: &b_sizes,
            data: &b,
            raw: None,
        },
    ];
    let (secs, row) = bench_node(
        "gemm",
        &ins,
        &[n as u64, m as u64],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        iters,
    )?;

    // Check the first output row, C[:, 0] = sum_p A[p, 0] B(p, :), against
    // the CPU (A is `[k, m]` with FCD k; B(p, j) is `b[j + p n]` as stored
    // `[n, k]` and `b[p + j k]` as stored `[k, n]`).
    let mut cpu = vec![0.0f32; n];
    for p in 0..k {
        let ap = a[p];
        for (j, c) in cpu.iter_mut().enumerate() {
            *c += ap * if tb { b[p + j * k] } else { b[j + p * n] };
        }
    }
    let num: f64 = row
        .iter()
        .zip(&cpu)
        .map(|(h, c)| f64::from(*h - *c).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    if rel > 0.03 {
        return Err(reng_core::Error::Other(format!(
            "correctness check failed: first row rel_L2 {rel:.4} (hpu[0]={} cpu[0]={})",
            row[0], cpu[0]
        )));
    }
    let flop = 2.0 * (m as f64) * (k as f64) * (n as f64);
    let tflops = flop / secs / 1e12;
    let bytes = 2.0 * ((m * k) + (k * n) + (m * n)) as f64;
    println!(
        "gemm m={m} k={k} n={n} transpose_b={tb}: {:.3} ms/launch, {tflops:.1} TFLOP/s ({:.1}% of {CEILING_TFLOPS} TFLOP/s), {:.2} TB/s moved",
        secs * 1e3,
        tflops / CEILING_TFLOPS * 100.0,
        bytes / secs / 1e12
    );
    Ok(())
}
