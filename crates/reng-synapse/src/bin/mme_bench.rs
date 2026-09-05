//! Measure steady-state bf16 matmul throughput on the Gaudi2 MME and compare
//! it to the roofline. Compile once, launch many.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-mme-bench -- [m] [k] [n] [iters]`
//! (default 4096 x 4096 x 4096, 50 launches; the prefill shapes are
//! `1024 2048 8192` and friends).

use reng_synapse::MatmulHpu;
use std::time::Instant;

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (m, k, n) = (arg(1, 4096usize), arg(2, 4096usize), arg(3, 4096usize));
    let iters = arg(4, 50usize) as u32;
    const CEILING_TFLOPS: f64 = 432.0; // Gaudi2 MME peak, bf16

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();

    println!("compiling {m}x{k} @ {k}x{n} bf16 matmul recipe...");
    let mm = MatmulHpu::new(m, k, n)?;

    let out = mm.run(&a, &b)?;

    // Cheap correctness check: recompute only C[0,0] on the CPU.
    let mut c00 = 0.0f32;
    for p in 0..k {
        c00 += a[p] * b[p * n];
    }
    let rel = (out[0] - c00).abs() / c00.abs().max(1e-6);
    println!("C[0,0] hpu={:.4} cpu={c00:.4} rel_err={rel:.4}", out[0]);
    if rel > 0.03 {
        return Err(reng_core::Error::Other(format!(
            "correctness check failed: C[0,0] rel_err {rel:.4}"
        )));
    }

    for _ in 0..5 {
        mm.launch_only()?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        mm.launch_only()?;
    }
    let dt = t0.elapsed().as_secs_f64();

    let flop = 2.0 * (m as f64) * (k as f64) * (n as f64) * f64::from(iters);
    let tflops = flop / dt / 1e12;
    let per_ms = dt / f64::from(iters) * 1e3;
    println!("{iters} launches in {dt:.3}s -> {per_ms:.3} ms/matmul");
    println!(
        "throughput = {tflops:.1} TFLOP/s ({:.1}% of {CEILING_TFLOPS} TFLOP/s bf16 MME ceiling)",
        tflops / CEILING_TFLOPS * 100.0
    );
    Ok(())
}
