//! Run a single bf16 matmul on the Gaudi2 MME and verify it against the CPU.
//!
//! Build and run on a Gaudi host with the SynapseAI stack:
//! `cargo run -p reng-synapse --features link-synapse --bin reng-hello-mme`.

use reng_synapse::{matmul_bf16, matmul_cpu};

fn main() -> reng_core::Result<()> {
    let (m, k, n) = (128usize, 256usize, 64usize);

    // Deterministic, small-magnitude inputs so bf16 rounding stays well-behaved.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    println!("running {m}x{k} @ {k}x{n} bf16 matmul on the Gaudi2 MME...");
    let hpu = matmul_bf16(&a, &b, m, k, n)?;
    let cpu = matmul_cpu(&a, &b, m, k, n);

    // L2-norm relative error is the right metric for a matmul: per-element
    // relative error is meaningless for near-zero outputs, where bf16 rounding
    // dominates. bf16 inputs with FP32 accumulation give a small norm error.
    let mut max_abs = 0.0f32;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (h, c) in hpu.iter().zip(cpu.iter()) {
        max_abs = max_abs.max((h - c).abs());
        let d = f64::from(h - c);
        num += d * d;
        den += f64::from(*c) * f64::from(*c);
    }
    let rel_l2 = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    println!("max_abs_err = {max_abs:.5}   rel_L2_err = {rel_l2:.5}");
    println!("hpu[0..4] = {:?}", &hpu[0..4]);
    println!("cpu[0..4] = {:?}", &cpu[0..4]);

    if rel_l2 < 0.02 {
        println!("PASS: HPU MME matches the CPU reference within bf16 tolerance");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "FAIL: rel_L2_err {rel_l2:.4} exceeds tolerance (layout or dtype bug?)"
        )))
    }
}
