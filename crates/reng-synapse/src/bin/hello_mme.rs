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

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (h, c) in hpu.iter().zip(cpu.iter()) {
        let ad = (h - c).abs();
        max_abs = max_abs.max(ad);
        max_rel = max_rel.max(ad / c.abs().max(1e-3));
    }
    println!("max_abs_err = {max_abs:.5}   max_rel_err = {max_rel:.5}");
    println!("hpu[0..4] = {:?}", &hpu[0..4]);
    println!("cpu[0..4] = {:?}", &cpu[0..4]);

    if max_rel < 0.05 {
        println!("PASS: HPU MME matches the CPU reference within bf16 tolerance");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "FAIL: max_rel_err {max_rel:.4} exceeds tolerance (layout or dtype bug?)"
        )))
    }
}
