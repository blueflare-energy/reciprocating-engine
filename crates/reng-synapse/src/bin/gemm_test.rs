//! Verify the general (non-square) gemm against a CPU reference at a shape
//! where m, k, n all differ and are all >= 128 (the reliable regime).
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-gemm-test`.

use reng_synapse::{gemm_bf16, gemm_cpu};

fn main() -> reng_core::Result<()> {
    let (m, k, n) = (256usize, 384usize, 512usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| (((i * 5 + 1) % 17) as f32 - 8.0) / 8.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (((i * 3 + 2) % 19) as f32 - 9.0) / 9.0)
        .collect();

    let hpu = gemm_bf16(&a, &b, m, k, n)?;
    let cpu = gemm_cpu(&a, &b, m, k, n);

    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| f64::from(*h - *c).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;

    println!("gemm_bf16 m={m} k={k} n={n}: rel_L2={rel:.4}");
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "gemm rel_L2 {rel} too high"
        )))
    }
}
