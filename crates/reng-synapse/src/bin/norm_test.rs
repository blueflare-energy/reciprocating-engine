//! Verify our direct RMSNorm (`rms_norm_fwd_bf16`) against a CPU reference at a
//! reliable size.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-norm-test -- [features] [tokens]`.

use reng_synapse::{rms_norm_bf16, rms_norm_cpu};

fn main() -> reng_core::Result<()> {
    let features = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let tokens = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let eps = 1e-6f32;

    let x: Vec<f32> = (0..features * tokens)
        .map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) / 7.0)
        .collect();
    let gamma: Vec<f32> = (0..features)
        .map(|i| 0.5 + ((i % 11) as f32) / 10.0)
        .collect();

    println!("rms_norm: features={features}, tokens={tokens}");
    let hpu = rms_norm_bf16(&x, &gamma, features, tokens, eps)?;
    let cpu = rms_norm_cpu(&x, &gamma, features, tokens, eps);

    let nan = hpu.iter().any(|v| !v.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| f64::from(*h - *c).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;

    println!("nan={nan}  rel_L2={rel:.4}");
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "rms_norm diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
