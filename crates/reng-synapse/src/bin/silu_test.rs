//! Verify our direct SiLU (`x * sigmoid(x)`, fused sigmoid+mult) against a CPU
//! reference at a reliable size.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-silu-test -- [rows] [cols]`.

use reng_synapse::{silu_bf16, silu_cpu};

fn main() -> reng_core::Result<()> {
    let rows = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let cols = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);

    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (((i * 13 + 5) % 41) as f32 - 20.0) / 5.0)
        .collect();

    println!("silu: rows={rows}, cols={cols}");
    let hpu = silu_bf16(&x, rows, cols)?;
    let cpu = silu_cpu(&x);

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
            "silu diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
