//! Verify the fused SwiGLU MLP block against a CPU reference. This is the first
//! multi-gemm on-device-dataflow test: three matmuls + SiLU + gating + residual
//! in one recipe, one launch, one readback.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-mlp-test -- [tokens] [hidden] [inter]`.

use reng_synapse::{swiglu_mlp_bf16, swiglu_mlp_cpu};

fn main() -> reng_core::Result<()> {
    let tokens = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let hidden = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let inter = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(512usize);

    // Small magnitudes so the SwiGLU output stays O(1) and bf16-representable.
    let s = 1.0 / (hidden as f32).sqrt();
    let x: Vec<f32> = (0..tokens * hidden)
        .map(|j| (((j * 7 + 3) % 23) as f32 - 11.0) / 11.0)
        .collect();
    let wgate: Vec<f32> = (0..hidden * inter)
        .map(|j| ((((j * 5 + 1) % 17) as f32 - 8.0) / 8.0) * s)
        .collect();
    let wup: Vec<f32> = (0..hidden * inter)
        .map(|j| ((((j * 11 + 4) % 19) as f32 - 9.0) / 9.0) * s)
        .collect();
    let wdown: Vec<f32> = (0..inter * hidden)
        .map(|j| ((((j * 13 + 2) % 21) as f32 - 10.0) / 10.0) * (1.0 / (inter as f32).sqrt()))
        .collect();

    println!("fused SwiGLU MLP: tokens={tokens}, hidden={hidden}, inter={inter}");
    let hpu = swiglu_mlp_bf16(&x, &wgate, &wup, &wdown, tokens, hidden, inter)?;
    let cpu = swiglu_mlp_cpu(&x, &wgate, &wup, &wdown, tokens, hidden, inter);

    let nan = hpu.iter().any(|v| !v.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| f64::from(*a - *b).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|b| f64::from(*b).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    let zeros = hpu.iter().filter(|v| **v == 0.0).count();

    println!(
        "nan={nan}  rel_L2={rel:.4}  zeros={zeros}/{}",
        tokens * hidden
    );
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "mlp diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
