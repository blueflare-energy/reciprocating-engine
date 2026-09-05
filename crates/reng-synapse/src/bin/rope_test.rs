//! Verify our direct RoPE (`rope_st2_fwd_bf16`, blockwise/rotate-half) against a
//! CPU reference at a reliable size.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-rope-test -- [head_dim] [seq]`.

use reng_synapse::{rope_bf16, rope_cpu};

fn main() -> reng_core::Result<()> {
    let head_dim = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128usize);
    let seq = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let half = head_dim / 2;
    let base = 10000.0f32;

    let x: Vec<f32> = (0..seq * head_dim)
        .map(|j| (((j * 7 + 3) % 23) as f32 - 11.0) / 11.0)
        .collect();
    // cos/sin caches, blockwise: freq index i = d mod (head_dim/2).
    let mut sin = vec![0.0f32; seq * head_dim];
    let mut cos = vec![0.0f32; seq * head_dim];
    for p in 0..seq {
        for d in 0..head_dim {
            let i = d % half;
            let theta = p as f32 * base.powf(-2.0 * (i as f32) / head_dim as f32);
            sin[p * head_dim + d] = theta.sin();
            cos[p * head_dim + d] = theta.cos();
        }
    }

    println!("rope (blockwise): head_dim={head_dim}, seq={seq}");
    let hpu = rope_bf16(&x, &sin, &cos, head_dim, seq)?;
    let cpu = rope_cpu(&x, &sin, &cos, head_dim, seq);

    let nan = hpu.iter().any(|v| !v.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| f64::from(*a - *b).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|b| f64::from(*b).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;

    println!("nan={nan}  rel_L2={rel:.4}");
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "rope diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
