//! Test our direct-C-API single-head attention (`softmax((Q*scale) @ K^T) @ V`)
//! against a CPU reference. This is the exact composition the PyTorch frameworks
//! garble on 1.24; the fused single-recipe path computes it correctly.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-attn-test -- [seq] [dim]`.

use reng_synapse::{attention_bf16, attention_cpu};

fn main() -> reng_core::Result<()> {
    let seq = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let dim = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256usize);
    let scale = 1.0 / (dim as f32).sqrt();

    let mk = |salt: usize| -> Vec<f32> {
        (0..seq * dim)
            .map(|i| (((i * 7 + salt) % 23) as f32 - 11.0) / 11.0)
            .collect()
    };
    let q = mk(1);
    let k = mk(2);
    let v = mk(3);

    println!("fused attention: softmax((Q*scale) @ K^T) @ V, seq={seq}, dim={dim}");
    let hpu = attention_bf16(&q, &k, &v, seq, dim, scale)?;
    let cpu = attention_cpu(&q, &k, &v, seq, dim, scale);

    let nan = hpu.iter().any(|x| !x.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| f64::from(*h - *c).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    let zeros = hpu.iter().filter(|x| **x == 0.0).count();
    let row0_zeros = hpu[0..dim].iter().filter(|x| **x == 0.0).count();

    println!(
        "nan={nan}  rel_L2={rel:.4}  zeros={zeros}/{}  row0_zeros={row0_zeros}/{dim}",
        seq * dim
    );
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "attention diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
