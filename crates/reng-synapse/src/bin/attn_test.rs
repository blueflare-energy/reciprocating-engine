//! Test our direct-C-API unfused attention (`gemm -> softmax -> gemm`) against
//! CPU. This is the exact composition the PyTorch frameworks garble on 1.24.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-attn-test`.

use reng_synapse::{attention_bf16, attention_cpu};

fn main() -> reng_core::Result<()> {
    let seq = 64usize;
    let dim = 64usize;
    let scale = 1.0 / (dim as f32).sqrt();

    let mk = |salt: usize| -> Vec<f32> {
        (0..seq * dim)
            .map(|i| (((i * 7 + salt) % 23) as f32 - 11.0) / 11.0)
            .collect()
    };
    let q = mk(1);
    let k = mk(2);
    let v = mk(3);

    println!("direct attention: softmax((Q*scale) @ K^T) @ V, seq={seq}, dim={dim}");
    let hpu = attention_bf16(&q, &k, &v, seq, dim, scale)?;
    let cpu = attention_cpu(&q, &k, &v, seq, dim, scale);

    let nan = hpu.iter().any(|x| !x.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| {
            let d = f64::from(*h - *c);
            d * d
        })
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c) * f64::from(*c)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;

    println!("nan={nan}  rel_L2={rel:.4}");
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS: our direct attention block computes correctly on this stack");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "DIVERGE: nan={nan} rel_L2={rel:.3} — attention composition is wrong"
        )))
    }
}
