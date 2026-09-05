//! Verify one fused transformer decoder layer (single head) against a CPU
//! reference: RMSNorm -> QKV -> RoPE -> attention -> O-proj -> residual ->
//! RMSNorm -> SwiGLU MLP -> residual, all in one recipe.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-layer-test -- [tokens] [hidden] [inter]`.

use reng_synapse::{LayerWeights, decoder_layer_bf16, decoder_layer_cpu};

fn seq(n: usize, mul: usize, add: usize, modulo: usize, scale: f32) -> Vec<f32> {
    let half = (modulo as f32 - 1.0) / 2.0;
    (0..n)
        .map(|j| ((((j * mul + add) % modulo) as f32) - half) / half * scale)
        .collect()
}

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (tokens, hidden, inter) = (arg(1, 256usize), arg(2, 256usize), arg(3, 512usize));
    let half = hidden / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let g1: Vec<f32> = (0..hidden).map(|i| 0.9 + ((i % 7) as f32) * 0.03).collect();
    let g2: Vec<f32> = (0..hidden).map(|i| 1.1 - ((i % 5) as f32) * 0.04).collect();
    let wq = seq(hidden * hidden, 5, 1, 17, fan);
    let wk = seq(hidden * hidden, 11, 4, 19, fan);
    let wv = seq(hidden * hidden, 13, 2, 21, fan);
    let wo = seq(hidden * hidden, 3, 5, 29, fan);
    let wg = seq(hidden * inter, 17, 6, 31, fan);
    let wu = seq(hidden * inter, 19, 7, 37, fan);
    let wd = seq(inter * hidden, 23, 8, 41, fan_i);
    let mut sin = vec![0.0f32; tokens * hidden];
    let mut cos = vec![0.0f32; tokens * hidden];
    for p in 0..tokens {
        for d in 0..hidden {
            let theta = p as f32 * 10000f32.powf(-2.0 * ((d % half) as f32) / hidden as f32);
            sin[p * hidden + d] = theta.sin();
            cos[p * hidden + d] = theta.cos();
        }
    }
    let w = LayerWeights {
        g1: &g1,
        g2: &g2,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        wg: &wg,
        wu: &wu,
        wd: &wd,
        sin: &sin,
        cos: &cos,
        scale: fan,
        eps: 1e-6,
    };

    println!("fused decoder layer (single head): tokens={tokens}, hidden={hidden}, inter={inter}");
    let hpu = decoder_layer_bf16(&x, &w, tokens, hidden, inter)?;
    let cpu = decoder_layer_cpu(&x, &w, tokens, hidden, inter);

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
            "decoder layer diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
