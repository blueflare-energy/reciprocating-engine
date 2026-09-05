//! Verify one fused multi-head (optionally grouped-query) transformer decoder
//! layer against a CPU reference: RMSNorm -> QKV -> per-head RoPE + attention
//! (split/concat) -> O-proj -> residual -> RMSNorm -> SwiGLU MLP -> residual,
//! all in one recipe.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-layer-test -- [tokens] [hidden] [inter] [n_heads] [n_kv_heads]`.

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
    let (tokens, hidden, inter, n_heads) = (
        arg(1, 256usize),
        arg(2, 256usize),
        arg(3, 512usize),
        arg(4, 2usize),
    );
    let n_kv_heads = arg(5, n_heads);
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let g1: Vec<f32> = (0..hidden).map(|i| 0.9 + ((i % 7) as f32) * 0.03).collect();
    let g2: Vec<f32> = (0..hidden).map(|i| 1.1 - ((i % 5) as f32) * 0.04).collect();
    let wq = seq(hidden * hidden, 5, 1, 17, fan);
    let wk = seq(hidden * kvd, 11, 4, 19, fan);
    let wv = seq(hidden * kvd, 13, 2, 21, fan);
    let wo = seq(hidden * hidden, 3, 5, 29, fan);
    let wg = seq(hidden * inter, 17, 6, 31, fan);
    let wu = seq(hidden * inter, 19, 7, 37, fan);
    let wd = seq(inter * hidden, 23, 8, 41, fan_i);
    // RoPE caches are per position and per head_dim, shared across heads.
    let mut sin = vec![0.0f32; tokens * hd];
    let mut cos = vec![0.0f32; tokens * hd];
    for p in 0..tokens {
        for d in 0..hd {
            let theta = p as f32 * 10000f32.powf(-2.0 * ((d % half) as f32) / hd as f32);
            sin[p * hd + d] = theta.sin();
            cos[p * hd + d] = theta.cos();
        }
    }
    let w = LayerWeights {
        n_heads,
        n_kv_heads,
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
        scale: 1.0 / (hd as f32).sqrt(),
        eps: 1e-6,
    };

    println!(
        "fused decoder layer: tokens={tokens}, hidden={hidden}, inter={inter}, n_heads={n_heads}, n_kv_heads={n_kv_heads} (head_dim {hd})"
    );
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
