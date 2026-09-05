//! Verify one fused multi-head (optionally grouped-query) transformer decoder
//! layer against a CPU reference: RMSNorm -> QKV -> per-head RoPE + attention
//! (split/concat) -> O-proj -> residual -> RMSNorm -> SwiGLU MLP -> residual,
//! all in one recipe.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-layer-test -- [tokens] [hidden] [inter] [n_heads] [n_kv_heads] [post_norm] [qk_norm]`.
//!
//! `post_norm` 1 puts the layer norms on the branch outputs (OLMo-2);
//! `qk_norm` 1 adds Qwen3 per-head q/k norms, 2 OLMo-2 full-width ones.

use reng_synapse::{LayerWeights, decoder_layer_bf16, decoder_layer_cpu, to_bf16};

/// Dense pseudo-random values in `(-scale, scale)` (xorshift64*), the same
/// generator as `reng-model-test`. A periodic pattern would make the
/// attention output cancel to about 1e-3, where the device RMSNorm of a
/// post-norm branch no longer matches the CPU (see `reng-norm-test`).
fn seq(n: usize, mul: usize, add: usize, modulo: usize, scale: f32) -> Vec<f32> {
    let mut s: u64 =
        0x9E37_79B9_7F4A_7C15 ^ ((mul as u64) << 40) ^ ((add as u64) << 20) ^ modulo as u64;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let u = ((r >> 40) as f32 + 0.5) / 16_777_216.0;
            (2.0 * u - 1.0) * scale
        })
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
    let post_norm = arg(6, 0usize) != 0;
    let qk_norm = arg(7, 0usize);
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let g1: Vec<f32> = (0..hidden).map(|i| 0.9 + ((i % 7) as f32) * 0.03).collect();
    let g2: Vec<f32> = (0..hidden).map(|i| 1.1 - ((i % 5) as f32) * 0.04).collect();
    // q/k norm gains: none, per head (`hd`, Qwen3) or over the whole
    // projection (OLMo-2).
    let gain = |n: usize, base: f32, step: f32, l: usize| -> Vec<f32> {
        (0..n)
            .map(|i| base + (((i + l) % 11) as f32) * step)
            .collect()
    };
    let (qn_len, kn_len) = match qk_norm {
        0 => (0, 0),
        1 => (hd, hd),
        _ => (hidden, kvd),
    };
    let qn = gain(qn_len, 0.95, 0.01, 0);
    let kn = gain(kn_len, 1.05, -0.01, 1);
    let wq = to_bf16(&seq(hidden * hidden, 5, 1, 17, fan));
    let wk = to_bf16(&seq(hidden * kvd, 11, 4, 19, fan));
    let wv = to_bf16(&seq(hidden * kvd, 13, 2, 21, fan));
    let wo = to_bf16(&seq(hidden * hidden, 3, 5, 29, fan));
    let wg = to_bf16(&seq(hidden * inter, 17, 6, 31, fan));
    let wu = to_bf16(&seq(hidden * inter, 19, 7, 37, fan));
    let wd = to_bf16(&seq(inter * hidden, 23, 8, 41, fan_i));
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
        head_dim: hd,
        g1: &g1,
        g2: &g2,
        post_norm,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        bq: &[],
        bk: &[],
        bv: &[],
        qn: &qn,
        kn: &kn,
        wg: &wg,
        wu: &wu,
        wd: &wd,
        sin: &sin,
        cos: &cos,
        scale: 1.0 / (hd as f32).sqrt(),
        use_rope: true,
        eps: 1e-6,
    };

    println!(
        "fused decoder layer: tokens={tokens}, hidden={hidden}, inter={inter}, n_heads={n_heads}, n_kv_heads={n_kv_heads} (head_dim {hd}), post_norm={post_norm}, qk_norm={qk_norm}"
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
