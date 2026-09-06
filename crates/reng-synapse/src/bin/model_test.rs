//! Verify a full synthetic decoder-only model (L fused layers + final norm +
//! LM head, one recipe, causal, optional GQA) against a CPU reference.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-model-test -- [tokens] [hidden] [inter] [n_heads] [vocab] [layers] [causal 0/1] [n_kv_heads]`.
//!
//! With `RENG_TEST_GEMMA` set the layers take Gemma's form: post norms on
//! both branches, the GELU-tanh gate, a sliding window of 100 positions on
//! the even layers and a second RoPE table on the odd ones.

use reng_synapse::{
    Activation, LayerWeights, ModelWeights, model_forward_bf16, model_forward_cpu,
    model_probe_bf16, model_probe_cpu, to_bf16,
};

/// Dense pseudo-random values in `(-scale, scale)` with no exact zeros and no
/// periodic structure (xorshift64*). Earlier periodic generators produced
/// weight/activation tiles of exact zeros in deep synthetic models, which
/// behaved unlike real weights on the device.
fn seq(n: usize, mul: usize, add: usize, modulo: usize, scale: f32) -> Vec<f32> {
    let mut s: u64 =
        0x9E37_79B9_7F4A_7C15 ^ ((mul as u64) << 40) ^ ((add as u64) << 20) ^ modulo as u64;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            // Top 24 bits -> (0, 1), then map to (-1, 1) avoiding exactly 0.
            let u = ((r >> 40) as f32 + 0.5) / 16_777_216.0;
            (2.0 * u - 1.0) * scale
        })
        .collect()
}

struct Owned {
    g1: Vec<f32>,
    g2: Vec<f32>,
    gpa: Vec<f32>,
    gpm: Vec<f32>,
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
    wo: Vec<u16>,
    wg: Vec<u16>,
    wu: Vec<u16>,
    wd: Vec<u16>,
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
    let (vocab, n_layers, causal) = (arg(5, 512usize), arg(6, 2usize), arg(7, 1usize) != 0);
    let n_kv_heads = arg(8, n_heads);
    let gemma = std::env::var_os("RENG_TEST_GEMMA").is_some();
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let rope = |base: f32| {
        let mut sin = vec![0.0f32; tokens * hd];
        let mut cos = vec![0.0f32; tokens * hd];
        for p in 0..tokens {
            for d in 0..hd {
                let theta = p as f32 * base.powf(-2.0 * ((d % half) as f32) / hd as f32);
                sin[p * hd + d] = theta.sin();
                cos[p * hd + d] = theta.cos();
            }
        }
        (sin, cos)
    };
    let (sin, cos) = rope(10000.0);
    let (sin_l, cos_l) = rope(1000.0);
    // Distinct weights per layer (different generator seeds).
    let gain = |l: usize, k: usize| -> Vec<f32> {
        if gemma {
            (0..hidden)
                .map(|i| 0.7 + (((i + k * l) % 9) as f32) * 0.08)
                .collect()
        } else {
            Vec::new()
        }
    };
    let owned: Vec<Owned> = (0..n_layers)
        .map(|l| Owned {
            g1: (0..hidden)
                .map(|i| 0.9 + (((i + l) % 7) as f32) * 0.03)
                .collect(),
            g2: (0..hidden)
                .map(|i| 1.1 - (((i + 2 * l) % 5) as f32) * 0.04)
                .collect(),
            gpa: gain(l, 3),
            gpm: gain(l, 5),
            wq: to_bf16(&seq(hidden * hidden, 5 + l, 1, 17, fan)),
            wk: to_bf16(&seq(hidden * kvd, 11 + l, 4, 19, fan)),
            wv: to_bf16(&seq(hidden * kvd, 13 + l, 2, 21, fan)),
            wo: to_bf16(&seq(hidden * hidden, 3 + l, 5, 29, fan)),
            wg: to_bf16(&seq(hidden * inter, 17 + l, 6, 31, fan)),
            wu: to_bf16(&seq(hidden * inter, 19 + l, 7, 37, fan)),
            wd: to_bf16(&seq(inter * hidden, 23 + l, 8, 41, fan_i)),
        })
        .collect();
    let layers: Vec<LayerWeights<'_>> = owned
        .iter()
        .enumerate()
        .map(|(l, o)| LayerWeights {
            n_heads,
            n_kv_heads,
            head_dim: hd,
            g1: &o.g1,
            g2: &o.g2,
            post_norm: false,
            g_post_attn: &o.gpa,
            g_post_mlp: &o.gpm,
            wq: &o.wq,
            wk: &o.wk,
            wv: &o.wv,
            wo: &o.wo,
            wo_pitch: 0,
            bq: &[],
            bk: &[],
            bv: &[],
            qn: &[],
            kn: &[],
            wg: &o.wg,
            wu: &o.wu,
            wd: &o.wd,
            wd_pitch: 0,
            sin: if gemma && l % 2 == 1 { &sin_l } else { &sin },
            cos: if gemma && l % 2 == 1 { &cos_l } else { &cos },
            scale: 1.0 / (hd as f32).sqrt(),
            use_rope: true,
            local_rope: gemma && l % 2 == 1,
            window: if gemma && l % 2 == 0 { Some(100) } else { None },
            act: if gemma {
                Activation::GeluTanh
            } else {
                Activation::Silu
            },
            attn_softcap: if gemma { Some(50.0) } else { None },
            eps: 1e-6,
        })
        .collect();
    // With the Gemma form the head softcaps the logits at 30 (the gain
    // carries the 1/30).
    let final_softcap = if gemma { Some(30.0) } else { None };
    let final_gamma: Vec<f32> = (0..hidden)
        .map(|i| (1.0 + ((i % 3) as f32) * 0.05) / final_softcap.unwrap_or(1.0))
        .collect();
    let lm_head = to_bf16(&seq(hidden * vocab, 29, 9, 43, fan));
    let m = ModelWeights {
        layers,
        final_gamma: &final_gamma,
        lm_head: &lm_head,
        final_softcap,
    };

    println!(
        "fused model: layers={n_layers}, tokens={tokens}, hidden={hidden}, inter={inter}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab={vocab}, causal={causal}, gemma={gemma}"
    );
    // Optional 9th arg: probe the residual stream after that layer instead of
    // the logits, and report where zeros sit in the read-back buffer.
    if let Some(upto) = std::env::args()
        .nth(9)
        .and_then(|a| a.parse::<usize>().ok())
    {
        let hpu = model_probe_bf16(&x, &m, tokens, hidden, inter, causal, upto)?;
        let cpu = model_probe_cpu(&x, &m, tokens, hidden, inter, causal, upto);
        let num: f64 = hpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| f64::from(*a - *b).powi(2))
            .sum();
        let den: f64 = cpu.iter().map(|b| f64::from(*b).powi(2)).sum();
        let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
        let zero_rows: Vec<usize> = (0..tokens)
            .filter(|&r| hpu[r * hidden..(r + 1) * hidden].iter().all(|v| *v == 0.0))
            .collect();
        let first_nonzero = hpu.iter().position(|v| *v != 0.0);
        println!(
            "probe after layer {upto}: rel_L2={rel:.4} zero_rows={} {:?} first_nonzero_elem={first_nonzero:?} (= {} bytes)",
            zero_rows.len(),
            zero_rows.iter().take(12).collect::<Vec<_>>(),
            first_nonzero.map_or(0, |p| p * 2)
        );
        return Ok(());
    }
    let hpu = model_forward_bf16(&x, &m, tokens, hidden, inter, vocab, causal)?;
    let cpu = model_forward_cpu(&x, &m, tokens, hidden, inter, vocab, causal);

    let nan = hpu.iter().any(|v| !v.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| f64::from(*a - *b).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|b| f64::from(*b).powi(2)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    let zeros = hpu.iter().filter(|v| **v == 0.0).count();
    // Top-1 agreement per token is what generation actually depends on.
    let mut agree = 0usize;
    for tk in 0..tokens {
        let argmax = |v: &[f32]| {
            v[tk * vocab..(tk + 1) * vocab]
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                    if x > b.1 { (i, x) } else { b }
                })
                .0
        };
        if argmax(&hpu) == argmax(&cpu) {
            agree += 1;
        }
    }

    println!(
        "nan={nan}  rel_L2={rel:.4}  zeros={zeros}/{}  top1_agree={agree}/{tokens}",
        tokens * vocab
    );
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "model diverges: nan={nan} rel_L2={rel}"
        )))
    }
}
