//! Verify a full synthetic decoder-only model (L fused layers + final norm +
//! LM head, one recipe, causal) against a CPU reference.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-model-test -- [tokens] [hidden] [inter] [n_heads] [vocab] [layers] [causal 0/1]`.

use reng_synapse::{LayerWeights, ModelWeights, model_forward_bf16, model_forward_cpu};

fn seq(n: usize, mul: usize, add: usize, modulo: usize, scale: f32) -> Vec<f32> {
    let half = (modulo as f32 - 1.0) / 2.0;
    (0..n)
        .map(|j| ((((j * mul + add) % modulo) as f32) - half) / half * scale)
        .collect()
}

struct Owned {
    g1: Vec<f32>,
    g2: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
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
    let hd = hidden / n_heads;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let mut sin = vec![0.0f32; tokens * hd];
    let mut cos = vec![0.0f32; tokens * hd];
    for p in 0..tokens {
        for d in 0..hd {
            let theta = p as f32 * 10000f32.powf(-2.0 * ((d % half) as f32) / hd as f32);
            sin[p * hd + d] = theta.sin();
            cos[p * hd + d] = theta.cos();
        }
    }
    // Distinct weights per layer (different generator seeds).
    let owned: Vec<Owned> = (0..n_layers)
        .map(|l| Owned {
            g1: (0..hidden)
                .map(|i| 0.9 + (((i + l) % 7) as f32) * 0.03)
                .collect(),
            g2: (0..hidden)
                .map(|i| 1.1 - (((i + 2 * l) % 5) as f32) * 0.04)
                .collect(),
            wq: seq(hidden * hidden, 5 + l, 1, 17, fan),
            wk: seq(hidden * hidden, 11 + l, 4, 19, fan),
            wv: seq(hidden * hidden, 13 + l, 2, 21, fan),
            wo: seq(hidden * hidden, 3 + l, 5, 29, fan),
            wg: seq(hidden * inter, 17 + l, 6, 31, fan),
            wu: seq(hidden * inter, 19 + l, 7, 37, fan),
            wd: seq(inter * hidden, 23 + l, 8, 41, fan_i),
        })
        .collect();
    let layers: Vec<LayerWeights<'_>> = owned
        .iter()
        .map(|o| LayerWeights {
            n_heads,
            g1: &o.g1,
            g2: &o.g2,
            wq: &o.wq,
            wk: &o.wk,
            wv: &o.wv,
            wo: &o.wo,
            wg: &o.wg,
            wu: &o.wu,
            wd: &o.wd,
            sin: &sin,
            cos: &cos,
            scale: 1.0 / (hd as f32).sqrt(),
            eps: 1e-6,
        })
        .collect();
    let final_gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + ((i % 3) as f32) * 0.05).collect();
    let lm_head = seq(hidden * vocab, 29, 9, 43, fan);
    let m = ModelWeights {
        layers,
        final_gamma: &final_gamma,
        lm_head: &lm_head,
    };

    println!(
        "fused model: layers={n_layers}, tokens={tokens}, hidden={hidden}, inter={inter}, n_heads={n_heads}, vocab={vocab}, causal={causal}"
    );
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
