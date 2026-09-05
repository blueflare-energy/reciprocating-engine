//! Verify a full synthetic decoder-only model (L fused layers + final norm +
//! LM head, one recipe, causal, optional GQA) against a CPU reference.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-model-test -- [tokens] [hidden] [inter] [n_heads] [vocab] [layers] [causal 0/1] [n_kv_heads]`.

use reng_synapse::{
    LayerWeights, ModelWeights, model_forward_bf16, model_forward_cpu, model_probe_bf16,
    model_probe_cpu,
};

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
    let n_kv_heads = arg(8, n_heads);
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
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
            wk: seq(hidden * kvd, 11 + l, 4, 19, fan),
            wv: seq(hidden * kvd, 13 + l, 2, 21, fan),
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
            n_kv_heads,
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
        "fused model: layers={n_layers}, tokens={tokens}, hidden={hidden}, inter={inter}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab={vocab}, causal={causal}"
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
