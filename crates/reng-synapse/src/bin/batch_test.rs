//! Verify batched decode: `B` synthetic sequences with different prompts and
//! prompt lengths (one, two and one prefill launches) are prefilled one at a
//! time, then advanced together for several steps; every sequence's step
//! logits must match the CPU reference over that sequence alone.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-batch-test -- [rows] [hidden] [inter] [n_heads] [vocab] [layers] [n_kv_heads] [capacity] [steps] [batch] [post_norm] [qk_norm]`
//!
//! `post_norm` 1 puts the layer norms on the branch outputs (OLMo-2);
//! `qk_norm` 1 adds Qwen3 per-head q/k norms, 2 OLMo-2 full-width ones.
//! With `RENG_TEST_GEMMA` set the layers take Gemma's form (see
//! `reng-model-test`): post norms on both branch outputs, GELU-tanh, a
//! window of 100 positions on the even layers and a second RoPE table on
//! the odd ones. With `RENG_TEST_BIAS` set the layers have Qwen2-style
//! attention biases (the fused q/k/v projection).

use reng_synapse::{
    Activation, BatchedModel, LayerWeights, ModelWeights, RopeTables, model_forward_cpu, to_bf16,
};
use std::time::Instant;

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

struct Owned {
    g1: Vec<f32>,
    g2: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    gpa: Vec<f32>,
    gpm: Vec<f32>,
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    wo: Vec<u16>,
    wg: Vec<u16>,
    wu: Vec<u16>,
    wd: Vec<u16>,
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x - *y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (rows, hidden, inter, n_heads) = (
        arg(1, 256usize),
        arg(2, 256usize),
        arg(3, 512usize),
        arg(4, 4usize),
    );
    let (vocab, n_layers, n_kv_heads) = (arg(5, 512usize), arg(6, 2usize), arg(7, 2usize));
    let capacity = arg(8, 512usize);
    let steps = arg(9, 6usize);
    // Prompt lengths: one launch, two launches, exactly one full block;
    // repeated cyclically up to `batch` sequences (10th arg, default 3).
    let batch = arg(10, 3usize);
    let post_norm = arg(11, 0usize) != 0;
    let qk_norm = arg(12, 0usize);
    let gemma = std::env::var_os("RENG_TEST_GEMMA").is_some();
    // Qwen2-style attention biases (the fused q/k/v projection).
    let bias = std::env::var_os("RENG_TEST_BIAS").is_some();
    let prompts: Vec<usize> = (0..batch)
        .map(|b| [40usize, rows + 44, rows][b % 3])
        .collect();
    let longest = prompts.iter().max().copied().unwrap() + steps;
    assert!(
        longest <= capacity,
        "sequence {longest} exceeds capacity {capacity}"
    );
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();
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

    let rope = |positions: usize, base: f32| {
        let mut sin = vec![0.0f32; positions * hd];
        let mut cos = vec![0.0f32; positions * hd];
        for p in 0..positions {
            for d in 0..hd {
                let theta = p as f32 * base.powf(-2.0 * ((d % half) as f32) / hd as f32);
                sin[p * hd + d] = theta.sin();
                cos[p * hd + d] = theta.cos();
            }
        }
        (sin, cos)
    };
    let (sin_seq, cos_seq) = rope(longest, 10000.0);
    let (sin_cap, cos_cap) = rope(capacity, 10000.0);
    let (sinl_seq, cosl_seq) = rope(longest, 1000.0);
    let (sinl_cap, cosl_cap) = rope(capacity, 1000.0);
    let post_gain = |l: usize, k: usize| -> Vec<f32> {
        if gemma {
            (0..hidden)
                .map(|i| 0.7 + (((i + k * l) % 9) as f32) * 0.08)
                .collect()
        } else {
            Vec::new()
        }
    };
    let bias_of = |n: usize, mul: usize| {
        if bias {
            seq(n, mul, 9, 43, 0.5)
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
            qn: gain(qn_len, 0.95, 0.01, l),
            kn: gain(kn_len, 1.05, -0.01, l + 1),
            gpa: post_gain(l, 3),
            gpm: post_gain(l, 5),
            wq: to_bf16(&seq(hidden * hidden, 5 + l, 1, 17, fan)),
            wk: to_bf16(&seq(hidden * kvd, 11 + l, 4, 19, fan)),
            wv: to_bf16(&seq(hidden * kvd, 13 + l, 2, 21, fan)),
            bq: bias_of(hidden, 29 + l),
            bk: bias_of(kvd, 31 + l),
            bv: bias_of(kvd, 37 + l),
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
            post_norm,
            g_post_attn: &o.gpa,
            g_post_mlp: &o.gpm,
            wq: &o.wq,
            wk: &o.wk,
            wv: &o.wv,
            wo: &o.wo,
            bq: &o.bq,
            bk: &o.bk,
            bv: &o.bv,
            qn: &o.qn,
            kn: &o.kn,
            wg: &o.wg,
            wu: &o.wu,
            wd: &o.wd,
            sin: if gemma && l % 2 == 1 {
                &sinl_seq
            } else {
                &sin_seq
            },
            cos: if gemma && l % 2 == 1 {
                &cosl_seq
            } else {
                &cos_seq
            },
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
    // Every sequence is a different input stream.
    let inputs: Vec<Vec<f32>> = (0..batch)
        .map(|b| seq(longest * hidden, 7 + b, 3 + 2 * b, 23, 1.0))
        .collect();
    println!(
        "batched model: layers={n_layers}, batch={batch}, rows={rows}, capacity={capacity}, prompts={prompts:?}, steps={steps}, hidden={hidden}, heads={n_heads}/{n_kv_heads} kv, vocab={vocab}, post_norm={post_norm}, qk_norm={qk_norm}, gemma={gemma}, bias={bias}"
    );

    let t0 = Instant::now();
    let rope_cap = if gemma {
        RopeTables {
            sin: &sin_cap,
            cos: &cos_cap,
            sin_local: &sinl_cap,
            cos_local: &cosl_cap,
        }
    } else {
        RopeTables::single(&sin_cap, &cos_cap)
    };
    let mut bm = BatchedModel::new(
        m.clone(),
        hidden,
        inter,
        vocab,
        batch,
        rows,
        capacity,
        &rope_cap,
    )?;
    println!("compile + upload: {:.2}s", t0.elapsed().as_secs_f32());

    for (b, &n) in prompts.iter().enumerate() {
        bm.reset(b);
        let t1 = Instant::now();
        bm.prefill(b, &inputs[b][..n * hidden])?;
        println!(
            "prefill sequence {b}: {n} tokens: {:.1} ms",
            t1.elapsed().as_secs_f32() * 1000.0
        );
        assert_eq!(bm.position(b), n);
    }
    // Batched steps: sequence b's next token is its input row at position pos.
    let mut step_logits: Vec<Vec<f32>> = vec![Vec::new(); batch];
    for s in 0..steps {
        let mut x = Vec::with_capacity(batch * hidden);
        for (b, &n) in prompts.iter().enumerate() {
            let pos = n + s;
            x.extend_from_slice(&inputs[b][pos * hidden..(pos + 1) * hidden]);
        }
        let t1 = Instant::now();
        let logits = bm.step(&x)?;
        println!("step {s}: {:.2} ms", t1.elapsed().as_secs_f32() * 1000.0);
        for b in 0..batch {
            step_logits[b].extend_from_slice(&logits[b * vocab..(b + 1) * vocab]);
        }
    }

    let mut worst = 0.0f32;
    let mut agree = 0usize;
    for (b, &n) in prompts.iter().enumerate() {
        let total = n + steps;
        let cpu = model_forward_cpu(
            &inputs[b][..total * hidden],
            &m,
            total,
            hidden,
            inter,
            vocab,
            true,
        );
        let cpu_steps = &cpu[n * vocab..total * vocab];
        let r = rel_l2(&step_logits[b], cpu_steps);
        worst = worst.max(r);
        for s in 0..steps {
            let argmax = |v: &[f32]| {
                v[s * vocab..(s + 1) * vocab]
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
                        if x > acc.1 { (i, x) } else { acc }
                    })
                    .0
            };
            if argmax(&step_logits[b]) == argmax(cpu_steps) {
                agree += 1;
            }
        }
        println!("sequence {b} ({n} prompt tokens): steps rel_L2={r:.4}");
    }
    println!(
        "worst rel_L2={worst:.4}  top1_agree={agree}/{}",
        batch * steps
    );
    if worst < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "batched decode diverges: worst rel_L2 {worst}"
        )))
    }
}
