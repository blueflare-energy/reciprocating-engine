//! Verify batched decode: `B` synthetic sequences with different prompts and
//! prompt lengths (one, two and one prefill launches) are prefilled one at a
//! time, then advanced together for several steps; every sequence's step
//! logits must match the CPU reference over that sequence alone.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-batch-test -- [rows] [hidden] [inter] [n_heads] [vocab] [layers] [n_kv_heads] [capacity] [steps] [batch] [post_norm] [qk_norm] [loop_steps]`
//!
//! `post_norm` 1 puts the layer norms on the branch outputs (OLMo-2);
//! `qk_norm` 1 adds Qwen3 per-head q/k norms, 2 OLMo-2 full-width ones.
//! With `RENG_TEST_GEMMA` set the layers take Gemma's form (see
//! `reng-model-test`): post norms on both branch outputs, GELU-tanh, a
//! window of 100 positions on the even layers and a second RoPE table on
//! the odd ones. With `RENG_TEST_BIAS` set the layers have Qwen2-style
//! attention biases (the fused q/k/v projection).
//! The embeddings are scaled by 2 on the device.
//!
//! The model gets a synthetic embedding table whose row `b * steps + s` is
//! sequence `b`'s input row at its step `s`, so when the device decode
//! loop is built (`RENG_DEVICE_LOOP` not off) the steps are fed as token
//! ids through it, one launch at a time, and checked like the per-step
//! path. Then `loop_steps` (default 4) greedy steps run as one device run
//! from the argmax of every sequence's last step: each sequence's
//! last-step logits are checked against the CPU reference over the
//! sequence the loop produced, and a second pass over the same prompts
//! feeds the same ids one launch at a time, which must reproduce the run's
//! ids and last logits exactly.

use reng_synapse::{
    Activation, BatchedModel, EmbedTable, LayerWeights, ModelWeights, RopeTables, bf16_to_f32,
    model_forward_cpu, to_bf16,
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
    let loop_steps = arg(13, 4usize);
    let gemma = std::env::var_os("RENG_TEST_GEMMA").is_some();
    // Qwen2-style attention biases (the fused q/k/v projection).
    let bias = std::env::var_os("RENG_TEST_BIAS").is_some();
    let prompts: Vec<usize> = (0..batch)
        .map(|b| [40usize, rows + 44, rows][b % 3])
        .collect();
    let longest = prompts.iter().max().copied().unwrap() + steps;
    assert!(
        longest + loop_steps <= capacity,
        "sequence {longest} + {loop_steps} loop steps exceeds capacity {capacity}"
    );
    assert!(
        batch * steps <= vocab,
        "the step rows are fed as their own ids: batch {batch} x steps {steps} must fit vocab {vocab}"
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
    let (sin_seq, cos_seq) = rope(longest + loop_steps, 10000.0);
    let (sin_cap, cos_cap) = rope(capacity, 10000.0);
    let (sinl_seq, cosl_seq) = rope(longest + loop_steps, 1000.0);
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
    // The embedding table: row `b * steps + s` is sequence b's input row
    // at step s (halved when the device scales by 2, an exact operation
    // in bf16), the rest dense random rows for the ids the loop produces.
    let embed_scale = if gemma { 2.0 } else { 1.0 };
    let mut embed = to_bf16(&seq(vocab * hidden, 31, 11, 47, 1.0 / embed_scale));
    for (b, &n) in prompts.iter().enumerate() {
        for s in 0..steps {
            let row = &inputs[b][(n + s) * hidden..(n + s + 1) * hidden];
            let at = (b * steps + s) * hidden;
            embed[at..at + hidden].copy_from_slice(&to_bf16(
                &row.iter().map(|v| v / embed_scale).collect::<Vec<f32>>(),
            ));
        }
    }
    let table = EmbedTable {
        rows: &embed,
        scale: embed_scale,
    };
    // The row of the table for `id`, as the device feeds it.
    let embed_row = |id: u32| -> Vec<f32> {
        let id = id as usize;
        embed[id * hidden..(id + 1) * hidden]
            .iter()
            .map(|&b| bf16_to_f32(b) * embed_scale)
            .collect()
    };
    let argmax = |row: &[f32]| -> u32 {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0 as u32
    };
    println!(
        "batched model: layers={n_layers}, batch={batch}, rows={rows}, capacity={capacity}, prompts={prompts:?}, steps={steps}, loop_steps={loop_steps}, hidden={hidden}, heads={n_heads}/{n_kv_heads} kv, vocab={vocab}, post_norm={post_norm}, qk_norm={qk_norm}, gemma={gemma}, bias={bias}"
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
        Some(&table),
    )?;
    println!(
        "compile + upload: {:.2}s{}",
        t0.elapsed().as_secs_f32(),
        if bm.has_loop() {
            " (device decode loop)"
        } else {
            ""
        }
    );

    // Prefill every sequence, then the batched steps: sequence b's next
    // token is its input row at position pos, fed as that row's id through
    // the loop (one launch at a time) or as the row itself.
    let run_steps = |bm: &mut BatchedModel<'_>| -> reng_core::Result<Vec<Vec<f32>>> {
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
        let mut step_logits: Vec<Vec<f32>> = vec![Vec::new(); batch];
        for s in 0..steps {
            let t1 = Instant::now();
            let logits = if bm.has_loop() {
                let ids: Vec<u32> = (0..batch).map(|b| (b * steps + s) as u32).collect();
                bm.run_ids_logits(&ids, 1)?.1
            } else {
                let mut x = Vec::with_capacity(batch * hidden);
                for (b, &n) in prompts.iter().enumerate() {
                    let pos = n + s;
                    x.extend_from_slice(&inputs[b][pos * hidden..(pos + 1) * hidden]);
                }
                bm.step(&x)?
            };
            println!(
                "step {s}: {:.2} ms{}",
                t1.elapsed().as_secs_f32() * 1000.0,
                if bm.has_loop() {
                    " (device loop, by id)"
                } else {
                    ""
                }
            );
            for b in 0..batch {
                step_logits[b].extend_from_slice(&logits[b * vocab..(b + 1) * vocab]);
            }
        }
        for (b, &n) in prompts.iter().enumerate() {
            assert_eq!(bm.position(b), n + steps);
        }
        Ok(step_logits)
    };
    let step_logits = run_steps(&mut bm)?;

    // The device loop: `loop_steps` greedy steps for every sequence as one
    // run, then the same ids one launch at a time over a fresh pass.
    let loop_check = if bm.has_loop() && loop_steps > 0 {
        let seeds: Vec<u32> = (0..batch)
            .map(|b| argmax(&step_logits[b][(steps - 1) * vocab..]))
            .collect();
        let t1 = Instant::now();
        let (ids, last) = bm.run_ids_logits(&seeds, loop_steps)?;
        println!(
            "loop: {loop_steps} steps x {batch} sequences from seeds {seeds:?}: {:.1} ms, ids {ids:?}",
            t1.elapsed().as_secs_f32() * 1000.0
        );
        assert_eq!(ids.len(), loop_steps * batch);
        assert_eq!(last.len(), batch * vocab);
        for (b, &n) in prompts.iter().enumerate() {
            assert_eq!(bm.position(b), n + steps + loop_steps);
        }
        let step_logits2 = run_steps(&mut bm)?;
        assert!(
            step_logits2 == step_logits,
            "second pass of the prompts and steps differs"
        );
        let mut single: Vec<u32> = Vec::with_capacity(loop_steps * batch);
        let mut next = seeds.clone();
        let mut single_last = Vec::new();
        for _ in 0..loop_steps {
            let (out, logits) = bm.run_ids_logits(&next, 1)?;
            single.extend_from_slice(&out);
            next = out;
            single_last = logits;
        }
        let same = single == ids && single_last == last;
        println!(
            "loop: one launch at a time gives ids {single:?}: {}",
            if same { "identical" } else { "DIFFERENT" }
        );
        Some((seeds, ids, last, same))
    } else {
        None
    };

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
    // The loop's last step of every sequence against the CPU reference
    // over the sequence it produced (the seed and all but the last id
    // appended to the prompt and steps).
    let mut loop_ok = true;
    if let Some((seeds, ids, last, same)) = loop_check {
        let mut worst_last = 0.0f32;
        let mut agree_loop = 0usize;
        let mut consistent = true;
        for (b, &n) in prompts.iter().enumerate() {
            let mut x_ext = inputs[b][..(n + steps) * hidden].to_vec();
            x_ext.extend(embed_row(seeds[b]));
            for j in 0..loop_steps - 1 {
                x_ext.extend(embed_row(ids[j * batch + b]));
            }
            let total = n + steps + loop_steps;
            let cpu_ext = model_forward_cpu(&x_ext, &m, total, hidden, inter, vocab, true);
            let cpu_last = &cpu_ext[(total - 1) * vocab..];
            let last_b = &last[b * vocab..(b + 1) * vocab];
            let rel_last = rel_l2(last_b, cpu_last);
            worst_last = worst_last.max(rel_last);
            let cpu_ids: Vec<u32> = (0..loop_steps)
                .map(|k| argmax(&cpu_ext[(n + steps + k) * vocab..(n + steps + k + 1) * vocab]))
                .collect();
            let dev_ids: Vec<u32> = (0..loop_steps).map(|j| ids[j * batch + b]).collect();
            agree_loop += dev_ids.iter().zip(&cpu_ids).filter(|(a, c)| a == c).count();
            let self_consistent = dev_ids[loop_steps - 1] == argmax(last_b);
            consistent &= self_consistent && last_b.iter().all(|v| v.is_finite());
            println!(
                "loop sequence {b}: last-step rel_L2={rel_last:.4}  ids {dev_ids:?} (cpu {cpu_ids:?})  last id is its logits' argmax: {self_consistent}"
            );
        }
        println!(
            "loop: worst last-step rel_L2={worst_last:.4}  top1_agree={agree_loop}/{}",
            batch * loop_steps
        );
        loop_ok = same && consistent && worst_last < 0.05;
    }
    if worst < 0.05 && loop_ok {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "batched decode diverges: worst rel_L2 {worst} loop_ok={loop_ok}"
        )))
    }
}
