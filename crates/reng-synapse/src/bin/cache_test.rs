//! Verify KV-cache decode: a synthetic causal model fed in blocks through one
//! compiled recipe (a full block, a partial block, then single rows) must
//! produce the same logits as the CPU reference over the whole sequence.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-cache-test -- [rows] [hidden] [inter] [n_heads] [vocab] [layers] [n_kv_heads] [capacity] [tail_blocks] [tail_size] [decode_rows]`.

use reng_synapse::{CachedModel, LayerWeights, ModelWeights, model_forward_cpu, to_bf16};
use std::time::Instant;

/// Dense pseudo-random values in `(-scale, scale)` (xorshift64*), the same
/// generator as `reng-model-test`.
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
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
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
    // Blocks: one full block, one of 8 rows, then `tail_rows` blocks of
    // `tail_size` rows (default 1: single-token decode steps).
    let tail_rows = arg(9, 3usize);
    let tail_size = arg(10, 1usize);
    // A separate decode recipe for blocks of up to this many rows (0: none).
    let decode_rows = arg(11, 0usize);
    let tokens = rows + 8 + tail_rows * tail_size;
    assert!(
        tokens <= capacity,
        "sequence {tokens} exceeds capacity {capacity}"
    );
    let hd = hidden / n_heads;
    let kvd = n_kv_heads * hd;
    let half = hd / 2;
    let fan = 1.0 / (hidden as f32).sqrt();
    let fan_i = 1.0 / (inter as f32).sqrt();

    let x = seq(tokens * hidden, 7, 3, 23, 1.0);
    let rope = |positions: usize| {
        let mut sin = vec![0.0f32; positions * hd];
        let mut cos = vec![0.0f32; positions * hd];
        for p in 0..positions {
            for d in 0..hd {
                let theta = p as f32 * 10000f32.powf(-2.0 * ((d % half) as f32) / hd as f32);
                sin[p * hd + d] = theta.sin();
                cos[p * hd + d] = theta.cos();
            }
        }
        (sin, cos)
    };
    // The CPU reference wants tables for the sequence; the cache for the capacity.
    let (sin_seq, cos_seq) = rope(tokens);
    let (sin_cap, cos_cap) = rope(capacity);
    let owned: Vec<Owned> = (0..n_layers)
        .map(|l| Owned {
            g1: (0..hidden)
                .map(|i| 0.9 + (((i + l) % 7) as f32) * 0.03)
                .collect(),
            g2: (0..hidden)
                .map(|i| 1.1 - (((i + 2 * l) % 5) as f32) * 0.04)
                .collect(),
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
        .map(|o| LayerWeights {
            n_heads,
            n_kv_heads,
            g1: &o.g1,
            g2: &o.g2,
            wq: &o.wq,
            wk: &o.wk,
            wv: &o.wv,
            wo: &o.wo,
            bq: &[],
            bk: &[],
            bv: &[],
            wg: &o.wg,
            wu: &o.wu,
            wd: &o.wd,
            sin: &sin_seq,
            cos: &cos_seq,
            scale: 1.0 / (hd as f32).sqrt(),
            eps: 1e-6,
        })
        .collect();
    let final_gamma: Vec<f32> = (0..hidden).map(|i| 1.0 + ((i % 3) as f32) * 0.05).collect();
    let lm_head = to_bf16(&seq(hidden * vocab, 29, 9, 43, fan));
    let m = ModelWeights {
        layers,
        final_gamma: &final_gamma,
        lm_head: &lm_head,
    };
    println!(
        "cached model: layers={n_layers}, rows={rows}, decode_rows={decode_rows}, capacity={capacity}, tokens={tokens}, hidden={hidden}, inter={inter}, heads={n_heads}/{n_kv_heads} kv, vocab={vocab}"
    );

    let t0 = Instant::now();
    let mut cm = CachedModel::new(
        &m,
        hidden,
        inter,
        vocab,
        rows,
        decode_rows,
        capacity,
        &sin_cap,
        &cos_cap,
    )?;
    println!("compile + upload: {:.2}s", t0.elapsed().as_secs_f32());

    let mut blocks: Vec<usize> = vec![rows, 8];
    blocks.extend(std::iter::repeat_n(tail_size, tail_rows));
    let mut hpu: Vec<f32> = Vec::with_capacity(tokens * vocab);
    let mut start = 0;
    for (bi, &n) in blocks.iter().enumerate() {
        let t1 = Instant::now();
        let logits = cm.step(&x[start * hidden..(start + n) * hidden])?;
        println!(
            "block {bi}: {n} rows at position {start}: {:.1} ms",
            t1.elapsed().as_secs_f32() * 1000.0
        );
        assert_eq!(logits.len(), n * vocab);
        hpu.extend_from_slice(&logits);
        start += n;
    }
    assert_eq!(cm.position(), tokens);

    let cpu = model_forward_cpu(&x, &m, tokens, hidden, inter, vocab, true);
    let nan = hpu.iter().any(|v| !v.is_finite());
    let rel = rel_l2(&hpu, &cpu);
    let mut agree = 0usize;
    let mut worst_block = 0.0f32;
    start = 0;
    let mut per_block: Vec<String> = Vec::new();
    for &n in &blocks {
        let (lo, hi) = (start * vocab, (start + n) * vocab);
        let r = rel_l2(&hpu[lo..hi], &cpu[lo..hi]);
        worst_block = worst_block.max(r);
        per_block.push(format!("{r:.3}"));
        start += n;
    }
    println!("per-block rel_L2: {}", per_block.join(" "));
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
        "nan={nan}  rel_L2={rel:.4}  worst_block_rel_L2={worst_block:.4}  top1_agree={agree}/{tokens}"
    );
    if !nan && rel < 0.05 && worst_block < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "cached decode diverges: nan={nan} rel_L2={rel} worst_block={worst_block}"
        )))
    }
}
