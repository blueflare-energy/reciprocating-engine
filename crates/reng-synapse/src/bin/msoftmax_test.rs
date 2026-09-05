//! Pin down the contract of `scaled_masked_softmax_fwd_bf16`: which of
//! `softmax(s * scale + m)`, `softmax(s / scale + m)` and `softmax(s * m)`
//! it computes over the FCD of `[keys, q, heads, 1]` scores with a mask of
//! the same shape, for a few `invScaleAttn` values and `isUseMax` settings.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-msoftmax-test`

use core::ffi::c_void;
use reng_synapse::{NodeInput, run_node};

#[repr(C)]
struct Params {
    inv_scale_attn: f32,
    grouped_batch_size: u32,
    is_use_max: u32,
    exp_mode: u32,
}

fn softmax_rows(v: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; v.len()];
    for (r, row) in v.chunks(n).enumerate() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
        let s: f32 = e.iter().sum();
        for (j, x) in e.iter().enumerate() {
            out[r * n + j] = x / s;
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

fn main() -> reng_core::Result<()> {
    let (keys, heads) = (133usize, 12usize);
    let n = keys * heads;
    let s: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) % 23) as f32 * 0.5 - 5.0)
        .collect();
    let keep = |i: usize| i % keys < 100;
    // Additive mask (0 keep, -30000 drop) and multiplicative mask (1 keep, 0 drop).
    let m_add: Vec<f32> = (0..n)
        .map(|i| if keep(i) { 0.0 } else { -30000.0 })
        .collect();
    let m_mul: Vec<f32> = (0..n).map(|i| if keep(i) { 1.0 } else { 0.0 }).collect();
    let sizes = [keys as u64, 1, heads as u64, 1];
    for (mask_name, m) in [("additive", &m_add), ("multiplicative", &m_mul)] {
        for mask_first in [false, true] {
            for inv_scale in [1.0f32, 0.125] {
                let p = Params {
                    inv_scale_attn: inv_scale,
                    grouped_batch_size: 1,
                    is_use_max: 1,
                    exp_mode: 0,
                };
                let a = NodeInput {
                    name: "S",
                    sizes: &sizes,
                    data: &s,
                    raw: None,
                };
                let b = NodeInput {
                    name: "M",
                    sizes: &sizes,
                    data: m,
                    raw: None,
                };
                let ins = if mask_first { [b, a] } else { [a, b] };
                let got = match run_node(
                    "scaled_masked_softmax_fwd_bf16",
                    &ins,
                    &sizes,
                    (&raw const p).cast::<c_void>(),
                    core::mem::size_of::<Params>() as u32,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        println!(
                            "{mask_name} mask_first {mask_first} inv {inv_scale}: rejected ({})",
                            e.to_string().chars().take(60).collect::<String>()
                        );
                        continue;
                    }
                };
                // Candidate formulas; a multiplicative mask of 0 means "drop".
                let dropped = |i: usize| -> f32 { if keep(i) { 0.0 } else { -30000.0 } };
                let cands: [(&str, Vec<f32>); 6] = [
                    (
                        "softmax(s*inv + m)",
                        s.iter().zip(m).map(|(x, y)| x * inv_scale + y).collect(),
                    ),
                    (
                        "softmax(s/inv + m)",
                        s.iter().zip(m).map(|(x, y)| x / inv_scale + y).collect(),
                    ),
                    (
                        "softmax(s*inv, drop where m==0)",
                        (0..n).map(|i| s[i] * inv_scale + dropped(i)).collect(),
                    ),
                    (
                        "softmax(s/inv, drop where m==0)",
                        (0..n).map(|i| s[i] / inv_scale + dropped(i)).collect(),
                    ),
                    (
                        "softmax(s*inv*m)",
                        s.iter().zip(m).map(|(x, y)| x * inv_scale * y).collect(),
                    ),
                    ("softmax(m)", m.clone()),
                ];
                let mut best = ("none", f32::INFINITY);
                for (name, v) in &cands {
                    let r = rel_l2(&got, &softmax_rows(v, keys));
                    if r < best.1 {
                        best = (name, r);
                    }
                }
                println!(
                    "{mask_name} mask_first {mask_first} inv {inv_scale}: best {} rel_L2 {:.4}; p[0..3] {:?} p[100] {:.4}",
                    best.0,
                    best.1,
                    &got[..3],
                    got[100]
                );
            }
        }
    }
    Ok(())
}
