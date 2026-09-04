//! Test our direct-C-API softmax against CPU on the current stack. Softmax is
//! the prime suspect for the 1.24 `!!!!` miscompute; if ours is correct here,
//! the bug is in the framework path, not the op itself.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-softmax-test`.

use reng_synapse::{softmax_bf16, softmax_cpu};

fn main() -> reng_core::Result<()> {
    let rows = 32usize;
    let cols = 512usize;
    // Include large-magnitude rows to exercise the overflow path that breaks
    // the fused softmax (values up to ~2000, per vllm-fork #275).
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let r = i / cols;
            let base = ((i % 97) as f32 - 48.0) / 12.0;
            base * (1.0 + 60.0 * (r as f32 / rows as f32)) // some rows scaled to ~2000
        })
        .collect();

    println!("direct softmax: {rows}x{cols}, row max |x| up to {:.0}", {
        let mut m = 0.0f32;
        for r in 0..rows {
            let rm = input[r * cols..r * cols + cols]
                .iter()
                .copied()
                .fold(0.0f32, |a, b| a.max(b.abs()));
            m = m.max(rm);
        }
        m
    });
    let hpu = softmax_bf16(&input, rows, cols)?;
    let cpu = softmax_cpu(&input, rows, cols);

    let nan = hpu.iter().any(|x| !x.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| {
            let d = f64::from(*h - *c);
            d * d
        })
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c) * f64::from(*c)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    // Each row of a valid softmax sums to 1.
    let mut worst_sum_err = 0.0f32;
    for r in 0..rows {
        let s: f32 = hpu[r * cols..r * cols + cols].iter().sum();
        worst_sum_err = worst_sum_err.max((s - 1.0).abs());
    }

    println!("nan={nan}  rel_L2={rel:.4}  worst_row_sum_err={worst_sum_err:.4}");
    if !nan && rel < 0.05 && worst_sum_err < 0.05 {
        println!("PASS: our direct softmax matches CPU on this stack");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "DIVERGE: nan={nan} rel_L2={rel:.3} row_sum_err={worst_sum_err:.3} — the softmax kernel is implicated"
        )))
    }
}
