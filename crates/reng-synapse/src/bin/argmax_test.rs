//! Probe the argmax kernels over `[vocab, rows]` with the maximum planted at
//! chosen positions and at chosen values (positive, zero, negative), and
//! report every wrong or out-of-range index.
//!
//! Finding (SynapseAI 1.24.1): `argmax_fwd_bf16` (and `argmin_fwd_bf16`) is
//! wrong for a single-row input, the decode shape, whenever the row's
//! maximum is small or negative (returns 0 or an index past the end, such
//! as 32384 for vocab 32064); multi-row inputs are right, and
//! `argmax_fwd_f32` is right in every case. The model head therefore casts
//! the logits to f32 before its argmax. The final verdict is for the f32
//! kernel, which is the one the engine relies on.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-argmax-test -- [vocab ...]`

use core::ffi::c_void;
use reng_synapse::{NodeInput, run_node_i32};

/// Synapse dtype code for f32 (`syn_type_single`).
const SYN_TYPE_F32: core::ffi::c_int = 1 << 2;

#[repr(C)]
struct ReductionParams {
    reduction_dimension: u32,
}

/// Row-major `[rows, vocab]` values in `[top - 40, top - 0.25]` with `top`
/// planted once per row; returns the data and the planted positions.
fn plant(vocab: usize, rows: usize, top: f32) -> (Vec<f32>, Vec<usize>) {
    let mut x = vec![0.0f32; vocab * rows];
    let mut want = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &mut x[r * vocab..(r + 1) * vocab];
        for (j, v) in row.iter_mut().enumerate() {
            *v = top - 40.0 + ((j * 7 + r * 13) % 318) as f32 * 0.125;
        }
        let at = match r % 4 {
            0 => vocab - 1,
            1 => vocab / 2,
            2 => 29889 % vocab,
            _ => (vocab - 130 + r) % vocab,
        };
        row[at] = top;
        want.push(at);
    }
    (x, want)
}

fn main() -> reng_core::Result<()> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("vocab size"))
        .collect();
    let vocabs = if args.is_empty() {
        vec![32000, 32064, 151936]
    } else {
        args
    };
    let p = ReductionParams {
        reduction_dimension: 0,
    };
    let (pp, ps) = (
        (&raw const p).cast::<c_void>(),
        core::mem::size_of::<ReductionParams>() as u32,
    );
    let mut bad = 0;
    for &vocab in &vocabs {
        for rows in [1usize, 4] {
            for top in [40.0f32, 0.5, 0.0, -0.5, -10.0] {
                let (x, want) = plant(vocab, rows, top);
                let neg: Vec<f32> = x.iter().map(|v| -v).collect();
                let f32_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                let sizes = [vocab as u64, rows as u64];
                let cases: [(&str, NodeInput<'_>); 3] = [
                    (
                        "argmax_fwd_bf16",
                        NodeInput {
                            name: "X",
                            sizes: &sizes,
                            data: &x,
                            raw: None,
                        },
                    ),
                    (
                        "argmin_fwd_bf16",
                        NodeInput {
                            name: "X",
                            sizes: &sizes,
                            data: &neg,
                            raw: None,
                        },
                    ),
                    (
                        "argmax_fwd_f32",
                        NodeInput {
                            name: "X",
                            sizes: &sizes,
                            data: &x,
                            raw: Some((SYN_TYPE_F32, &f32_bytes)),
                        },
                    ),
                ];
                for (guid, input) in cases {
                    let got = match run_node_i32(guid, &[input], &[1, rows as u64], pp, ps) {
                        Ok(v) => v,
                        Err(e) => {
                            let m = e.to_string();
                            println!(
                                "{guid} vocab {vocab} rows {rows} top {top}: rejected ({})",
                                &m[..m.len().min(60)]
                            );
                            continue;
                        }
                    };
                    let wrong: Vec<String> = (0..rows)
                        .filter(|&r| usize::try_from(got[r]).ok() != Some(want[r]))
                        .map(|r| format!("row {r}: got {} want {}", got[r], want[r]))
                        .collect();
                    println!(
                        "{guid} vocab {vocab} rows {rows} top {top}: {}{}",
                        if wrong.is_empty() { "ok" } else { "WRONG" },
                        if wrong.is_empty() {
                            String::new()
                        } else {
                            format!(" ({})", wrong[..wrong.len().min(2)].join("; "))
                        }
                    );
                    if guid == "argmax_fwd_f32" {
                        bad += wrong.len();
                    }
                }
            }
        }
    }
    println!(
        "{}",
        if bad == 0 {
            "PASS"
        } else {
            "FAIL (argmax_fwd_bf16)"
        }
    );
    Ok(())
}
