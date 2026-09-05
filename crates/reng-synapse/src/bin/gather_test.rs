//! Pin down the kernels the device-resident decode loop is built from,
//! each as a single node against a host reference:
//!
//! - `gather_fwd_bf16` (`ns_GatherKernel::Params { axis }`, FCD-first
//!   axis): rows of a `[fcd, n]` table by an int32 index vector (the
//!   embedding row of a token id, the RoPE rows of a position) and
//!   elements of a 1-D vector by an int32 index vector (a mask row as a
//!   window into a static pattern);
//! - `sub_fwd_i32`, `add_fwd_i32` and `mult_fwd_i32` with a broadcast
//!   `[1]` operand (index arithmetic from a position tensor);
//! - `cast_f32_to_bf16` rounding (round half to nearest even, as the
//!   host's `f32_to_bf16`), for Gemma's embedding scale.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-gather-test`

use core::ffi::c_void;
use reng_synapse::{NodeInput, SYN_TYPE_INT32, bf16_to_f32, f32_to_bf16, run_node, run_node_i32};

/// Synapse dtype code for f32 (`syn_type_single`).
const SYN_TYPE_F32: core::ffi::c_int = 1 << 2;

#[repr(C)]
struct GatherParams {
    axis: i32,
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn report(name: &str, ok: bool, detail: &str) -> bool {
    println!("{name}: {} {detail}", if ok { "ok" } else { "WRONG" });
    ok
}

/// `gather_fwd_bf16` of rows `idx` out of a `[fcd, n]` bf16 table along
/// the outer axis (FCD-first axis 1): expect `[fcd, idx.len()]`.
fn gather_rows(fcd: usize, n: usize, idx: &[i32], axis: i32) -> reng_core::Result<Vec<f32>> {
    let table: Vec<f32> = (0..n * fcd)
        .map(|j| ((j / fcd) as f32) + ((j % fcd) as f32) * 0.001)
        .collect();
    let bytes = i32_bytes(idx);
    let p = GatherParams { axis };
    run_node(
        "gather_fwd_bf16",
        &[
            NodeInput {
                name: "T",
                sizes: &[fcd as u64, n as u64],
                data: &table,
                raw: None,
            },
            NodeInput {
                name: "I",
                sizes: &[idx.len() as u64],
                data: &[],
                raw: Some((SYN_TYPE_INT32, &bytes)),
            },
        ],
        &[fcd as u64, idx.len() as u64],
        (&raw const p).cast::<c_void>(),
        core::mem::size_of::<GatherParams>() as u32,
    )
}

fn main() -> reng_core::Result<()> {
    let mut all = true;

    // Row gather: a `[64, 300]` table, rows 137, 0 and 299 (bf16 keeps
    // the integer row index exact below 256, so check against the
    // rounded expectation).
    for axis in [1i32, 0] {
        let idx = [137i32, 0, 299];
        match gather_rows(64, 300, &idx, axis) {
            Ok(out) => {
                let want =
                    |r: usize, d: usize| bf16_to_f32(f32_to_bf16(r as f32 + d as f32 * 0.001));
                let bad = (0..idx.len())
                    .flat_map(|k| (0..64).map(move |d| (k, d)))
                    .filter(|&(k, d)| (out[k * 64 + d] - want(idx[k] as usize, d)).abs() > 1e-6)
                    .count();
                let ok = report(
                    &format!("gather rows axis {axis}"),
                    bad == 0,
                    &format!("({bad} wrong of {}; first {:?})", out.len(), &out[..4]),
                );
                if axis == 1 {
                    all &= ok;
                }
            }
            Err(e) => {
                println!("gather rows axis {axis}: rejected ({e})");
                if axis == 1 {
                    all = false;
                }
            }
        }
    }

    // Embedding-sized table: `[hidden 128, vocab 4096]`, one id.
    {
        let (fcd, n) = (128usize, 4096usize);
        let idx = [3999i32];
        match gather_rows(fcd, n, &idx, 1) {
            Ok(out) => {
                let want = |d: usize| bf16_to_f32(f32_to_bf16(3999.0 + d as f32 * 0.001));
                let bad = (0..fcd)
                    .filter(|&d| (out[d] - want(d)).abs() > 1e-6)
                    .count();
                all &= report("gather embedding row", bad == 0, &format!("({bad} wrong)"));
            }
            Err(e) => {
                println!("gather embedding row: rejected ({e})");
                all = false;
            }
        }
    }

    // Element gather from a 1-D pattern: the causal mask row of position p
    // over `keys` keys is `pattern[keys - 1 - p + k]` with
    // `pattern = [0; keys] ++ [NEG; keys]`.
    {
        let keys = 1025usize;
        let neg = -30000.0f32;
        let pattern: Vec<f32> = (0..2 * keys)
            .map(|j| if j < keys { 0.0 } else { neg })
            .collect();
        for p in [0usize, 7, 1023] {
            let idx: Vec<i32> = (0..keys).map(|k| (keys - 1 - p + k) as i32).collect();
            let bytes = i32_bytes(&idx);
            let gp = GatherParams { axis: 0 };
            match run_node(
                "gather_fwd_bf16",
                &[
                    NodeInput {
                        name: "P",
                        sizes: &[2 * keys as u64],
                        data: &pattern,
                        raw: None,
                    },
                    NodeInput {
                        name: "I",
                        sizes: &[keys as u64],
                        data: &[],
                        raw: Some((SYN_TYPE_INT32, &bytes)),
                    },
                ],
                &[keys as u64],
                (&raw const gp).cast::<c_void>(),
                core::mem::size_of::<GatherParams>() as u32,
            ) {
                Ok(out) => {
                    let bad = (0..keys)
                        .filter(|&k| {
                            let want = if k <= p {
                                0.0
                            } else {
                                bf16_to_f32(f32_to_bf16(neg))
                            };
                            out[k] != want
                        })
                        .count();
                    all &= report(
                        &format!("gather mask row p {p}"),
                        bad == 0,
                        &format!(
                            "({bad} wrong; out[p..p+2] {:?})",
                            &out[p..(p + 2).min(keys)]
                        ),
                    );
                }
                Err(e) => {
                    println!("gather mask row p {p}: rejected ({e})");
                    all = false;
                }
            }
        }
    }

    // Int32 index arithmetic with a broadcast `[1]` operand.
    {
        let keys = 1025usize;
        let base: Vec<i32> = (0..keys).map(|k| (keys - 1 + k) as i32).collect();
        let pos = [513i32];
        let (bb, pb) = (i32_bytes(&base), i32_bytes(&pos));
        for guid in ["sub_fwd_i32", "add_fwd_i32"] {
            let want = |b: i32| {
                if guid == "sub_fwd_i32" {
                    b - 513
                } else {
                    b + 513
                }
            };
            match run_node_i32(
                guid,
                &[
                    NodeInput {
                        name: "B",
                        sizes: &[keys as u64],
                        data: &[],
                        raw: Some((SYN_TYPE_INT32, &bb)),
                    },
                    NodeInput {
                        name: "P",
                        sizes: &[1],
                        data: &[],
                        raw: Some((SYN_TYPE_INT32, &pb)),
                    },
                ],
                &[keys as u64],
                core::ptr::null(),
                0,
            ) {
                Ok(out) => {
                    let bad = (0..keys).filter(|&k| out[k] != want(base[k])).count();
                    all &= report(
                        &format!("{guid} [{keys}] op [1]"),
                        bad == 0,
                        &format!("({bad} wrong; first {:?})", &out[..3]),
                    );
                }
                Err(e) => {
                    println!("{guid} [{keys}] op [1]: rejected ({e})");
                    all = false;
                }
            }
        }
        // `[3, groups]` scatter triples (g, 0, 0) + (0, 0, 1) * pos.
        let groups = 8usize;
        let mut kb: Vec<i32> = Vec::new();
        let mut sel: Vec<i32> = Vec::new();
        for g in 0..groups {
            kb.extend_from_slice(&[g as i32, 0, 0]);
            sel.extend_from_slice(&[0, 0, 1]);
        }
        let (kbb, selb) = (i32_bytes(&kb), i32_bytes(&sel));
        let posb = i32_bytes(&[77]);
        let prod = run_node_i32(
            "mult_fwd_i32",
            &[
                NodeInput {
                    name: "S",
                    sizes: &[3, groups as u64],
                    data: &[],
                    raw: Some((SYN_TYPE_INT32, &selb)),
                },
                NodeInput {
                    name: "P",
                    sizes: &[1, 1],
                    data: &[],
                    raw: Some((SYN_TYPE_INT32, &posb)),
                },
            ],
            &[3, groups as u64],
            core::ptr::null(),
            0,
        );
        match prod {
            Ok(out) => {
                let bad = (0..3 * groups).filter(|&j| out[j] != sel[j] * 77).count();
                all &= report(
                    "mult_fwd_i32 [3, g] x [1, 1]",
                    bad == 0,
                    &format!("({bad} wrong; first {:?})", &out[..6]),
                );
            }
            Err(e) => {
                println!("mult_fwd_i32 [3, g] x [1, 1]: rejected ({e})");
                all = false;
            }
        }
        match run_node_i32(
            "add_fwd_i32",
            &[
                NodeInput {
                    name: "B",
                    sizes: &[3, groups as u64],
                    data: &[],
                    raw: Some((SYN_TYPE_INT32, &kbb)),
                },
                NodeInput {
                    name: "S",
                    sizes: &[3, groups as u64],
                    data: &[],
                    raw: Some((SYN_TYPE_INT32, &selb)),
                },
            ],
            &[3, groups as u64],
            core::ptr::null(),
            0,
        ) {
            Ok(out) => {
                let bad = (0..3 * groups)
                    .filter(|&j| out[j] != kb[j] + sel[j])
                    .count();
                all &= report(
                    "add_fwd_i32 [3, g] + [3, g]",
                    bad == 0,
                    &format!("({bad} wrong; first {:?})", &out[..6]),
                );
            }
            Err(e) => {
                println!("add_fwd_i32 [3, g] + [3, g]: rejected ({e})");
                all = false;
            }
        }
    }

    // f32 -> bf16 cast rounding against the host's round-to-nearest-even,
    // on values that sit at and around bf16 ties.
    {
        let vals: Vec<f32> = (0..256)
            .map(|j| {
                let base = 1.0f32 + (j as f32) / 128.0;
                base + (j % 3) as f32 * 0.001_953_125 + (j % 5) as f32 * 1e-4
            })
            .chain([25.298_222f32 * 0.123_45, -30000.0, 1e-3, 3.0e4, 0.0])
            .collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        match run_node(
            "cast_f32_to_bf16",
            &[NodeInput {
                name: "X",
                sizes: &[vals.len() as u64],
                data: &[],
                raw: Some((SYN_TYPE_F32, &bytes)),
            }],
            &[vals.len() as u64],
            core::ptr::null(),
            0,
        ) {
            Ok(out) => {
                let bad: Vec<String> = vals
                    .iter()
                    .zip(&out)
                    .filter(|(v, o)| bf16_to_f32(f32_to_bf16(**v)) != **o)
                    .map(|(v, o)| format!("{v} -> {o} (host {})", bf16_to_f32(f32_to_bf16(*v))))
                    .collect();
                all &= report(
                    "cast_f32_to_bf16 rounding",
                    bad.is_empty(),
                    &format!(
                        "({} of {} differ{})",
                        bad.len(),
                        vals.len(),
                        if bad.is_empty() {
                            String::new()
                        } else {
                            format!(": {}", bad[..bad.len().min(3)].join(", "))
                        }
                    ),
                );
            }
            Err(e) => {
                println!("cast_f32_to_bf16: rejected ({e})");
                all = false;
            }
        }
    }

    println!("{}", if all { "PASS" } else { "FAIL" });
    Ok(())
}
