//! Pin the kernel contracts a batched (all heads in one node) attention layer
//! needs, each against a CPU reference:
//!
//! 0. `batch_gemm` 4-D with the inner batch dim of B broadcast (query heads
//!    of one GQA group against a single K head), `transpose_b`;
//! 1. `batch_gemm` with A broadcast against a batched B (one activation
//!    against per-head weight blocks: all head projections in one node);
//! 2. `add_fwd_bf16` broadcasting a `[n, m, 1, 1]` mask over `[n, m, g, h]`;
//! 3. `softmax_fwd_bf16` over the FCD of a 4-D tensor;
//! 4. `rope_st2_fwd_bf16` on `[hd, rows, heads]` with a 2-D `[hd, rows]` table;
//! 5. `transpose` `[hd, heads, rows] -> [hd, rows, heads]`;
//! 6. `slice` of a batch dim: `[hd, rows, 5, g]` rows 0..3 of dim 2;
//! 7. `softmax_fwd_bf16` with a second (mask) input, added before the softmax.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-batched-test -- [case]`

use core::ffi::c_void;
use reng_synapse::{NodeInput, run_node, synGEMMParams, synSoftmaxParams};

#[repr(C)]
struct RopeParams {
    offset: u32,
    mode: i32,
}

/// `synTransposeParams`: `permutation[i]` is the source dim of output dim
/// `i`, over `MAX_DIMENSIONS_NUM` (5) entries.
#[repr(C)]
struct TransposeParams {
    permutation: [u32; 5],
    tensor_dim: u32,
}

/// `synSliceParams`: per listed axis, `[start, end)` with a step.
#[repr(C)]
struct SliceParams {
    axes: [u32; 5],
    starts: [u32; 5],
    ends: [u32; 5],
    steps: [u32; 5],
}

fn rel(h: &[f32], c: &[f32]) -> f32 {
    let num: f64 = h
        .iter()
        .zip(c)
        .map(|(x, y)| f64::from(*x - *y).powi(2))
        .sum();
    let den: f64 = c.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * (5 + seed) + 1) % 17) as f32 - 8.0) / 8.0)
        .collect()
}

fn bf16(v: f32) -> f32 {
    reng_synapse::bf16_to_f32(reng_synapse::f32_to_bf16(v))
}

fn params<T>(p: &T) -> (*const c_void, u32) {
    (
        (p as *const T).cast::<c_void>(),
        core::mem::size_of::<T>() as u32,
    )
}

/// Case 0: `C[g][j] = A[g][j] @ B[g]^T`, A `[k, m, hpg, groups]`, B `[k, n, 1, groups]`.
fn case_gqa() -> reng_core::Result<f32> {
    let (m, k, n, hpg, groups) = (32usize, 64usize, 512usize, 3usize, 3usize);
    let a = ramp(k * m * hpg * groups, 0);
    let b = ramp(k * n * groups, 1);
    let p = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "batch_gemm",
        &[
            NodeInput {
                name: "A",
                sizes: &[k as u64, m as u64, hpg as u64, groups as u64],
                data: &a,
            },
            NodeInput {
                name: "B",
                sizes: &[k as u64, n as u64, 1, groups as u64],
                data: &b,
            },
        ],
        &[n as u64, m as u64, hpg as u64, groups as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; n * m * hpg * groups];
    for g in 0..groups {
        for j in 0..hpg {
            for i in 0..m {
                for col in 0..n {
                    let mut s = 0.0;
                    for q in 0..k {
                        s += a[q + k * (i + m * (j + hpg * g))] * b[q + k * (col + n * g)];
                    }
                    c[col + n * (i + m * (j + hpg * g))] = s;
                }
            }
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 1: `C[h] = A @ B[h]`, A `[k, m, 1]` broadcast, B `[n, k, heads]`.
fn case_a_broadcast() -> reng_core::Result<f32> {
    let (m, k, n, heads) = (32usize, 256usize, 64usize, 9usize);
    let a = ramp(k * m, 2);
    let b = ramp(n * k * heads, 3);
    let p = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "batch_gemm",
        &[
            NodeInput {
                name: "A",
                sizes: &[k as u64, m as u64, 1],
                data: &a,
            },
            NodeInput {
                name: "B",
                sizes: &[n as u64, k as u64, heads as u64],
                data: &b,
            },
        ],
        &[n as u64, m as u64, heads as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; n * m * heads];
    for h in 0..heads {
        for i in 0..m {
            for col in 0..n {
                let mut s = 0.0;
                for q in 0..k {
                    s += a[q + k * i] * b[col + n * (q + k * h)];
                }
                c[col + n * (i + m * h)] = s;
            }
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 2: `[n, m, g, h] + [n, m, 1, 1]`.
fn case_broadcast_add() -> reng_core::Result<f32> {
    let (n, m, g, h) = (512usize, 32usize, 3usize, 3usize);
    let x = ramp(n * m * g * h, 4);
    let mask: Vec<f32> = (0..n * m)
        .map(|i| if i % 3 == 0 { -30000.0 } else { 0.0 })
        .collect();
    let hpu = run_node(
        "add_fwd_bf16",
        &[
            NodeInput {
                name: "X",
                sizes: &[n as u64, m as u64, g as u64, h as u64],
                data: &x,
            },
            NodeInput {
                name: "M",
                sizes: &[n as u64, m as u64, 1, 1],
                data: &mask,
            },
        ],
        &[n as u64, m as u64, g as u64, h as u64],
        core::ptr::null(),
        0,
    )?;
    let c: Vec<f32> = (0..n * m * g * h)
        .map(|i| bf16(bf16(x[i]) + mask[i % (n * m)]))
        .collect();
    Ok(rel(&hpu, &c))
}

/// Case 3: softmax over the FCD of `[n, m, g, h]`.
fn case_softmax4d() -> reng_core::Result<f32> {
    let (n, m, g, h) = (512usize, 32usize, 3usize, 3usize);
    let x: Vec<f32> = ramp(n * m * g * h, 5).iter().map(|v| v * 4.0).collect();
    let p = synSoftmaxParams { dim: 0 };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "softmax_fwd_bf16",
        &[NodeInput {
            name: "X",
            sizes: &[n as u64, m as u64, g as u64, h as u64],
            data: &x,
        }],
        &[n as u64, m as u64, g as u64, h as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; x.len()];
    for row in 0..m * g * h {
        let sl = &x[row * n..(row + 1) * n];
        let mx = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = sl.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        for (j, v) in e.iter().enumerate() {
            c[row * n + j] = v / sum;
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 4: RoPE on `[hd, rows, heads]` with a `[hd, rows]` table.
fn case_rope3d() -> reng_core::Result<f32> {
    let (hd, rows, heads) = (64usize, 32usize, 9usize);
    let x = ramp(hd * rows * heads, 6);
    let half = hd / 2;
    let mut sin = vec![0.0f32; hd * rows];
    let mut cos = vec![0.0f32; hd * rows];
    for r in 0..rows {
        for d in 0..hd {
            let ang = r as f32 * 10000f32.powf(-2.0 * ((d % half) as f32) / hd as f32);
            sin[r * hd + d] = ang.sin();
            cos[r * hd + d] = ang.cos();
        }
    }
    let p = RopeParams { offset: 0, mode: 0 };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "rope_st2_fwd_bf16",
        &[
            NodeInput {
                name: "X",
                sizes: &[hd as u64, rows as u64, heads as u64],
                data: &x,
            },
            NodeInput {
                name: "SIN",
                sizes: &[hd as u64, rows as u64],
                data: &sin,
            },
            NodeInput {
                name: "COS",
                sizes: &[hd as u64, rows as u64],
                data: &cos,
            },
        ],
        &[hd as u64, rows as u64, heads as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; x.len()];
    for h in 0..heads {
        for r in 0..rows {
            let b = hd * (r + rows * h);
            for d in 0..hd {
                let rot = if d < half {
                    -x[b + d + half]
                } else {
                    x[b + d - half]
                };
                c[b + d] = x[b + d] * cos[r * hd + d] + rot * sin[r * hd + d];
            }
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 5: `[hd, heads, rows] -> [hd, rows, heads]`.
fn case_transpose() -> reng_core::Result<f32> {
    let (hd, heads, rows) = (64usize, 9usize, 32usize);
    let x = ramp(hd * heads * rows, 7);
    let mut p = TransposeParams {
        permutation: [0; 5],
        tensor_dim: 3,
    };
    p.permutation[0] = 0;
    p.permutation[1] = 2;
    p.permutation[2] = 1;
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "transpose",
        &[NodeInput {
            name: "X",
            sizes: &[hd as u64, heads as u64, rows as u64],
            data: &x,
        }],
        &[hd as u64, rows as u64, heads as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; x.len()];
    for r in 0..rows {
        for h in 0..heads {
            for d in 0..hd {
                c[d + hd * (r + rows * h)] = bf16(x[d + hd * (h + heads * r)]);
            }
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 6: keep entries 0..3 of dim 2 of `[hd, rows, 5, g]`.
fn case_slice() -> reng_core::Result<f32> {
    let (hd, rows, n, g, keep) = (64usize, 32usize, 5usize, 3usize, 3usize);
    let x = ramp(hd * rows * n * g, 8);
    let p = SliceParams {
        axes: [0, 1, 2, 3, 0],
        starts: [0, 0, 0, 0, 0],
        ends: [hd as u32, rows as u32, keep as u32, g as u32, 0],
        steps: [1, 1, 1, 1, 0],
    };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "slice",
        &[NodeInput {
            name: "X",
            sizes: &[hd as u64, rows as u64, n as u64, g as u64],
            data: &x,
        }],
        &[hd as u64, rows as u64, keep as u64, g as u64],
        pp,
        ps,
    )?;
    let mut c = Vec::with_capacity(hd * rows * keep * g);
    for gi in 0..g {
        for j in 0..keep {
            for r in 0..rows {
                for d in 0..hd {
                    c.push(bf16(x[d + hd * (r + rows * (j + n * gi))]));
                }
            }
        }
    }
    Ok(rel(&hpu, &c))
}

/// Case 7: softmax over the FCD with a mask as a second input.
fn case_softmax_mask() -> reng_core::Result<f32> {
    let (n, m) = (512usize, 32usize);
    let x: Vec<f32> = ramp(n * m, 9).iter().map(|v| v * 4.0).collect();
    let mask: Vec<f32> = (0..n * m)
        .map(|i| if i % 5 == 0 { -30000.0 } else { 0.0 })
        .collect();
    let p = synSoftmaxParams { dim: 0 };
    let (pp, ps) = params(&p);
    let hpu = run_node(
        "softmax_fwd_bf16",
        &[
            NodeInput {
                name: "X",
                sizes: &[n as u64, m as u64],
                data: &x,
            },
            NodeInput {
                name: "M",
                sizes: &[n as u64, m as u64],
                data: &mask,
            },
        ],
        &[n as u64, m as u64],
        pp,
        ps,
    )?;
    let mut c = vec![0.0f32; x.len()];
    for row in 0..m {
        let sl: Vec<f32> = (0..n).map(|j| x[row * n + j] + mask[row * n + j]).collect();
        let mx = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = sl.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        for (j, v) in e.iter().enumerate() {
            c[row * n + j] = v / sum;
        }
    }
    Ok(rel(&hpu, &c))
}

fn main() -> reng_core::Result<()> {
    let names = [
        "batch_gemm GQA inner-batch broadcast (transpose_b)",
        "batch_gemm A broadcast against batched B",
        "add_fwd_bf16 broadcast [n,m,1,1] over [n,m,g,h]",
        "softmax_fwd_bf16 over FCD of 4-D",
        "rope_st2_fwd_bf16 on [hd,rows,heads] with [hd,rows] table",
        "transpose [hd,heads,rows] -> [hd,rows,heads]",
        "slice dim 2 of [hd,rows,5,g] to 0..3",
        "softmax_fwd_bf16 with a mask input",
    ];
    let selected: Vec<usize> = match std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(i) if i < names.len() => vec![i],
        _ => (0..names.len()).collect(),
    };
    let mut failed = 0;
    for ci in selected {
        let res = match ci {
            0 => case_gqa(),
            1 => case_a_broadcast(),
            2 => case_broadcast_add(),
            3 => case_softmax4d(),
            4 => case_rope3d(),
            5 => case_transpose(),
            6 => case_slice(),
            _ => case_softmax_mask(),
        };
        match res {
            Ok(r) if r < 0.02 => println!("case {ci}: {}: rel_L2={r:.4} ok", names[ci]),
            Ok(r) => {
                failed += 1;
                println!("case {ci}: {}: rel_L2={r:.4} MISMATCH", names[ci]);
            }
            Err(e) => {
                failed += 1;
                println!("case {ci}: {}: FAILED ({e})", names[ci]);
            }
        }
    }
    if failed == 0 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "{failed} contract checks failed"
        )))
    }
}
