//! Pin the contract of the optional third input of the MME nodes: a bias
//! added to the product, a candidate for the projection gemms of a model
//! with attention biases (Qwen2) in place of a broadcast `add` node after
//! each of them. Cases: `gemm` with a bias `[N]` over an output `[N, M]`
//! (the FCD), `[N, 1]` and `[N, M]`, `gemm` without `transpose_b`, and
//! `batch_gemm` over per-head weight blocks with a `[hd, 1, hpg, groups]`
//! bias. Small-integer operands make every product exact in f32 on both
//! sides, so the outputs tell whether the bias joins the f32 accumulator
//! before the bf16 rounding or the rounded product (a separate bf16 add).
//!
//! Findings (SynapseAI 1.24, 2026-09-05): every case compiles and every
//! output element is `bf16(bf16(A B^T) + bias)`, the bias added to the
//! rounded product, never to the f32 accumulator; the profiler shows why:
//! the graph compiler's `addMmeBias` pass takes the bias off the MME node
//! and adds it in a TPC `add_fwd_bf16` node of its own (`<node>_add_bias_
//! complex/add_fwd_bf16_0`; for `batch_gemm` a `tile` broadcast plus the
//! add). So the third input is the separate add node written differently,
//! and one the TPC fuser does not merge into the RoPE kernels the way it
//! merges explicit adds; the engine keeps the explicit adds (see
//! `QkvProj` in `model.rs`).
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-gemm-bias-test -- [case]`
//!
//! Without a case every case runs, each as its own recipe and launch.

use core::ffi::c_void;
use reng_synapse::{NodeInput, bf16_to_f32, f32_to_bf16, run_node, synGEMMParams};

/// Round through bf16, as the device sees an uploaded f32 value.
fn r(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// Host `out[i, j] = sum_p a[i, p] * b[j, p] + bias[j]` in f32 (exact for
/// the small-integer operands), returned unrounded together with the
/// product alone.
fn reference(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    per_row: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut prod = vec![0.0f32; m * n];
    let mut sum = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: f32 = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
            prod[i * n + j] = s;
            sum[i * n + j] = s + if per_row { bias[i * n + j] } else { bias[j] };
        }
    }
    (prod, sum)
}

/// One probe: a node over `ins` (name, device sizes, host data) into an
/// output of `out` sizes with `params`; `per_row` for a full `[N, M]` bias.
struct Case<'a> {
    label: &'a str,
    guid: &'a str,
    ins: Vec<(&'a str, Vec<u64>, &'a [f32])>,
    out: Vec<u64>,
    params: &'a synGEMMParams,
    per_row: bool,
}

struct Verdict {
    total: usize,
    pre: usize,
    post: usize,
    neither: usize,
    discriminating: usize,
    max_abs: f32,
}

/// Compare the device output with the two rounding hypotheses: the bias
/// added before the bf16 rounding (`bf16(s + b)`) or after it
/// (`bf16(bf16(s) + b)`).
fn judge(out: &[f32], prod: &[f32], sum: &[f32], bias_of: impl Fn(usize) -> f32) -> Verdict {
    let mut v = Verdict {
        total: out.len(),
        pre: 0,
        post: 0,
        neither: 0,
        discriminating: 0,
        max_abs: 0.0,
    };
    for (idx, (&o, (&s, &sb))) in out.iter().zip(prod.iter().zip(sum)).enumerate() {
        let pre = r(sb);
        let post = r(r(s) + bias_of(idx));
        if pre != post {
            v.discriminating += 1;
        }
        let hit_pre = o == pre;
        let hit_post = o == post;
        v.pre += usize::from(hit_pre);
        v.post += usize::from(hit_post);
        v.neither += usize::from(!hit_pre && !hit_post);
        v.max_abs = v.max_abs.max((o - sb).abs());
    }
    v
}

fn main() -> reng_core::Result<()> {
    let selected: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    // M tokens, K inputs, N outputs; N splits into heads for the batched case.
    let (m, k, n) = (7usize, 256usize, 96usize);
    let (hd, hpg, groups) = (16u64, 3u64, 2u64);
    assert_eq!(hd * hpg * groups, n as u64);
    // Operands in {1, 2} and {1, 2, 3}: products of 256 to 1536, exact in
    // f32, mostly beyond 512 where a bf16 ulp is 4 or 8.
    let a: Vec<f32> = (0..m * k)
        .map(|i| 1.0 + ((i * 7 + i / k) % 2) as f32)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| 1.0 + ((i * 11 + i / k * 5) % 3) as f32)
        .collect();
    // Biases in about (-2, 2), bf16 values: at those products the two
    // rounding orders differ for a good share of the elements.
    let bias: Vec<f32> = (0..n)
        .map(|j| r(((j * 13) % 41) as f32 * 0.1 - 2.0))
        .collect();
    let bias_full: Vec<f32> = (0..m * n)
        .map(|i| r(((i * 7) % 37) as f32 * 0.1 - 1.8))
        .collect();
    let (prod, sum) = reference(&a, &b, &bias, m, k, n, false);
    let (_, sum_full) = reference(&a, &b, &bias_full, m, k, n, true);

    let bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let nt = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let params = |p: &synGEMMParams| {
        (
            std::ptr::from_ref(p).cast::<c_void>(),
            core::mem::size_of::<synGEMMParams>() as u32,
        )
    };
    // B in its non-transposed device layout `[N, K]` (host `[K, N]`).
    let b_nt: Vec<f32> = (0..k * n).map(|i| b[(i % n) * k + i / n]).collect();
    // Bias as the per-head block of the batched case: the same `[N]`
    // values viewed as `[hd, 1, hpg, groups]`.
    let (mu, ku, nu) = (m as u64, k as u64, n as u64);
    let cases = [
        Case {
            label: "gemm bias [N]",
            guid: "gemm",
            ins: vec![
                ("A", vec![ku, mu], &a),
                ("B", vec![ku, nu], &b),
                ("BIAS", vec![nu], &bias),
            ],
            out: vec![nu, mu],
            params: &bt,
            per_row: false,
        },
        Case {
            label: "gemm bias [N, 1]",
            guid: "gemm",
            ins: vec![
                ("A", vec![ku, mu], &a),
                ("B", vec![ku, nu], &b),
                ("BIAS", vec![nu, 1], &bias),
            ],
            out: vec![nu, mu],
            params: &bt,
            per_row: false,
        },
        Case {
            label: "gemm bias [N, M]",
            guid: "gemm",
            ins: vec![
                ("A", vec![ku, mu], &a),
                ("B", vec![ku, nu], &b),
                ("BIAS", vec![nu, mu], &bias_full),
            ],
            out: vec![nu, mu],
            params: &bt,
            per_row: true,
        },
        Case {
            label: "gemm (no transpose_b) bias [N]",
            guid: "gemm",
            ins: vec![
                ("A", vec![ku, mu], &a),
                ("B", vec![nu, ku], &b_nt),
                ("BIAS", vec![nu], &bias),
            ],
            out: vec![nu, mu],
            params: &nt,
            per_row: false,
        },
        Case {
            label: "batch_gemm per-head bias [hd, 1, hpg, groups]",
            guid: "batch_gemm",
            ins: vec![
                ("A", vec![ku, mu, 1, 1], &a),
                ("B", vec![ku, hd, hpg, groups], &b),
                ("BIAS", vec![hd, 1, hpg, groups], &bias),
            ],
            out: vec![hd, mu, hpg, groups],
            params: &bt,
            per_row: false,
        },
    ];
    let mut failed = 0usize;
    for (ci, case) in cases.iter().enumerate() {
        if selected.is_some_and(|s| s != ci) {
            continue;
        }
        let (label, guid, out) = (case.label, case.guid, &case.out);
        let inputs: Vec<NodeInput<'_>> = case
            .ins
            .iter()
            .map(|(name, sizes, data)| NodeInput {
                name,
                sizes,
                data,
                raw: None,
            })
            .collect();
        let (pp, ps) = params(case.params);
        let outv = match run_node(guid, &inputs, out, pp, ps) {
            Ok(v) => v,
            Err(e) => {
                println!("case {ci} {label}: FAILED {e}");
                failed += 1;
                continue;
            }
        };
        // The batched output `[hd, M, hpg, groups]` (host `[groups, hpg, M,
        // hd]`) is reordered to the flat `[M, N]` before the comparison.
        let flat: Vec<f32> = if out.len() == 4 {
            let hd = hd as usize;
            let mut f = vec![0.0f32; m * n];
            for (idx, &v) in outv.iter().enumerate() {
                let d = idx % hd;
                let i = (idx / hd) % m;
                let head = idx / (hd * m);
                f[i * n + head * hd + d] = v;
            }
            f
        } else {
            outv
        };
        let (sum_ref, bias_of): (&[f32], Box<dyn Fn(usize) -> f32>) = if case.per_row {
            (&sum_full, Box::new(|idx| bias_full[idx]))
        } else {
            (&sum, Box::new(|idx| bias[idx % n]))
        };
        let v = judge(&flat, &prod, sum_ref, bias_of);
        let ok = v.neither == 0 && v.max_abs < 8.0;
        println!(
            "case {ci} {label}: {} elements, {} discriminating; match bias-before-rounding {}, \
             bias-after-rounding {}, neither {}; max |dev - (A B^T + bias)| {:.3}: {}",
            v.total,
            v.discriminating,
            v.pre,
            v.post,
            v.neither,
            v.max_abs,
            if ok { "ok" } else { "MISMATCH" }
        );
        if !ok {
            failed += 1;
        }
    }
    if failed == 0 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "{failed} gemm bias case(s) failed"
        )))
    }
}
