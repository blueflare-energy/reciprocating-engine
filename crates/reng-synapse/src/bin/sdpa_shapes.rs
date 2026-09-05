//! Probe the mask and batch shapes `sdpa_recomp_fwd_bf16` accepts around
//! the engine's attention layouts (see `reng-sdpa-test` for the kernel's
//! basic contract): whether a size-1 query dim of the mask broadcasts,
//! whether the mask may carry a batch dim (the batched decode recipe has
//! one mask row per sequence), whether the mask may carry the heads dim,
//! and whether a K/V heads dim of 1 broadcasts over the query heads (the
//! engine's natural GQA view). Also checks the `broadcast` guid (the
//! device-side mask tiling) and, with `--time`, times the kernel at model
//! shapes.
//!
//! Findings (SynapseAI 1.24, 2026-09-05): every layout passes at rel_L2
//! 0.003 to 0.005 against the host: a size-1 query dim of the mask
//! broadcasts, the mask may carry the batch dim and the heads dim, K/V
//! with a heads dim of 1 broadcast over the query heads of the same
//! group (the engine's `[hd, keys, 1, groups]` cache against
//! `[hd, t, hpg, groups]` queries, so no grouped view or mask tiling is
//! needed), and five-dimensional tensors work with one mask row per
//! sequence. The timing loop (50 launches of one recipe queued back to
//! back) hung once at the grouped view `[128, 7, 4, 1]` over 1025 keys
//! (the readback never completed) and once raised a device error at the
//! 5-D `[128, 1, 7, 4, 8]` shape (every TPC reported
//! `tpc_illegal_instruction`; the driver reset the card), while the same
//! shapes run for hundreds of sequential launches inside the engine.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-sdpa-shapes -- [--time]`

use core::ffi::c_void;
use reng_synapse::{NodeInput, bench_node, run_node};

/// `ns_Sdpa::Params`, see `reng-sdpa-test`.
#[repr(C)]
struct SdpaParams {
    scale: f32,
    is_causal: u8,
    _pad0: [u8; 3],
    dropout_ratio: f32,
    dropout_seed: u32,
    disable_mask_out: u8,
    _pad1: [u8; 3],
    is_inference: u8,
    _pad2: [u8; 3],
}

fn params(scale: f32) -> SdpaParams {
    SdpaParams {
        scale,
        is_causal: 0,
        _pad0: [0; 3],
        dropout_ratio: 0.0,
        dropout_seed: 0,
        disable_mask_out: 1,
        _pad1: [0; 3],
        is_inference: 1,
        _pad2: [0; 3],
    }
}

/// Host attention over one batch element of `[d, n, h, 1]` tensors (row
/// major: `((head * n) + pos) * d + i`); query head `hd` reads kv head
/// `hd / (h / hkv)`; `mask(iq, ik)` is additive.
#[allow(clippy::too_many_arguments)]
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &dyn Fn(usize, usize) -> f32,
    d: usize,
    nq: usize,
    nk: usize,
    h: usize,
    hkv: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; d * nq * h];
    for hd in 0..h {
        let kh = hd / (h / hkv);
        for iq in 0..nq {
            let qrow = &q[(hd * nq + iq) * d..(hd * nq + iq + 1) * d];
            let mut sc = vec![0.0f32; nk];
            for ik in 0..nk {
                let krow = &k[(kh * nk + ik) * d..(kh * nk + ik + 1) * d];
                let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                sc[ik] = dot * scale + mask(iq, ik);
            }
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|x| (x - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            let orow = &mut out[(hd * nq + iq) * d..(hd * nq + iq + 1) * d];
            for ik in 0..nk {
                let vrow = &v[(kh * nk + ik) * d..(kh * nk + ik + 1) * d];
                let p = e[ik] / s;
                for i in 0..d {
                    orow[i] += p * vrow[i];
                }
            }
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13 + 3) % 29) as f32 - 14.0) / 28.0)
        .collect()
}

/// The reference mask of batch element `b`: query `iq` admits the keys up
/// to `nk - nq + iq - 9 * b` and no further back than 120 before that, so
/// the batch elements differ and a wrong broadcast shows.
fn full_mask(nk: usize, nq: usize, b: usize, iq: usize, ik: usize) -> f32 {
    let last = nk - nq + iq - 9 * b;
    if ik <= last && ik + 120 > last {
        0.0
    } else {
        -30000.0
    }
}

/// How the mask input is laid out.
#[derive(Clone, Copy)]
enum Mask {
    /// `[nk, nq * hpg, 1, nb]`: tiled over the query heads of a group.
    Tiled,
    /// `[nk, 1, 1, nb]`: one row per batch element (decode).
    Untiled,
    /// `[nk, nq * hpg, groups, nb]`: every head its own rows.
    Full,
    /// `[nk, nq, 1, 1]`: the engine's own layout (natural GQA view).
    Natural,
}

/// Q as `[d, nq * hpg, hkv, nb]` (grouped) or `[d, nq, hpg, hkv]` with
/// K/V `[d, nk, 1, hkv]` (natural), which takes `nb` as a fifth dim.
#[derive(Clone, Copy)]
enum View {
    Grouped,
    Natural,
}

#[allow(clippy::too_many_arguments)]
fn probe(
    label: &str,
    d: usize,
    nq: usize,
    nk: usize,
    h: usize,
    hkv: usize,
    nb: usize,
    view: View,
    mask: Mask,
) {
    let hpg = h / hkv;
    let per_b = d * nk * hkv;
    let k = fill(per_b * nb, 1);
    let v = fill(per_b * nb, 2);
    let q = fill(d * nq * h * nb, 3);
    let scale = 1.0 / (d as f32).sqrt();
    let (qs, ks): (Vec<u64>, Vec<u64>) = match view {
        View::Grouped => (
            vec![d as u64, (nq * hpg) as u64, hkv as u64, nb as u64],
            vec![d as u64, nk as u64, hkv as u64, nb as u64],
        ),
        View::Natural if nb == 1 => (
            vec![d as u64, nq as u64, hpg as u64, hkv as u64],
            vec![d as u64, nk as u64, 1, hkv as u64],
        ),
        View::Natural => (
            vec![d as u64, nq as u64, hpg as u64, hkv as u64, nb as u64],
            vec![d as u64, nk as u64, 1, hkv as u64, nb as u64],
        ),
    };
    let (ms, mdata): (Vec<u64>, Vec<f32>) = match mask {
        Mask::Tiled => (
            vec![nk as u64, (nq * hpg) as u64, 1, nb as u64],
            (0..nb * nq * hpg * nk)
                .map(|i| {
                    let (ik, r, b) = (i % nk, (i / nk) % (nq * hpg), i / (nk * nq * hpg));
                    full_mask(nk, nq, b, r % nq, ik)
                })
                .collect(),
        ),
        Mask::Untiled => {
            assert_eq!(nq, 1);
            (
                vec![nk as u64, 1, 1, nb as u64],
                (0..nb * nk)
                    .map(|i| full_mask(nk, nq, i / nk, 0, i % nk))
                    .collect(),
            )
        }
        Mask::Full => (
            vec![nk as u64, (nq * hpg) as u64, hkv as u64, nb as u64],
            (0..nb * hkv * nq * hpg * nk)
                .map(|i| {
                    let (ik, r) = (i % nk, (i / nk) % (nq * hpg));
                    let b = i / (nk * nq * hpg * hkv);
                    full_mask(nk, nq, b, r % nq, ik)
                })
                .collect(),
        ),
        Mask::Natural if nb == 1 => (
            vec![nk as u64, nq as u64, 1, 1],
            (0..nq * nk)
                .map(|i| full_mask(nk, nq, 0, i / nk, i % nk))
                .collect(),
        ),
        Mask::Natural => (
            vec![nk as u64, nq as u64, 1, 1, nb as u64],
            (0..nb * nq * nk)
                .map(|i| full_mask(nk, nq, i / (nq * nk), (i / nk) % nq, i % nk))
                .collect(),
        ),
    };
    let p = params(scale);
    let ins = [
        NodeInput {
            name: "Q",
            sizes: &qs,
            data: &q,
            raw: None,
        },
        NodeInput {
            name: "K",
            sizes: &ks,
            data: &k,
            raw: None,
        },
        NodeInput {
            name: "V",
            sizes: &ks,
            data: &v,
            raw: None,
        },
        NodeInput {
            name: "M",
            sizes: &ms,
            data: &mdata,
            raw: None,
        },
    ];
    match run_node(
        "sdpa_recomp_fwd_bf16",
        &ins,
        &qs,
        (&raw const p).cast::<c_void>(),
        core::mem::size_of::<SdpaParams>() as u32,
    ) {
        Ok(got) => {
            let per_q = d * nq * h;
            let want: Vec<f32> = (0..nb)
                .flat_map(|b| {
                    host_attention(
                        &q[b * per_q..(b + 1) * per_q],
                        &k[b * per_b..(b + 1) * per_b],
                        &v[b * per_b..(b + 1) * per_b],
                        &|iq, ik| full_mask(nk, nq, b, iq, ik),
                        d,
                        nq,
                        nk,
                        h,
                        hkv,
                        scale,
                    )
                })
                .collect();
            println!("{label}: rel_L2 {:.4}", rel_l2(&got, &want));
        }
        Err(e) => println!(
            "{label}: rejected ({})",
            e.to_string()
                .chars()
                .filter(|c| !c.is_control())
                .take(60)
                .collect::<String>()
        ),
    }
}

/// The `broadcast` guid: `[nk, nq, 1, 1]` to `[nk, nq, hpg, 1]`.
fn probe_broadcast(nk: usize, nq: usize, hpg: usize) {
    let src: Vec<f32> = (0..nk * nq).map(|i| (i % 13) as f32 - 6.0).collect();
    let ins = [NodeInput {
        name: "M",
        sizes: &[nk as u64, nq as u64, 1, 1],
        data: &src,
        raw: None,
    }];
    let outs = [nk as u64, nq as u64, hpg as u64, 1];
    match run_node("broadcast", &ins, &outs, core::ptr::null(), 0) {
        Ok(got) => {
            let want: Vec<f32> = (0..hpg).flat_map(|_| src.iter().copied()).collect();
            println!(
                "broadcast [{nk}, {nq}, 1, 1] -> [{nk}, {nq}, {hpg}, 1]: rel_L2 {:.4}",
                rel_l2(&got, &want)
            );
        }
        Err(e) => println!(
            "broadcast: rejected ({})",
            e.to_string()
                .chars()
                .filter(|c| !c.is_control())
                .take(60)
                .collect::<String>()
        ),
    }
}

/// Time a model shape: the natural view (a fifth dim for `nb > 1`) or the
/// grouped view with a tiled mask.
#[allow(clippy::too_many_arguments)]
fn time(
    label: &str,
    d: usize,
    nq: usize,
    nk: usize,
    hpg: usize,
    groups: usize,
    nb: usize,
    view: View,
) {
    let k = fill(d * nk * groups * nb, 1);
    let v = fill(d * nk * groups * nb, 2);
    let q = fill(d * nq * hpg * groups * nb, 3);
    let (qs, ks, ms): (Vec<u64>, Vec<u64>, Vec<u64>) = match view {
        View::Grouped => (
            vec![d as u64, (nq * hpg) as u64, groups as u64, nb as u64],
            vec![d as u64, nk as u64, groups as u64, nb as u64],
            vec![nk as u64, (nq * hpg) as u64, 1, nb as u64],
        ),
        View::Natural if nb == 1 => (
            vec![d as u64, nq as u64, hpg as u64, groups as u64],
            vec![d as u64, nk as u64, 1, groups as u64],
            vec![nk as u64, nq as u64, 1, 1],
        ),
        View::Natural => (
            vec![d as u64, nq as u64, hpg as u64, groups as u64, nb as u64],
            vec![d as u64, nk as u64, 1, groups as u64, nb as u64],
            vec![nk as u64, nq as u64, 1, 1, nb as u64],
        ),
    };
    let mask = vec![0.0f32; ms.iter().product::<u64>() as usize];
    let p = params(1.0);
    let ins = [
        NodeInput {
            name: "Q",
            sizes: &qs,
            data: &q,
            raw: None,
        },
        NodeInput {
            name: "K",
            sizes: &ks,
            data: &k,
            raw: None,
        },
        NodeInput {
            name: "V",
            sizes: &ks,
            data: &v,
            raw: None,
        },
        NodeInput {
            name: "M",
            sizes: &ms,
            data: &mask,
            raw: None,
        },
    ];
    match bench_node(
        "sdpa_recomp_fwd_bf16",
        &ins,
        &qs,
        (&raw const p).cast::<c_void>(),
        core::mem::size_of::<SdpaParams>() as u32,
        50,
    ) {
        Ok((secs, _)) => println!("time {label}: {:.1} us per launch", secs * 1e6),
        Err(e) => println!(
            "time {label}: rejected ({})",
            e.to_string()
                .chars()
                .filter(|c| !c.is_control())
                .take(60)
                .collect::<String>()
        ),
    }
}

fn main() -> reng_core::Result<()> {
    let (d, nk, h, hkv) = (64usize, 133usize, 8usize, 2usize);
    // Known good (reng-sdpa-test): grouped view, tiled mask, one batch.
    probe(
        "grouped tiled nq 8",
        d,
        8,
        nk,
        h,
        hkv,
        1,
        View::Grouped,
        Mask::Tiled,
    );
    probe(
        "grouped tiled nq 1",
        d,
        1,
        nk,
        h,
        hkv,
        1,
        View::Grouped,
        Mask::Tiled,
    );
    // Does a size-1 query dim of the mask broadcast over the hpg rows?
    probe(
        "grouped untiled nq 1",
        d,
        1,
        nk,
        h,
        hkv,
        1,
        View::Grouped,
        Mask::Untiled,
    );
    // Batch dim on Q/K/V with a per-batch mask, tiled and untiled, and
    // with the heads dim too.
    probe(
        "grouped tiled nq 1 B 2",
        d,
        1,
        nk,
        h,
        hkv,
        2,
        View::Grouped,
        Mask::Tiled,
    );
    probe(
        "grouped untiled nq 1 B 2",
        d,
        1,
        nk,
        h,
        hkv,
        2,
        View::Grouped,
        Mask::Untiled,
    );
    probe(
        "grouped full nq 1 B 2",
        d,
        1,
        nk,
        h,
        hkv,
        2,
        View::Grouped,
        Mask::Full,
    );
    probe(
        "grouped tiled nq 8 B 2",
        d,
        8,
        nk,
        h,
        hkv,
        2,
        View::Grouped,
        Mask::Tiled,
    );
    // The engine's natural GQA view: K/V heads dim 1 against hpg.
    probe(
        "natural nq 1",
        d,
        1,
        nk,
        h,
        hkv,
        1,
        View::Natural,
        Mask::Natural,
    );
    probe(
        "natural nq 8",
        d,
        8,
        nk,
        h,
        hkv,
        1,
        View::Natural,
        Mask::Natural,
    );
    // MHA in the natural view (hpg 1): 4-D with a size-1 heads dim.
    probe(
        "natural mha nq 8",
        d,
        8,
        nk,
        4,
        4,
        1,
        View::Natural,
        Mask::Natural,
    );
    // Device-side mask tiling.
    probe_broadcast(nk, 8, 4);
    probe_broadcast(1025, 256, 7);
    // Five-dimensional tensors, the batched decode recipe's layout: the
    // sequence batch outermost, one mask row per sequence.
    probe(
        "natural 5d nq 1 B 2",
        d,
        1,
        nk,
        h,
        hkv,
        2,
        View::Natural,
        Mask::Natural,
    );
    probe(
        "natural 5d nq 8 B 2",
        d,
        8,
        nk,
        h,
        hkv,
        2,
        View::Natural,
        Mask::Natural,
    );
    // The natural view at every target model's shape (hd, hpg, groups) over
    // a 1025-key cache (capacity 1024 plus the trash slot) and a 4097-key
    // one: decode (one query), a 256-row prefill block, eight decode
    // sequences (5-D); correctness against the host, then the launch time.
    let models = [
        ("qwen7b", 128usize, 7usize, 4usize),
        ("qwen1.5b", 128, 6, 2),
        ("qwen3-0.6b", 128, 2, 8),
        ("llama3b", 128, 3, 8),
        ("dsl8b", 128, 4, 8),
        ("smollm1.7b", 64, 1, 32),
        ("smollm135m", 64, 3, 3),
        ("phi3mini", 96, 1, 32),
        ("gemma270m", 256, 4, 1),
    ];
    for (name, hd, hpg, groups) in models {
        let h = hpg * groups;
        for nk in [1025usize, 4097] {
            probe(
                &format!("{name} decode {nk}"),
                hd,
                1,
                nk,
                h,
                groups,
                1,
                View::Natural,
                Mask::Natural,
            );
        }
        probe(
            &format!("{name} prefill 256 x 1025"),
            hd,
            256,
            1025,
            h,
            groups,
            1,
            View::Natural,
            Mask::Natural,
        );
        probe(
            &format!("{name} decode 1025 B 8"),
            hd,
            1,
            1025,
            h,
            groups,
            8,
            View::Natural,
            Mask::Natural,
        );
    }
    // Launch times, only on request: `bench_node` queues 50 launches of
    // one recipe back to back, which at two of these shapes did not end
    // well (see the module doc).
    if !std::env::args().any(|a| a == "--time") {
        return Ok(());
    }
    for (name, hd, hpg, groups) in models {
        time(
            &format!("{name} decode 1025"),
            hd,
            1,
            1025,
            hpg,
            groups,
            1,
            View::Natural,
        );
        time(
            &format!("{name} decode 4097"),
            hd,
            1,
            4097,
            hpg,
            groups,
            1,
            View::Natural,
        );
        time(
            &format!("{name} prefill 256 x 1025"),
            hd,
            256,
            1025,
            hpg,
            groups,
            1,
            View::Natural,
        );
        time(
            &format!("{name} decode 1025 B 8"),
            hd,
            1,
            1025,
            hpg,
            groups,
            8,
            View::Natural,
        );
    }
    Ok(())
}
