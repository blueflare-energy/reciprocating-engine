//! Probe the fused attention guid `sdpa_recomp_fwd_bf16` (libSynapse):
//! inputs Q, K, V as `[head_dim, positions, heads, batch]`, an optional
//! additive mask, params `ns_Sdpa::Params`; compares the output with a host
//! attention over the same data for the decode shape (one query) and a
//! prefill block (eight queries), with and without the mask input.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-sdpa-test`

use core::ffi::c_void;
use reng_synapse::{NodeInput, bench_node, run_node};

/// `ns_Sdpa::Params`: `float scale; bool is_causal; ParamsOptionalMaskOut
/// dropout { float ratio; unsigned seed; bool disableMaskOut }; bool
/// is_inference` (C layout with padding).
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

#[allow(clippy::too_many_arguments)]
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    d: usize,
    nq: usize,
    nk: usize,
    h: usize,
    hkv: usize,
    scale: f32,
) -> Vec<f32> {
    // Row-major host layout of a `[d, n, h, 1]` device tensor: index
    // ((head * n) + pos) * d + i; query head hd reads kv head hd / (h / hkv).
    let mut out = vec![0.0f32; d * nq * h];
    for hd in 0..h {
        let kh = hd / (h / hkv);
        for iq in 0..nq {
            let qrow = &q[(hd * nq + iq) * d..(hd * nq + iq + 1) * d];
            let mut sc = vec![0.0f32; nk];
            for ik in 0..nk {
                let krow = &k[(kh * nk + ik) * d..(kh * nk + ik + 1) * d];
                let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                sc[ik] = dot * scale + mask.map_or(0.0, |m| m[iq * nk + ik]);
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13 + 3) % 29) as f32 - 14.0) / 28.0)
        .collect()
}

fn main() -> reng_core::Result<()> {
    let d = 64usize;
    // (h, hkv, nk, nq): MHA and GQA, decode and prefill shapes.
    for (h, hkv, nk, nq) in [
        (4usize, 4usize, 133usize, 1usize),
        (4, 4, 133, 8),
        (8, 2, 133, 1),
        (8, 2, 133, 8),
    ] {
        let k = fill(d * nk * hkv, 1);
        let v = fill(d * nk * hkv, 2);
        let q = fill(d * nq * h, 3);
        let scale = 1.0 / (d as f32).sqrt();
        let mask: Vec<f32> = (0..nq * nk)
            .map(|i| {
                let (iq, ik) = (i / nk, i % nk);
                if ik <= nk - nq + iq { 0.0 } else { -30000.0 }
            })
            .collect();
        let qs = [d as u64, nq as u64, h as u64, 1];
        let ks = [d as u64, nk as u64, hkv as u64, 1];
        let ms = [nk as u64, nq as u64, 1, 1];
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
                data: &mask,
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
                let want = host_attention(&q, &k, &v, Some(&mask), d, nq, nk, h, hkv, scale);
                println!(
                    "h {h} kv {hkv} nk {nk} nq {nq}: rel_L2 {:.4}",
                    rel_l2(&got, &want)
                );
            }
            Err(e) => println!(
                "h {h} kv {hkv} nk {nk} nq {nq}: rejected ({})",
                e.to_string().chars().take(70).collect::<String>()
            ),
        }
    }
    // GQA through the grouped view: the hpg query heads of a kv group are
    // extra query rows, Q' = [d, nq * hpg, groups, 1] (a free reshape of
    // [d, nq, hpg, groups]), K/V per group, mask tiled hpg times along q.
    for (h, hkv, nk, nq) in [(8usize, 2usize, 133usize, 1usize), (8, 2, 133, 8)] {
        let hpg = h / hkv;
        let k = fill(d * nk * hkv, 1);
        let v = fill(d * nk * hkv, 2);
        let q = fill(d * nq * h, 3);
        let scale = 1.0 / (d as f32).sqrt();
        let mask: Vec<f32> = (0..nq * nk)
            .map(|i| {
                let (iq, ik) = (i / nk, i % nk);
                if ik <= nk - nq + iq { 0.0 } else { -30000.0 }
            })
            .collect();
        let mask_t: Vec<f32> = (0..hpg).flat_map(|_| mask.iter().copied()).collect();
        let qs = [d as u64, (nq * hpg) as u64, hkv as u64, 1];
        let ks = [d as u64, nk as u64, hkv as u64, 1];
        let ms = [nk as u64, (nq * hpg) as u64, 1, 1];
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
                data: &mask_t,
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
                let want = host_attention(&q, &k, &v, Some(&mask), d, nq, nk, h, hkv, scale);
                println!(
                    "grouped view h {h} kv {hkv} nk {nk} nq {nq}: rel_L2 {:.4}",
                    rel_l2(&got, &want)
                );
            }
            Err(e) => println!(
                "grouped view h {h} kv {hkv} nk {nk} nq {nq}: rejected ({})",
                e.to_string().chars().take(70).collect::<String>()
            ),
        }
    }
    // Timing at model shapes (head_dim 128): decode over 1024 keys and a
    // 256-query prefill block, MHA 12 heads and GQA 16/2.
    let d = 128usize;
    for (h, hkv, nk, nq) in [
        (12usize, 12usize, 1024usize, 1usize),
        (16, 2, 1024, 1),
        (12, 12, 1024, 256),
        (16, 2, 1024, 256),
    ] {
        let hpg = h / hkv;
        let k = fill(d * nk * hkv, 1);
        let v = fill(d * nk * hkv, 2);
        let q = fill(d * nq * h, 3);
        let mask = vec![0.0f32; nq * hpg * nk];
        let qs = [d as u64, (nq * hpg) as u64, hkv as u64, 1];
        let ks = [d as u64, nk as u64, hkv as u64, 1];
        let ms = [nk as u64, (nq * hpg) as u64, 1, 1];
        let p = params(1.0 / (d as f32).sqrt());
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
            Ok((secs, _)) => println!(
                "time h {h} kv {hkv} nk {nk} nq {nq}: {:.1} us per launch",
                secs * 1e6
            ),
            Err(e) => println!(
                "time h {h} kv {hkv} nk {nk} nq {nq}: rejected ({})",
                e.to_string().chars().take(70).collect::<String>()
            ),
        }
    }
    Ok(())
}
