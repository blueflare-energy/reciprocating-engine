//! Check and time kernels that write a block of rows into the KV cache:
//! `scatter_nd_update_fwd_bf16` (one index tuple per row and group, as the
//! engine uses it) against `index_copy_fwd_bf16` (one index vector shared by
//! every group).
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-scatter-bench -- [t] [keys] [hd] [groups] [iters]`

use core::ffi::c_void;
use reng_synapse::{NodeInput, SYN_TYPE_INT32, bench_node, run_node};

#[repr(C)]
struct ScatterParams {
    mode: i32,
}

#[repr(C)]
struct IndexCopyParams {
    axis: u32,
}

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: u64| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (t, keys, hd, g) = (arg(1, 1024), arg(2, 1041), arg(3, 64), arg(4, 32));
    let iters = arg(5, 30) as usize;

    // Small semantic check of index_copy: rows [1, 4, 5] of a [4, 6, 1, 2]
    // cache take the source's three rows, per group.
    {
        let (hd, keys, t, g) = (4u64, 6u64, 3u64, 2u64);
        let cache: Vec<f32> = (0..hd * keys * g).map(|i| -(i as f32)).collect();
        let src: Vec<f32> = (0..hd * t * g).map(|i| 100.0 + i as f32).collect();
        let idx: Vec<i32> = vec![1, 4, 5];
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = IndexCopyParams { axis: 1 };
        let ins = [
            NodeInput {
                name: "SELF",
                sizes: &[hd, keys, 1, g],
                data: &cache,
                raw: None,
            },
            NodeInput {
                name: "INDEX",
                sizes: &[t],
                data: &[],
                raw: Some((SYN_TYPE_INT32, &idx_bytes)),
            },
            NodeInput {
                name: "SRC",
                sizes: &[hd, t, 1, g],
                data: &src,
                raw: None,
            },
        ];
        match run_node(
            "index_copy_fwd_bf16",
            &ins,
            &[hd, keys, 1, g],
            (&raw const params).cast::<c_void>(),
            core::mem::size_of::<IndexCopyParams>() as u32,
        ) {
            Ok(out) => {
                let mut bad = 0;
                for gi in 0..g {
                    for k in 0..keys {
                        for d in 0..hd {
                            let o = out[(d + k * hd + gi * hd * keys) as usize];
                            let want = match idx.iter().position(|&p| p as u64 == k) {
                                Some(r) => src[(d + r as u64 * hd + gi * hd * t) as usize],
                                None => cache[(d + k * hd + gi * hd * keys) as usize],
                            };
                            if (o - want).abs() > 0.5 {
                                bad += 1;
                            }
                        }
                    }
                }
                println!("index_copy semantic check: {bad} wrong of {}", out.len());
            }
            Err(e) => println!("index_copy_fwd_bf16 unavailable: {e}"),
        }
    }

    println!("t={t} keys={keys} hd={hd} groups={g}");
    let cache: Vec<f32> = (0..hd * keys * g).map(|i| (i % 5) as f32).collect();
    let src: Vec<f32> = (0..hd * t * g).map(|i| (i % 7) as f32).collect();
    // ScatterND: tuples (g, 0, position) for update r + t * g.
    let mut tuples: Vec<u8> = Vec::new();
    for gi in 0..g {
        for r in 0..t {
            for v in [gi as i32, 0i32, r as i32] {
                tuples.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let sp = ScatterParams { mode: 0 };
    let ins = [
        NodeInput {
            name: "CACHE",
            sizes: &[hd, keys, 1, g],
            data: &cache,
            raw: None,
        },
        NodeInput {
            name: "KIDX",
            sizes: &[3, t * g],
            data: &[],
            raw: Some((SYN_TYPE_INT32, &tuples)),
        },
        NodeInput {
            name: "UPD",
            sizes: &[hd, t * g],
            data: &src,
            raw: None,
        },
    ];
    let (secs, _) = bench_node(
        "scatter_nd_update_fwd_bf16",
        &ins,
        &[hd, keys, 1, g],
        (&raw const sp).cast::<c_void>(),
        core::mem::size_of::<ScatterParams>() as u32,
        iters,
    )?;
    println!("scatter_nd_update: {:.3} ms/launch", secs * 1e3);

    let idx: Vec<u8> = (0..t as i32).flat_map(|v| v.to_le_bytes()).collect();
    let ip = IndexCopyParams { axis: 1 };
    let ins = [
        NodeInput {
            name: "SELF",
            sizes: &[hd, keys, 1, g],
            data: &cache,
            raw: None,
        },
        NodeInput {
            name: "INDEX",
            sizes: &[t],
            data: &[],
            raw: Some((SYN_TYPE_INT32, &idx)),
        },
        NodeInput {
            name: "SRC",
            sizes: &[hd, t, 1, g],
            data: &src,
            raw: None,
        },
    ];
    match bench_node(
        "index_copy_fwd_bf16",
        &ins,
        &[hd, keys, 1, g],
        (&raw const ip).cast::<c_void>(),
        core::mem::size_of::<IndexCopyParams>() as u32,
        iters,
    ) {
        Ok((secs, _)) => println!("index_copy: {:.3} ms/launch", secs * 1e3),
        Err(e) => println!("index_copy bench failed: {e}"),
    }
    Ok(())
}
