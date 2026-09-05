//! Time the attention gemms at prefill shapes in their possible
//! orientations: `qk` (scores = q k^T per head) and `av` (context = probs v)
//! as the engine builds them, and `av` with both operands transposed so the
//! MME's N dim is the token count instead of the head dim.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-attn-bench -- [t] [keys] [hd] [hpg] [groups] [iters]`

use core::ffi::c_void;
use reng_synapse::{NodeInput, bench_node, synGEMMParams};

#[allow(clippy::too_many_arguments)]
fn run(
    label: &str,
    a_sizes: &[u64],
    b_sizes: &[u64],
    out_sizes: &[u64],
    ta: bool,
    tb: bool,
    flop: f64,
    iters: usize,
) -> reng_core::Result<()> {
    let a: Vec<f32> = (0..a_sizes.iter().product::<u64>())
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let b: Vec<f32> = (0..b_sizes.iter().product::<u64>())
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let params = synGEMMParams {
        transpose_a: ta,
        transpose_b: tb,
    };
    let ins = [
        NodeInput {
            name: "A",
            sizes: a_sizes,
            data: &a,
            raw: None,
        },
        NodeInput {
            name: "B",
            sizes: b_sizes,
            data: &b,
            raw: None,
        },
    ];
    let (secs, _) = bench_node(
        "batch_gemm",
        &ins,
        out_sizes,
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
        iters,
    )?;
    println!(
        "{label}: {:.3} ms/launch, {:.1} TFLOP/s",
        secs * 1e3,
        flop / secs / 1e12
    );
    Ok(())
}

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: u64| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (t, keys, hd, hpg, g) = (
        arg(1, 1024),
        arg(2, 1024),
        arg(3, 64),
        arg(4, 1),
        arg(5, 32),
    );
    let iters = arg(6, 50) as usize;
    let heads = (hpg * g) as f64;
    let flop = 2.0 * (t as f64) * (keys as f64) * (hd as f64) * heads;
    println!("t={t} keys={keys} hd={hd} hpg={hpg} groups={g}");
    // scores[keys, t] = q[hd, t]^T k[hd, keys]: A = q, B = k (transposed).
    run(
        "qk   A=q[hd,t,hpg,g]      B=k[hd,keys,1,g] tb",
        &[hd, t, hpg, g],
        &[hd, keys, 1, g],
        &[keys, t, hpg, g],
        false,
        true,
        flop,
        iters,
    )?;
    // context[hd, t] = probs[keys, t] over v[hd, keys]: A = probs, B = v.
    run(
        "av   A=probs[keys,t,hpg,g] B=v[hd,keys,1,g]",
        &[keys, t, hpg, g],
        &[hd, keys, 1, g],
        &[hd, t, hpg, g],
        false,
        false,
        flop,
        iters,
    )?;
    // context^T[t, hd] = v[hd, keys]^T probs[keys, t]^T: A = v (transposed),
    // B = probs (transposed); N = t.
    run(
        "avT  A=v[hd,keys,1,g] ta   B=probs[keys,t,hpg,g] tb",
        &[hd, keys, 1, g],
        &[keys, t, hpg, g],
        &[t, hd, hpg, g],
        true,
        true,
        flop,
        iters,
    )?;
    // context^T[t, hd] with v stored transposed, [keys, hd]: A = vT, B = probs (transposed).
    run(
        "avT2 A=vT[keys,hd,1,g]     B=probs[keys,t,hpg,g] tb",
        &[keys, hd, 1, g],
        &[keys, t, hpg, g],
        &[t, hd, hpg, g],
        false,
        true,
        flop,
        iters,
    )?;
    // Heads of a group merged into M: probs as [keys, t * hpg, 1, g].
    if hpg > 1 {
        run(
            "avM  A=probs[keys,t*hpg,1,g] B=v[hd,keys,1,g]",
            &[keys, t * hpg, 1, g],
            &[hd, keys, 1, g],
            &[hd, t * hpg, 1, g],
            false,
            false,
            flop,
            iters,
        )?;
    }
    Ok(())
}
