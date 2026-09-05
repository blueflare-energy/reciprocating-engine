//! Test whether a multi-node graph (`A * W^depth`), built directly through the
//! SynapseAI C API, computes correctly — the frameworks (transformers, vLLM)
//! garble the composed graph on the 1.24 stack; this checks our direct path.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-mme-chain`.

use reng_synapse::{matmul_chain_bf16, matmul_chain_cpu};

fn main() -> reng_core::Result<()> {
    let depth = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(12usize);
    let d = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(512usize);
    let s = 1.0 / (d as f32).sqrt(); // keep the chain O(1) in magnitude

    let a: Vec<f32> = (0..d * d)
        .map(|i| ((((i * 7 + 3) % 15) as f32 - 7.0) / 7.0) * s)
        .collect();
    let w: Vec<f32> = (0..d * d)
        .map(|i| ((((i * 13 + 5) % 15) as f32 - 7.0) / 7.0) * s)
        .collect();

    println!("direct chain: A * W^{depth}, d={d} ({depth} chained gemm nodes, one recipe)");
    let hpu = matmul_chain_bf16(&a, &w, d, depth)?;
    let cpu = matmul_chain_cpu(&a, &w, d, depth);

    let nan = hpu.iter().any(|x| !x.is_finite());
    let num: f64 = hpu
        .iter()
        .zip(&cpu)
        .map(|(h, c)| {
            let dd = f64::from(*h - *c);
            dd * dd
        })
        .sum();
    let den: f64 = cpu.iter().map(|c| f64::from(*c) * f64::from(*c)).sum();
    let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
    let hmax = hpu.iter().copied().fold(0.0f32, |m, x| m.max(x.abs()));
    let cmax = cpu.iter().copied().fold(0.0f32, |m, x| m.max(x.abs()));

    println!("nan={nan}  rel_L2={rel:.4}  hpu_absmax={hmax:.4}  cpu_absmax={cmax:.4}");
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);

    if !nan && rel < 0.5 {
        println!("PASS: our direct {depth}-node graph composes correctly here");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "DIVERGE: nan={nan} rel_L2={rel:.3} — our direct multi-op graph reproduces the miscompute"
        )))
    }
}
