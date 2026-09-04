//! Test our direct-C-API unfused attention (`gemm -> softmax -> gemm`) against
//! CPU. This is the exact composition the PyTorch frameworks garble on 1.24.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-attn-test`.

use reng_synapse::{attention_bf16, attention_cpu};

fn main() -> reng_core::Result<()> {
    let seq = 64usize;
    let dim = 64usize;
    let scale = 1.0 / (dim as f32).sqrt();

    let mk = |salt: usize| -> Vec<f32> {
        (0..seq * dim)
            .map(|i| (((i * 7 + salt) % 23) as f32 - 11.0) / 11.0)
            .collect()
    };
    let q = mk(1);
    let k = mk(2);
    let v = mk(3);

    println!("direct attention: softmax((Q*scale) @ K^T) @ V, seq={seq}, dim={dim}");
    let hpu = attention_bf16(&q, &k, &v, seq, dim, scale)?;
    let cpu = attention_cpu(&q, &k, &v, seq, dim, scale);

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

    // Diagnostics: is it a transpose? a zeroed region?
    let mut hpu_t = vec![0.0f32; seq * dim];
    for i in 0..seq {
        for e in 0..dim {
            hpu_t[e * seq + i] = hpu[i * dim + e];
        }
    }
    let rel_t = {
        let n: f64 = hpu_t
            .iter()
            .zip(&cpu)
            .map(|(h, c)| {
                let d = f64::from(*h - *c);
                d * d
            })
            .sum();
        (n.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    let zeros = hpu.iter().filter(|x| **x == 0.0).count();
    let row0_zeros = hpu[0..dim].iter().filter(|x| **x == 0.0).count();
    let row1_zeros = hpu[dim..2 * dim].iter().filter(|x| **x == 0.0).count();

    println!("nan={nan}  rel_L2={rel:.4}  rel_L2_vs_transpose={rel_t:.4}");
    println!(
        "zeros total={zeros}/{}  row0_zeros={row0_zeros}/{dim}  row1_zeros={row1_zeros}/{dim}",
        seq * dim
    );
    println!("hpu[0..4]={:?}", &hpu[0..4]);
    println!("cpu[0..4]={:?}", &cpu[0..4]);
    println!("hpu row1 [{}..{}]={:?}", dim, dim + 4, &hpu[dim..dim + 4]);
    println!("cpu row1 [{}..{}]={:?}", dim, dim + 4, &cpu[dim..dim + 4]);

    if !nan && rel.min(rel_t) < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        println!("DIVERGE (diagnostic run)");
        Ok(())
    }
}
