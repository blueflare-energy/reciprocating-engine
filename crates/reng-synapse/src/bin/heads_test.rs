//! Verify the `split` / `concat` head round trip (rotated head order) that
//! multi-head attention is built on. On divergence, prints which input head
//! block each output block matches (or whether it is zeroed) to distinguish a
//! permutation surprise from a readback race.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-heads-test -- [tokens] [hidden] [n_heads]`.

use reng_synapse::{bf16_to_f32, f32_to_bf16, split_rotate_concat_bf16, split_rotate_concat_cpu};

fn main() -> reng_core::Result<()> {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (tokens, hidden, n_heads) = (arg(1, 256usize), arg(2, 256usize), arg(3, 2usize));
    let hd = hidden / n_heads;
    // The device round-trips the input through bf16, and split/concat move
    // data losslessly, so the reference must use the bf16-rounded input and the
    // comparison can be exact.
    let x: Vec<f32> = (0..tokens * hidden)
        .map(|j| bf16_to_f32(f32_to_bf16((((j * 7 + 3) % 23) as f32 - 11.0) / 11.0)))
        .collect();

    println!("split/concat rotate: tokens={tokens}, hidden={hidden}, n_heads={n_heads}");
    let hpu = split_rotate_concat_bf16(&x, tokens, hidden, n_heads)?;
    let cpu = split_rotate_concat_cpu(&x, tokens, hidden, n_heads);
    let close = |a: f32, b: f32| a == b;
    let mism = hpu
        .iter()
        .zip(&cpu)
        .filter(|(a, b)| !close(**a, **b))
        .count();
    println!("mismatches vs rotated ref: {mism}/{}", tokens * hidden);

    if mism != 0 {
        // Per output block j: fraction matching each input block i, and zeros.
        for j in 0..n_heads {
            let mut line = format!("  out block {j}:");
            for i in 0..n_heads {
                let mut ok = 0usize;
                for tk in 0..tokens {
                    for d in 0..hd {
                        if close(hpu[tk * hidden + j * hd + d], x[tk * hidden + i * hd + d]) {
                            ok += 1;
                        }
                    }
                }
                line.push_str(&format!(
                    "  =in{i}: {:.1}%",
                    100.0 * ok as f32 / (tokens * hd) as f32
                ));
            }
            let zeros = (0..tokens)
                .flat_map(|tk| (0..hd).map(move |d| tk * hidden + j * hd + d))
                .filter(|&idx| hpu[idx] == 0.0)
                .count();
            line.push_str(&format!(
                "  zeros: {:.1}%",
                100.0 * zeros as f32 / (tokens * hd) as f32
            ));
            println!("{line}");
        }
        // Row profile: which token rows are wrong?
        let bad_rows = (0..tokens)
            .filter(|&tk| (0..hidden).any(|c| !close(hpu[tk * hidden + c], cpu[tk * hidden + c])))
            .count();
        println!("  token rows with any mismatch: {bad_rows}/{tokens}");
        return Err(reng_core::Error::Other(format!(
            "split/concat axis semantics wrong: {mism} mismatches"
        )));
    }
    println!("PASS");
    Ok(())
}
