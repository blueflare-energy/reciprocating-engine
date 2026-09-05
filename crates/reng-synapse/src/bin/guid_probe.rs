//! Try to compile and run one TPC or complex guid with bf16 inputs of the
//! given shapes, to learn whether a kernel exists and what it accepts.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-guid-probe -- <guid> <out sizes> <in sizes>... [--params <hex bytes>]`
//!
//! Sizes are comma-separated FCD-first dims, e.g. `64,1024,32,1`. Prints
//! the first eight output values on success or the failing call.

use reng_synapse::{NodeInput, run_node};

fn sizes(s: &str) -> Vec<u64> {
    s.split(',').map(|d| d.parse().expect("dim")).collect()
}

fn main() -> reng_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut params: Vec<u8> = Vec::new();
    let mut shapes: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--params" {
            let hex = &args[i + 1];
            params = (0..hex.len())
                .step_by(2)
                .map(|j| u8::from_str_radix(&hex[j..j + 2], 16).expect("hex"))
                .collect();
            i += 2;
        } else {
            shapes.push(args[i].clone());
            i += 1;
        }
    }
    assert!(shapes.len() >= 3, "guid, out sizes, at least one input");
    let guid = &shapes[0];
    let out = sizes(&shapes[1]);
    let in_sizes: Vec<Vec<u64>> = shapes[2..].iter().map(|s| sizes(s)).collect();
    let names: Vec<String> = (0..in_sizes.len()).map(|k| format!("IN{k}")).collect();
    let data: Vec<Vec<f32>> = in_sizes
        .iter()
        .enumerate()
        .map(|(k, s)| {
            (0..s.iter().product::<u64>())
                .map(|j| ((j % 11) as f32 - 5.0) * 0.05 * (k as f32 + 1.0))
                .collect()
        })
        .collect();
    let ins: Vec<NodeInput<'_>> = (0..in_sizes.len())
        .map(|k| NodeInput {
            name: &names[k],
            sizes: &in_sizes[k],
            data: &data[k],
            raw: None,
        })
        .collect();
    let (p, n) = if params.is_empty() {
        (core::ptr::null(), 0u32)
    } else {
        (params.as_ptr().cast(), params.len() as u32)
    };
    match run_node(guid, &ins, &out, p, n) {
        Ok(v) => {
            println!(
                "{guid}: OK, {} outputs, first {:?}",
                v.len(),
                &v[..v.len().min(8)]
            );
            Ok(())
        }
        Err(e) => {
            println!("{guid}: FAILED {e}");
            Ok(())
        }
    }
}
