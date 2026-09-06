//! Quantize a checkpoint's projections to fp8 on the host and report what
//! it costs: bytes before and after, and the relative error of the round
//! trip. Nothing here touches a device, so it runs anywhere the checkpoint
//! does.
//!
//! `reng-fp8-quantize <model_dir> [--fp8 <spec>] [--per-matrix] [--layers <n>]`
//!
//! `<spec>` is the value the `RENG_FP8` switch takes: a format (`e4m3`,
//! `e5m2`), a scale scheme (`pcs`, `hw`, `unit`) and `backoff=<f32>`,
//! colon- or comma-separated; the default is `e4m3:pcs`. `--per-matrix`
//! prints a line per matrix of the first `--layers` layers (default 1)
//! instead of only the totals.

use reng_core::{Error, Result};
use reng_model::{Fp8Config, LlamaConfig, fp8_switch, load_weights};
use std::path::Path;

const USAGE: &str =
    "usage: reng-fp8-quantize <model_dir> [--fp8 <spec>] [--per-matrix] [--layers <n>]";

struct Args {
    dir: String,
    fp8: Fp8Config,
    per_matrix: bool,
    layers: usize,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        return Err(Error::Other(USAGE.into()));
    }
    let mut a = Args {
        dir: argv[0].clone(),
        fp8: Fp8Config::default(),
        per_matrix: false,
        layers: 1,
    };
    // The switch reads the environment first, so `RENG_FP8=e5m2` works
    // here as it does for the device binaries; `--fp8 <spec>` overrides it.
    if let Some(c) = fp8_switch(std::env::var("RENG_FP8").ok().as_deref(), true)? {
        a.fp8 = c;
    }
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--per-matrix" => {
                a.per_matrix = true;
                i += 1;
            }
            "--fp8" => {
                let v = argv.get(i + 1).ok_or_else(|| Error::Other(USAGE.into()))?;
                a.fp8 = fp8_switch(Some(v), true)?
                    .ok_or_else(|| Error::Other(format!("--fp8 {v}: that switch is off")))?;
                i += 2;
            }
            "--layers" => {
                let v = argv.get(i + 1).ok_or_else(|| Error::Other(USAGE.into()))?;
                a.layers = v
                    .parse()
                    .map_err(|e| Error::Other(format!("--layers: {e}")))?;
                i += 2;
            }
            other => return Err(Error::Other(format!("unknown flag {other}\n{USAGE}"))),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let dir = Path::new(&a.dir);
    let cfg = LlamaConfig::load(dir)?;
    let t0 = std::time::Instant::now();
    let mut w = load_weights(dir, &cfg)?;
    println!(
        "{}: {} layers, hidden {}, intermediate {}, loaded in {:.2} s",
        a.dir,
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        t0.elapsed().as_secs_f64()
    );
    println!("fp8 scheme: {}", a.fp8);
    let report = w.quantize_fp8(&cfg, a.fp8)?;
    println!("fp8: {report}");
    let saved = report.bf16_bytes as f64 - (report.fp8_bytes + report.scale_bytes) as f64;
    println!(
        "projection bytes: {:.3} GiB -> {:.3} GiB ({:.2}x, {:.3} GiB saved)",
        report.bf16_bytes as f64 / (1 << 30) as f64,
        (report.fp8_bytes + report.scale_bytes) as f64 / (1 << 30) as f64,
        report.bf16_bytes as f64 / (report.fp8_bytes + report.scale_bytes) as f64,
        saved / (1 << 30) as f64
    );
    if a.per_matrix {
        let fp8 = w.fp8.as_ref().expect("quantized above");
        for (li, layer) in fp8.iter().take(a.layers).enumerate() {
            let bf16 = [
                &w.layers[li].wq,
                &w.layers[li].wk,
                &w.layers[li].wv,
                &w.layers[li].wo,
                &w.layers[li].wg,
                &w.layers[li].wu,
                &w.layers[li].wd,
            ];
            for ((name, q), src) in layer.matrices().iter().zip(bf16) {
                let errs = reng_fp8::sample_row_errors(src, q, reng_model::FP8_ERROR_ROWS);
                let mean = errs.iter().map(|e| f64::from(e.mean_rel)).sum::<f64>()
                    / errs.len().max(1) as f64;
                let max = errs.iter().fold(0.0f32, |m, e| m.max(e.max_rel));
                let absmax = reng_fp8::row_absmax(src, q.rows, q.cols)
                    .into_iter()
                    .fold(0.0f32, f32::max);
                println!(
                    "  l{li} {name:<3} [{}, {}] bias {} absmax {absmax:.4} \
                     mean_rel {mean:.4} max_rel {max:.4}",
                    q.rows, q.cols, q.exp_bias
                );
            }
        }
    }
    Ok(())
}
