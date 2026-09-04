//! `reng`: the command-line entry point for the Reciprocating Engine.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reng_ceiling::{
    Bottleneck, Ceiling, HardwareSpec, Precision, ceiling_grid, decode_ceiling, fits,
    model_from_hf_config, prefill_ceiling, vram_bytes,
};
use reng_hal::enumerate_devices;

#[derive(Parser)]
#[command(name = "reng", version, about = "Reciprocating Engine control CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List Gaudi2 accelerators visible to the host.
    Devices,
    /// Compute first-principles prefill and decode ceilings for one scenario.
    Ceiling {
        /// Path to a HuggingFace config.json.
        #[arg(long)]
        config: PathBuf,
        /// Prompt length for prefill, in tokens.
        #[arg(long, default_value_t = 4096)]
        seq: u32,
        /// Context length for decode, in tokens.
        #[arg(long, default_value_t = 4096)]
        ctx: u32,
        /// Batch size (concurrent sequences).
        #[arg(long, default_value_t = 1)]
        batch: u32,
        /// Weight precision: bf16, fp16, fp8, fp32, int8, int4.
        #[arg(long, default_value = "bf16")]
        precision: String,
        /// KV-cache precision.
        #[arg(long, default_value = "bf16")]
        kv_precision: String,
        /// Number of cards, for the aggregate-HBM fit check.
        #[arg(long, default_value_t = 1)]
        cards: u32,
    },
    /// Print a context-by-batch grid of ceilings (Chart 1/2).
    Grid {
        /// Path to a HuggingFace config.json.
        #[arg(long)]
        config: PathBuf,
        /// Largest context to chart; halves down to 256.
        #[arg(long, default_value_t = 32768)]
        max_context: u32,
        /// Weight precision.
        #[arg(long, default_value = "bf16")]
        precision: String,
        /// KV-cache precision.
        #[arg(long, default_value = "bf16")]
        kv_precision: String,
        /// Number of cards (tensor-parallel, idealized linear scaling).
        #[arg(long, default_value_t = 1)]
        cards: u32,
        /// Which ceiling to chart: prefill or decode.
        #[arg(long, default_value = "decode")]
        mode: String,
    },
}

fn main() -> reng_core::Result<()> {
    match Cli::parse().cmd {
        Cmd::Devices => devices(),
        Cmd::Ceiling {
            config,
            seq,
            ctx,
            batch,
            precision,
            kv_precision,
            cards,
        } => ceiling(&config, seq, ctx, batch, &precision, &kv_precision, cards),
        Cmd::Grid {
            config,
            max_context,
            precision,
            kv_precision,
            cards,
            mode,
        } => grid(
            &config,
            max_context,
            &precision,
            &kv_precision,
            cards,
            &mode,
        ),
    }
}

fn devices() -> reng_core::Result<()> {
    let devs = enumerate_devices()?;
    if devs.is_empty() {
        println!("No Gaudi2 accelerators found.");
        return Ok(());
    }
    println!("{:<6} {:<16} {:<10}", "INDEX", "PCI", "STEPPING");
    for d in devs {
        println!(
            "{:<6} {:<16} {:<10}",
            d.id.0,
            d.pci_addr.as_deref().unwrap_or("-"),
            d.stepping.map_or_else(|| "-".to_string(), |s| s.0),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ceiling(
    config: &Path,
    seq: u32,
    ctx: u32,
    batch: u32,
    precision: &str,
    kv_precision: &str,
    cards: u32,
) -> reng_core::Result<()> {
    let model = model_from_hf_config(&std::fs::read_to_string(config)?)?;
    let hw = HardwareSpec::gaudi2();
    let prec = Precision::parse(precision)?;
    let kv = Precision::parse(kv_precision)?;

    println!("model: {}", model.name);
    println!(
        "  total params : {:.2} B",
        model.total_params() as f64 / 1e9
    );
    println!(
        "  active params: {:.2} B",
        model.active_params() as f64 / 1e9
    );

    let vram = vram_bytes(&model, prec, kv, batch, ctx) as f64 / 1e9;
    let ok = fits(&hw, &model, prec, kv, batch, ctx, cards);
    println!(
        "footprint: {vram:.1} GB ({precision} weights, {kv_precision} KV, batch {batch}, ctx {ctx}) -> fits {cards} x {}: {}",
        hw.name,
        if ok { "YES" } else { "NO" },
    );

    let p = prefill_ceiling(&hw, &model, prec, kv, batch, seq);
    println!("prefill (batch {batch}, seq {seq}):");
    print_ceiling(&p);
    let d = decode_ceiling(&hw, &model, prec, kv, batch, ctx);
    println!("decode  (batch {batch}, ctx {ctx}):");
    print_ceiling(&d);
    println!("(ceilings are single-card hardware bounds at 100% utilization)");
    Ok(())
}

fn print_ceiling(c: &Ceiling) {
    println!(
        "  {:>12.1} tok/s   {:>9.3} ms   bottleneck: {}   AI={:.1} FLOP/byte",
        c.tokens_per_s,
        c.latency_s * 1e3,
        bottleneck_name(c.bottleneck),
        c.arithmetic_intensity,
    );
}

fn grid(
    config: &Path,
    max_context: u32,
    precision: &str,
    kv_precision: &str,
    cards: u32,
    mode: &str,
) -> reng_core::Result<()> {
    let model = model_from_hf_config(&std::fs::read_to_string(config)?)?;
    let hw = HardwareSpec::gaudi2();
    let prec = Precision::parse(precision)?;
    let kv = Precision::parse(kv_precision)?;
    let decode = match mode {
        "decode" => true,
        "prefill" => false,
        other => {
            return Err(reng_core::Error::Other(format!(
                "mode must be prefill or decode, got {other:?}"
            )));
        }
    };

    let cells = ceiling_grid(&hw, &model, prec, kv, max_context, cards);
    println!(
        "{} ceiling (tok/s) for {} on {cards} x {} [{precision} weights, {kv_precision} KV]",
        if decode { "decode" } else { "prefill" },
        model.name,
        hw.name,
    );
    if cells.is_empty() {
        println!("  (model does not fit in aggregate HBM at any batch/context)");
        return Ok(());
    }

    let contexts: Vec<u32> = {
        let s: BTreeSet<u32> = cells.iter().map(|c| c.context).collect();
        s.iter().rev().copied().collect::<Vec<_>>()
    };
    let batches: BTreeSet<u32> = cells.iter().map(|c| c.batch).collect();

    // Header: batch\ctx, then one column per context.
    print!("{:>7} ", "batch\\ctx");
    for c in &contexts {
        print!("{:>10}", human(f64::from(*c)));
    }
    println!();

    for b in &batches {
        print!("{b:>7} ");
        for c in &contexts {
            match cells.iter().find(|x| x.batch == *b && x.context == *c) {
                Some(cell) => {
                    let ce: &Ceiling = if decode { &cell.decode } else { &cell.prefill };
                    let mark = match ce.bottleneck {
                        Bottleneck::Compute => 'C',
                        Bottleneck::HbmBandwidth => 'H',
                    };
                    print!("{:>9}{}", human(ce.tokens_per_s), mark);
                }
                None => print!("{:>10}", "-"),
            }
        }
        println!();
    }
    println!("(C = compute-bound, H = HBM-bandwidth-bound; blank = exceeds HBM)");
    Ok(())
}

fn bottleneck_name(b: Bottleneck) -> &'static str {
    match b {
        Bottleneck::Compute => "compute (MME)",
        Bottleneck::HbmBandwidth => "HBM bandwidth",
    }
}

/// Human-readable count: 1_500 -> "1.5k", 2_300_000 -> "2.3M".
fn human(x: f64) -> String {
    if x >= 1e6 {
        format!("{:.1}M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.1}k", x / 1e3)
    } else {
        format!("{x:.0}")
    }
}
