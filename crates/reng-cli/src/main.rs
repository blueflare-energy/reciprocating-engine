//! `reng`: the command-line entry point for the Reciprocating Engine.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reng_ceiling::{
    Bottleneck, Ceiling, HardwareSpec, Precision, decode_ceiling, fits, model_from_hf_config,
    prefill_ceiling, vram_bytes,
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
    /// Compute first-principles prefill and decode ceilings for a model.
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
    let json = std::fs::read_to_string(config)?;
    let model = model_from_hf_config(&json)?;
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
    let bn = match c.bottleneck {
        Bottleneck::Compute => "compute (MME)",
        Bottleneck::HbmBandwidth => "HBM bandwidth",
    };
    println!(
        "  {:>12.1} tok/s   {:>9.3} ms   bottleneck: {bn}   AI={:.1} FLOP/byte",
        c.tokens_per_s,
        c.latency_s * 1e3,
        c.arithmetic_intensity,
    );
}
