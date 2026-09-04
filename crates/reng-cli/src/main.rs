//! `reng`: the command-line entry point for the Reciprocating Engine.

use clap::{Parser, Subcommand};
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
}

fn main() -> reng_core::Result<()> {
    match Cli::parse().cmd {
        Cmd::Devices => devices(),
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
