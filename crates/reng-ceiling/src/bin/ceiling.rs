//! `reng-ceiling`: the parallelism strategies for one model on N cards, and
//! the best of them for each objective.
//!
//! ```console
//! $ reng-ceiling ~/models/Qwen2.5-32B --cards 8 --batch 8 --ctx 192
//! ```
//!
//! Reads `config.json` from the model directory, so it needs no hardware and
//! no weights.

use std::path::{Path, PathBuf};

use clap::Parser;
use reng_ceiling::strategy::{
    Choice, CollectiveFloor, LaunchFloor, Objective, Plan, Scenario, choose, plans,
};
use reng_ceiling::{HardwareSpec, Precision, model_from_hf_config};
use reng_core::{Error, Result};

#[derive(Parser)]
#[command(
    name = "reng-ceiling",
    version,
    about = "N-card parallelism strategies and their ceilings"
)]
struct Cli {
    /// Model directory (the one holding config.json), or the config itself.
    model_dir: PathBuf,
    /// Cards to spend. The measured tables carry 1, 2, 4 and 8; any other
    /// world borrows the largest measured world below it, and every row that
    /// does says so.
    #[arg(long, default_value_t = 1)]
    cards: u32,
    /// Batch per replica, for the aggregate objective.
    #[arg(long, default_value_t = 1)]
    batch: u32,
    /// Context length the KV cache is charged at.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,
    /// Weight precision: bf16, fp16, fp8, fp32, int8, int4.
    #[arg(long, default_value = "bf16")]
    precision: String,
    /// KV-cache precision. Defaults to the weight precision.
    #[arg(long)]
    kv_precision: Option<String>,
    /// A JSON collective-latency table to use instead of the measured one.
    #[arg(long)]
    collectives: Option<PathBuf>,
    /// A JSON host launch-floor table to use instead of the measured one.
    #[arg(long)]
    launch: Option<PathBuf>,
    /// Print the whole table as JSON instead of text (for tools/sweep_tp.py).
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = if cli.model_dir.is_dir() {
        cli.model_dir.join("config.json")
    } else {
        cli.model_dir.clone()
    };
    let json = std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("{}: {e}", path.display())))?;
    let model = model_from_hf_config(&json)?;
    let prec = Precision::parse(&cli.precision)?;
    let kv = match &cli.kv_precision {
        None => prec,
        Some(name) => Precision::parse(name)?,
    };
    let floor = match &cli.collectives {
        None => CollectiveFloor::measured(),
        Some(p) => CollectiveFloor::from_json(
            &std::fs::read_to_string(p)
                .map_err(|e| Error::Other(format!("{}: {e}", p.display())))?,
        )?,
    };
    let launch = match &cli.launch {
        None => LaunchFloor::measured(),
        Some(p) => LaunchFloor::from_json(
            &std::fs::read_to_string(p)
                .map_err(|e| Error::Other(format!("{}: {e}", p.display())))?,
        )?,
    };
    if cli.cards == 0 {
        return Err(Error::Other("--cards must be at least 1".into()));
    }
    let hw = HardwareSpec::gaudi2();
    let s = Scenario {
        hw: &hw,
        model: &model,
        prec,
        kv,
        batch: cli.batch,
        ctx: cli.ctx,
        floor: &floor,
        launch: &launch,
    };

    let name = model_name(&model.name, &cli.model_dir);
    if cli.json {
        println!("{}", to_json(&name, &s, cli.cards));
        return Ok(());
    }
    println!(
        "{name}: {} layers, hidden {}, {} q heads, {} kv heads, intermediate {}, vocab {}",
        model.layers, model.hidden, model.n_heads, model.n_kv_heads, model.ff, model.vocab
    );
    if let Some(m) = &model.moe {
        println!(
            "  mixture of experts: {} experts, {} per token, expert intermediate {}, {} shared",
            m.n_experts, m.top_k, m.expert_ff, m.shared_experts
        );
    }
    println!(
        "  {} active params, {} resident; {:.2} GB of weights at {} per weight{}",
        model.active_params(),
        model.total_params(),
        model.total_params() as f64 * prec.weight_bytes() / 1e9,
        prec.weight_bytes(),
        if model.tied_embeddings {
            " (embedding and LM head tied: counted once)"
        } else {
            ""
        },
    );
    println!(
        "  {} cards, batch {} per replica, context {}, {:?} weights and {:?} cache",
        cli.cards, cli.batch, cli.ctx, prec, kv
    );
    println!("  collective floor: {}", floor.source);
    println!("  host launch floor: {}", launch.source);
    println!();

    println!(
        "strategy       1-stream     ms/token   physical     launch     aggregate    per card  \
         admissible"
    );
    println!(
        "                  tok/s    practical      tok/s      floor         tok/s          GB"
    );
    for p in plans(&s, cli.cards) {
        print_row(&p);
    }
    println!();

    for obj in [Objective::SingleStream, Objective::Aggregate] {
        print_choice(&choose(&s, cli.cards, obj), cli.cards);
    }
    Ok(())
}

/// The same table as the text form, for a tool to read.
fn to_json(name: &str, s: &Scenario, cards: u32) -> String {
    let plans: Vec<serde_json::Value> = plans(s, cards).iter().map(plan_json).collect();
    let pick = |obj: Objective| {
        let c = choose(s, cards, obj);
        serde_json::json!({
            "admissible": c.admissible,
            "strategy": c.admissible.then(|| c.plan.strategy.to_string()),
            "tok_s": c.admissible.then(|| c.plan.rate(obj).tokens_per_s),
            "physical_tok_s": c.admissible.then(|| c.plan.rate(obj).physical_tokens_per_s),
            "step_ms": c.admissible.then(|| c.plan.rate(obj).practical_s * 1e3),
            "launch_floor_ms": c.plan.rate(obj).launch_s * 1e3,
            "launch_bound": c.plan.rate(obj).launch_bound,
            "projected": c.plan.projected,
            "rejected": c.rejected.iter()
                .map(|(s, why)| serde_json::json!({"strategy": s, "why": why}))
                .collect::<Vec<_>>(),
            "notes": c.plan.notes,
            "reasons": c.reasons,
        })
    };
    let v = serde_json::json!({
        "model": name,
        "layers": s.model.layers,
        "hidden": s.model.hidden,
        "n_heads": s.model.n_heads,
        "n_kv_heads": s.model.n_kv_heads,
        "intermediate": s.model.ff,
        "vocab": s.model.vocab,
        "moe": s.model.moe.as_ref().map(|m| serde_json::json!({
            "n_experts": m.n_experts, "top_k": m.top_k,
            "expert_ff": m.expert_ff, "shared_experts": m.shared_experts,
        })),
        "active_params": s.model.active_params(),
        "total_params": s.model.total_params(),
        "cards": cards,
        "batch": s.batch,
        "ctx": s.ctx,
        "precision": format!("{:?}", s.prec),
        "kv_precision": format!("{:?}", s.kv),
        "collectives": s.floor.source,
        "launch_floor": s.launch.source,
        "plans": plans,
        "picks": {
            "single_stream": pick(Objective::SingleStream),
            "aggregate": pick(Objective::Aggregate),
        },
    });
    serde_json::to_string_pretty(&v).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}

fn plan_json(p: &Plan) -> serde_json::Value {
    serde_json::json!({
        "strategy": p.strategy.to_string(),
        "cards": p.cards,
        "batch": p.batch,
        "admissible": p.admissible(),
        "rejected": p.rejected,
        "single_stream_tok_s": p.single_stream.tokens_per_s,
        "single_stream_physical_tok_s": p.single_stream.physical_tokens_per_s,
        "single_stream_ms": p.single_stream.practical_s * 1e3,
        "single_stream_physical_ms": p.single_stream.physical_s * 1e3,
        "single_stream_launch_floor_ms": p.single_stream.launch_s * 1e3,
        "single_stream_launch_tok_s": p.single_stream.launch_tokens_per_s(1.0),
        "single_stream_launch_bound": p.single_stream.launch_bound,
        "aggregate_tok_s": p.aggregate.tokens_per_s,
        "aggregate_physical_tok_s": p.aggregate.physical_tokens_per_s,
        "aggregate_step_ms": p.aggregate.practical_s * 1e3,
        "aggregate_launch_bound": p.aggregate.launch_bound,
        "collective_ms": p.collective_s * 1e3,
        "resident_gb": p.resident_bytes_per_card as f64 / 1e9,
        "bottleneck": format!("{:?}", p.single_stream.bottleneck),
        "projected": p.projected,
        "notes": p.notes,
    })
}

fn model_name(from_config: &str, dir: &Path) -> String {
    if from_config == "model" || from_config.is_empty() {
        dir.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| from_config.to_string())
    } else {
        from_config.to_string()
    }
}

fn print_row(p: &Plan) {
    let mark = if p.projected { " (projected)" } else { "" };
    let state = match &p.rejected {
        None => format!("yes{mark}"),
        Some(why) => format!("no: {why}"),
    };
    let launch = match p.single_stream.launch_tokens_per_s(1.0) {
        Some(t) => format!("{t:.1}"),
        None => "-".to_string(),
    };
    println!(
        "{:<12} {:>10.1} {:>12.2} {:>10.1} {:>10} {:>13.1} {:>11.1}  {}",
        p.strategy.to_string(),
        p.single_stream.tokens_per_s,
        p.single_stream.practical_s * 1e3,
        p.single_stream.physical_tokens_per_s,
        launch,
        p.aggregate.tokens_per_s,
        p.resident_bytes_per_card as f64 / 1e9,
        state
    );
    for n in &p.notes {
        println!("               {n}");
    }
}

fn print_choice(c: &Choice, cards: u32) {
    let p = &c.plan;
    let r = p.rate(c.objective);
    if !c.admissible {
        println!(
            "no pick, {} on {cards} cards: no split of this model is admissible here",
            c.objective.label()
        );
        for (strategy, why) in &c.rejected {
            println!("  - {strategy}: {why}");
        }
        for reason in &c.reasons {
            println!("  - {reason}");
        }
        println!(
            "  - the {} row's own numbers, for the shape only:",
            p.strategy
        );
        for note in &p.notes {
            println!("      {note}");
        }
        println!();
        return;
    }
    println!(
        "pick, {} on {cards} cards: {}{} at {:.1} tok/s ({:.2} ms per step practical, \
         {:.1} tok/s physical{})",
        c.objective.label(),
        p.strategy,
        if p.projected { " (projected)" } else { "" },
        r.tokens_per_s,
        r.practical_s * 1e3,
        r.physical_tokens_per_s,
        if r.launch_bound {
            ", host launch bound"
        } else {
            ""
        }
    );
    for reason in &c.reasons {
        println!("  - {reason}");
    }
    println!();
}
