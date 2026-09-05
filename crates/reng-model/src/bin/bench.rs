//! Measure prefill and decode throughput of a model on one Gaudi2 card at
//! batch 1 and report each next to its roofline ceiling from `reng-ceiling`.
//!
//! The output JSON is the `customBiggerIsBetter` list that
//! benchmark-action/github-action-benchmark charts, so every merge to main
//! appends a point to the tok/s and percent-of-ceiling series.
//!
//! `reng-bench <model_dir> <out.json> [--prompt <tokens>] [--new <tokens>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--warmup <steps>] [--batch <n>]`
//!
//! With `--batch B > 1` the batched decoder is measured instead: every
//! sequence is prefilled with the prompt (one at a time), then all advance
//! together; decode tok/s counts every sequence's token.

use reng_ceiling::{
    HardwareSpec, Precision, decode_ceiling, model_from_hf_config, prefill_ceiling,
};
use reng_model::{BatchedGenerator, Generator, LlamaConfig, load_weights};
use std::path::Path;
use std::time::Instant;

struct Args {
    dir: String,
    out_path: String,
    prompt: usize,
    n_new: usize,
    rows: usize,
    decode_rows: usize,
    capacity: usize,
    warmup: usize,
    batch: usize,
}

fn parse_args() -> reng_core::Result<Args> {
    let usage = "usage: reng-bench <model_dir> <out.json> [--prompt <tokens>] [--new <tokens>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--warmup <steps>] [--batch <n>]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(reng_core::Error::Other(usage.into()));
    }
    let mut a = Args {
        dir: args[0].clone(),
        out_path: args[1].clone(),
        prompt: 128,
        n_new: 64,
        rows: 256,
        decode_rows: 1,
        capacity: 1024,
        warmup: 4,
        batch: 1,
    };
    let mut i = 2;
    while i + 1 < args.len() {
        let val: usize = args[i + 1]
            .parse()
            .map_err(|e| reng_core::Error::Other(format!("{}: {e}", args[i])))?;
        match args[i].as_str() {
            "--prompt" => a.prompt = val,
            "--new" => a.n_new = val,
            "--rows" => a.rows = val,
            "--decode-rows" => a.decode_rows = val,
            "--capacity" => a.capacity = val,
            "--warmup" => a.warmup = val,
            "--batch" => a.batch = val,
            other => return Err(reng_core::Error::Other(format!("unknown flag {other}"))),
        }
        i += 2;
    }
    if i != args.len() {
        return Err(reng_core::Error::Other(usage.into()));
    }
    assert!(a.prompt >= 1 && a.n_new >= 1);
    assert!(
        a.prompt + a.n_new <= a.capacity,
        "prompt + new must fit the cache capacity"
    );
    Ok(a)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[v.len() / 2]
}

fn main() -> reng_core::Result<()> {
    let a = parse_args()?;
    let dir = Path::new(&a.dir);
    let cfg = LlamaConfig::load(dir)?;
    let config_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| reng_core::Error::Other(format!("config.json: {e}")))?;
    let shape = model_from_hf_config(&config_text)?;
    let model_name = dir
        .file_name()
        .map_or_else(|| shape.name.clone(), |n| n.to_string_lossy().into_owned());
    let hw = HardwareSpec::gaudi2();
    let w = load_weights(dir, &cfg)?;

    // A synthetic prompt: the ids only need to be valid, throughput does not
    // depend on their values.
    let vocab = cfg.vocab_size as u32;
    let prompt: Vec<u32> = (0..a.prompt as u32)
        .map(|i| (i * 7919 + 13) % vocab)
        .collect();
    let batch = a.batch.max(1);

    let t0 = Instant::now();
    let (prefill_s, decode_s, step_ms) = if batch == 1 {
        let mut g = Generator::new(&w, &cfg, a.rows, a.decode_rows, a.capacity)?;
        println!(
            "{model_name}: {} layers, hidden {}, vocab {}; compiled rows {} / decode rows {} / capacity {} in {:.2}s",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            a.rows,
            a.decode_rows,
            a.capacity,
            t0.elapsed().as_secs_f64()
        );
        // Warm both recipes (the first launches of a recipe are slower), then
        // measure prefill as one fresh sequence.
        for _ in 0..a.warmup {
            g.feed(&prompt)?;
            g.feed(&prompt[..1])?;
            g.reset()?;
        }
        let t1 = Instant::now();
        g.feed(&prompt)?;
        let prefill_s = t1.elapsed().as_secs_f64();
        // Decode: one token per step, greedy on the engine's own output.
        let mut next = prompt[a.prompt - 1];
        let mut step_s: Vec<f64> = Vec::with_capacity(a.n_new);
        let t2 = Instant::now();
        for _ in 0..a.n_new {
            let t = Instant::now();
            let logits = g.feed(&[next])?;
            step_s.push(t.elapsed().as_secs_f64());
            next = reng_model::argmax_rows(&logits, cfg.vocab_size)[0] as u32;
        }
        (
            prefill_s,
            t2.elapsed().as_secs_f64(),
            median(&mut step_s) * 1e3,
        )
    } else {
        let mut g = BatchedGenerator::new(&w, &cfg, batch, a.rows, a.capacity)?;
        println!(
            "{model_name}: {} layers, hidden {}, vocab {}; compiled batch {} / rows {} / capacity {} in {:.2}s",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            batch,
            a.rows,
            a.capacity,
            t0.elapsed().as_secs_f64()
        );
        let mut next: Vec<u32> = vec![prompt[a.prompt - 1]; batch];
        for _ in 0..a.warmup {
            for b in 0..batch {
                g.prefill(b, &prompt)?;
            }
            g.step(&next)?;
        }
        // Prefill: every sequence, one at a time (the wide recipe).
        let t1 = Instant::now();
        for b in 0..batch {
            g.prefill(b, &prompt)?;
        }
        let prefill_s = t1.elapsed().as_secs_f64() / batch as f64;
        let mut step_s: Vec<f64> = Vec::with_capacity(a.n_new);
        let t2 = Instant::now();
        for _ in 0..a.n_new {
            let t = Instant::now();
            let logits = g.step(&next)?;
            step_s.push(t.elapsed().as_secs_f64());
            let ids = reng_model::argmax_rows(&logits, cfg.vocab_size);
            for (n, id) in next.iter_mut().zip(ids) {
                *n = id as u32;
            }
        }
        (
            prefill_s,
            t2.elapsed().as_secs_f64(),
            median(&mut step_s) * 1e3,
        )
    };
    let prefill_tps = a.prompt as f64 / prefill_s;
    let decode_tps = (batch * a.n_new) as f64 / decode_s;

    let ctx = (a.prompt + a.n_new / 2) as u32;
    let c_pre = prefill_ceiling(
        &hw,
        &shape,
        Precision::Bf16,
        Precision::Bf16,
        1,
        a.prompt as u32,
    );
    let c_dec = decode_ceiling(
        &hw,
        &shape,
        Precision::Bf16,
        Precision::Bf16,
        batch as u32,
        ctx,
    );
    let pre_pct = 100.0 * prefill_tps / c_pre.tokens_per_s;
    let dec_pct = 100.0 * decode_tps / c_dec.tokens_per_s;

    println!(
        "prefill {} tokens: {prefill_s:.4}s = {prefill_tps:.0} tok/s  (ceiling {:.0} tok/s, {:?}) = {pre_pct:.2}%",
        a.prompt, c_pre.tokens_per_s, c_pre.bottleneck
    );
    println!(
        "decode {} tokens x batch {batch} at ctx ~{ctx}: {decode_tps:.1} tok/s, median step {step_ms:.2} ms  (ceiling {:.0} tok/s, {:?}) = {dec_pct:.2}%",
        a.n_new, c_dec.tokens_per_s, c_dec.bottleneck
    );

    let entry = |name: &str, unit: &str, value: f64, extra: String| {
        serde_json::json!({
            "name": format!("{model_name} {name}"),
            "unit": unit,
            "value": value,
            "extra": extra,
        })
    };
    let out = serde_json::json!([
        entry(
            "prefill tok/s (b1)",
            "tok/s",
            prefill_tps,
            format!(
                "{} tokens; ceiling {:.0} tok/s",
                a.prompt, c_pre.tokens_per_s
            )
        ),
        entry(
            "prefill % of ceiling (b1)",
            "%",
            pre_pct,
            format!("{:?} bound", c_pre.bottleneck)
        ),
        entry(
            &format!("decode tok/s (b{batch})"),
            "tok/s",
            decode_tps,
            format!(
                "ctx ~{ctx}; median step {step_ms:.2} ms; ceiling {:.0} tok/s",
                c_dec.tokens_per_s
            )
        ),
        entry(
            &format!("decode % of ceiling (b{batch})"),
            "%",
            dec_pct,
            format!("{:?} bound", c_dec.bottleneck)
        ),
    ]);
    std::fs::write(&a.out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| reng_core::Error::Other(format!("{}: {e}", a.out_path)))?;
    println!("wrote {}", a.out_path);
    Ok(())
}
