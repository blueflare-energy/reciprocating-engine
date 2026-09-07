//! Measure prefill and decode throughput of a model on one Gaudi2 card at
//! batch 1 and report each next to its roofline ceiling from `reng-ceiling`.
//!
//! The output JSON is the `customBiggerIsBetter` list that
//! benchmark-action/github-action-benchmark charts, so every merge to main
//! appends a point to the tok/s and percent-of-ceiling series.
//!
//! `reng-bench <model_dir> <out.json> [--prompt <tokens>] [--new <tokens>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--warmup <steps>] [--batch <n>] [--stagger <k>]`
//!
//! With `--batch B > 1` the batched decoder is measured instead: every
//! sequence is prefilled with the prompt (one at a time), then all advance
//! together; decode tok/s counts every sequence's token.
//!
//! The ids the decode phase produced are summarised as an FNV-1a hash (and
//! their first few printed) so that two runs, for example with and without
//! `RENG_DEVICE_LOOP`, can be checked for identical output. `--stagger k`
//! (batched only) gives sequence `b` its own synthetic prompt, `b % k`
//! tokens shorter than `--prompt`, so that the slots hold different ids
//! at different positions; it is for such checks, not for timing (the
//! prefill rate is still reported per `--prompt` tokens).

use reng_ceiling::{
    HardwareSpec, Precision, decode_ceiling, model_from_hf_config, prefill_ceiling,
};
use reng_model::{
    BatchedGenerator, CachePath, CachePlan, Fp8Config, Generator, LlamaConfig, fp8_switch,
    load_weights_fp8, take_fp8_flag,
};
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
    stagger: usize,
    fp8: Option<Fp8Config>,
}

fn parse_args() -> reng_core::Result<Args> {
    let usage = "usage: reng-bench <model_dir> <out.json> [--prompt <tokens>] [--new <tokens>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--warmup <steps>] [--batch <n>] [--stagger <k>] [--fp8]";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let fp8 = fp8_switch(
        std::env::var("RENG_FP8").ok().as_deref(),
        take_fp8_flag(&mut args),
    )?;
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
        stagger: 0,
        fp8,
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
            "--stagger" => a.stagger = val,
            other => return Err(reng_core::Error::Other(format!("unknown flag {other}"))),
        }
        i += 2;
    }
    if i != args.len() {
        return Err(reng_core::Error::Other(usage.into()));
    }
    assert!(a.prompt >= 1 && a.n_new >= 1);
    assert!(
        a.stagger <= a.prompt,
        "stagger must leave every sequence at least one prompt token"
    );
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

/// FNV-1a over the ids' little-endian bytes.
fn fnv1a64(ids: &[u32]) -> u64 {
    ids.iter()
        .flat_map(|id| id.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

fn main() -> reng_core::Result<()> {
    let t_start = Instant::now();
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
    let t_load = Instant::now();
    let (w, fp8_report) = load_weights_fp8(dir, &cfg, a.fp8)?;
    let load_s = t_load.elapsed().as_secs_f64();
    if let Some(r) = fp8_report {
        println!(
            "fp8 {}: {r}",
            a.fp8.expect("a report means the switch was on")
        );
    }
    let (mapped, owned) = w.footprint();
    println!(
        "loaded weights in {load_s:.2}s (mapped {:.2} GB, owned {:.2} GB)",
        mapped as f64 / 1e9,
        owned as f64 / 1e9
    );

    let batch_for_plan = a.batch.max(1);
    let plan = CachePlan::new(
        &cfg,
        w.device_bytes(),
        a.rows,
        if batch_for_plan == 1 {
            CachePath::Single
        } else {
            CachePath::Batched(batch_for_plan)
        },
    );
    // What this run allocates: `--capacity` itself at batch 1, and the
    // cache bucket the run grows into on the batched path, whose
    // `--capacity` is a ceiling.
    let planned = plan.allocated_capacity(a.prompt + a.n_new, a.capacity);
    println!("{}", plan.summary(planned));
    if let Some(msg) = cfg.context_warning(a.prompt + a.n_new) {
        eprintln!("{msg}");
        println!("{msg}");
    }
    plan.check(planned)?;

    // A synthetic prompt: the ids only need to be valid, throughput does not
    // depend on their values.
    let vocab = cfg.vocab_size as u32;
    let prompt: Vec<u32> = (0..a.prompt as u32)
        .map(|i| (i * 7919 + 13) % vocab)
        .collect();
    let batch = a.batch.max(1);
    // With `--stagger k`, sequence b's prompt is its own stream and b % k
    // tokens shorter.
    let prompt_of = |b: usize| -> Vec<u32> {
        if a.stagger == 0 {
            return prompt.clone();
        }
        let n = a.prompt - b % a.stagger;
        (0..n as u32)
            .map(|i| (i * 7919 + 13 + (b as u32) * 7) % vocab)
            .collect()
    };
    // Seconds from process start to the first token of the first prompt
    // (the launch cost: load, compile, upload, one prefill).
    let mut first_token_s: Option<f64> = None;
    let note_first = |first: &mut Option<f64>| {
        if first.is_none() {
            *first = Some(t_start.elapsed().as_secs_f64());
        }
    };

    let t0 = Instant::now();
    // The reported step time: the median over the per-step launches, or
    // the mean over one device-loop run (its steps are not timed singly).
    let mut step_kind = "median";
    let (prefill_s, decode_s, step_ms, ids) = if batch == 1 {
        let mut g = Generator::new(&w, &cfg, a.rows, a.decode_rows, a.capacity)?;
        println!(
            "{model_name}: {} layers, hidden {}, vocab {}; compiled rows {} / decode rows {} / capacity {} in {:.2}s{}",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            a.rows,
            a.decode_rows,
            a.capacity,
            t0.elapsed().as_secs_f64(),
            if g.device_loop() {
                " (device decode loop)"
            } else {
                ""
            }
        );
        // Warm both recipes (the first launches of a recipe are slower), then
        // measure prefill as one fresh sequence.
        for _ in 0..a.warmup {
            g.feed(&prompt)?;
            note_first(&mut first_token_s);
            g.feed(&prompt[..1])?;
            g.reset();
        }
        let t1 = Instant::now();
        let mut next = g.feed_id(&prompt)?;
        let prefill_s = t1.elapsed().as_secs_f64();
        note_first(&mut first_token_s);
        // Decode: one token per step, greedy on the engine's own output
        // (argmax on the device); with the device loop all steps run from
        // one call and one readback.
        if g.device_loop() {
            step_kind = "mean";
            let t2 = Instant::now();
            let ids = g.generate(next, a.n_new)?;
            let s = t2.elapsed().as_secs_f64();
            (prefill_s, s, s / a.n_new as f64 * 1e3, ids)
        } else {
            let mut step_s: Vec<f64> = Vec::with_capacity(a.n_new);
            let mut ids = Vec::with_capacity(a.n_new);
            let t2 = Instant::now();
            for _ in 0..a.n_new {
                let t = Instant::now();
                next = g.feed_id(&[next])?;
                step_s.push(t.elapsed().as_secs_f64());
                ids.push(next);
            }
            (
                prefill_s,
                t2.elapsed().as_secs_f64(),
                median(&mut step_s) * 1e3,
                ids,
            )
        }
    } else {
        let mut g = BatchedGenerator::new(&w, &cfg, batch, a.rows, a.capacity)?;
        println!(
            "{model_name}: {} layers, hidden {}, vocab {}; compiled batch {} / rows {} / capacity {} in {:.2}s{}",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            batch,
            a.rows,
            a.capacity,
            t0.elapsed().as_secs_f64(),
            if g.device_loop() {
                " (device decode loop)"
            } else {
                ""
            }
        );
        let mut next: Vec<u32> = vec![prompt[a.prompt - 1]; batch];
        for _ in 0..a.warmup {
            for b in 0..batch {
                g.prefill_id(b, &prompt_of(b))?;
                note_first(&mut first_token_s);
            }
            g.step_ids(&next)?;
        }
        // Prefill: every sequence, one at a time (the wide recipe).
        let t1 = Instant::now();
        for (b, n) in next.iter_mut().enumerate() {
            *n = g.prefill_id(b, &prompt_of(b))?;
            note_first(&mut first_token_s);
        }
        let prefill_s = t1.elapsed().as_secs_f64() / batch as f64;
        // Decode: every sequence one token per step, greedy on its own
        // output; with the device loop all steps run from one call and
        // one readback.
        if g.device_loop() {
            step_kind = "mean";
            let t2 = Instant::now();
            let ids = g.generate(&next, a.n_new)?;
            let s = t2.elapsed().as_secs_f64();
            (prefill_s, s, s / a.n_new as f64 * 1e3, ids)
        } else {
            let mut step_s: Vec<f64> = Vec::with_capacity(a.n_new);
            let mut ids = Vec::with_capacity(a.n_new * batch);
            let t2 = Instant::now();
            for _ in 0..a.n_new {
                let t = Instant::now();
                next = g.step_ids(&next)?;
                step_s.push(t.elapsed().as_secs_f64());
                ids.extend_from_slice(&next);
            }
            (
                prefill_s,
                t2.elapsed().as_secs_f64(),
                median(&mut step_s) * 1e3,
                ids,
            )
        }
    };
    let prefill_tps = a.prompt as f64 / prefill_s;
    let decode_tps = (batch * a.n_new) as f64 / decode_s;
    println!(
        "decode ids: {} ids step by step, fnv1a64 {:016x}, first {:?}",
        ids.len(),
        fnv1a64(&ids),
        &ids[..ids.len().min(8)]
    );
    println!(
        "first token {:.2}s after start (weights loaded in {load_s:.2}s)",
        first_token_s.unwrap_or(f64::NAN)
    );

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
        "decode {} tokens x batch {batch} at ctx ~{ctx}: {decode_tps:.1} tok/s, {step_kind} step {step_ms:.2} ms  (ceiling {:.0} tok/s, {:?}) = {dec_pct:.2}%",
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
                "ctx ~{ctx}; {step_kind} step {step_ms:.2} ms; ceiling {:.0} tok/s",
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
