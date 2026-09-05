//! Greedy generation on Gaudi2 through the fused engine. By default the
//! model is compiled once with a KV cache ([`Generator`]): the prompt goes in
//! as one block per `--rows` tokens and every generated token as a block of
//! one. `--recompute` instead re-runs prefill over the whole sequence at
//! every step (no cache; each step compiles a fresh recipe), the slow
//! cross-check the cached path was validated against.
//!
//! Without `--ref` the loop is free-running and the ids it produces are
//! written to `out.json`. With `--ref <ref.json>` (from `generate.py`) the
//! loop is teacher-forced: step `i` scores `prompt ++ ref[..i]` and the
//! engine's top-1 is compared with `ref[i]`. A mismatch counts as a failure
//! only when the reference's own f32 top-1/top-2 margin at that step is at
//! least `--margin` (default 0.5 logits); below that the two candidates are
//! within bf16 rounding of each other and the mismatch is reported as a
//! near-tie instead.
//!
//! `reng-generate <model_dir> <out.json> <n_new> [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--recompute] <id> [<id> ...]`

use reng_model::{Generator, LlamaConfig, argmax_rows, load_weights, prefill_logits};
use std::path::Path;
use std::time::Instant;

const DEFAULT_MARGIN: f32 = 0.5;
const DEFAULT_ROWS: usize = 256;
const DEFAULT_DECODE_ROWS: usize = 16;
const DEFAULT_CAPACITY: usize = 1024;

#[derive(serde::Deserialize)]
struct RefStep {
    top1: u32,
    top2: u32,
    margin: f32,
}

#[derive(serde::Deserialize)]
struct Reference {
    generated: Vec<u32>,
    #[serde(default)]
    steps: Vec<RefStep>,
}

struct Args {
    dir: String,
    out_path: String,
    n_new: usize,
    ref_path: Option<String>,
    margin: f32,
    rows: usize,
    decode_rows: usize,
    capacity: usize,
    recompute: bool,
    ids: Vec<u32>,
}

fn parse_args() -> reng_core::Result<Args> {
    let usage = "usage: reng-generate <model_dir> <out.json> <n_new> [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--recompute] <id> [<id> ...]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err(reng_core::Error::Other(usage.into()));
    }
    let n_new: usize = args[2]
        .parse()
        .map_err(|e| reng_core::Error::Other(format!("n_new: {e}")))?;
    let mut ref_path = None;
    let mut margin = DEFAULT_MARGIN;
    let mut rows = DEFAULT_ROWS;
    let mut decode_rows = DEFAULT_DECODE_ROWS;
    let mut capacity = DEFAULT_CAPACITY;
    let mut recompute = false;
    let mut i = 3;
    while i < args.len() && args[i].starts_with("--") {
        if args[i] == "--recompute" {
            recompute = true;
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .ok_or_else(|| reng_core::Error::Other(usage.into()))?;
        let num = |what: &str| {
            val.parse::<usize>()
                .map_err(|e| reng_core::Error::Other(format!("{what}: {e}")))
        };
        match args[i].as_str() {
            "--ref" => ref_path = Some(val.clone()),
            "--margin" => {
                margin = val
                    .parse()
                    .map_err(|e| reng_core::Error::Other(format!("margin: {e}")))?;
            }
            "--rows" => rows = num("rows")?,
            "--decode-rows" => decode_rows = num("decode-rows")?,
            "--capacity" => capacity = num("capacity")?,
            other => {
                return Err(reng_core::Error::Other(format!("unknown flag {other}")));
            }
        }
        i += 2;
    }
    let ids: Vec<u32> = args[i..]
        .iter()
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| reng_core::Error::Other(format!("token id: {e}")))?;
    if ids.is_empty() {
        return Err(reng_core::Error::Other("no token ids given".into()));
    }
    Ok(Args {
        dir: args[0].clone(),
        out_path: args[1].clone(),
        n_new,
        ref_path,
        margin,
        rows,
        decode_rows,
        capacity,
        recompute,
        ids,
    })
}

fn load_reference(path: &str) -> reng_core::Result<Reference> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| reng_core::Error::Other(format!("{path}: {e}")))?;
    serde_json::from_str(&text).map_err(|e| reng_core::Error::Other(format!("{path}: {e}")))
}

fn main() -> reng_core::Result<()> {
    let a = parse_args()?;
    let reference = a.ref_path.as_deref().map(load_reference).transpose()?;
    let prompt_len = a.ids.len();
    let n_new = match &reference {
        Some(r) => a.n_new.min(r.generated.len()),
        None => a.n_new,
    };

    let cfg = LlamaConfig::load(Path::new(&a.dir))?;
    let w = load_weights(Path::new(&a.dir), &cfg)?;
    let vocab = cfg.vocab_size;
    let mut ids = a.ids.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(n_new);
    let mut step_secs: Vec<f32> = Vec::with_capacity(n_new);
    let mut near_ties = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut cached = if a.recompute {
        None
    } else {
        assert!(
            prompt_len + n_new <= a.capacity,
            "prompt {prompt_len} + {n_new} new tokens exceed the cache capacity {}",
            a.capacity
        );
        let t0 = Instant::now();
        let g = Generator::new(&w, &cfg, a.rows, a.decode_rows, a.capacity)?;
        println!(
            "compiled cached model (rows {}, decode rows {}, capacity {}): {:.2}s",
            a.rows,
            a.decode_rows,
            a.capacity,
            t0.elapsed().as_secs_f32()
        );
        Some(g)
    };
    // What the cached model has not seen yet: the whole prompt, then the
    // token appended at the previous step.
    let mut pending = 0;
    for step in 0..n_new {
        let t0 = Instant::now();
        let last_logits = match cached.as_mut() {
            Some(g) => g.feed(&ids[pending..])?,
            None => {
                let logits = prefill_logits(&w, &cfg, &ids)?;
                logits[(ids.len() - 1) * vocab..].to_vec()
            }
        };
        pending = ids.len();
        let last = argmax_rows(&last_logits, vocab)[0] as u32;
        step_secs.push(t0.elapsed().as_secs_f32());
        generated.push(last);
        match &reference {
            None => {
                println!(
                    "step {step}: next id {last}  ({:.1} ms)",
                    step_secs[step] * 1000.0
                );
                ids.push(last);
            }
            Some(r) => {
                let want = r.generated[step];
                let verdict = if last == want {
                    "match"
                } else if let Some(s) = r.steps.get(step) {
                    let tie = s.margin < a.margin && (last == s.top1 || last == s.top2);
                    if tie {
                        near_ties += 1;
                        "near-tie"
                    } else {
                        failures.push(format!(
                            "step {step}: got {last}, want {want} (ref margin {:.3})",
                            s.margin
                        ));
                        "DIVERGE"
                    }
                } else {
                    failures.push(format!("step {step}: got {last}, want {want}"));
                    "DIVERGE"
                };
                let margin = r.steps.get(step).map_or(f32::NAN, |s| s.margin);
                println!(
                    "step {step}: engine {last}  ref {want}  margin {margin:.3}  {verdict}  ({:.1} ms)",
                    step_secs[step] * 1000.0
                );
                ids.push(want);
            }
        }
    }
    let out = serde_json::json!({
        "prompt": &ids[..prompt_len],
        "generated": generated,
        "teacher_forced": reference.is_some(),
        "kv_cache": !a.recompute,
        "step_seconds": step_secs,
    });
    std::fs::write(&a.out_path, out.to_string())
        .map_err(|e| reng_core::Error::Other(format!("{}: {e}", a.out_path)))?;
    println!("generated ids: {generated:?}");

    if let Some(r) = &reference {
        let exact = generated
            .iter()
            .zip(&r.generated)
            .filter(|(g, w)| g == w)
            .count();
        println!(
            "vs reference: {exact}/{n_new} exact, {near_ties} near-tie (ref margin < {}), {} diverge",
            a.margin,
            failures.len()
        );
        if failures.is_empty() {
            println!("PASS");
        } else {
            return Err(reng_core::Error::Other(format!(
                "generation diverges from reference: {}",
                failures.join("; ")
            )));
        }
    }
    Ok(())
}
