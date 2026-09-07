//! Greedy generation on Gaudi2 through the fused engine. By default the
//! model is compiled once with a KV cache ([`Generator`]): the prompt goes in
//! as one block per `--rows` tokens and every generated token as a block of
//! one. `--recompute` instead re-runs prefill over the whole sequence at
//! every step (no cache; each step compiles a fresh recipe), the slow
//! cross-check the cached path was validated against.
//!
//! Without `--ref` the loop is free-running and the ids it produces are
//! written to `out.json`; with the device decode loop (`RENG_DEVICE_LOOP`,
//! the default) every token after the first comes out of one device run
//! and one readback. With `--ref <ref.json>` (from `generate.py`) the
//! loop is teacher-forced: step `i` scores `prompt ++ ref[..i]` and the
//! engine's top-1 is compared with `ref[i]`. A mismatch counts as a failure
//! unless the engine's token is within `--margin` logits (default 0.5) of
//! the reference's best candidate in the reference's own f32 logits; such a
//! near-tie is within bf16 rounding and is reported as one instead.
//!
//! `--fp8` (or `RENG_FP8=1`) quantizes the projections at load; see
//! [`reng_model::fp8_switch`] for the values the variable takes.
//!
//! `reng-generate <model_dir> <out.json> <n_new> [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--recompute] [--fp8] <id> [<id> ...]`

use reng_model::{
    CachePath, CachePlan, Fp8Config, Generator, LlamaConfig, argmax_rows, fp8_switch,
    load_weights_fp8, prefill_logits, take_fp8_flag,
};
use std::path::Path;
use std::time::Instant;

const DEFAULT_MARGIN: f32 = 0.5;
const DEFAULT_ROWS: usize = 256;
const DEFAULT_DECODE_ROWS: usize = 1;
const DEFAULT_CAPACITY: usize = 1024;

#[derive(serde::Deserialize)]
struct RefStep {
    top1: u32,
    top2: u32,
    margin: f32,
    /// Top candidates and their f32 logits, best first (newer references).
    #[serde(default)]
    top_ids: Vec<u32>,
    #[serde(default)]
    top_logits: Vec<f32>,
}

impl RefStep {
    /// Whether `id` is within `margin` logits of the reference's best.
    fn near_tie(&self, id: u32, margin: f32) -> bool {
        if self.top_ids.is_empty() {
            return self.margin < margin && (id == self.top1 || id == self.top2);
        }
        let best = self.top_logits[0];
        self.top_ids
            .iter()
            .zip(&self.top_logits)
            .any(|(&t, &l)| t == id && best - l < margin)
    }
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
    fp8: Option<Fp8Config>,
    ids: Vec<u32>,
}

fn parse_args() -> reng_core::Result<Args> {
    let usage = "usage: reng-generate <model_dir> <out.json> <n_new> [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--decode-rows <n>] [--capacity <n>] [--recompute] [--fp8] <id> [<id> ...]";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let fp8 = fp8_switch(
        std::env::var("RENG_FP8").ok().as_deref(),
        take_fp8_flag(&mut args),
    )?;
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
        fp8,
        ids,
    })
}

/// Compare the engine's `last` at `step` with the reference and print the
/// verdict; a divergence is recorded in `failures`, a near-tie counted.
fn verdict(
    r: &Reference,
    step: usize,
    last: u32,
    margin: f32,
    secs: f32,
    near_ties: &mut usize,
    failures: &mut Vec<String>,
) {
    let want = r.generated[step];
    let verdict = if last == want {
        "match"
    } else if let Some(s) = r.steps.get(step) {
        if s.near_tie(last, margin) {
            *near_ties += 1;
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
    let ref_margin = r.steps.get(step).map_or(f32::NAN, |s| s.margin);
    println!(
        "step {step}: engine {last}  ref {want}  margin {ref_margin:.3}  {verdict}  ({:.1} ms)",
        secs * 1000.0
    );
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

    let t_start = Instant::now();
    let cfg = LlamaConfig::load(Path::new(&a.dir))?;
    let (w, fp8_report) = load_weights_fp8(Path::new(&a.dir), &cfg, a.fp8)?;
    if let Some(r) = fp8_report {
        println!(
            "fp8 {}: {r}",
            a.fp8.expect("a report means the switch was on")
        );
    }
    let (mapped, owned) = w.footprint();
    println!(
        "loaded weights in {:.2}s (mapped {:.2} GB, owned {:.2} GB)",
        t_start.elapsed().as_secs_f32(),
        mapped as f64 / 1e9,
        owned as f64 / 1e9
    );
    let vocab = cfg.vocab_size;
    let plan = CachePlan::new(&cfg, w.device_bytes(), a.rows, CachePath::Single);
    if !a.recompute {
        println!("{}", plan.summary(a.capacity));
    }
    if let Some(msg) = cfg.context_warning(prompt_len + n_new) {
        eprintln!("{msg}");
        println!("{msg}");
    }
    let mut ids = a.ids.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(n_new);
    let mut step_secs: Vec<f32> = Vec::with_capacity(n_new);
    let mut near_ties = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut cached = if a.recompute {
        None
    } else {
        if prompt_len + n_new > a.capacity {
            return Err(reng_core::Error::Other(format!(
                "a prompt of {prompt_len} tokens plus {n_new} new tokens exceeds the cache \
                 capacity {} (--capacity); this model holds up to {} positions",
                a.capacity,
                plan.max_capacity()
            )));
        }
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
    let mut step = 0;
    while step < n_new {
        let t0 = Instant::now();
        // Free-running with the device loop: after the first token, the
        // rest come out of one run.
        let whole_run =
            reference.is_none() && step > 0 && cached.as_ref().is_some_and(Generator::device_loop);
        let run = if whole_run {
            let g = cached.as_mut().expect("checked above");
            let last = ids[ids.len() - 1];
            let out = g.generate(last, n_new - step)?;
            pending = ids.len() + out.len() - 1;
            out
        } else {
            let last = match cached.as_mut() {
                Some(g) => g.feed_id(&ids[pending..])?,
                None => {
                    let logits = prefill_logits(&w, &cfg, &ids)?;
                    argmax_rows(&logits[(ids.len() - 1) * vocab..], vocab)[0] as u32
                }
            };
            pending = ids.len();
            vec![last]
        };
        let per_step = t0.elapsed().as_secs_f32() / run.len() as f32;
        if step == 0 {
            println!(
                "first token {:.2}s after start",
                t_start.elapsed().as_secs_f32()
            );
        }
        for &last in &run {
            step_secs.push(per_step);
            generated.push(last);
            match &reference {
                None => {
                    println!(
                        "step {step}: next id {last}  ({:.1} ms{})",
                        per_step * 1000.0,
                        if run.len() > 1 { ", device loop" } else { "" }
                    );
                    ids.push(last);
                }
                Some(r) => {
                    verdict(
                        r,
                        step,
                        last,
                        a.margin,
                        per_step,
                        &mut near_ties,
                        &mut failures,
                    );
                    ids.push(r.generated[step]);
                }
            }
            step += 1;
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
