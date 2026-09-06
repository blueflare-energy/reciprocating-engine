//! Greedy generation with a model split over several Gaudi2 cards (tensor
//! parallelism over HCCL, see `reng_synapse::tp`). One process per card:
//! the coordinator (this binary without `--rank`) spawns one worker per
//! module id, in rank order, hands them a directory to hand-shake and
//! exchange the HCCL unique id through, echoes their output under
//! `[r<rank>]` prefixes, and checks that every rank produced the same ids
//! (each rank computes the argmax after the last all-reduce, so they must).
//!
//! `reng-tp <model_dir> <n_new> [<id> ...] --modules 4,1 [--prompt-file <json>] [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--capacity <n>] [--batch <n>] [--bench <tokens>] [--out <out.json>] [--timeout <s>] [--no-numa]`
//!
//! The prompt is the trailing ids, or `--prompt-file <json>`: the
//! `"prompt"` array of a `generate.py` reference file (any trailing ids
//! are appended to it), which is how a long prompt gets in without a
//! thousand arguments.
//!
//! With `--ref` the run is teacher-forced against a `generate.py`
//! reference, as `reng-generate --ref` (step `i` scores `prompt ++
//! ref[..i]`; a mismatch within `--margin` logits of the reference's best
//! is a near-tie, not a failure). With `--batch B` every sequence of the
//! batch gets the prompt and advances in lockstep (the batched decode
//! form); the ids reported are sequence 0's. With `--bench <tokens>`
//! every rank then times that many decode steps in the four modes of
//! `reng_synapse::tp::Mode` and rank 0 reports the per-layer split
//! (recipe A, the all-reduces, recipe B) that the differences give. One
//! module id runs the same graphs on one card without a communicator.
//!
//! Worker (spawned by the coordinator; the same binary):
//! `reng-tp --rank r --world n --module m --dir DIR <the coordinator's arguments>`

use reng_model::{LlamaConfig, TpGenerator, load_weights};
use reng_synapse::hccl::{EXIT_ACQUIRE, Group, Rank, abort_and_die};
use reng_synapse::tp::{Mode, rss_bytes};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_MARGIN: f32 = 0.5;
const DEFAULT_ROWS: usize = 256;
const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_TIMEOUT_S: u64 = 3600;

#[derive(serde::Deserialize)]
struct RefStep {
    top1: u32,
    top2: u32,
    margin: f32,
    #[serde(default)]
    top_ids: Vec<u32>,
    #[serde(default)]
    top_logits: Vec<f32>,
}

impl RefStep {
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

struct Opts {
    dir: String,
    n_new: usize,
    ids: Vec<u32>,
    modules: Vec<u32>,
    prompt_file: Option<String>,
    ref_path: Option<String>,
    margin: f32,
    rows: usize,
    capacity: usize,
    batch: usize,
    bench: usize,
    out_path: Option<String>,
    timeout_s: u64,
    numa: bool,
    /// Worker mode.
    rank: Option<usize>,
    world: usize,
    module: u32,
    hand_dir: Option<PathBuf>,
    /// Every argument after the program name, forwarded to the workers.
    raw: Vec<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: reng-tp <model_dir> <n_new> [<id>...] --modules 4,1 [--prompt-file <json>] [--ref <ref.json>] [--margin <f32>] [--rows <n>] [--capacity <n>] [--batch <n>] [--bench <tokens>] [--out <out.json>] [--timeout <s>] [--no-numa]"
    );
    std::process::exit(2)
}

#[allow(clippy::too_many_lines)]
fn parse() -> Opts {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut o = Opts {
        dir: String::new(),
        n_new: 0,
        ids: Vec::new(),
        modules: Vec::new(),
        prompt_file: None,
        ref_path: None,
        margin: DEFAULT_MARGIN,
        rows: DEFAULT_ROWS,
        capacity: DEFAULT_CAPACITY,
        batch: 1,
        bench: 0,
        out_path: None,
        timeout_s: DEFAULT_TIMEOUT_S,
        numa: true,
        rank: None,
        world: 0,
        module: 0,
        hand_dir: None,
        raw: raw.clone(),
    };
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    let value = |i: &mut usize| -> String {
        *i += 1;
        raw.get(*i).cloned().unwrap_or_else(|| usage())
    };
    let num = |s: &str| -> usize { s.parse().unwrap_or_else(|_| usage()) };
    while i < raw.len() {
        match raw[i].as_str() {
            "--modules" => {
                for m in value(&mut i).split(',') {
                    o.modules.push(m.trim().parse().unwrap_or_else(|_| usage()));
                }
            }
            "--prompt-file" => o.prompt_file = Some(value(&mut i)),
            "--ref" => o.ref_path = Some(value(&mut i)),
            "--margin" => o.margin = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--rows" => o.rows = num(&value(&mut i)),
            "--capacity" => o.capacity = num(&value(&mut i)),
            "--batch" => o.batch = num(&value(&mut i)).max(1),
            "--bench" => o.bench = num(&value(&mut i)),
            "--out" => o.out_path = Some(value(&mut i)),
            "--timeout" => o.timeout_s = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--no-numa" => o.numa = false,
            "--rank" => o.rank = Some(num(&value(&mut i))),
            "--world" => o.world = num(&value(&mut i)),
            "--module" => o.module = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--dir" => o.hand_dir = Some(PathBuf::from(value(&mut i))),
            a if a.starts_with("--") => usage(),
            a => positional.push(a.to_owned()),
        }
        i += 1;
    }
    if positional.len() < if o.prompt_file.is_some() { 2 } else { 3 } {
        usage();
    }
    o.dir = positional[0].clone();
    o.n_new = num(&positional[1]);
    o.ids = positional[2..].iter().map(|s| num(s) as u32).collect();
    if let Some(path) = &o.prompt_file {
        let mut ids = prompt_ids(path);
        ids.extend_from_slice(&o.ids);
        o.ids = ids;
    }
    if o.rank.is_none() && o.modules.is_empty() {
        usage();
    }
    // The workers get everything but the coordinator-only options.
    let mut fwd = Vec::new();
    let mut j = 0;
    while j < raw.len() {
        match raw[j].as_str() {
            "--modules" | "--timeout" => j += 2,
            "--no-numa" => j += 1,
            _ => {
                fwd.push(raw[j].clone());
                j += 1;
            }
        }
    }
    o.raw = fwd;
    o
}

/// The `"prompt"` array of a `generate.py` reference file, as ids.
fn prompt_ids(path: &str) -> Vec<u32> {
    let bail = |e: String| -> ! {
        eprintln!("{path}: {e}");
        std::process::exit(2)
    };
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| bail(e.to_string()));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| bail(e.to_string()));
    let Some(a) = v.get("prompt").and_then(|p| p.as_array()) else {
        bail(String::from("no \"prompt\" array"))
    };
    a.iter()
        .map(|x| {
            u32::try_from(x.as_u64().unwrap_or_else(|| bail(format!("prompt id {x}"))))
                .unwrap_or_else(|_| bail(format!("prompt id {x}")))
        })
        .collect()
}

fn load_reference(path: &str) -> reng_core::Result<Reference> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| reng_core::Error::Other(format!("{path}: {e}")))?;
    serde_json::from_str(&text).map_err(|e| reng_core::Error::Other(format!("{path}: {e}")))
}

fn main() {
    let o = parse();
    let code = match o.rank {
        Some(rank) => match worker(&o, rank) {
            Ok(()) => 0,
            Err(e) if e.to_string().starts_with("acquire:") => {
                println!("RESULT: rank {rank} ACQUIRE-FAILED: {e}");
                EXIT_ACQUIRE
            }
            Err(e) => {
                println!("RESULT: rank {rank} ERROR: {e}");
                2
            }
        },
        None => coordinate(&o),
    };
    std::process::exit(code);
}

/// Compare the engine's `last` at `step` with the reference (see
/// `reng-generate`): a divergence goes into `failures`, a near-tie is
/// counted.
fn verdict(
    r: &Reference,
    step: usize,
    last: u32,
    margin: f32,
    ms: f64,
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
        "step {step}: engine {last}  ref {want}  margin {ref_margin:.3}  {verdict}  ({ms:.1} ms)"
    );
}

#[allow(clippy::too_many_lines)]
fn worker(o: &Opts, rank: usize) -> reng_core::Result<()> {
    let dir = o.hand_dir.as_deref().unwrap_or_else(|| usage());
    if o.world == 0 {
        usage();
    }
    let reference = o.ref_path.as_deref().map(load_reference).transpose()?;
    let n_new = match &reference {
        Some(r) => o.n_new.min(r.generated.len()),
        None => o.n_new,
    };
    let t_start = Instant::now();
    let joined = Rank::join(rank, o.world, o.module, dir)?;
    let t_join = t_start.elapsed().as_secs_f64();

    let model_dir = Path::new(&o.dir);
    let cfg = LlamaConfig::load(model_dir)?;
    let t0 = Instant::now();
    let full = load_weights(model_dir, &cfg)?;
    let shard = full.shard(&cfg, rank, o.world);
    let scfg = cfg.shard(rank, o.world);
    let (mapped, owned) = shard.footprint();
    let rss_load = rss_bytes().unwrap_or(0);
    println!(
        "rank {rank}: joined in {t_join:.2} s; shard {rank}/{} loaded in {:.2} s (viewed {:.2} GB, owned {:.2} GB; {} heads, {} kv heads, inter {}); rss {:.2} GB",
        o.world,
        t0.elapsed().as_secs_f64(),
        mapped as f64 / 1e9,
        owned as f64 / 1e9,
        scfg.num_attention_heads,
        scfg.n_kv_heads(),
        scfg.intermediate_size,
        rss_load as f64 / 1e9
    );
    assert!(
        o.ids.len() + n_new.max(o.bench + 2) <= o.capacity,
        "prompt {} + {n_new} new tokens exceed the cache capacity {}",
        o.ids.len(),
        o.capacity
    );
    let t1 = Instant::now();
    let mut g = TpGenerator::new(joined, &shard, &scfg, o.batch, o.rows, o.capacity)?;
    let rss_up = rss_bytes().unwrap_or(0);
    let (upload_s, device_bytes) = (g.model().upload_s, g.model().device_bytes);
    println!(
        "rank {rank}: recipes and shard on the card in {:.2} s (layers 1.. uploaded in {:.2} s, {:.2} GB in the store); rss {:.2} GB; {:.1} s since start",
        t1.elapsed().as_secs_f64(),
        upload_s,
        device_bytes as f64 / 1e9,
        rss_up as f64 / 1e9,
        t_start.elapsed().as_secs_f64()
    );

    match generate_and_check(&mut g, o, rank, dir, n_new, reference.as_ref(), t_start) {
        Ok(()) => Ok(()),
        Err(e) => {
            // The probe's rule: after a failure with a live communicator
            // the orderly teardown can hang inside libSynapse for minutes
            // while still holding the card. Abort the communicator and
            // leave at once instead.
            println!("RESULT: rank {rank} ERROR: {e}");
            abort_and_die(2);
        }
    }
}

/// Prefill the prompt into every sequence, run the decode loop (teacher
/// forced against `reference` when there is one), write this rank's ids
/// for the coordinator to compare, and run the bench when asked.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn generate_and_check(
    g: &mut TpGenerator<'_>,
    o: &Opts,
    rank: usize,
    dir: &Path,
    n_new: usize,
    reference: Option<&Reference>,
    t_start: Instant,
) -> reng_core::Result<()> {
    let nb = o.batch;
    let mut generated: Vec<u32> = Vec::with_capacity(n_new);
    let mut step_ms: Vec<f64> = Vec::with_capacity(n_new);
    let mut near_ties = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let t2 = Instant::now();
    // Every sequence gets the prompt; the ids reported are sequence 0's.
    let mut first = 0;
    for b in 0..nb {
        first = g.prefill(b, &o.ids)?;
    }
    let ms = t2.elapsed().as_secs_f64() * 1e3 / nb as f64;
    println!(
        "rank {rank}: first token {:.2} s after start",
        t_start.elapsed().as_secs_f64()
    );
    let quiet = rank != 0;
    generated.push(first);
    step_ms.push(ms);
    match reference {
        Some(r) => {
            if !quiet {
                verdict(r, 0, first, o.margin, ms, &mut near_ties, &mut failures);
            }
            for step in 1..n_new {
                let t = Instant::now();
                let (ids, _) = g.generate(&vec![r.generated[step - 1]; nb], 1)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if !quiet {
                    verdict(r, step, ids[0], o.margin, ms, &mut near_ties, &mut failures);
                }
                generated.push(ids[0]);
                step_ms.push(ms);
            }
        }
        None => {
            if !quiet {
                println!("step 0: next id {first}  ({ms:.1} ms)");
            }
            if n_new > 1 {
                let t = Instant::now();
                let (ids, times) = g.generate(&vec![first; nb], n_new - 1)?;
                let per = t.elapsed().as_secs_f64() * 1e3 / (n_new - 1) as f64;
                for k in 0..n_new - 1 {
                    let id = ids[k * nb];
                    if !quiet {
                        println!("step {}: next id {id}  ({per:.1} ms, device loop)", k + 1);
                    }
                    generated.push(id);
                    step_ms.push(per);
                }
                if !quiet {
                    println!(
                        "rank {rank}: {} steps x {nb} enqueued in {:.1} ms, {:.1} ms total",
                        n_new - 1,
                        times.enqueue * 1e3,
                        times.total * 1e3
                    );
                }
            }
        }
    }
    println!("rank {rank}: generated ids: {generated:?}");
    let ids_json = serde_json::json!({
        "prompt": &o.ids,
        "generated": generated,
        "teacher_forced": reference.is_some(),
        "step_ms": step_ms,
    });
    std::fs::write(dir.join(format!("rank{rank}.ids")), ids_json.to_string())?;

    if o.bench > 0 {
        bench(g, o, rank)?;
    }

    if let Some(r) = reference {
        if !quiet {
            let exact = generated
                .iter()
                .zip(&r.generated)
                .filter(|(a, b)| a == b)
                .count();
            println!(
                "vs reference: {exact}/{n_new} exact, {near_ties} near-tie (ref margin < {}), {} diverge",
                o.margin,
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
    }
    Ok(())
}

/// Time `o.bench` decode steps after the prompt in each mode and print
/// the per-step and per-layer split (rank 0).
fn bench(g: &mut TpGenerator<'_>, o: &Opts, rank: usize) -> reng_core::Result<()> {
    let n = o.bench;
    let nb = g.batch();
    let n_layers = LlamaConfig::load(Path::new(&o.dir))?.num_hidden_layers as f64;
    let mut per_step = Vec::new();
    for mode in [
        Mode::Full,
        Mode::NoAllReduce,
        Mode::AttnOnly,
        Mode::NoLayers,
    ] {
        let mut seed = 0;
        for b in 0..nb {
            g.reset(b);
            seed = g.prefill(b, &o.ids)?;
        }
        let seeds = vec![seed; nb];
        g.model().mode = mode;
        // Warm-up, then the timed run.
        let _ = g.generate(&seeds, 2)?;
        let t = Instant::now();
        let (ids, times) = g.generate(&seeds, n)?;
        let s = t.elapsed().as_secs_f64();
        g.model().mode = Mode::Full;
        let ms = s * 1e3 / n as f64;
        per_step.push(ms);
        if rank == 0 {
            println!(
                "bench {mode:?}: {n} steps x {nb} in {s:.3} s = {:.1} tok/s, {ms:.2} ms/step (enqueue {:.2} ms/step, ids {:?}..)",
                (n * nb) as f64 / s,
                times.enqueue * 1e3 / n as f64,
                &ids[..ids.len().min(4)]
            );
        }
    }
    if rank == 0 {
        let [full, no_ar, a_only, no_layers] = [per_step[0], per_step[1], per_step[2], per_step[3]];
        let world = g.model().rank().1;
        println!(
            "bench split (world {world}, batch {nb}, {n_layers} layers): step {full:.2} ms; per layer: recipe A {:.1} us, all-reduces {:.1} us, recipe B {:.1} us; embed + head {no_layers:.2} ms",
            (a_only - no_layers) / n_layers * 1e3,
            (full - no_ar) / n_layers * 1e3,
            (no_ar - a_only) / n_layers * 1e3
        );
        println!(
            "RESULT: decode b{nb} {:.1} tok/s ({full:.2} ms/step)",
            1e3 / full * nb as f64
        );
    }
    Ok(())
}

/// Spawn the ranks, wait for them, and compare their ids.
fn coordinate(o: &Opts) -> i32 {
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("current_exe: {e}");
        std::process::exit(2)
    });
    let prompt = if o.ids.len() > 8 {
        format!("{} ids {:?}..", o.ids.len(), &o.ids[..8])
    } else {
        format!("{:?}", o.ids)
    };
    println!(
        "coordinator: modules {:?} (rank order), model {}, {} new tokens, prompt {prompt}, rows {}, capacity {}, timeout {} s",
        o.modules, o.dir, o.n_new, o.rows, o.capacity, o.timeout_s
    );
    let started = Instant::now();
    let deadline = started + Duration::from_secs(o.timeout_s);
    let max_attempts = 3;
    for attempt in 1..=max_attempts {
        let dir = std::env::temp_dir().join(format!("reng-tp-{}-{attempt}", std::process::id()));
        println!(
            "coordinator: attempt {attempt}/{max_attempts}, directory {}",
            dir.display()
        );
        let mut group = match Group::spawn(&exe, &o.modules, &o.raw, &dir, o.numa) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        };
        if !group.wait_acquired(deadline) {
            let codes = group.wait_all(deadline);
            println!("coordinator: acquire phase failed, exit codes {codes:?}");
            if group.all_exited_with(&[EXIT_ACQUIRE, 0])
                && attempt < max_attempts
                && Instant::now() + Duration::from_secs(60) < deadline
            {
                println!("coordinator: waiting 60 s before relaunching the group");
                std::thread::sleep(Duration::from_secs(60));
                continue;
            }
            println!("VERDICT: FAIL (could not acquire every card)");
            return 1;
        }
        let codes = group.wait_all(deadline);
        println!(
            "coordinator: exit codes {codes:?} in {:.1} s",
            group.elapsed()
        );
        let mut verdict = if codes.iter().all(|&c| c == 0) { 0 } else { 1 };
        // Every rank's ids must agree with rank 0's.
        let mut ids: Vec<Option<serde_json::Value>> = Vec::new();
        for r in 0..o.modules.len() {
            let v = std::fs::read_to_string(dir.join(format!("rank{r}.ids")))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
            ids.push(v);
        }
        match &ids[0] {
            Some(v0) => {
                let gen0 = v0.get("generated").cloned().unwrap_or_default();
                println!("generated ids: {gen0}");
                for (r, v) in ids.iter().enumerate().skip(1) {
                    match v {
                        Some(v) if v.get("generated") == Some(&gen0) => {}
                        Some(v) => {
                            println!(
                                "coordinator: rank {r} DISAGREES: {}",
                                v.get("generated").cloned().unwrap_or_default()
                            );
                            verdict = 1;
                        }
                        None => {
                            println!("coordinator: rank {r} produced no ids");
                            verdict = 1;
                        }
                    }
                }
                if verdict == 0 {
                    println!("coordinator: all {} ranks agree", o.modules.len());
                }
                if let Some(path) = &o.out_path {
                    if let Err(e) = std::fs::write(path, v0.to_string()) {
                        eprintln!("{path}: {e}");
                        verdict = 1;
                    }
                }
            }
            None => {
                println!("coordinator: rank 0 produced no ids");
                verdict = 1;
            }
        }
        println!("VERDICT: {}", if verdict == 0 { "PASS" } else { "FAIL" });
        return verdict;
    }
    1
}
