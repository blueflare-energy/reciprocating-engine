//! Prefill a HF Llama-family model on Gaudi2 through the fused engine and
//! report per-position argmax tokens, optionally comparing against a reference
//! JSON produced by an HF transformers run of the same token ids.
//!
//! `reng-prefill <model_dir> <out.json> [--ref <ref.json>] [--fp8] [--cached]
//! [--rows <n>] <id> [<id> ...]`
//!
//! `--fp8` (or `RENG_FP8=1`) quantizes the projections at load; see
//! [`reng_model::fp8_switch`] for the values the variable takes.
//!
//! The reference JSON has `argmax` (per position), `last_logits` (full row),
//! and `last_top5` (ids). The engine's output JSON has the same fields.
//!
//! Prompts up to `max_nocache_prefill` run as one recipe of `tokens` rows
//! with a full `[tokens, tokens]` mask. Longer ones (and `--cached`) run in
//! blocks over the KV cache instead: the one-recipe form would need tensor
//! strides wider than 32 bits and a DRAM workspace the card does not have.

use reng_model::{
    CachePath, CachePlan, LlamaConfig, MIN_PREFILL_TOKENS, argmax_rows, fp8_switch,
    load_weights_fp8, max_nocache_prefill, prefill_logits, prefill_logits_cached, take_fp8_flag,
};
use std::path::Path;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Reference {
    argmax: Vec<usize>,
    last_logits: Vec<f32>,
    last_top5: Vec<usize>,
}

fn top5(row: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(5);
    idx
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(1e-12)) as f32
}

fn main() -> reng_core::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let fp8 = fp8_switch(
        std::env::var("RENG_FP8").ok().as_deref(),
        take_fp8_flag(&mut args),
    )?;
    if args.len() < 3 {
        return Err(reng_core::Error::Other(
            "usage: reng-prefill <model_dir> <out.json> [--ref <ref.json>] [--fp8] [--cached] \
             [--rows <n>] <id> [<id> ...]"
                .into(),
        ));
    }
    let dir = Path::new(&args[0]);
    let out_path = &args[1];
    let mut ref_path: Option<String> = None;
    let mut force_cached = false;
    let mut rows = MIN_PREFILL_TOKENS;
    let mut id_start = 2;
    while id_start < args.len() {
        match args[id_start].as_str() {
            "--ref" => {
                ref_path = args.get(id_start + 1).cloned();
                id_start += 2;
            }
            "--cached" => {
                force_cached = true;
                id_start += 1;
            }
            "--rows" => {
                rows = args
                    .get(id_start + 1)
                    .and_then(|s| s.parse().ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| {
                        reng_core::Error::Other("--rows needs a number above zero".into())
                    })?;
                id_start += 2;
            }
            _ => break,
        }
    }
    let ids: Vec<u32> = args[id_start..]
        .iter()
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| reng_core::Error::Other(format!("token id: {e}")))?;
    if ids.is_empty() {
        return Err(reng_core::Error::Other("no token ids given".into()));
    }

    let t0 = Instant::now();
    let cfg = LlamaConfig::load(dir)?;
    let (w, fp8_report) = load_weights_fp8(dir, &cfg, fp8)?;
    if let Some(r) = fp8_report {
        println!(
            "fp8 {}: {r}",
            fp8.expect("a report means the switch was on")
        );
    }
    let (mapped, owned) = w.footprint();
    println!(
        "loaded {} layers, hidden {}, inter {}, heads {}/{} kv, head_dim {}, vocab {} in {:.2}s (mapped {:.2} GB, owned {:.2} GB)",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.n_kv_heads(),
        cfg.head_dim(),
        cfg.vocab_size,
        t0.elapsed().as_secs_f32(),
        mapped as f64 / 1e9,
        owned as f64 / 1e9
    );

    let tokens = ids.len();
    if let Some(msg) = cfg.context_warning(tokens) {
        eprintln!("{msg}");
        println!("{msg}");
    }
    let limit = max_nocache_prefill(&cfg);
    let cached = force_cached || tokens > limit;
    if cached {
        let plan = CachePlan::new(&cfg, w.device_bytes(), rows, CachePath::Single);
        println!("{}", plan.summary(tokens));
        if !force_cached {
            println!(
                "prompt of {tokens} tokens is past the {limit}-token limit of the one-recipe \
                 prefill (a [{tokens}, {tokens}, {}] score tensor needs a tensor stride wider \
                 than 32 bits, and its workspace more DRAM than the card has): running it in \
                 blocks of {rows} over the KV cache",
                cfg.num_attention_heads
            );
        }
        // Every position's logits come back in host f32: 4 * vocab bytes
        // a token, which is the binding cost of a long prompt on this
        // path and fails by aborting, not with an error.
        let host = 4.0 * cfg.vocab_size as f64 * tokens as f64;
        if host > 8e9 {
            let msg = format!(
                "warning: {tokens} positions of f32 logits are {:.1} GB of host memory in \
                 one allocation ({} bytes a token); this path returns every position's \
                 logits, and the process aborts if the allocator cannot serve it",
                host / 1e9,
                4 * cfg.vocab_size
            );
            eprintln!("{msg}");
            println!("{msg}");
        }
    }
    let t1 = Instant::now();
    let logits = if cached {
        prefill_logits_cached(&w, &cfg, &ids, rows)?
    } else {
        prefill_logits(&w, &cfg, &ids)?
    };
    let dt = t1.elapsed().as_secs_f32();
    let vocab = cfg.vocab_size;
    if cached {
        println!(
            "prefill of {tokens} tokens (cached, {} blocks of {rows}): {dt:.2}s",
            tokens.div_ceil(rows)
        );
    } else {
        println!("prefill of {tokens} tokens (graph build + compile + run): {dt:.2}s");
    }

    let am = argmax_rows(&logits, vocab);
    let last = &logits[(tokens - 1) * vocab..];
    let t5 = top5(last);
    let zeros = logits.iter().filter(|v| **v == 0.0).count();
    // Per-position diagnostics: which token rows are entirely zero, and the
    // row norms around the boundary, to localise an intra-graph zeroing.
    let row_norm = |r: usize| -> f32 {
        logits[r * vocab..(r + 1) * vocab]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    };
    let zero_rows: Vec<usize> = (0..tokens).filter(|&r| row_norm(r) == 0.0).collect();
    println!("argmax: {am:?}");
    println!("last top5: {t5:?}   zeros={zeros}/{}", logits.len());
    println!(
        "zero rows: {} of {tokens} {:?}; row norms 0..8: {:?}",
        zero_rows.len(),
        zero_rows.iter().take(40).collect::<Vec<_>>(),
        (0..tokens.min(8)).map(row_norm).collect::<Vec<_>>()
    );

    let out = serde_json::json!({
        "argmax": am,
        "last_logits": last,
        "last_top5": t5,
        "tokens": tokens,
        "prefill_seconds": dt,
    });
    std::fs::write(out_path, out.to_string())
        .map_err(|e| reng_core::Error::Other(format!("{out_path}: {e}")))?;

    if let Some(rp) = ref_path {
        let text = std::fs::read_to_string(&rp)
            .map_err(|e| reng_core::Error::Other(format!("{rp}: {e}")))?;
        let r: Reference = serde_json::from_str(&text)
            .map_err(|e| reng_core::Error::Other(format!("{rp}: {e}")))?;
        let agree = am.iter().zip(&r.argmax).filter(|(a, b)| a == b).count();
        let cos = cosine(last, &r.last_logits);
        let top1_ok = t5.first() == r.last_top5.first();
        println!(
            "vs reference: argmax agree {agree}/{tokens}, last-position top1 {}, last-logits cosine {cos:.4}, ref top5 {:?}",
            if top1_ok { "MATCH" } else { "DIFFER" },
            r.last_top5
        );
        if top1_ok && cos > 0.98 && agree * 10 >= tokens * 8 {
            println!("PASS");
        } else {
            return Err(reng_core::Error::Other(format!(
                "engine diverges from reference: agree {agree}/{tokens}, cosine {cos}"
            )));
        }
    }
    Ok(())
}
