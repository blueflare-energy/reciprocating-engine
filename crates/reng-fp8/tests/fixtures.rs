//! The quantizer against PyTorch's own fp8 cast.
//!
//! `tools/fp8_fixtures.py` writes `fixtures/fp8_cast.json`: random and
//! Llama-like weight blocks plus real rows of a SmolLM2-135M checkpoint
//! (read out of the safetensors file with numpy), each with the per-output
//! -channel scales, the bytes `torch.Tensor.to(torch.float8_e4m3fn)` /
//! `float8_e5m2` produced for them, and the per-row relative error of the
//! round trip. Regenerate it with
//!
//! ```text
//! python3 tools/fp8_fixtures.py --model <hf_model_dir> \
//!     --out crates/reng-fp8/fixtures/fp8_cast.json
//! ```
//!
//! Torch's `float8_e4m3fn` and Gaudi's `hf8` share the exponent bias, the
//! round-to-nearest-even and every finite code up to 240; they part above
//! it, where Gaudi reserves the top exponent field for infinity and NaN and
//! its clipping cast saturates. The fixture stores torch's own bytes for
//! the cases that reach that range, and the test below checks the
//! divergence is exactly the saturation and nothing else.

use reng_fp8::{Fp8Format, Quantized, Scaling, quantize, row_errors};
use serde_json::Value;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/fp8_cast.json");

fn doc() -> Value {
    let text = std::fs::read_to_string(FIXTURE).expect("the fixture file");
    serde_json::from_str(&text).expect("the fixture parses")
}

fn hex_u16(s: &str) -> Vec<u16> {
    (0..s.len() / 4)
        .map(|i| u16::from_str_radix(&s[i * 4..i * 4 + 4], 16).expect("hex"))
        .collect()
}

fn hex_u8(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .expect(key)
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

/// One fixture case, with the weights of the case it shares them with
/// resolved.
struct Case {
    name: String,
    fmt: Fp8Format,
    scaling: Scaling,
    rows: usize,
    cols: usize,
    exp_bias: u32,
    w: Vec<u16>,
    scales: Vec<f32>,
    codes: Vec<u8>,
    torch_raw: Option<Vec<u8>>,
    mean_rel: Vec<f64>,
    max_rel: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let doc = doc();
    let raw = doc["cases"].as_array().expect("cases").clone();
    let mut out: Vec<Case> = Vec::with_capacity(raw.len());
    for c in &raw {
        let name = c["name"].as_str().expect("name").to_owned();
        let from = c["bf16_from"].as_str().unwrap_or("");
        let w = if from.is_empty() {
            hex_u16(c["bf16_hex"].as_str().expect("bf16_hex"))
        } else {
            out.iter()
                .find(|o| o.name == from)
                .unwrap_or_else(|| panic!("{name}: shares weights with unknown case {from}"))
                .w
                .clone()
        };
        out.push(Case {
            name,
            fmt: Fp8Format::from_name(c["format"].as_str().expect("format")).expect("format"),
            scaling: Scaling::from_name(c["scaling"].as_str().expect("scaling")).expect("scaling"),
            rows: c["rows"].as_u64().expect("rows") as usize,
            cols: c["cols"].as_u64().expect("cols") as usize,
            exp_bias: c["exp_bias"].as_u64().expect("exp_bias") as u32,
            w,
            scales: c["scale_bits"]
                .as_array()
                .expect("scale_bits")
                .iter()
                .map(|x| f32::from_bits(x.as_u64().expect("bits") as u32))
                .collect(),
            codes: hex_u8(c["codes_hex"].as_str().expect("codes_hex")),
            torch_raw: c["torch_raw_hex"].as_str().map(hex_u8),
            mean_rel: floats(c, "mean_rel"),
            max_rel: floats(c, "max_rel"),
        });
    }
    out
}

fn quantized(c: &Case) -> Quantized {
    quantize(&c.w, c.rows, c.cols, c.fmt, c.scaling)
}

#[test]
fn the_fixture_is_the_one_the_generator_writes() {
    let doc = doc();
    assert_eq!(doc["generator"], "tools/fp8_fixtures.py");
    let n = doc["cases"].as_array().expect("cases").len();
    assert!(n >= 12, "{n} cases");
    // The checkpoint rows are part of the fixture, not an optional extra.
    let names: Vec<&str> = doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["name"].as_str().expect("name"))
        .collect();
    assert!(
        names.iter().filter(|n| n.starts_with("ckpt_")).count() >= 6,
        "no checkpoint cases in {names:?}"
    );
}

#[test]
fn scales_match_torchs_input_bit_for_bit() {
    for c in cases() {
        let q = quantized(&c);
        assert_eq!(q.rows, c.rows, "{}", c.name);
        assert_eq!(q.exp_bias, c.exp_bias, "{} exponent bias", c.name);
        assert_eq!(
            q.scales.iter().map(|s| s.to_bits()).collect::<Vec<u32>>(),
            c.scales.iter().map(|s| s.to_bits()).collect::<Vec<u32>>(),
            "{} scales",
            c.name
        );
    }
}

#[test]
fn codes_match_the_torch_cast() {
    for c in cases() {
        let q = quantized(&c);
        assert_eq!(q.codes.len(), c.codes.len(), "{}", c.name);
        let bad: Vec<(usize, u8, u8)> = q
            .codes
            .iter()
            .zip(&c.codes)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (&a, &b))| (i, a, b))
            .take(4)
            .collect();
        assert!(
            bad.is_empty(),
            "{}: {} of {} codes differ from torch, first (index, ours, torch): {bad:?}",
            c.name,
            q.codes.iter().zip(&c.codes).filter(|(a, b)| a != b).count(),
            q.codes.len()
        );
    }
}

#[test]
fn the_only_divergence_from_torch_is_the_saturation() {
    let mut checked = 0;
    for c in cases() {
        let Some(raw) = c.torch_raw.as_ref() else {
            continue;
        };
        checked += 1;
        let q = quantized(&c);
        let reserved = match c.fmt {
            Fp8Format::E4M3 => 0x78u8,
            Fp8Format::E5M2 => 0x7cu8,
        };
        let mut saturated = 0;
        for (i, (&ours, &theirs)) in q.codes.iter().zip(raw).enumerate() {
            if theirs & reserved == reserved {
                // Out of Gaudi's finite range: we clip, torch does not.
                saturated += 1;
                assert_eq!(
                    ours,
                    (theirs & 0x80) | c.fmt.max_finite_code(),
                    "{} index {i}",
                    c.name
                );
            } else {
                assert_eq!(ours, theirs, "{} index {i}", c.name);
            }
        }
        assert!(saturated > 0, "{}: nothing saturated", c.name);
    }
    assert!(checked >= 2, "no saturation cases in the fixture");
}

#[test]
fn row_error_statistics_match_numpy() {
    for c in cases() {
        let q = quantized(&c);
        let errs = row_errors(&c.w, &q);
        assert_eq!(errs.len(), c.mean_rel.len(), "{}", c.name);
        for (r, e) in errs.iter().enumerate() {
            let (want_mean, want_max) = (c.mean_rel[r], c.max_rel[r]);
            assert!(
                (f64::from(e.mean_rel) - want_mean).abs() <= 1e-6 + 1e-5 * want_mean,
                "{} row {r}: mean_rel {} against numpy {want_mean}",
                c.name,
                e.mean_rel
            );
            assert!(
                (f64::from(e.max_rel) - want_max).abs() <= 1e-6 + 1e-5 * want_max,
                "{} row {r}: max_rel {} against numpy {want_max}",
                c.name,
                e.max_rel
            );
        }
    }
}

#[test]
fn checkpoint_rows_quantize_within_the_formats_resolution() {
    // E4M3 keeps three mantissa bits, so a weight inside the scaled range
    // is within 1/16 of its value; the mean over a real row is about 2.2%.
    // A row's smallest weights fall into the subnormal range, where the
    // relative error grows, which is what the per-row maximum shows.
    let mut seen = 0;
    for c in cases() {
        if !c.name.starts_with("ckpt_") || c.scaling != Scaling::PerChannel {
            continue;
        }
        seen += 1;
        let q = quantized(&c);
        let errs = row_errors(&c.w, &q);
        for (r, e) in errs.iter().enumerate() {
            assert!(
                e.mean_rel < 0.03,
                "{} row {r}: mean relative error {}",
                c.name,
                e.mean_rel
            );
            assert!(
                e.max_rel <= 0.25,
                "{} row {r}: max relative error {}",
                c.name,
                e.max_rel
            );
        }
        // The round trip reproduces every weight of the row to that
        // accuracy, and the largest one within the format's resolution.
        let back = q.dequantize();
        for r in 0..q.rows {
            let base = r * q.cols;
            let (mut am, mut ai) = (0.0f32, 0usize);
            for i in 0..q.cols {
                let v = reng_fp8::bf16_to_f32(c.w[base + i]).abs();
                if v > am {
                    am = v;
                    ai = i;
                }
            }
            let want = reng_fp8::bf16_to_f32(c.w[base + ai]);
            let got = back[base + ai];
            assert!(
                (got - want).abs() <= want.abs() * 1e-6,
                "{} row {r}: the absmax weight {want} came back as {got}",
                c.name
            );
        }
    }
    assert!(seen >= 3, "no per-channel checkpoint cases");
}
