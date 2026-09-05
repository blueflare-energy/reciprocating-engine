//! Pin down which GELU `gelu_fwd_bf16` computes (the erf form or the
//! tanh approximation `gelu_pytorch_tanh` that Gemma uses), and probe the
//! two helper contracts the Gemma graphs may need: `mult_fwd_bf16` with a
//! `[1, 1]` constant broadcast over a matrix, and `tanh_fwd_bf16` on a 4-D
//! tensor.
//!
//! The two GELU forms agree to within a bf16 ulp for |x| <= 2, so the
//! input cycles through values that separate them (x = -4, -3, -2.5, -1:
//! at x = -3 the erf form gives -0.0040497 and the tanh form -0.0036374).
//!
//! Finding (SynapseAI 1.24.1): `gelu_fwd_bf16` compiles only with two
//! outputs (the second is the "retain" tensor of its backward pass) and
//! with `ns_GeluKernel::Params { approximation = 0 }` computes the erf form
//! exactly; `approximation = 1` (and no params) is a coarse fast
//! approximation that is neither form (0.0 at x = -3). No variant computes
//! `gelu_pytorch_tanh`, so the engine composes it from `mult`, `add` and
//! `sigmoid` nodes (`x * sigmoid(1.5957691 * (x + 0.044715 * x^3))`, an
//! exact identity), which is what the broadcast probe here underwrites.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-gelu-test`

use reng_synapse::{NodeInput, bf16_to_f32, f32_to_bf16, run_node, run_node_extra};

/// erf by its Taylor series in f64 (converges for the |x| <= 4 used here).
fn erf(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..120 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

fn gelu_erf(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / 2f64.sqrt()))
}

fn gelu_tanh(x: f64) -> f64 {
    let u = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

fn bf16(v: f64) -> f32 {
    bf16_to_f32(f32_to_bf16(v as f32))
}

/// Count elements of `got` equal to the bf16 rounding of `f(x)`.
fn matches(x: &[f32], got: &[f32], f: fn(f64) -> f64) -> usize {
    x.iter()
        .zip(got)
        .filter(|&(&xi, &g)| g == bf16(f(f64::from(xi))))
        .count()
}

fn main() -> reng_core::Result<()> {
    let (rows, cols) = (256usize, 256usize);
    let sizes = [cols as u64, rows as u64];
    let pattern = [
        -4.0f32, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
        -3.5,
    ];
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| pattern[(i * 7 + i / cols) % pattern.len()])
        .collect();
    let mut verdicts = Vec::new();

    // Probe A: gelu_fwd_bf16 with each `ns_GeluKernel` params variant
    // (`Params { bool approximation }`, `ParamsV2` adds an accuracy mode):
    // which form each computes.
    let n = x.len();
    let variants: [(&str, &[u8], bool); 8] = [
        ("no params", &[], false),
        ("approximation=1 (1 byte)", &[1], false),
        ("approximation=0 (1 byte)", &[0], false),
        (
            "V2 approximation=1, default accuracy (8 bytes)",
            &[1, 0, 0, 0, 0, 0, 0, 0],
            false,
        ),
        (
            "V2 approximation=0, default accuracy (8 bytes)",
            &[0, 0, 0, 0, 0, 0, 0, 0],
            false,
        ),
        ("no params, 2 outputs", &[], true),
        ("approximation=1 (1 byte), 2 outputs", &[1], true),
        ("approximation=0 (1 byte), 2 outputs", &[0], true),
    ];
    let mut tanh_variant = None;
    let mut erf_variant = None;
    for (label, params, two_outs) in variants {
        let (pp, ps) = if params.is_empty() {
            (core::ptr::null(), 0u32)
        } else {
            (params.as_ptr().cast(), params.len() as u32)
        };
        let input = [NodeInput {
            name: "X",
            sizes: &sizes,
            data: &x,
            raw: None,
        }];
        let run = if two_outs {
            run_node_extra("gelu_fwd_bf16", &input, &sizes, &[&sizes], pp, ps)
        } else {
            run_node("gelu_fwd_bf16", &input, &sizes, pp, ps)
        };
        let got = match run {
            Ok(v) => v,
            Err(e) => {
                let m = e.to_string();
                println!(
                    "gelu_fwd_bf16 {label}: rejected ({})",
                    &m[..m.len().min(80)]
                );
                continue;
            }
        };
        let m_erf = matches(&x, &got, gelu_erf);
        let m_tanh = matches(&x, &got, gelu_tanh);
        let form = if m_tanh == n {
            "tanh"
        } else if m_erf == n {
            "erf"
        } else if m_tanh > m_erf && m_tanh * 100 >= n * 99 {
            "tanh (within rounding)"
        } else if m_erf > m_tanh && m_erf * 100 >= n * 99 {
            "erf (within rounding)"
        } else {
            "neither"
        };
        println!(
            "gelu_fwd_bf16 {label}: {m_erf}/{n} match the erf form, {m_tanh}/{n} the tanh form: {form}"
        );
        for xv in [-4.0f32, -3.0, -2.5, -1.0, 1.0, 3.0] {
            let i = x.iter().position(|&v| v == xv).unwrap();
            println!(
                "  x = {xv:+.2}: device {:+.7}  erf {:+.7} (bf16 {:+.7})  tanh {:+.7} (bf16 {:+.7})",
                got[i],
                gelu_erf(f64::from(xv)),
                bf16(gelu_erf(f64::from(xv))),
                gelu_tanh(f64::from(xv)),
                bf16(gelu_tanh(f64::from(xv)))
            );
        }
        if form.starts_with("tanh") && tanh_variant.is_none() {
            tanh_variant = Some(label);
        }
        if form.starts_with("erf") && erf_variant.is_none() {
            erf_variant = Some(label);
        }
    }
    println!(
        "gelu_fwd_bf16: tanh form from {}; erf form from {}",
        tanh_variant.unwrap_or("no variant (the engine composes it)"),
        erf_variant.unwrap_or("no variant")
    );
    verdicts.push(tanh_variant.is_some() || erf_variant.is_some());

    // Probe B: mult_fwd_bf16 with a [1, 1] constant broadcast over [cols, rows].
    let c = [1.5f32];
    let want: Vec<f32> = x.iter().map(|v| bf16(f64::from(v * 1.5))).collect();
    match run_node(
        "mult_fwd_bf16",
        &[
            NodeInput {
                name: "X",
                sizes: &sizes,
                data: &x,
                raw: None,
            },
            NodeInput {
                name: "C",
                sizes: &[1, 1],
                data: &c,
                raw: None,
            },
        ],
        &sizes,
        core::ptr::null(),
        0,
    ) {
        Ok(got) => {
            let ok = got.iter().zip(&want).filter(|(a, b)| a == b).count();
            println!("mult_fwd_bf16 [{cols}, {rows}] x [1, 1]: {ok}/{n} exact");
            verdicts.push(ok == n);
        }
        Err(e) => {
            println!("mult_fwd_bf16 [{cols}, {rows}] x [1, 1]: rejected ({e})");
            verdicts.push(false);
        }
    }

    // Probe C: tanh_fwd_bf16 on a 4-D tensor.
    let sizes4 = [64u64, 16, 4, 16];
    let x4: Vec<f32> = (0..65536)
        .map(|i| ((i * 13 + 5) % 41) as f32 / 8.0 - 2.5)
        .collect();
    match run_node(
        "tanh_fwd_bf16",
        &[NodeInput {
            name: "X",
            sizes: &sizes4,
            data: &x4,
            raw: None,
        }],
        &sizes4,
        core::ptr::null(),
        0,
    ) {
        Ok(got) => {
            let close = got
                .iter()
                .zip(&x4)
                .filter(|&(&g, &v)| (g - v.tanh()).abs() <= 0.01 + v.tanh().abs() / 128.0)
                .count();
            println!("tanh_fwd_bf16 [64, 16, 4, 16]: {close}/65536 within a bf16 ulp");
            verdicts.push(close == 65536);
        }
        Err(e) => {
            println!("tanh_fwd_bf16 [64, 16, 4, 16]: rejected ({e})");
            verdicts.push(false);
        }
    }

    if verdicts.iter().all(|&v| v) {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "probe verdicts (gelu, broadcast mult, 4-D tanh): {verdicts:?}"
        )))
    }
}
