//! The device probes that decide how a quantized weight reaches the MME.
//!
//! `reng-fp8-probe [check] [time] [--format e4m3|e5m2] [--iters <n>]
//! [--min-speedup <x>]`
//!
//! `check` (the default) runs the correctness probes: the device's reading
//! of every fp8 code against [`reng_fp8::decode`], the device's clipping
//! cast against [`reng_fp8::encode`], the weight-only pair the engine
//! would like (expected to be refused), the plain `gemm` over two fp8
//! operands whose scale is the hardware exponent bias, and the
//! `fp8_gemm_bf16` complex guid in both its operand forms (`[A, B]` and
//! `[A, B, scaleA, scaleB]` with a per-output-channel `scaleB`), each at
//! one and at 256 rows. `time` measures the fp8 and bf16 `gemm` at the
//! decode shapes `K = 4096`, `N = 4096` and `N = 14336`, and checks the
//! `N = 14336`, `m = 1` ratio against `--min-speedup` (default 1.2x).
//!
//! Every check prints one line ending in PASS or FAIL and the binary exits
//! non-zero if any of them failed. It needs a card
//! (`RENG_MODULE_ID=<module>`).

use reng_core::{Error, Result};
use reng_fp8::{Fp8Format, Quantized, Scaling, bf16_to_f32, f32_to_bf16, quantize};
use reng_synapse::fp8::{
    Fp8Operand, bench_gemm_bf16, bench_gemm_fp8, cast_round_trip, decode_on_device, gemm_fp8,
    gemm_fp8_scaled, gemm_mixed,
};

/// The floor the `N = 14336`, `m = 1` fp8-over-bf16 ratio has to clear.
/// `fp8-research.md` 4.3 measured 1.6x to 2.1x there; a form that lost the
/// halved weight traffic would land near 1.0x.
const DEFAULT_SPEEDUP_FLOOR: f64 = 1.2;

/// A deterministic Llama-like weight row: a narrow normal-ish spread with
/// a few large channels, in bf16.
fn weights(rows: usize, cols: usize) -> Vec<u16> {
    (0..rows * cols)
        .map(|i| {
            let x = (((i * 2_654_435_761_usize) % 65_536) as f32 / 65_536.0 - 0.5) * 0.08;
            let outlier = if i % 977 == 0 { 12.0 } else { 1.0 };
            f32_to_bf16(x * outlier)
        })
        .collect()
}

/// Deterministic activations in the range post-norm activations live in.
fn activations(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7919) % 2003) as f32 - 1000.0) / 700.0)
        .collect()
}

/// `C[N, m] = A[K, m]^T-free gemm against B[K, N] transposed`, in f32 over
/// the dequantized operands: the reference every gemm probe is read
/// against.
fn gemm_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * m];
    for row in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[row * k + t] * b[j * k + t];
            }
            c[row * n + j] = acc;
        }
    }
    c
}

/// Relative L2 of `got` against `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| f64::from(g - w).powi(2))
        .sum();
    let den: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

struct Checks {
    ran: usize,
    failed: usize,
}

impl Checks {
    fn line(&mut self, name: &str, detail: &str, ok: bool) {
        self.ran += 1;
        if !ok {
            self.failed += 1;
        }
        println!("{name:<28} {detail} {}", if ok { "PASS" } else { "FAIL" });
    }
}

/// The device's reading of every one of the 256 codes.
fn check_decode(c: &mut Checks, fmt: Fp8Format) {
    let codes: Vec<u8> = (0..256).map(|i| i as u8).collect();
    // A 256-wide vector; the device wants at least a plausible shape.
    match decode_on_device(&codes, &[256, 1], fmt) {
        Ok(got) => {
            let want: Vec<f32> = codes.iter().map(|&x| reng_fp8::decode(x, fmt)).collect();
            let mut bad = 0;
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                // Infinities and NaN are the reserved codes; compare the
                // finite ones exactly (bf16 holds every fp8 value).
                if w.is_finite() && (g - w).abs() > 0.0 {
                    bad += 1;
                    if bad == 1 {
                        println!("  first mismatch at code {i:#04x}: device {g}, host {w}");
                    }
                }
            }
            c.line(
                &format!("decode-{}", fmt.name()),
                &format!("{} of 256 codes differ:", bad),
                bad == 0,
            );
        }
        Err(e) => c.line(&format!("decode-{}", fmt.name()), &format!("{e}:"), false),
    }
}

/// The device's clipping cast against the host encoder.
fn check_cast(c: &mut Checks, fmt: Fp8Format) {
    let mut x: Vec<f32> = activations(256);
    // Values that exercise the saturation and the subnormal range.
    x[0] = 250.0;
    x[1] = -300.0;
    x[2] = 0.001;
    x[3] = 1.375;
    x[4] = -3.3;
    match cast_round_trip(&x, &[256, 1], fmt) {
        Ok(got) => {
            let want: Vec<f32> = x
                .iter()
                .map(|&v| {
                    // The graph rounds to bf16 on the way in, as the
                    // engine's uploads do.
                    reng_fp8::decode(reng_fp8::encode(bf16_to_f32(f32_to_bf16(v)), fmt), fmt)
                })
                .collect();
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| (*g - *w).abs() > 0.0)
                .count();
            c.line(
                &format!("cast-clip-{}", fmt.name()),
                &format!("{bad} of {} values differ:", x.len()),
                bad == 0,
            );
        }
        Err(e) => c.line(
            &format!("cast-clip-{}", fmt.name()),
            &format!("{e}:"),
            false,
        ),
    }
}

/// The pair the engine would like: bf16 activation, fp8 weight. The
/// research says the operand validator refuses it; the check records what
/// this stack does rather than assuming.
fn check_mixed(c: &mut Checks, q: &Quantized, m: usize, k: usize, n: usize) {
    let a = activations(m * k);
    let b = Fp8Operand {
        name: "W",
        sizes: &[k as u64, n as u64],
        codes: &q.codes,
        format: q.format,
        exp_bias: q.exp_bias,
    };
    let out = [n as u64, m as u64];
    match gemm_mixed(&a, &[k as u64, m as u64], &b, &out, true) {
        Ok(_) => c.line(
            "gemm-mixed-bf16-fp8",
            "the stack accepted a mixed operand pair, which changes the plan:",
            false,
        ),
        Err(e) => {
            println!("  refused as expected: {e}");
            c.line("gemm-mixed-bf16-fp8", "refused:", true);
        }
    }
}

/// What the device reads out of an fp8 operand: the code decoded at the
/// format's own bias, times `2^(default - exp_bias)`, which is the factor
/// the MME descriptor patcher applies for a tensor whose
/// `SYN_FP_QUANT_METADATA` carries that bias (`fp8-research.md` 4.2: bias
/// 11 gives `out/ref 2^-4`, bias 3 gives `2^4`, bias 15 gives `2^-8`).
/// Nothing else of the quantization reaches the plain `gemm`: the `scale`
/// field and per-channel metadata are accepted by the API and ignored by
/// this path (probes `gemm-scale0.5-B`, `gemm-pcs-B`).
///
/// # Panics
///
/// Panics on a [`Scaling::PerChannel`] matrix, whose scales the plain
/// `gemm` cannot see: comparing such a weight's `dequantize()` against a
/// plain-gemm result would fail by the row scales (a factor of thousands
/// on a Llama-shaped matrix), which is a bug in the probe and not a
/// finding about the card.
fn device_values(q: &Quantized) -> Vec<f32> {
    assert_ne!(
        q.scaling,
        Scaling::PerChannel,
        "the plain gemm does not see per-channel scales; \
         use the complex guid's scaleB operand for those"
    );
    let factor = 2.0f32.powi(q.format.exponent_bias() as i32 - q.exp_bias as i32);
    q.codes
        .iter()
        .map(|&c| reng_fp8::decode(c, q.format) * factor)
        .collect()
}

/// The plain `gemm` over two fp8 operands, against a CPU product of the
/// values the device reads.
///
/// The weight is the [`Scaling::HwExpBias`] one, because the exponent
/// bias is the only scale mechanism this form has: the plain `gemm`
/// ignores the `scale` field and any per-channel metadata
/// (`fp8-research.md` 4.2, probes `gemm-scale0.5-B` and `gemm-pcs-B`, both
/// `out/ref 1.0`) and honours only `SYN_FP_QUANT_METADATA`'s `expBias`
/// (`gemm-expbias11-B`, `out/ref 2^-4`). So the reference is
/// `Quantized::dequantize`, whose scale is exactly the `2^(7 - bias)` the
/// MME descriptor patcher applies, and the check is discriminating: on
/// this weight the scheme picks bias 15, so a stack that dropped the
/// metadata would be off by 2^8 and `rel_L2` would be about 255.
///
/// Passing the [`Scaling::PerChannel`] weight here instead would compare
/// an unscaled device product against a scaled CPU reference and fail by
/// the ratio of the row scales; the per-channel scales belong to
/// [`check_gemm_scaled`]'s complex guid.
fn check_gemm(c: &mut Checks, q: &Quantized, m: usize, k: usize, n: usize, fmt: Fp8Format) {
    assert_eq!(
        q.scaling,
        Scaling::HwExpBias,
        "the plain gemm carries its scale as the exponent bias"
    );
    let a_f32 = activations(m * k);
    // The activation is cast on the host at the format's own bias, which
    // is what an in-recipe `cast_bf16_to_hf8` would do.
    let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16(x)).collect();
    let a_q = quantize(&a_bf16, m, k, fmt, Scaling::Unit);
    let a_deq = device_values(&a_q);
    let b_deq = device_values(q);
    let want = gemm_cpu(&a_deq, &b_deq, m, k, n);
    let a = Fp8Operand {
        name: "A",
        sizes: &[k as u64, m as u64],
        codes: &a_q.codes,
        format: fmt,
        exp_bias: a_q.exp_bias,
    };
    let b = Fp8Operand {
        name: "W",
        sizes: &[k as u64, n as u64],
        codes: &q.codes,
        format: fmt,
        exp_bias: q.exp_bias,
    };
    match gemm_fp8(&a, &b, &[n as u64, m as u64], true) {
        Ok(got) => {
            let rel = rel_l2(&got, &want);
            c.line(
                &format!("gemm-fp8-m{m}"),
                &format!(
                    "rel_L2 {rel:.4} at weight exp_bias {} against the dequantized product:",
                    q.exp_bias
                ),
                rel < 0.02,
            );
        }
        Err(e) => c.line(&format!("gemm-fp8-m{m}"), &format!("{e}:"), false),
    }
}

/// The complex guid, in both operand forms the research validated
/// (`fp8-research.md` 7.1b): `[A, B]` and `[A, B, scaleA, scaleB]`. Three
/// inputs is not a form - a lone `scaleB` would land in the `scaleA` slot -
/// so `gemm_fp8_scaled` sends a unit `scaleA` with it, and the
/// per-channel run below is what would catch a regression: the `[N]`
/// vector applied to A instead of B is either a compile failure (`[512]`
/// against A's `[256, 1]`) or a wrong product.
///
/// The per-channel scales are what makes an arbitrary (non power-of-16)
/// weight scale possible, which is this form's whole reason to exist.
fn check_gemm_scaled(c: &mut Checks, q: &Quantized, m: usize, k: usize, n: usize, fmt: Fp8Format) {
    assert_eq!(
        q.scaling,
        Scaling::PerChannel,
        "the complex guid carries its scale as the scaleB operand"
    );
    let a_f32 = activations(m * k);
    let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16(x)).collect();
    let a_q = quantize(&a_bf16, m, k, fmt, Scaling::Unit);
    let a_deq = device_values(&a_q);
    let a = Fp8Operand {
        name: "A",
        sizes: &[k as u64, m as u64],
        codes: &a_q.codes,
        format: fmt,
        exp_bias: a_q.exp_bias,
    };
    // The codes without their scales; the guid multiplies them in.
    let b = Fp8Operand {
        name: "W",
        sizes: &[k as u64, n as u64],
        codes: &q.codes,
        format: fmt,
        exp_bias: fmt.exponent_bias(),
    };
    let out = [n as u64, m as u64];

    // Two inputs: no scale operand at all, so the product is the one over
    // the codes as the device reads them (bias 7 here), the weight's own
    // per-channel scales left out.
    let raw: Vec<f32> = q.codes.iter().map(|&x| reng_fp8::decode(x, fmt)).collect();
    let want_raw = gemm_cpu(&a_deq, &raw, m, k, n);
    match gemm_fp8_scaled(&a, &b, None, None, &out, true) {
        Ok(got) => {
            let rel = rel_l2(&got, &want_raw);
            c.line(
                &format!("fp8_gemm_bf16-2in-m{m}"),
                &format!("rel_L2 {rel:.4} against the unscaled product:"),
                rel < 0.02,
            );
        }
        Err(e) => c.line(&format!("fp8_gemm_bf16-2in-m{m}"), &format!("{e}:"), false),
    }

    // Four inputs: a unit scaleA and the quantizer's `[N]` scaleB, which
    // must reproduce the dequantized product column by column.
    let b_deq = q.dequantize();
    let want = gemm_cpu(&a_deq, &b_deq, m, k, n);
    let scales = q.scale_operand().unwrap_or(&[]);
    match gemm_fp8_scaled(&a, &b, Some(&[1.0]), Some(scales), &out, true) {
        Ok(got) => {
            let rel = rel_l2(&got, &want);
            c.line(
                &format!("fp8_gemm_bf16-pcs-m{m}"),
                &format!("rel_L2 {rel:.4} against the dequantized product:"),
                rel < 0.02,
            );
        }
        Err(e) => c.line(&format!("fp8_gemm_bf16-pcs-m{m}"), &format!("{e}:"), false),
    }
}

/// The fp8 and bf16 `gemm` at the decode shapes. The ratio at the widest
/// shape is a check, not context: the point of fp8 is the halved weight
/// traffic, and a recipe-cache or driver change that turned the E4M3 gemm
/// back into a bf16-speed form would otherwise leave every line PASS.
/// `fp8-research.md` 4.3 measured 0.064-0.073 ms bf16 against 0.034-0.040
/// ms E4M3 at `K = 4096, N = 14336, m = 1`, so 1.6x to 2.1x; the floor is
/// well under that because `time_launches` adds the same fixed launch
/// overhead to both rows, which compresses the ratio toward 1.0.
fn timings(c: &mut Checks, iters: usize, fmt: Fp8Format, floor: f64) -> Result<()> {
    println!(
        "{:<8} {:>6} {:>8} {:>10} {:>10} {:>8}",
        "form", "m", "K x N", "ms/launch", "weight MB", "GB/s"
    );
    for &(k, n) in &[(4096usize, 4096usize), (4096, 14336)] {
        let w = weights(n, k);
        let q = quantize(&w, n, k, fmt, Scaling::PerChannel);
        for &m in &[1usize, 256] {
            let a_bf16: Vec<u16> = activations(m * k).iter().map(|&x| f32_to_bf16(x)).collect();
            let a_q = quantize(&a_bf16, m, k, fmt, Scaling::Unit);
            let a = Fp8Operand {
                name: "A",
                sizes: &[k as u64, m as u64],
                codes: &a_q.codes,
                format: fmt,
                exp_bias: a_q.exp_bias,
            };
            let b = Fp8Operand {
                name: "W",
                sizes: &[k as u64, n as u64],
                codes: &q.codes,
                format: fmt,
                exp_bias: q.exp_bias,
            };
            let out = [n as u64, m as u64];
            let t8 = bench_gemm_fp8(&a, &b, &out, true, iters)?;
            let t16 = bench_gemm_bf16(
                &a_bf16,
                &[k as u64, m as u64],
                &w,
                &[k as u64, n as u64],
                &out,
                true,
                iters,
            )?;
            let mb8 = (k * n) as f64 / 1e6;
            let mb16 = 2.0 * mb8;
            println!(
                "{:<8} {m:>6} {:>8} {:>10.4} {:>10.0} {:>8.0}",
                fmt.name(),
                format!("{k}x{n}"),
                t8 * 1e3,
                mb8,
                mb8 / 1e3 / t8
            );
            println!(
                "{:<8} {m:>6} {:>8} {:>10.4} {:>10.0} {:>8.0}",
                "bf16",
                format!("{k}x{n}"),
                t16 * 1e3,
                mb16,
                mb16 / 1e3 / t16
            );
            println!("  speedup bf16 -> {}: {:.2}x", fmt.name(), t16 / t8);
            // The widest shape at batch 1 is the decode case the design
            // rests on; the others are recorded but not gated (at
            // 4096 x 4096 the launch overhead is most of the time).
            if (k, n, m) == (4096, 14336, 1) {
                let ratio = t16 / t8;
                c.line(
                    &format!("speedup-{k}x{n}-m{m}"),
                    &format!(
                        "{ratio:.2}x bf16 -> {} against a {floor:.2}x floor:",
                        fmt.name()
                    ),
                    ratio >= floor,
                );
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut fmt = Fp8Format::E4M3;
    let mut iters = 20usize;
    let mut floor = DEFAULT_SPEEDUP_FLOOR;
    let mut do_check = !argv.iter().any(|a| a == "time");
    let do_time = argv.iter().any(|a| a == "time");
    if argv.iter().any(|a| a == "check") {
        do_check = true;
    }
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--format" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| Error::Other("--format needs a value".into()))?;
                fmt = Fp8Format::from_name(v)
                    .ok_or_else(|| Error::Other(format!("unknown format {v}")))?;
                i += 2;
            }
            "--iters" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| Error::Other("--iters needs a value".into()))?;
                iters = v
                    .parse()
                    .map_err(|e| Error::Other(format!("--iters: {e}")))?;
                i += 2;
            }
            "--min-speedup" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| Error::Other("--min-speedup needs a value".into()))?;
                floor = v
                    .parse()
                    .map_err(|e| Error::Other(format!("--min-speedup: {e}")))?;
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut c = Checks { ran: 0, failed: 0 };
    if do_check {
        let (k, n) = (256usize, 512usize);
        let w = weights(n, k);
        // Two quantizations of the same weight, one per gemm form: the
        // plain `gemm` can only carry a power-of-16 scale as the exponent
        // bias, the complex guid takes the per-channel vector.
        let q = quantize(&w, n, k, fmt, Scaling::PerChannel);
        let q_hw = quantize(&w, n, k, fmt, Scaling::HwExpBias);
        println!(
            "weight [{n}, {k}] {}: {} exp_bias {}, {} exp_bias {} (scale {:.3e})",
            fmt.name(),
            q.scaling.name(),
            q.exp_bias,
            q_hw.scaling.name(),
            q_hw.exp_bias,
            q_hw.scales[0],
        );
        check_decode(&mut c, fmt);
        check_cast(&mut c, fmt);
        check_mixed(&mut c, &q, 1, k, n);
        for m in [1usize, 256] {
            check_gemm(&mut c, &q_hw, m, k, n, fmt);
            // The complex guid carries the per-output-channel scales, which
            // is an E4M3 question: E5M2 is here for the plain gemm and the
            // casts only, and whether `fp8_gemm_bf16` takes E5M2 operands
            // has no bearing on the weight path.
            if fmt == Fp8Format::E4M3 {
                check_gemm_scaled(&mut c, &q, m, k, n, fmt);
            }
        }
    }
    if do_time {
        timings(&mut c, iters, fmt, floor)?;
    }
    if c.failed > 0 {
        return Err(Error::Other(format!(
            "{} of {} checks failed",
            c.failed, c.ran
        )));
    }
    if c.ran == 0 {
        return Err(Error::Other(
            "nothing to do: pass `check`, `time` or both".into(),
        ));
    }
    println!("reng-fp8-probe: {} checks passed", c.ran);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe's own shapes.
    const K: usize = 256;
    const N: usize = 512;

    /// For a weight whose scale is the hardware exponent bias, the CPU
    /// reference `check_gemm` uses is exactly `Quantized::dequantize`,
    /// because the bias factor is that matrix's scale. This is what makes
    /// the check a check: the bias is not the format default here, so a
    /// device that dropped `SYN_FP_QUANT_METADATA` would be off by 2^8.
    #[test]
    fn the_plain_gemm_reference_is_the_dequantized_weight() {
        let w = weights(N, K);
        let q = quantize(&w, N, K, Fp8Format::E4M3, Scaling::HwExpBias);
        assert_ne!(
            q.exp_bias,
            Fp8Format::E4M3.exponent_bias(),
            "this weight has to pick a non-default bias or the check proves nothing"
        );
        assert_eq!(device_values(&q), q.dequantize());
    }

    /// A unit-scaled activation is read as its codes decode, whatever the
    /// row count.
    #[test]
    fn a_unit_scaled_operand_is_read_as_its_codes() {
        let a: Vec<u16> = activations(K).iter().map(|&x| f32_to_bf16(x)).collect();
        let q = quantize(&a, 1, K, Fp8Format::E4M3, Scaling::Unit);
        let want: Vec<f32> = q
            .codes
            .iter()
            .map(|&c| reng_fp8::decode(c, Fp8Format::E4M3))
            .collect();
        assert_eq!(device_values(&q), want);
    }

    /// The per-channel scales are invisible to the plain `gemm`, so
    /// building its reference out of a per-channel matrix is a probe bug
    /// rather than a device finding: it is refused outright.
    #[test]
    #[should_panic(expected = "does not see per-channel scales")]
    fn the_plain_gemm_reference_refuses_a_per_channel_weight() {
        let w = weights(N, K);
        let q = quantize(&w, N, K, Fp8Format::E4M3, Scaling::PerChannel);
        let _ = device_values(&q);
    }

    /// How far wrong that would be, on this probe's own weight: the row
    /// scales run to 1e-4, so a per-channel reference against the plain
    /// gemm misses by three orders of magnitude, nowhere near the 0.02
    /// acceptance.
    #[test]
    fn a_per_channel_reference_would_miss_the_plain_gemm_by_thousands() {
        let w = weights(N, K);
        let q = quantize(&w, N, K, Fp8Format::E4M3, Scaling::PerChannel);
        let a: Vec<u16> = activations(K).iter().map(|&x| f32_to_bf16(x)).collect();
        let a_q = quantize(&a, 1, K, Fp8Format::E4M3, Scaling::Unit);
        let a_deq = device_values(&a_q);
        let read: Vec<f32> = q
            .codes
            .iter()
            .map(|&c| reng_fp8::decode(c, Fp8Format::E4M3))
            .collect();
        let device = gemm_cpu(&a_deq, &read, 1, K, N);
        let scaled = gemm_cpu(&a_deq, &q.dequantize(), 1, K, N);
        let rel = rel_l2(&device, &scaled);
        assert!(rel > 100.0, "rel_L2 {rel}");
    }
}
