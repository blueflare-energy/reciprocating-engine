//! Host-side FP8 weight quantization: a bf16 `[out, in]` checkpoint matrix
//! becomes one byte per element plus a per-output-channel f32 scale.
//!
//! Two formats, both of which the Gaudi2 MME multiplies into a bf16 output:
//! [`Fp8Format::E4M3`] (the vendor's `hf8`, `syn_type_fp8_143`, exponent
//! bias 7, largest finite 240 because the top exponent is reserved for
//! infinity and NaN) and [`Fp8Format::E5M2`] (`f8`, `syn_type_fp8_152`,
//! exponent bias 15, largest finite 57344). E4M3 is the inference format;
//! E5M2 keeps two mantissa bits and is here for completeness. The encoder
//! rounds to nearest even and saturates to the largest finite value, which
//! is what the device cast kernel does with
//! `ns_CastKernel::ParamsV3 { CAST_ROUND_HALF_NE, 0, CAST_CLIP }`; the
//! device's default cast returns infinity above the range instead, so the
//! recipe must ask for the clipping form to agree with this encoder.
//!
//! Three scale schemes ([`Scaling`]), all of which leave the codes'
//! meaning as `weight = decode(code) * scale[channel]`:
//!
//! - [`Scaling::PerChannel`]: absmax of each output channel, an arbitrary
//!   f32. The device applies it as the `scaleB` input of the `fp8_gemm_bf16`
//!   complex guid, which takes a `[N]` or `[N, 1]` vector.
//! - [`Scaling::HwExpBias`]: one power-of-16 factor for the whole matrix,
//!   expressed as an exponent bias out of `{3, 7, 11, 15}` (the four values
//!   Gaudi2 accepts for E4M3; E5M2 has only 15). The device applies it for
//!   free by patching the MME descriptor from the tensor's
//!   `SYN_FP_QUANT_METADATA`, so this scheme costs no node and no time.
//! - [`Scaling::Unit`]: no scaling, for measuring what the scales buy.
//!
//! The crate has no dependencies: the quantizer is plain numerics and is
//! built and tested without the vendor stack, so the graph crate can depend
//! on it rather than the other way round.

/// Convert bf16 bits to `f32` (exact: bf16 is the top half of an f32).
#[must_use]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

/// Convert an `f32` to bf16 bits, rounding to nearest even.
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0; // NaN
    }
    let round_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round_bias) >> 16) as u16
}

/// An 8-bit floating-point format. The names follow the sign / exponent /
/// mantissa bit counts, as the vendor's `synDataType` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1-4-3, exponent bias 7, largest finite 240 (the vendor's `hf8`).
    E4M3,
    /// 1-5-2, exponent bias 15, largest finite 57344 (the vendor's `f8`).
    E5M2,
}

impl Fp8Format {
    /// Mantissa bits.
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::E4M3 => 3,
            Self::E5M2 => 2,
        }
    }

    /// Exponent bits.
    #[must_use]
    pub const fn exponent_bits(self) -> u32 {
        8 - 1 - self.mantissa_bits()
    }

    /// The format's own exponent bias, the one the cast kernel encodes at
    /// and the default of the tensor metadata.
    #[must_use]
    pub const fn exponent_bias(self) -> u32 {
        match self {
            Self::E4M3 => 7,
            Self::E5M2 => 15,
        }
    }

    /// The code of the largest finite magnitude: `0x77` (240) for E4M3,
    /// `0x7b` (57344) for E5M2. The top exponent field is reserved for
    /// infinity and NaN in both.
    #[must_use]
    pub const fn max_finite_code(self) -> u8 {
        let mbits = self.mantissa_bits();
        let exp_field = ((1u32 << self.exponent_bits()) - 2) << mbits;
        (exp_field | ((1 << mbits) - 1)) as u8
    }

    /// The largest finite magnitude: 240 (E4M3) or 57344 (E5M2).
    #[must_use]
    pub fn max_finite(self) -> f32 {
        decode(self.max_finite_code(), self)
    }

    /// A NaN code: the top exponent field with a non-zero mantissa.
    #[must_use]
    pub const fn nan_code(self) -> u8 {
        0x7f
    }

    /// The `synDataType` of the format: `syn_type_fp8_143` (8192) for E4M3,
    /// `syn_type_fp8_152` (16384) for E5M2.
    #[must_use]
    pub const fn syn_data_type(self) -> i32 {
        match self {
            Self::E4M3 => 1 << 13,
            Self::E5M2 => 1 << 14,
        }
    }

    /// The exponent biases Gaudi2 accepts in `SYN_FP_QUANT_METADATA` for
    /// this format, smallest range first. E4M3 has four (the API rejects
    /// any other value); E5M2 has only its default.
    #[must_use]
    pub const fn hw_exponent_biases(self) -> &'static [u32] {
        match self {
            Self::E4M3 => &[15, 11, 7, 3],
            Self::E5M2 => &[15],
        }
    }

    /// The short name used by the switch and by the report lines.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
        }
    }

    /// The format of a switch value (`e4m3`, `e5m2`, case-insensitive).
    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "e4m3" | "fp8_143" | "hf8" => Some(Self::E4M3),
            "e5m2" | "fp8_152" | "f8" => Some(Self::E5M2),
            _ => None,
        }
    }
}

/// Encode one value, rounding to nearest even and saturating to the
/// largest finite magnitude (infinities and out-of-range values included;
/// NaN stays NaN). This is the device cast kernel's `CAST_CLIP` behaviour,
/// bit for bit.
#[must_use]
pub fn encode(x: f32, fmt: Fp8Format) -> u8 {
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_nan() {
        return sign | fmt.nan_code();
    }
    let bits = x.to_bits() & 0x7fff_ffff;
    // Zero, and an f32 subnormal, which is far below the smallest fp8
    // subnormal (2^-9 for E4M3, 2^-16 for E5M2) and rounds to zero.
    if bits >> 23 == 0 {
        return sign;
    }
    let a = f32::from_bits(bits);
    let mbits = fmt.mantissa_bits() as i32;
    let bias = fmt.exponent_bias() as i32;
    // Exponents of the smallest and the largest normal of the format
    // (-6 and 7 for E4M3, -14 and 15 for E5M2); subnormals share the
    // smallest normal's quantum.
    let emin = 1 - bias;
    let emax = ((1i32 << fmt.exponent_bits()) - 2) - bias;
    let e = ((bits >> 23) as i32) - 127;
    if e > emax {
        // Everything above the top binade, infinity included.
        return sign | fmt.max_finite_code();
    }
    let quantum_exp = e.max(emin) - mbits;
    // Exact: the quotient of a value by the power of two one mantissa
    // width below its own exponent is a small integer plus a fraction.
    let mut n = (a / pow2(quantum_exp)).round_ties_even() as u32;
    let mut e_eff = e.max(emin);
    // A mantissa that rounded up to the next power of two carries into the
    // exponent (only possible for a normal value).
    if n >= 2 << mbits {
        n >>= 1;
        e_eff += 1;
    }
    if e_eff > emax {
        return sign | fmt.max_finite_code();
    }
    let code = if e < emin {
        // Subnormal: the exponent field stays zero and `n` is the
        // mantissa. A value that rounded up to `1 << mbits` becomes the
        // smallest normal, which the same bit pattern spells.
        n
    } else {
        (((e_eff + bias) as u32) << mbits) | (n - (1 << mbits))
    };
    sign | code as u8
}

/// Decode one code to `f32`: the finite values exactly, the reserved top
/// exponent field as an infinity (zero mantissa) or a NaN.
#[must_use]
pub fn decode(code: u8, fmt: Fp8Format) -> f32 {
    let mbits = fmt.mantissa_bits();
    let bias = fmt.exponent_bias() as i32;
    let sign = if code & 0x80 == 0 { 1.0f32 } else { -1.0f32 };
    let mant = u32::from(code) & ((1 << mbits) - 1);
    let exp_field = (u32::from(code) & 0x7f) >> mbits;
    if exp_field == (1 << fmt.exponent_bits()) - 1 {
        return if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    let frac = mant as f32 / (1u32 << mbits) as f32;
    if exp_field == 0 {
        sign * frac * pow2(1 - bias)
    } else {
        sign * (1.0 + frac) * pow2(exp_field as i32 - bias)
    }
}

/// `2^e` as an exact `f32` for the exponents this crate uses
/// (`-149 ..= 127`).
fn pow2(e: i32) -> f32 {
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        // Subnormal powers of two, needed for the smallest fp8 quanta.
        f32::from_bits(1u32 << (e + 149))
    }
}

/// How a quantized matrix carries its scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// One absmax scale per output channel (row of the `[out, in]`
    /// matrix), an arbitrary f32. The device applies it as the `scaleB`
    /// input of `fp8_gemm_bf16`.
    PerChannel,
    /// One power-of-16 factor for the whole matrix, expressed as an
    /// exponent bias out of [`Fp8Format::hw_exponent_biases`] and applied
    /// by the MME descriptor for free.
    HwExpBias,
    /// No scaling: the codes are the encoding of the weights themselves.
    Unit,
}

impl Scaling {
    /// The scheme of a switch value (`pcs`, `hw`, `unit`).
    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pcs" | "per-channel" | "perchannel" => Some(Self::PerChannel),
            "hw" | "hw-exp-bias" | "expbias" => Some(Self::HwExpBias),
            "unit" | "none" => Some(Self::Unit),
            _ => None,
        }
    }

    /// The short name used by the switch and by the report lines.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PerChannel => "pcs",
            Self::HwExpBias => "hw",
            Self::Unit => "unit",
        }
    }
}

/// A quantized `[rows, cols]` weight matrix: one byte per element in the
/// checkpoint's own row-major order, one f32 scale per row, and the
/// exponent bias the device tensor's `SYN_FP_QUANT_METADATA` must carry.
///
/// The codes mean `weight[r, c] = decode(codes[r * cols + c]) * scales[r]`
/// whatever the scheme: a [`Scaling::HwExpBias`] matrix repeats one
/// power of two in every entry of `scales` (the device gets it from the
/// exponent bias instead of from a scale operand) and a [`Scaling::Unit`]
/// one repeats 1.0.
#[derive(Clone, Debug)]
pub struct Quantized {
    pub format: Fp8Format,
    pub scaling: Scaling,
    /// Output channels: rows of the checkpoint's `[out, in]` matrix.
    pub rows: usize,
    /// Input width.
    pub cols: usize,
    /// `rows * cols` codes, row-major.
    pub codes: Vec<u8>,
    /// One multiplier per row, `rows` entries.
    pub scales: Vec<f32>,
    /// The exponent bias of the device tensor's quantization metadata:
    /// the chosen one for [`Scaling::HwExpBias`], the format's own
    /// otherwise.
    pub exp_bias: u32,
}

impl Quantized {
    /// Bytes of codes (the device weight buffer's size).
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// The `[rows]` scale vector as the `fp8_gemm_bf16` `scaleB` operand
    /// wants it; `None` for a scheme whose scale rides on the exponent
    /// bias or is 1.
    #[must_use]
    pub fn scale_operand(&self) -> Option<&[f32]> {
        match self.scaling {
            Scaling::PerChannel => Some(&self.scales),
            Scaling::HwExpBias | Scaling::Unit => None,
        }
    }

    /// The matrix back in f32, `rows * cols` row-major.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for (r, row) in out.chunks_mut(self.cols).enumerate() {
            let s = self.scales[r];
            for (o, &c) in row.iter_mut().zip(&self.codes[r * self.cols..]) {
                *o = decode(c, self.format) * s;
            }
        }
        out
    }

    /// The matrix back in bf16 (one rounding per element), the form a
    /// dequantizing device path would produce.
    #[must_use]
    pub fn dequantize_bf16(&self) -> Vec<u16> {
        self.dequantize().iter().map(|&x| f32_to_bf16(x)).collect()
    }
}

/// Mean and maximum relative error of one output channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowError {
    /// Mean of `|w - dequant(w)| / |w|` over the row's non-zero weights.
    pub mean_rel: f32,
    /// Maximum of the same.
    pub max_rel: f32,
}

/// Per-row relative error of a quantized matrix against the bf16 weights
/// it came from. Zero weights are skipped (their relative error is not
/// defined); a row that is all zeros reports `0.0` for both.
///
/// # Panics
///
/// Panics if `w` does not hold `q.rows * q.cols` elements.
#[must_use]
pub fn row_errors(w: &[u16], q: &Quantized) -> Vec<RowError> {
    assert_eq!(w.len(), q.rows * q.cols, "weight length against the shape");
    (0..q.rows)
        .map(|r| {
            let base = r * q.cols;
            let s = q.scales[r];
            let (mut sum, mut max, mut n) = (0.0f64, 0.0f32, 0usize);
            for c in 0..q.cols {
                let want = bf16_to_f32(w[base + c]);
                if want == 0.0 {
                    continue;
                }
                let got = decode(q.codes[base + c], q.format) * s;
                let rel = ((got - want) / want).abs();
                sum += f64::from(rel);
                max = max.max(rel);
                n += 1;
            }
            RowError {
                mean_rel: if n == 0 { 0.0 } else { (sum / n as f64) as f32 },
                max_rel: max,
            }
        })
        .collect()
}

/// Per-row relative error over at most `max_rows` evenly spaced rows, the
/// cheap form of [`row_errors`] for a matrix too big to walk twice at load
/// time. Returns the sampled rows' errors, in row order.
///
/// # Panics
///
/// Panics if `w` does not hold `q.rows * q.cols` elements or `max_rows` is
/// zero.
#[must_use]
pub fn sample_row_errors(w: &[u16], q: &Quantized, max_rows: usize) -> Vec<RowError> {
    assert_eq!(w.len(), q.rows * q.cols, "weight length against the shape");
    assert!(max_rows > 0, "max_rows must be positive");
    let n = max_rows.min(q.rows);
    let step = q.rows.div_ceil(n);
    (0..q.rows)
        .step_by(step)
        .map(|r| {
            let base = r * q.cols;
            let s = q.scales[r];
            let (mut sum, mut max, mut count) = (0.0f64, 0.0f32, 0usize);
            for c in 0..q.cols {
                let want = bf16_to_f32(w[base + c]);
                if want == 0.0 {
                    continue;
                }
                let got = decode(q.codes[base + c], q.format) * s;
                let rel = ((got - want) / want).abs();
                sum += f64::from(rel);
                max = max.max(rel);
                count += 1;
            }
            RowError {
                mean_rel: if count == 0 {
                    0.0
                } else {
                    (sum / count as f64) as f32
                },
                max_rel: max,
            }
        })
        .collect()
}

/// The largest magnitude of each row of a bf16 `[rows, cols]` matrix.
///
/// # Panics
///
/// Panics if `w` does not hold `rows * cols` elements.
#[must_use]
pub fn row_absmax(w: &[u16], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "weight length against the shape");
    let mut out = vec![0.0f32; rows];
    absmax_range(w, cols, &mut out);
    out
}

/// Row absmax of a contiguous block of rows.
fn absmax_range(w: &[u16], cols: usize, out: &mut [f32]) {
    for (r, o) in out.iter_mut().enumerate() {
        let mut m = 0.0f32;
        for &b in &w[r * cols..(r + 1) * cols] {
            let v = bf16_to_f32(b).abs();
            if v > m {
                m = v;
            }
        }
        *o = m;
    }
}

/// The largest exponent bias of `fmt`'s hardware set whose range still
/// covers `absmax`: the value with the finest quantum that does not
/// saturate. The range at bias `b` is `max_finite * 2^(default - b)`
/// (0.9375, 15, 240 and 3840 for E4M3). A matrix larger than the widest
/// range gets the widest bias and saturates.
#[must_use]
pub fn choose_exp_bias(absmax: f32, fmt: Fp8Format) -> u32 {
    let biases = fmt.hw_exponent_biases();
    let default = fmt.exponent_bias() as i32;
    for &b in biases {
        if absmax <= fmt.max_finite() * pow2(default - b as i32) {
            return b;
        }
    }
    *biases.last().expect("a non-empty bias set")
}

/// The default backoff of the absmax scales: 1.0, so the largest weight of
/// a channel lands exactly on the format's largest finite value. The
/// vendor's own recipe uses 0.5 for weights (and 0.25 for activations),
/// which trades one bit of weight resolution for headroom the engine does
/// not need on a static weight; [`quantize_with`] takes the factor.
pub const DEFAULT_BACKOFF: f32 = 1.0;

/// Quantize a bf16 `[rows, cols]` matrix (the checkpoint's own `[out, in]`
/// layout, so a row is one output channel) with the default backoff.
///
/// # Panics
///
/// Panics if `w` does not hold `rows * cols` elements or a dimension is
/// zero.
#[must_use]
pub fn quantize(
    w: &[u16],
    rows: usize,
    cols: usize,
    fmt: Fp8Format,
    scaling: Scaling,
) -> Quantized {
    quantize_with(w, rows, cols, fmt, scaling, DEFAULT_BACKOFF)
}

/// [`quantize`] with an explicit backoff: the scales stretch the channel's
/// absmax to `max_finite * backoff` instead of to `max_finite`.
///
/// The rows are quantized in parallel over up to eight threads for a
/// matrix big enough to pay for them (two passes over the weights: the
/// absmax of every row, then the encoding).
///
/// # Panics
///
/// Panics if `w` does not hold `rows * cols` elements, a dimension is
/// zero, or `backoff` is not positive and finite.
#[must_use]
pub fn quantize_with(
    w: &[u16],
    rows: usize,
    cols: usize,
    fmt: Fp8Format,
    scaling: Scaling,
    backoff: f32,
) -> Quantized {
    assert!(rows > 0 && cols > 0, "empty shape {rows} x {cols}");
    assert_eq!(w.len(), rows * cols, "weight length against the shape");
    assert!(
        backoff > 0.0 && backoff.is_finite(),
        "backoff {backoff} is not positive and finite"
    );
    let mut absmax = vec![0.0f32; rows];
    absmax_parallel(w, rows, cols, &mut absmax);
    let full = fmt.max_finite() * backoff;
    let tensor_absmax = absmax.iter().copied().fold(0.0f32, f32::max);
    let (scales, exp_bias) = match scaling {
        Scaling::PerChannel => {
            let s = absmax
                .iter()
                .map(|&m| if m > 0.0 { m / full } else { 1.0 })
                .collect();
            (s, fmt.exponent_bias())
        }
        Scaling::HwExpBias => {
            let b = choose_exp_bias(tensor_absmax / backoff, fmt);
            let s = pow2(fmt.exponent_bias() as i32 - b as i32);
            (vec![s; rows], b)
        }
        Scaling::Unit => (vec![1.0f32; rows], fmt.exponent_bias()),
    };
    let mut codes = vec![0u8; rows * cols];
    encode_parallel(w, cols, &scales, fmt, &mut codes);
    Quantized {
        format: fmt,
        scaling,
        rows,
        cols,
        codes,
        scales,
        exp_bias,
    }
}

/// Thread count for a `[rows, cols]` pass: up to eight, one per megabyte
/// of weights, never more than there are rows.
fn split(rows: usize, cols: usize) -> usize {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 1 << 20; // elements
    let by_size = (rows * cols).div_ceil(PER_THREAD).max(1);
    by_size.min(THREADS).min(rows)
}

/// Row absmax over up to eight threads.
fn absmax_parallel(w: &[u16], rows: usize, cols: usize, out: &mut [f32]) {
    let threads = split(rows, cols);
    if threads <= 1 {
        absmax_range(w, cols, out);
        return;
    }
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk).enumerate() {
            let src = &w[i * chunk * cols..(i * chunk + part.len()) * cols];
            s.spawn(move || absmax_range(src, cols, part));
        }
    });
}

/// Encode a block of rows with one scale each.
fn encode_range(w: &[u16], cols: usize, scales: &[f32], fmt: Fp8Format, out: &mut [u8]) {
    for (r, &s) in scales.iter().enumerate() {
        let src = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (o, &b) in dst.iter_mut().zip(src) {
            // Division, not a reciprocal multiply: one rounding, and the
            // same arithmetic the fixture generator does in float32.
            *o = encode(bf16_to_f32(b) / s, fmt);
        }
    }
}

/// Encoding over up to eight threads.
fn encode_parallel(w: &[u16], cols: usize, scales: &[f32], fmt: Fp8Format, out: &mut [u8]) {
    let rows = scales.len();
    let threads = split(rows, cols);
    if threads <= 1 {
        encode_range(w, cols, scales, fmt, out);
        return;
    }
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk * cols).enumerate() {
            let r0 = i * chunk;
            let n = part.len() / cols;
            let src = &w[r0 * cols..(r0 + n) * cols];
            let sc = &scales[r0..r0 + n];
            s.spawn(move || encode_range(src, cols, sc, fmt, part));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bf16 bits of an f32, for building test matrices.
    fn b(x: f32) -> u16 {
        f32_to_bf16(x)
    }

    #[test]
    fn e4m3_codes_match_the_device_cast() {
        // 1.375, -3.3, 100, 0.01, -240, 250 and 300 are the values the
        // device `cast_bf16_to_hf8` was probed with in the CLIP form
        // (`ns_CastKernel::ParamsV3`), with the codes it returned; the
        // rest follow the same encoding. 17 is a tie between 16 and 18 and
        // goes to the even mantissa.
        let cases: &[(f32, u8)] = &[
            (0.0, 0x00),
            (1.0, 0x38),
            (1.375, 0x3b),
            (-3.3, 0xc5),
            (100.0, 0x6c),
            (0.01, 0x05),
            (0.0625, 0x18),
            (17.0, 0x58),
            (240.0, 0x77),
            (-240.0, 0xf7),
            (250.0, 0x77),
            (300.0, 0x77),
            (0.001, 0x01),
            (1e-9, 0x00),
            (f32::INFINITY, 0x77),
            (f32::NEG_INFINITY, 0xf7),
        ];
        for &(x, code) in cases {
            assert_eq!(encode(x, Fp8Format::E4M3), code, "encode({x})");
        }
        assert_eq!(encode(f32::NAN, Fp8Format::E4M3) & 0x7f, 0x7f);
    }

    #[test]
    fn e5m2_codes_match_the_device_cast() {
        // Probe cast-f8-raw: 1.375 -> 1.5, 250 -> 256, 300 -> 320,
        // -240 -> -256.
        let cases: &[(f32, u8)] = &[
            (1.375, 0x3e),
            (250.0, 0x5c),
            (300.0, 0x5d),
            (-240.0, 0xdc),
            (1.0, 0x3c),
        ];
        for &(x, code) in cases {
            assert_eq!(encode(x, Fp8Format::E5M2), code, "encode({x})");
        }
        assert_eq!(decode(0x3e, Fp8Format::E5M2), 1.5);
        assert_eq!(decode(0x5c, Fp8Format::E5M2), 256.0);
        assert_eq!(decode(0x5d, Fp8Format::E5M2), 320.0);
        assert_eq!(decode(0xdc, Fp8Format::E5M2), -256.0);
    }

    #[test]
    fn format_constants() {
        assert_eq!(Fp8Format::E4M3.max_finite(), 240.0);
        assert_eq!(Fp8Format::E4M3.max_finite_code(), 0x77);
        assert_eq!(Fp8Format::E4M3.syn_data_type(), 8192);
        assert_eq!(Fp8Format::E5M2.max_finite(), 57344.0);
        assert_eq!(Fp8Format::E5M2.max_finite_code(), 0x7b);
        assert_eq!(Fp8Format::E5M2.syn_data_type(), 16384);
        assert!(decode(0x78, Fp8Format::E4M3).is_infinite());
        assert!(decode(0x7f, Fp8Format::E4M3).is_nan());
        assert_eq!(Fp8Format::from_name("E4M3"), Some(Fp8Format::E4M3));
        assert_eq!(Fp8Format::from_name("f8"), Some(Fp8Format::E5M2));
        assert_eq!(Fp8Format::from_name("fp4"), None);
        assert_eq!(Scaling::from_name("hw"), Some(Scaling::HwExpBias));
        assert_eq!(Scaling::from_name("pcs"), Some(Scaling::PerChannel));
        assert_eq!(Scaling::from_name("unit"), Some(Scaling::Unit));
        assert_eq!(Scaling::from_name("bogus"), None);
    }

    #[test]
    fn every_code_round_trips_through_the_encoder() {
        // Decoding a finite code and encoding the value back is the
        // identity, which is what makes `dequantize` a faithful inverse.
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for code in 0u16..256 {
                let code = code as u8;
                let v = decode(code, fmt);
                if !v.is_finite() {
                    continue;
                }
                let back = encode(v, fmt);
                // Negative zero encodes to 0x80, which decodes to -0.0.
                assert_eq!(back, code, "{fmt:?} code {code:#04x} value {v}");
            }
        }
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // Halfway between 1.0 (mantissa 0) and 1.125 (mantissa 1) in E4M3.
        assert_eq!(encode(1.0625, Fp8Format::E4M3), 0x38);
        // Halfway between 1.125 (odd) and 1.25 (even) rounds up.
        assert_eq!(encode(1.1875, Fp8Format::E4M3), 0x3a);
        // A hair above the halfway point always rounds away from zero.
        assert_eq!(encode(1.0626, Fp8Format::E4M3), 0x39);
    }

    #[test]
    fn per_channel_scales_stretch_each_row_to_full_scale() {
        // Two rows with very different magnitudes: per-channel scaling
        // gives each of them the format's whole range.
        let rows = 2;
        let cols = 4;
        let w: Vec<u16> = [0.5f32, -1.0, 0.25, 0.0, 100.0, -50.0, 12.5, 0.0]
            .iter()
            .map(|&x| b(x))
            .collect();
        let q = quantize(&w, rows, cols, Fp8Format::E4M3, Scaling::PerChannel);
        assert_eq!(q.scales.len(), rows);
        assert_eq!(q.scales[0], 1.0 / 240.0);
        assert_eq!(q.scales[1], 100.0 / 240.0);
        assert_eq!(q.exp_bias, 7);
        // The absmax element of each row lands on the largest finite code.
        assert_eq!(q.codes[1] & 0x7f, 0x77);
        assert_eq!(q.codes[4] & 0x7f, 0x77);
        assert_eq!(q.scale_operand().map(<[f32]>::len), Some(2));
        let d = q.dequantize();
        assert!((d[0] - 0.5).abs() < 0.5 * 0.07);
        assert!((d[4] - 100.0).abs() < 100.0 * 0.07);
    }

    #[test]
    fn hw_exp_bias_picks_the_finest_range_that_fits() {
        let f = Fp8Format::E4M3;
        // Ranges: bias 15 -> 0.9375, 11 -> 15, 7 -> 240, 3 -> 3840.
        assert_eq!(choose_exp_bias(0.5, f), 15);
        assert_eq!(choose_exp_bias(0.9375, f), 15);
        assert_eq!(choose_exp_bias(1.0, f), 11);
        assert_eq!(choose_exp_bias(15.0, f), 11);
        assert_eq!(choose_exp_bias(16.0, f), 7);
        assert_eq!(choose_exp_bias(1000.0, f), 3);
        assert_eq!(choose_exp_bias(1e9, f), 3);
        // E5M2 has one accepted bias on Gaudi2.
        assert_eq!(choose_exp_bias(1e9, Fp8Format::E5M2), 15);

        let w: Vec<u16> = (0..64).map(|i| b((i as f32 - 32.0) / 64.0)).collect();
        let q = quantize(&w, 4, 16, Fp8Format::E4M3, Scaling::HwExpBias);
        assert_eq!(q.exp_bias, 15);
        assert!(q.scales.iter().all(|&s| s == 1.0 / 256.0));
        assert_eq!(q.scale_operand(), None);
    }

    #[test]
    fn unit_scaling_leaves_the_codes_alone() {
        let w: Vec<u16> = [1.0f32, 2.0, 0.5, 100.0].iter().map(|&x| b(x)).collect();
        let q = quantize(&w, 1, 4, Fp8Format::E4M3, Scaling::Unit);
        assert_eq!(q.scales, vec![1.0]);
        assert_eq!(q.codes[0], encode(1.0, Fp8Format::E4M3));
        assert_eq!(q.codes[3], encode(100.0, Fp8Format::E4M3));
        assert_eq!(q.scale_operand(), None);
    }

    #[test]
    fn dequantize_of_quantize_is_within_the_format_resolution() {
        // E4M3 keeps three mantissa bits for every normal value, so the
        // worst relative error of an in-range weight is one half of the
        // 1/8 spacing, 1/16 = 6.25%; the mean is far below it.
        let rows = 32;
        let cols = 128;
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let x = ((i * 2654435761usize) % 65536) as f32 / 65536.0 - 0.5;
                b(x * (1.0 + (i / cols) as f32))
            })
            .collect();
        let q = quantize(&w, rows, cols, Fp8Format::E4M3, Scaling::PerChannel);
        let errs = row_errors(&w, &q);
        assert_eq!(errs.len(), rows);
        for (r, e) in errs.iter().enumerate() {
            assert!(e.max_rel <= 0.0626, "row {r} max_rel {}", e.max_rel);
            assert!(e.mean_rel < 0.03, "row {r} mean_rel {}", e.mean_rel);
        }
        // The bf16 round trip is the same values rounded once more.
        let bf = q.dequantize_bf16();
        let f = q.dequantize();
        assert_eq!(bf.len(), f.len());
        assert_eq!(bf[0], f32_to_bf16(f[0]));
    }

    #[test]
    fn the_parallel_path_agrees_with_the_serial_one() {
        // Big enough to take the eight-thread path (over a megabyte).
        let rows = 512;
        let cols = 4096;
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| b((((i * 7919) % 2003) as f32 - 1000.0) / 977.0))
            .collect();
        assert!(split(rows, cols) > 1, "the parallel path was not taken");
        let par = quantize(&w, rows, cols, Fp8Format::E4M3, Scaling::PerChannel);
        // The serial reference: one row block at a time.
        let mut absmax = vec![0.0f32; rows];
        absmax_range(&w, cols, &mut absmax);
        assert_eq!(absmax, row_absmax(&w, rows, cols));
        let scales: Vec<f32> = absmax.iter().map(|&m| m / 240.0).collect();
        assert_eq!(par.scales, scales);
        let mut codes = vec![0u8; rows * cols];
        encode_range(&w, cols, &scales, Fp8Format::E4M3, &mut codes);
        assert_eq!(par.codes, codes);
    }

    #[test]
    fn sampling_agrees_with_the_full_error_pass() {
        let (rows, cols) = (16, 64);
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| b((((i * 37) % 101) as f32 - 50.0) / 50.0))
            .collect();
        let q = quantize(&w, rows, cols, Fp8Format::E4M3, Scaling::PerChannel);
        let all = row_errors(&w, &q);
        // Every row, and then every fourth row.
        assert_eq!(sample_row_errors(&w, &q, rows), all);
        let s = sample_row_errors(&w, &q, 4);
        assert_eq!(s.len(), 4);
        for (k, e) in s.iter().enumerate() {
            assert_eq!(*e, all[k * 4]);
        }
        // More rows asked for than there are: every row, once.
        assert_eq!(sample_row_errors(&w, &q, 1000).len(), rows);
    }

    #[test]
    fn zero_rows_keep_a_unit_scale() {
        let w = vec![0u16; 8];
        let q = quantize(&w, 2, 4, Fp8Format::E4M3, Scaling::PerChannel);
        assert_eq!(q.scales, vec![1.0, 1.0]);
        assert_eq!(q.codes, vec![0u8; 8]);
        let e = row_errors(&w, &q);
        assert_eq!(
            e[0],
            RowError {
                mean_rel: 0.0,
                max_rel: 0.0
            }
        );
    }

    #[test]
    fn a_backoff_leaves_headroom() {
        let w: Vec<u16> = [1.0f32, -0.5, 0.25, 0.125].iter().map(|&x| b(x)).collect();
        let q = quantize_with(&w, 1, 4, Fp8Format::E4M3, Scaling::PerChannel, 0.5);
        assert_eq!(q.scales[0], 1.0 / 120.0);
        // The absmax element now sits at half of full scale, code 0x6f.
        assert_eq!(q.codes[0], 0x6f);
        assert_eq!(decode(q.codes[0], Fp8Format::E4M3), 120.0);
    }
}
