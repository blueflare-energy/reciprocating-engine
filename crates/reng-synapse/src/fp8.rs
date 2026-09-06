//! FP8 operands on the Gaudi2 MME: the graph-side host code, and the
//! probes that pin down the gemm form a quantized projection will use.
//!
//! What the host side owns (and what this module builds) is the operand:
//! a persistent tensor of `syn_type_fp8_143` (E4M3) or `syn_type_fp8_152`
//! (E5M2) codes produced by [`reng_fp8`], carrying the exponent bias as
//! `SYN_FP_QUANT_METADATA` (see [`crate::model::Gb::input_fp8`]).
//!
//! What the host side does not yet own is the node that consumes it. Two
//! forms exist on this stack and the choice between them is a device
//! measurement, not a host one:
//!
//! - the plain MME guid `gemm` with **both** operands fp8, which means the
//!   activation is cast inside the recipe (`cast_bf16_to_hf8`), and the
//!   only scale is the exponent bias (a power of 16);
//! - the complex guid `fp8_gemm_bf16`, which takes f32 scale operands and
//!   applies a per-output-channel `scaleB` of shape `[N]`.
//!
//! A mixed pair (bf16 activation, fp8 weight) is rejected: the MME's
//! operand validator refuses it at node creation and the complex guid
//! refuses it at compile time. [`gemm_mixed`] builds that pair on purpose
//! so the probe can record the refusal rather than assume it.
//!
//! Every function here builds one graph, compiles it and launches it once,
//! so each is a device probe. `reng-fp8-probe` (in `reng-model`, which has
//! the quantizer and the checkpoints) drives them.

use crate::ffi::*;
use crate::model::Gb;
use crate::runtime::{Out, OutKind, Runtime};
use core::ffi::{c_int, c_uint, c_void};
use reng_core::Result;
use reng_fp8::Fp8Format;

/// `ns_CastKernel::ParamsV3` (`perf_lib_layer_params.h`), the only cast
/// params form that reaches the fp8 saturation mode: round mode,
/// stochastic-rounding seed, saturation mode.
#[repr(C)]
struct CastParamsV3 {
    round_mode: c_int,
    seed: c_uint,
    mode: c_int,
}

/// `CastF32RoundMode_t::CAST_ROUND_HALF_NE`.
const CAST_ROUND_HALF_NE: c_int = 0;
/// `CastSatMode_t::CAST_CLIP`: saturate to the largest finite value
/// instead of returning an infinity, which is what [`reng_fp8::encode`]
/// does on the host.
const CAST_CLIP: c_int = 1;

impl CastParamsV3 {
    const fn clipping() -> Self {
        Self {
            round_mode: CAST_ROUND_HALF_NE,
            seed: 0,
            mode: CAST_CLIP,
        }
    }
}

/// `ns_Fp8Gemm::Params`, the params of the `fp8_gemm_<out>` complex guid.
#[repr(C)]
struct Fp8GemmParams {
    transpose_a: bool,
    transpose_b: bool,
}

/// The guid of the bf16-to-fp8 cast for a format.
const fn cast_to_guid(fmt: Fp8Format) -> &'static str {
    match fmt {
        Fp8Format::E4M3 => "cast_bf16_to_hf8",
        Fp8Format::E5M2 => "cast_bf16_to_f8",
    }
}

/// The guid of the fp8-to-bf16 cast for a format.
const fn cast_from_guid(fmt: Fp8Format) -> &'static str {
    match fmt {
        Fp8Format::E4M3 => "cast_hf8_to_bf16",
        Fp8Format::E5M2 => "cast_f8_to_bf16",
    }
}

/// One fp8 operand of a probe: the device sizes (fastest-changing
/// dimension first, the engine's convention) and the codes a
/// [`reng_fp8::Quantized`] holds, with the exponent bias its metadata
/// carries.
pub struct Fp8Operand<'a> {
    pub name: &'a str,
    pub sizes: &'a [u64],
    pub codes: &'a [u8],
    pub format: Fp8Format,
    pub exp_bias: u32,
}

impl Fp8Operand<'_> {
    fn add(&self, gb: &mut Gb<'_>) -> Result<synTensor> {
        gb.input_fp8(
            self.name,
            self.sizes,
            std::borrow::Cow::Owned(self.codes.to_vec()),
            self.format,
            self.exp_bias,
        )
    }
}

/// Rows of a device tensor: the outermost dimension.
fn rows(sizes: &[u64]) -> usize {
    *sizes.last().unwrap_or(&1) as usize
}

/// Run `gb` with `out` as the read-back tensor and return every row.
fn run(gb: Gb<'_>, out_name: std::ffi::CString, out_sizes: &[u64]) -> Result<Vec<f32>> {
    let out = Out {
        name: out_name,
        sizes: out_sizes.to_vec(),
        kind: OutKind::Bf16,
    };
    Runtime::new(gb, out)?.launch_and_read(rows(out_sizes))
}

/// Decode fp8 codes on the device: upload them as an fp8 tensor and cast
/// them back to bf16. Against [`reng_fp8::decode`] this pins the device's
/// reading of every code, the exponent bias included.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn decode_on_device(codes: &[u8], sizes: &[u64], fmt: Fp8Format) -> Result<Vec<f32>> {
    let mut gb = Gb::new()?;
    let t_in = Fp8Operand {
        name: "CODES",
        sizes,
        codes,
        format: fmt,
        exp_bias: fmt.exponent_bias(),
    }
    .add(&mut gb)?;
    let (t_out, n_out) = gb.output("OUT", sizes, SYN_TYPE_BF16)?;
    gb.node(
        cast_from_guid(fmt),
        "decode",
        &[t_in],
        &[t_out],
        core::ptr::null(),
        0,
    )?;
    run(gb, n_out, sizes)
}

/// Encode on the device and decode again: `cast_bf16_to_*` in the
/// clipping form into a persistent fp8 tensor, then `cast_*_to_bf16`.
/// Against `decode(encode(x))` on the host this pins the device encoder;
/// the intermediate must be persistent, or the graph compiler elides the
/// pair.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn cast_round_trip(x: &[f32], sizes: &[u64], fmt: Fp8Format) -> Result<Vec<f32>> {
    let mut gb = Gb::new()?;
    let t_in = gb.input("X", sizes, x)?;
    let t_mid = gb.scratch_typed("CODES", sizes, fmt.syn_data_type())?;
    let (t_out, n_out) = gb.output("OUT", sizes, SYN_TYPE_BF16)?;
    let params = CastParamsV3::clipping();
    gb.node(
        cast_to_guid(fmt),
        "encode",
        &[t_in],
        &[t_mid],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<CastParamsV3>() as u32,
    )?;
    gb.node(
        cast_from_guid(fmt),
        "decode",
        &[t_mid],
        &[t_out],
        core::ptr::null(),
        0,
    )?;
    run(gb, n_out, sizes)
}

/// The plain MME guid `gemm` over two fp8 operands into bf16, with
/// `transpose_b` as the engine's projections use it (`[K, m] x [K, N] ->
/// [N, m]`).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails; a rejected operand pair
/// fails at node creation.
pub fn gemm_fp8(
    a: &Fp8Operand<'_>,
    b: &Fp8Operand<'_>,
    out_sizes: &[u64],
    transpose_b: bool,
) -> Result<Vec<f32>> {
    let mut gb = Gb::new()?;
    let t_a = a.add(&mut gb)?;
    let t_b = b.add(&mut gb)?;
    let (t_out, n_out) = gb.output("OUT", out_sizes, SYN_TYPE_BF16)?;
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b,
    };
    gb.node(
        "gemm",
        "gemm_fp8",
        &[t_a, t_b],
        &[t_out],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    run(gb, n_out, out_sizes)
}

/// The weight-only pair the engine would like: a bf16 activation and an
/// fp8 weight in one `gemm`. The probe calls it to record how the stack
/// refuses it.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails, which on this stack
/// includes the node creation itself.
pub fn gemm_mixed(
    a: &[f32],
    a_sizes: &[u64],
    b: &Fp8Operand<'_>,
    out_sizes: &[u64],
    transpose_b: bool,
) -> Result<Vec<f32>> {
    let mut gb = Gb::new()?;
    let t_a = gb.input("A", a_sizes, a)?;
    let t_b = b.add(&mut gb)?;
    let (t_out, n_out) = gb.output("OUT", out_sizes, SYN_TYPE_BF16)?;
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b,
    };
    gb.node(
        "gemm",
        "gemm_mixed",
        &[t_a, t_b],
        &[t_out],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    run(gb, n_out, out_sizes)
}

/// The complex guid `fp8_gemm_bf16`: two fp8 operands and, optionally, f32
/// scale operands that multiply the product. `scale_b` of length `N`
/// applies one scale per output channel, which is what a
/// [`reng_fp8::Scaling::PerChannel`] weight needs.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if a scale vector is empty.
pub fn gemm_fp8_scaled(
    a: &Fp8Operand<'_>,
    b: &Fp8Operand<'_>,
    scale_a: Option<&[f32]>,
    scale_b: Option<&[f32]>,
    out_sizes: &[u64],
    transpose_b: bool,
) -> Result<Vec<f32>> {
    let mut gb = Gb::new()?;
    let t_a = a.add(&mut gb)?;
    let t_b = b.add(&mut gb)?;
    let mut ins = vec![t_a, t_b];
    for (name, s) in [("SA", scale_a), ("SB", scale_b)] {
        let Some(s) = s else { continue };
        assert!(!s.is_empty(), "{name}: an empty scale vector");
        // SAFETY: an f32 slice is readable as four times as many bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s))
        };
        ins.push(gb.input_raw(name, &[s.len() as u64], SYN_TYPE_F32, bytes)?);
    }
    let (t_out, n_out) = gb.output("OUT", out_sizes, SYN_TYPE_BF16)?;
    let params = Fp8GemmParams {
        transpose_a: false,
        transpose_b,
    };
    gb.node(
        "fp8_gemm_bf16",
        "gemm_cguid",
        &ins,
        &[t_out],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<Fp8GemmParams>() as u32,
    )?;
    run(gb, n_out, out_sizes)
}

/// Seconds per launch of an fp8 `gemm` at the given shapes: compile once,
/// three warm-up launches, `iters` launches back to back, then one read
/// back (which waits for all of them).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn bench_gemm_fp8(
    a: &Fp8Operand<'_>,
    b: &Fp8Operand<'_>,
    out_sizes: &[u64],
    transpose_b: bool,
    iters: usize,
) -> Result<f64> {
    let mut gb = Gb::new()?;
    let t_a = a.add(&mut gb)?;
    let t_b = b.add(&mut gb)?;
    let (t_out, n_out) = gb.output("OUT", out_sizes, SYN_TYPE_BF16)?;
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b,
    };
    gb.node(
        "gemm",
        "gemm_fp8",
        &[t_a, t_b],
        &[t_out],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    let out = Out {
        name: n_out,
        sizes: out_sizes.to_vec(),
        kind: OutKind::Bf16,
    };
    time_launches(Runtime::new(gb, out)?, iters)
}

/// [`bench_gemm_fp8`]'s bf16 counterpart at the same shapes, the baseline
/// the fp8 timing is read against. `a` and `b` are the bf16 bits of the
/// operands in the same device layout.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn bench_gemm_bf16(
    a: &[u16],
    a_sizes: &[u64],
    b: &[u16],
    b_sizes: &[u64],
    out_sizes: &[u64],
    transpose_b: bool,
    iters: usize,
) -> Result<f64> {
    let mut gb = Gb::new()?;
    let t_a = gb.input_bf16("A", a_sizes, std::borrow::Cow::Borrowed(a))?;
    let t_b = gb.input_bf16("B", b_sizes, std::borrow::Cow::Borrowed(b))?;
    let (t_out, n_out) = gb.output("OUT", out_sizes, SYN_TYPE_BF16)?;
    let params = synGEMMParams {
        transpose_a: false,
        transpose_b,
    };
    gb.node(
        "gemm",
        "gemm_bf16",
        &[t_a, t_b],
        &[t_out],
        (&raw const params).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    let out = Out {
        name: n_out,
        sizes: out_sizes.to_vec(),
        kind: OutKind::Bf16,
    };
    time_launches(Runtime::new(gb, out)?, iters)
}

fn time_launches(mut rt: Runtime<'_>, iters: usize) -> Result<f64> {
    for _ in 0..3 {
        rt.launch_and_read(1)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        rt.launch_only()?;
    }
    rt.launch_and_read(1)?;
    Ok(t0.elapsed().as_secs_f64() / (iters as f64 + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendor structs this module passes by pointer must have the
    /// layout the headers give them; a mismatch would be read as garbage
    /// by the graph compiler with no diagnostic.
    #[test]
    fn vendor_struct_layouts() {
        assert_eq!(core::mem::size_of::<synFpQuantParam>(), 16);
        assert_eq!(core::mem::align_of::<synFpQuantParam>(), 8);
        assert_eq!(core::mem::size_of::<synFpQuantMetadata>(), 24);
        assert_eq!(core::mem::size_of::<CastParamsV3>(), 12);
        assert_eq!(core::mem::size_of::<Fp8GemmParams>(), 2);
        assert_eq!(SYN_FP_QUANT_METADATA, 2);
        assert_eq!(SYN_TYPE_FP8_143, 8192);
        assert_eq!(SYN_TYPE_FP8_152, 16384);
        assert_eq!(Fp8Format::E4M3.syn_data_type(), SYN_TYPE_FP8_143);
        assert_eq!(Fp8Format::E5M2.syn_data_type(), SYN_TYPE_FP8_152);
    }

    #[test]
    fn cast_guids_follow_the_vendor_naming() {
        assert_eq!(cast_to_guid(Fp8Format::E4M3), "cast_bf16_to_hf8");
        assert_eq!(cast_from_guid(Fp8Format::E4M3), "cast_hf8_to_bf16");
        assert_eq!(cast_to_guid(Fp8Format::E5M2), "cast_bf16_to_f8");
        assert_eq!(cast_from_guid(Fp8Format::E5M2), "cast_f8_to_bf16");
    }
}
