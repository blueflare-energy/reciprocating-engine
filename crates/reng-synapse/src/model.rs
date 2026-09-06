//! A full decoder-only transformer forward pass as ONE fused SynapseAI recipe:
//! `L` decoder layers, a final RMSNorm, and the LM head, with a causal mask.
//! One graph, one launch; activations stay in HBM and only the
//! `[tokens, vocab]` logits are read back.
//!
//! The graph is assembled by a small builder that appends layers, so the same
//! code path serves a 2-layer synthetic model and a real 30-layer one. The
//! embedding gather is done on the host (a row copy per token; trivially exact
//! and cheap), so the graph starts from `[tokens, hidden]` activations.
//! Grouped-query attention is supported: query head `j` uses K/V head
//! `j / (n_heads / n_kv_heads)`, the HF `repeat_kv` convention. Attention
//! is batched over heads in 4-D tensors (`[.., heads-per-group, groups]`)
//! so each step of it is one node for all heads.
//!
//! With a KV cache (see `cached.rs`) each layer updates its cache tensors
//! `[head_dim, capacity + 1, 1, kv_heads]` in place: an ONNX ScatterND
//! update writes the block's rotated keys and values at their positions
//! (index tensor uploaded per step). Attention runs over the whole cache
//! with a mask that admits positions up to each query's own. No DMA node
//! touches the cache.
//!
//! Attention is the four nodes qk `batch_gemm`, mask add, softmax and av
//! `batch_gemm`, or one fused `sdpa_recomp_fwd_bf16` node over the same
//! tensors: the default in the single-sequence decode recipe, everywhere
//! with `RENG_SDPA=1` and nowhere with `RENG_SDPA=0` (read when the graph
//! is built; see [`fused_sdpa`]).
//!
//! Launch and readback (including the completion protocol) live in
//! `runtime.rs`. Diagnostic environment switches (all off by default):
//! `RENG_DEVSYNC` (device-wide sync after launch), `RENG_SETTLE_MS` (sleep
//! before readback), `RENG_EVBRIDGE` (event-gated readback on a second
//! stream), `RENG_SERIALIZE` (explicit dependency chain over all nodes),
//! `RENG_PERSIST_LAYERS` (every layer output is a persistent tensor instead
//! of a workspace tensor), `RENG_READBACK_TRACE` (print readback poll counts).

use crate::ffi::*;
use crate::heads::AxisParams;
use crate::runtime::{Out, OutKind, Runtime};
use crate::{Activation, LayerWeights, Stride, gather_columns};
use crate::{bf16_to_f32, scale_bf16};
use core::ffi::c_void;
use reng_core::{Error, Result};
use std::ffi::CString;

macro_rules! syn {
    ($call:expr) => {{
        let st = unsafe { $call };
        if st != SYN_SUCCESS {
            return Err(Error::Other(format!(
                concat!(stringify!($call), " -> synStatus {}"),
                st
            )));
        }
    }};
}

/// `ns_LayerNormKernel::ParamsRmsNorm` (`perf_lib_layer_params.h`), the
/// params of the forward `rms_norm_fwd_*` guids. With the backward kernel's
/// `ns_RmsNorm` layout (epsilon first) the node compiles and runs but
/// applies a fixed epsilon of 1e-5 whatever is passed. Padding is explicit
/// and zeroed: the raw bytes go into the recipe cache key.
#[repr(C)]
struct RmsNormParams {
    eps_valid: u8,
    _pad0: [u8; 3],
    eps: f32,
    /// Bitmaps of the normalised and parameter axes (CWHN); the FCD.
    norm_axis_bmp: i32,
    param_axis_bmp: i32,
    normalized_shape_dims: u32,
    fast_math: u8,
    _pad1: [u8; 3],
}

impl RmsNormParams {
    const fn new(eps: f32) -> Self {
        Self {
            eps_valid: 1,
            _pad0: [0; 3],
            eps,
            norm_axis_bmp: 1,
            param_axis_bmp: 1,
            normalized_shape_dims: 1,
            fast_math: 0,
            _pad1: [0; 3],
        }
    }
}

#[repr(C)]
struct RopeParams {
    offset: u32,
    mode: i32,
}

/// `ns_Sdpa::Params` of the fused attention guid `sdpa_recomp_fwd_bf16`:
/// `float scale; bool is_causal; ns_DropoutKernel::ParamsOptionalMaskOut
/// dropout { float ratio; unsigned seed; bool disableMaskOut }; bool
/// is_inference`, 24 bytes with C padding (explicit and zeroed: the raw
/// bytes go into the recipe cache key).
#[repr(C)]
struct SdpaParams {
    scale: f32,
    is_causal: u8,
    _pad0: [u8; 3],
    dropout_ratio: f32,
    dropout_seed: u32,
    disable_mask_out: u8,
    _pad1: [u8; 3],
    is_inference: u8,
    _pad2: [u8; 3],
}

impl SdpaParams {
    /// Inference without dropout and with an explicit mask (no causal
    /// flag: with a KV cache the key axis is the whole cache).
    const fn inference(scale: f32) -> Self {
        Self {
            scale,
            is_causal: 0,
            _pad0: [0; 3],
            dropout_ratio: 0.0,
            dropout_seed: 0,
            disable_mask_out: 1,
            _pad1: [0; 3],
            is_inference: 1,
            _pad2: [0; 3],
        }
    }
}

/// Whether a graph being built takes the fused attention node: what
/// `RENG_SDPA` says ([`crate::sdpa_switch`]), or `default` when it is
/// unset. The default is `true` for the single-sequence decode recipe
/// only: measured on Qwen2.5-1.5B, SmolLM2-1.7B, Llama-3.2-3B and
/// Qwen2.5-7B, the fused node never slows a batch-1 decode step and
/// speeds Llama-3.2-3B's up by 2%, while prefill blocks and batched
/// decode are a wash at best (Qwen2.5-1.5B, with two KV groups, loses 5%
/// on 256-row prefill blocks and 2.5% at batch 8).
pub(crate) fn fused_sdpa(default: bool) -> bool {
    crate::sdpa_switch(std::env::var("RENG_SDPA").ok().as_deref()).unwrap_or(default)
}

/// Additive attention-mask value for disallowed keys; `exp` of it underflows
/// to exactly zero in bf16 softmax while staying representable.
pub(crate) const MASK_NEG: f32 = -30000.0;

/// `wq` with the attention scale folded in: a scaled bf16 copy, or the
/// checkpoint slice itself when the model has a q norm, which would divide
/// the scale out of `wq` again (it rides on the norm gain instead).
fn scaled_wq<'a>(w: &LayerWeights<'a>) -> std::borrow::Cow<'a, [u16]> {
    if w.qn.is_empty() {
        std::borrow::Cow::Owned(scale_bf16(w.wq, score_scale(w)))
    } else {
        std::borrow::Cow::Borrowed(w.wq)
    }
}

/// The `wq` input of a layer graph: the checkpoint's rows, scaled while
/// they are uploaded when the scale rides on `wq` (no q norm), else as
/// they are (the scale is on the q norm gain).
fn wq_input<'a>(
    gb: &mut Gb<'a>,
    name: &str,
    sizes: &[u64],
    w: &LayerWeights<'a>,
) -> Result<synTensor> {
    if w.qn.is_empty() {
        gb.input_bf16_scaled(name, sizes, w.wq, score_scale(w))
    } else {
        gb.input_bf16(name, sizes, std::borrow::Cow::Borrowed(w.wq))
    }
}

/// The `wo` or `wd` input of a layer graph: the checkpoint's `[out, in]`
/// matrix as it is, or (`pitch > 0`) the column window of a tensor-
/// parallel shard, `out_rows` rows of `in_cols` elements gathered from
/// the mapped checkpoint while they are uploaded (see
/// [`LayerWeights::wo_pitch`]).
fn col_input<'a>(
    gb: &mut Gb<'a>,
    name: &str,
    sizes: &[u64],
    data: &'a [u16],
    out_rows: usize,
    in_cols: usize,
    pitch: usize,
) -> Result<synTensor> {
    if pitch == 0 {
        gb.input_bf16(name, sizes, std::borrow::Cow::Borrowed(data))
    } else {
        let st = Stride {
            rows: out_rows,
            cols: in_cols,
            pitch,
        };
        gb.input_bf16_strided(name, sizes, data, st, None)
    }
}

/// A `[out, in]` matrix for the CPU reference: the slice itself, or the
/// gathered column window of a strided one.
fn col_view(
    data: &[u16],
    out_rows: usize,
    in_cols: usize,
    pitch: usize,
) -> std::borrow::Cow<'_, [u16]> {
    if pitch == 0 {
        std::borrow::Cow::Borrowed(data)
    } else {
        std::borrow::Cow::Owned(gather_columns(
            data,
            Stride {
                rows: out_rows,
                cols: in_cols,
                pitch,
            },
        ))
    }
}

/// The factor on `q` before the scores: the attention scale, divided by
/// the attention softcap when the layer has one (the scores then come out
/// of the gemm as `scores / cap`, ready for the `tanh`).
fn score_scale(w: &LayerWeights<'_>) -> f32 {
    w.scale / w.attn_softcap.unwrap_or(1.0)
}

/// How a layer's q/k/v projections are built: from the checkpoint's three
/// matrices, or (a model with attention biases, Qwen2) from their row-block
/// concatenation as one gemm. Three small decode gemms run concurrently
/// on the MME at a fraction of the rate of one over the concatenated
/// weight (5.8 us against 3.9 us for the 6.3 MB of Qwen2.5-1.5B), which
/// is worth about 2 us of a 74 us layer. The biases stay the three
/// broadcast `add` nodes on the per-head tensors: the TPC fuser merges
/// them into the RoPE kernels, whereas a bias given to the gemm as its
/// third input (which `reng-gemm-bias-test` shows to be numerically the
/// same add) is split out by the graph compiler into an add stage of its
/// own that costs a handoff. Bias-free models keep the three gemms, so
/// their graphs are unchanged.
enum QkvProj {
    /// The three weights as separate gemm operands: the flat `[in, out]`
    /// matrices, or the per-head blocks of `RENG_HEAD_BLOCKS`.
    Separate(synTensor, synTensor, synTensor),
    /// The concatenated weight `[hidden, (n_heads + 2 n_kv_heads) *
    /// head_dim]`, the q rows carrying the attention scale.
    Fused(synTensor),
}

/// Whether a layer's projections take the fused form of [`QkvProj`]: it
/// has attention biases and neither the per-head blocks nor the
/// full-width q/k norm (which wants the flat q and k alone) is in use.
fn fused_qkv(w: &LayerWeights<'_>, head_blocks: bool) -> bool {
    !w.bq.is_empty() && !head_blocks && !wide_qk_norm(w)
}

/// The q, k and v projections as one `[(n_heads + 2 n_kv_heads) *
/// head_dim, hidden]` row-block matrix (the q rows scaled as `scaled_wq`),
/// the B operand of the fused projection gemm. An owned copy, as the
/// scaled `wq` is anyway; the runtime frees it after the upload.
fn qkv_weight(w: &LayerWeights<'_>) -> Vec<u16> {
    let wq = scaled_wq(w);
    let mut m = Vec::with_capacity(wq.len() + w.wk.len() + w.wv.len());
    m.extend_from_slice(&wq);
    m.extend_from_slice(w.wk);
    m.extend_from_slice(w.wv);
    m
}

/// Gemma-2's attention softcap on scores that already carry `1 / cap`:
/// `tanh(x) * cap` into a new tensor of `sizes` (`cap` a broadcast
/// constant of the same rank).
fn softcap_nodes(
    gb: &mut Gb<'_>,
    p: &dyn Fn(&str) -> String,
    x: synTensor,
    sizes: &[u64],
    cap: f32,
) -> Result<synTensor> {
    let none = (core::ptr::null::<c_void>(), 0u32);
    let ones = vec![1u64; sizes.len()];
    let t_cap = gb.input(&p("softcap"), &ones, &[cap])?;
    let th = gb.mid(&p("sc_tanh"), sizes, SYN_TYPE_BF16)?;
    let out = gb.mid(&p("sc_capped"), sizes, SYN_TYPE_BF16)?;
    gb.node("tanh_fwd_bf16", &p("sc_tanh"), &[x], &[th], none.0, none.1)?;
    gb.node(
        "mult_fwd_bf16",
        &p("sc_cap"),
        &[th, t_cap],
        &[out],
        none.0,
        none.1,
    )?;
    Ok(out)
}

/// Whether the layer's q/k norms take the OLMo-2 full-width form (gains
/// over the whole projection) rather than the Qwen3 per-head form; see
/// [`LayerWeights::qn`].
///
/// # Panics
///
/// Panics if a gain has neither length, the two gains take different
/// forms, or the full-width form meets attention biases.
fn wide_qk_norm(w: &LayerWeights<'_>) -> bool {
    if w.qn.is_empty() {
        assert!(w.kn.is_empty(), "k_norm gain without a q_norm gain");
        return false;
    }
    let hd = w.head_dim;
    let (qw, kvd) = (w.n_heads * hd, w.n_kv_heads * hd);
    let wide = w.qn.len() != hd;
    if wide {
        assert!(
            w.qn.len() == qw && w.kn.len() == kvd,
            "q/k norm gains of {} and {} for widths {qw} and {kvd}",
            w.qn.len(),
            w.kn.len()
        );
        assert!(
            w.bq.is_empty(),
            "full-width q/k norms with attention biases are not supported"
        );
    } else {
        assert!(
            w.kn.len() == hd,
            "k_norm gain of {} for head_dim {hd}",
            w.kn.len()
        );
    }
    wide
}

fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok()
}

/// The sliding window of the windowed layers, which must all agree (one
/// windowed mask per graph); `None` when no layer has one.
///
/// # Panics
///
/// Panics if two layers carry different windows.
pub(crate) fn common_window(layers: &[LayerWeights<'_>]) -> Option<usize> {
    let window = layers.iter().find_map(|l| l.window);
    assert!(
        layers
            .iter()
            .all(|l| l.window.is_none() || l.window == window),
        "layers with different sliding windows"
    );
    window
}

/// Whether some layer reads the second ("local") RoPE table.
pub(crate) fn uses_local_rope(layers: &[LayerWeights<'_>]) -> bool {
    layers.iter().any(|l| l.local_rope)
}

/// Whether some layer uses full causal attention (reads the plain mask).
pub(crate) fn uses_full_mask(layers: &[LayerWeights<'_>]) -> bool {
    layers.iter().any(|l| l.window.is_none())
}

/// `gelu_pytorch_tanh` of `x` into `out` (bf16, same `sizes`) as
/// `x * sigmoid(c1 * x + c3 * x^3)` with `c1 = 2 * sqrt(2 / pi)` and
/// `c3 = 0.044715 * c1`: `0.5 * (1 + tanh(u)) == sigmoid(2u)`, an exact
/// identity. Seven elementwise nodes; `gelu_fwd_bf16` computes only the
/// erf form (reng-gelu-test), so Gemma's activation is composed.
fn gelu_tanh_nodes(
    gb: &mut Gb<'_>,
    p: &dyn Fn(&str) -> String,
    x: synTensor,
    sizes: &[u64],
    out: synTensor,
) -> Result<()> {
    const C1: f32 = 1.595_769;
    const C3: f32 = 0.071_354_82;
    let bf = SYN_TYPE_BF16;
    let none = (core::ptr::null::<c_void>(), 0u32);
    let ones = vec![1u64; sizes.len()];
    let t_c1 = gb.input(&p("gelu_c1"), &ones, &[C1])?;
    let t_c3 = gb.input(&p("gelu_c3"), &ones, &[C3])?;
    let x2 = gb.mid(&p("gelu_x2"), sizes, bf)?;
    let x3 = gb.mid(&p("gelu_x3"), sizes, bf)?;
    let a = gb.mid(&p("gelu_a"), sizes, bf)?;
    let b = gb.mid(&p("gelu_b"), sizes, bf)?;
    let u = gb.mid(&p("gelu_u"), sizes, bf)?;
    let sg = gb.mid(&p("gelu_sig"), sizes, bf)?;
    gb.node(
        "mult_fwd_bf16",
        &p("gelu_sq"),
        &[x, x],
        &[x2],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gelu_cube"),
        &[x2, x],
        &[x3],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gelu_c3x3"),
        &[x3, t_c3],
        &[a],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gelu_c1x"),
        &[x, t_c1],
        &[b],
        none.0,
        none.1,
    )?;
    gb.node(
        "add_fwd_bf16",
        &p("gelu_sum"),
        &[a, b],
        &[u],
        none.0,
        none.1,
    )?;
    gb.node(
        "sigmoid_fwd_bf16",
        &p("gelu_sigmoid"),
        &[u],
        &[sg],
        none.0,
        none.1,
    )?;
    gb.node(
        "mult_fwd_bf16",
        &p("gelu_out"),
        &[x, sg],
        &[out],
        none.0,
        none.1,
    )?;
    Ok(())
}

/// The gate activation on `x` into `out` (bf16, same `sizes`): SiLU as
/// `x * sigmoid(x)` (two nodes) or the composed GELU-tanh.
fn activation_nodes(
    gb: &mut Gb<'_>,
    p: &dyn Fn(&str) -> String,
    act: Activation,
    x: synTensor,
    sizes: &[u64],
    out: synTensor,
) -> Result<()> {
    match act {
        Activation::Silu => {
            let none = (core::ptr::null::<c_void>(), 0u32);
            let sg = gb.mid(&p("sg"), sizes, SYN_TYPE_BF16)?;
            gb.node("sigmoid_fwd_bf16", &p("sig"), &[x], &[sg], none.0, none.1)?;
            gb.node(
                "mult_fwd_bf16",
                &p("silu"),
                &[x, sg],
                &[out],
                none.0,
                none.1,
            )?;
            Ok(())
        }
        Activation::GeluTanh => gelu_tanh_nodes(gb, p, x, sizes, out),
    }
}

/// An RMSNorm of `x` with gain `gain` (`[hidden]`, uploaded as input
/// `name`) into a new `[hidden, ...]` tensor of `sizes` (Gemma's post
/// norms on a branch output before its residual add).
fn post_norm_node(
    gb: &mut Gb<'_>,
    p: &dyn Fn(&str) -> String,
    name: &str,
    gain: &[f32],
    x: synTensor,
    sizes: &[u64],
    rms: &RmsNormParams,
) -> Result<synTensor> {
    let t_g = gb.input(&p(&format!("g_{name}")), &[sizes[0]], gain)?;
    let out = gb.mid(&p(&format!("{name}_out")), sizes, SYN_TYPE_BF16)?;
    let mut inv_sizes = sizes.to_vec();
    inv_sizes[0] = 1;
    let inv = gb.mid(&p(&format!("{name}_inv")), &inv_sizes, SYN_TYPE_F32)?;
    gb.node(
        "rms_norm_fwd_bf16",
        &p(name),
        &[x, t_g],
        &[out, inv],
        (&raw const *rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    )?;
    Ok(out)
}

/// Bytes per element of a device dtype the engine's persistent tensors
/// take: 2 for bf16, 4 for f32 and int32.
fn elem_bytes(dtype: core::ffi::c_int) -> usize {
    if dtype == SYN_TYPE_BF16 { 2 } else { 4 }
}

pub(crate) fn make_tensor(
    graph: synGraphHandle,
    name: &str,
    sizes: &[u64],
    dtype: core::ffi::c_int,
    persistent: bool,
) -> Result<(synTensor, CString)> {
    let section = if persistent {
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        sec
    } else {
        core::ptr::null_mut()
    };
    make_tensor_in(graph, name, sizes, dtype, section)
}

/// Create a tensor; a non-null `section` makes it persistent in that section
/// (at offset 0), so two tensors in one section alias the same memory.
pub(crate) fn make_tensor_in(
    graph: synGraphHandle,
    name: &str,
    sizes: &[u64],
    dtype: core::ffi::c_int,
    section: synSectionHandle,
) -> Result<(synTensor, CString)> {
    let cname = CString::new(name).unwrap();
    let mut t: synTensor = core::ptr::null_mut();
    syn!(synTensorHandleCreate(
        &mut t,
        graph,
        SYN_TENSOR_DATA,
        cname.as_ptr()
    ));
    if !section.is_null() {
        syn!(synTensorAssignToSection(t, section, 0));
    }
    let mut geo = synTensorGeometry {
        sizes: [0; HABANA_DIM_MAX],
        dims: sizes.len() as u32,
    };
    geo.sizes[..sizes.len()].copy_from_slice(sizes);
    syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
    syn!(synTensorSetDeviceDataType(t, dtype));
    Ok((t, cname))
}

/// Accumulates a graph: persistent inputs (with their host data), persistent
/// scratch tensors (device-resident, not read back), internal tensors, and
/// nodes. Launch plumbing lives in [`Runtime`].
pub(crate) struct Gb<'a> {
    pub graph: synGraphHandle,
    pub names: Vec<CString>,
    pub sizes: Vec<Vec<u64>>,
    /// Host data per bf16 input: a checkpoint slice borrowed as is, or an
    /// owned conversion (per-step inputs, scaled copies).
    pub data: Vec<std::borrow::Cow<'a, [u16]>>,
    /// Raw device bytes for non-bf16 inputs (else `data` is converted).
    pub raw: Vec<Option<Vec<u8>>>,
    /// Per input: a factor applied to the bf16 data while it is staged for
    /// the upload (the attention scale on `wq`), so the host keeps a view of
    /// the checkpoint instead of a scaled copy. `None` for the others.
    pub scales: Vec<Option<f32>>,
    /// Per input: the column window of a strided host source (see
    /// [`Stride`]); `None` for a contiguous one.
    pub strides: Vec<Option<Stride>>,
    pub scratch_names: Vec<CString>,
    pub scratch_sizes: Vec<Vec<u64>>,
    /// Bytes per element of each scratch tensor (2 for bf16, 4 for f32 and
    /// int32); sizes count elements.
    pub scratch_elem: Vec<usize>,
    /// For a scratch tensor that shares another scratch tensor's section
    /// (in-place update), that tensor's name; the runtime binds both to one
    /// buffer.
    pub scratch_alias: Vec<Option<String>>,
    /// Section of each scratch tensor (by name), for aliasing.
    scratch_sections: std::collections::HashMap<String, synSectionHandle>,
    /// Node ids in creation order (a valid topological order for our graphs).
    node_ids: Vec<synNodeId>,
    /// Running digest of the graph's structure (every tensor's name, sizes,
    /// dtype and persistence, every node's guid, name, operands and params),
    /// the key of the on-disk recipe cache (see `Runtime`).
    digest: std::hash::DefaultHasher,
    digest2: std::hash::DefaultHasher,
}

impl<'a> Gb<'a> {
    pub fn new() -> Result<Self> {
        // A second graph in the same process finds the library initialised.
        let st = unsafe { synInitialize() };
        if st != SYN_SUCCESS && st != SYN_OBJECT_ALREADY_INITIALIZED {
            return Err(Error::Other(format!("synInitialize -> synStatus {st}")));
        }
        let mut graph: synGraphHandle = core::ptr::null_mut();
        syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));
        Ok(Self {
            graph,
            names: Vec::new(),
            sizes: Vec::new(),
            data: Vec::new(),
            raw: Vec::new(),
            scales: Vec::new(),
            strides: Vec::new(),
            scratch_names: Vec::new(),
            scratch_sizes: Vec::new(),
            scratch_elem: Vec::new(),
            scratch_alias: Vec::new(),
            scratch_sections: std::collections::HashMap::new(),
            node_ids: Vec::new(),
            digest: std::hash::DefaultHasher::new(),
            digest2: {
                let mut h = std::hash::DefaultHasher::new();
                std::hash::Hash::hash(&0x5eed_u64, &mut h);
                h
            },
        })
    }

    /// Fold `bytes` into the structural digest.
    fn note(&mut self, bytes: &[u8]) {
        use std::hash::Hash;
        bytes.hash(&mut self.digest);
        bytes.hash(&mut self.digest2);
    }

    /// Record a tensor's declaration in the digest: kind, name, sizes,
    /// dtype and, for aliased scratch tensors, the tensor they alias.
    fn note_tensor(
        &mut self,
        kind: &str,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
        of: &str,
    ) {
        let mut b = Vec::new();
        b.extend_from_slice(kind.as_bytes());
        b.push(0);
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        for d in sizes {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&dtype.to_le_bytes());
        b.extend_from_slice(of.as_bytes());
        self.note(&b);
    }

    /// The structural digest as 32 hex digits. Two graphs with the same
    /// digest compile to interchangeable recipes: the compiled program
    /// depends on the structure only, never on the weights' values.
    pub fn cache_key(&self) -> String {
        use std::hash::Hasher;
        format!(
            "{:016x}{:016x}",
            self.digest.finish(),
            self.digest2.finish()
        )
    }

    /// A persistent output tensor that is not an input (read back, or
    /// written by the graph and read through [`Runtime::read_bf16_range`]).
    pub fn output(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
    ) -> Result<(synTensor, CString)> {
        self.note_tensor("out", name, sizes, dtype, "");
        make_tensor(self.graph, name, sizes, dtype, true)
    }

    /// A persistent bf16 input tensor whose host data (converted from f32
    /// here) is uploaded at launch.
    pub fn input(&mut self, name: &str, sizes: &[u64], data: &[f32]) -> Result<synTensor> {
        self.input_bf16(name, sizes, std::borrow::Cow::Owned(crate::to_bf16(data)))
    }

    /// A persistent bf16 input tensor from bf16 host data, borrowed when the
    /// caller keeps it alive (checkpoint weights) so nothing is copied
    /// before the upload.
    pub fn input_bf16(
        &mut self,
        name: &str,
        sizes: &[u64],
        data: std::borrow::Cow<'a, [u16]>,
    ) -> Result<synTensor> {
        assert_eq!(
            sizes.iter().product::<u64>() as usize,
            data.len(),
            "input {name}: sizes {sizes:?} against {} elements",
            data.len()
        );
        self.note_tensor("in", name, sizes, SYN_TYPE_BF16, "");
        let (t, cname) = make_tensor(self.graph, name, sizes, SYN_TYPE_BF16, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(data);
        self.raw.push(None);
        self.scales.push(None);
        self.strides.push(None);
        Ok(t)
    }

    /// A persistent bf16 input whose host data is the column window `st`
    /// of the row-major matrix starting at `data` (see [`Stride`]),
    /// gathered row by row while it is staged for the upload and, with a
    /// `scale`, scaled on the way like [`Gb::input_bf16_scaled`]. `sizes`
    /// describe the device tensor, `rows * cols` elements.
    pub fn input_bf16_strided(
        &mut self,
        name: &str,
        sizes: &[u64],
        data: &'a [u16],
        st: Stride,
        scale: Option<f32>,
    ) -> Result<synTensor> {
        assert_eq!(
            sizes.iter().product::<u64>() as usize,
            st.rows * st.cols,
            "input {name}: sizes {sizes:?} against a {} x {} window",
            st.rows,
            st.cols
        );
        assert!(
            st.pitch >= st.cols && data.len() >= (st.rows - 1) * st.pitch + st.cols,
            "input {name}: {} elements do not hold the window {st:?}",
            data.len()
        );
        let of = scale.map_or(String::new(), |s| format!("scale={}", s.to_bits()));
        self.note_tensor("in", name, sizes, SYN_TYPE_BF16, &of);
        let (t, cname) = make_tensor(self.graph, name, sizes, SYN_TYPE_BF16, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(std::borrow::Cow::Borrowed(data));
        self.raw.push(None);
        self.scales.push(scale);
        self.strides.push(Some(st));
        Ok(t)
    }

    /// A persistent bf16 input whose data is multiplied by `scale` (in f32,
    /// rounded to bf16 like [`crate::scale_bf16`]) while it is staged for
    /// the upload; the host slice stays borrowed. The scale is part of the
    /// recipe cache key.
    pub fn input_bf16_scaled(
        &mut self,
        name: &str,
        sizes: &[u64],
        data: &'a [u16],
        scale: f32,
    ) -> Result<synTensor> {
        assert_eq!(
            sizes.iter().product::<u64>() as usize,
            data.len(),
            "input {name}: sizes {sizes:?} against {} elements",
            data.len()
        );
        self.note_tensor(
            "in",
            name,
            sizes,
            SYN_TYPE_BF16,
            &format!("scale={}", scale.to_bits()),
        );
        let (t, cname) = make_tensor(self.graph, name, sizes, SYN_TYPE_BF16, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(std::borrow::Cow::Borrowed(data));
        self.raw.push(None);
        self.scales.push(Some(scale));
        self.strides.push(None);
        Ok(t)
    }

    /// A persistent input tensor of another dtype whose bytes are uploaded
    /// as given (index tensors for scatter/gather probes).
    pub fn input_raw(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
        bytes: &[u8],
    ) -> Result<synTensor> {
        self.note_tensor("in", name, sizes, dtype, "");
        let (t, cname) = make_tensor(self.graph, name, sizes, dtype, true)?;
        self.names.push(cname);
        self.sizes.push(sizes.to_vec());
        self.data.push(std::borrow::Cow::Owned(Vec::new()));
        self.raw.push(Some(bytes.to_vec()));
        self.scales.push(None);
        self.strides.push(None);
        Ok(t)
    }

    /// A persistent bf16 tensor that gets its own device buffer at launch but
    /// is neither uploaded nor read back (device-resident intermediate).
    pub fn scratch(&mut self, name: &str, sizes: &[u64]) -> Result<synTensor> {
        self.scratch_typed(name, sizes, SYN_TYPE_BF16)
    }

    /// [`Gb::scratch`] with an explicit dtype (bf16 or f32).
    pub fn scratch_typed(
        &mut self,
        name: &str,
        sizes: &[u64],
        dtype: core::ffi::c_int,
    ) -> Result<synTensor> {
        self.note_tensor("scratch", name, sizes, dtype, "");
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, self.graph));
        syn!(synSectionSetPersistent(sec, true));
        let (t, cname) = make_tensor_in(self.graph, name, sizes, dtype, sec)?;
        self.scratch_sections.insert(name.to_owned(), sec);
        self.scratch_names.push(cname);
        self.scratch_sizes.push(sizes.to_vec());
        self.scratch_elem.push(elem_bytes(dtype));
        self.scratch_alias.push(None);
        Ok(t)
    }

    /// A persistent bf16 tensor in the same section (same memory) as the
    /// scratch tensor named `of`: the output side of an in-place update.
    ///
    /// # Panics
    ///
    /// Panics if `of` is not a scratch tensor of this graph.
    pub fn scratch_alias(&mut self, name: &str, sizes: &[u64], of: &str) -> Result<synTensor> {
        self.scratch_alias_typed(name, sizes, of, SYN_TYPE_BF16)
    }

    /// [`Gb::scratch_alias`] with an explicit dtype (bf16 or f32), which
    /// must be the aliased tensor's.
    ///
    /// # Panics
    ///
    /// Panics if `of` is not a scratch tensor of this graph.
    pub fn scratch_alias_typed(
        &mut self,
        name: &str,
        sizes: &[u64],
        of: &str,
        dtype: core::ffi::c_int,
    ) -> Result<synTensor> {
        let sec = *self
            .scratch_sections
            .get(of)
            .unwrap_or_else(|| panic!("no scratch tensor named {of}"));
        self.note_tensor("alias", name, sizes, dtype, of);
        let (t, cname) = make_tensor_in(self.graph, name, sizes, dtype, sec)?;
        self.scratch_names.push(cname);
        self.scratch_sizes.push(sizes.to_vec());
        self.scratch_elem.push(elem_bytes(dtype));
        self.scratch_alias.push(Some(of.to_owned()));
        Ok(t)
    }

    /// A graph-internal tensor. Diagnostic `RENG_PERSIST_ALL` makes every
    /// intermediate a persistent scratch tensor instead, so a run can be
    /// dumped tensor by tensor (`RENG_DUMP_SCRATCH`).
    pub fn mid(&mut self, name: &str, sizes: &[u64], dtype: core::ffi::c_int) -> Result<synTensor> {
        if env_on("RENG_PERSIST_ALL") {
            return self.scratch_typed(name, sizes, dtype);
        }
        self.note_tensor("mid", name, sizes, dtype, "");
        Ok(make_tensor(self.graph, name, sizes, dtype, false)?.0)
    }

    /// Append a node and return its id (for explicit dependency edges).
    pub fn node(
        &mut self,
        guid: &str,
        name: &str,
        ins: &[synTensor],
        outs: &[synTensor],
        params: *const c_void,
        params_size: u32,
    ) -> Result<synNodeId> {
        let g = CString::new(guid).unwrap();
        let n = CString::new(name).unwrap();
        // Digest: guid, name, operand tensor names, raw params.
        let mut b = Vec::new();
        b.extend_from_slice(guid.as_bytes());
        b.push(0);
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        for (which, list) in [(b'i', ins), (b'o', outs)] {
            for &t in list {
                let mut buf = [0u8; 256];
                syn!(synTensorGetName(
                    t,
                    buf.len() as u64,
                    buf.as_mut_ptr().cast::<core::ffi::c_char>()
                ));
                let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
                b.push(which);
                b.extend_from_slice(&buf[..end]);
                b.push(0);
            }
        }
        if params_size > 0 {
            // SAFETY: the caller passes `params_size` readable bytes.
            b.extend_from_slice(unsafe {
                core::slice::from_raw_parts(params.cast::<u8>(), params_size as usize)
            });
        }
        self.note(&b);
        let mut id: synNodeId = 0;
        syn!(synNodeCreateWithId(
            self.graph,
            ins.as_ptr(),
            outs.as_ptr(),
            ins.len() as u32,
            outs.len() as u32,
            params,
            params_size,
            g.as_ptr(),
            n.as_ptr(),
            &mut id,
            core::ptr::null(),
            core::ptr::null(),
        ));
        self.node_ids.push(id);
        Ok(id)
    }

    /// Diagnostic (`RENG_SERIALIZE`): an explicit control dependency from every
    /// node to the next one in creation order, forcing sequential execution.
    pub fn serialize_if_requested(&mut self) -> Result<()> {
        if !env_on("RENG_SERIALIZE") {
            return Ok(());
        }
        self.note(b"RENG_SERIALIZE");
        for w in self.node_ids.windows(2) {
            syn!(synNodeDependencySet(self.graph, &w[0], &w[1], 1, 1));
        }
        Ok(())
    }
}

/// Shared per-graph tensors a layer needs besides its own weights.
pub(crate) struct Shared {
    pub sin: synTensor,
    pub cos: synTensor,
    /// The second RoPE table pair, read by the layers with `local_rope`
    /// (Gemma-3 sliding layers); present when some layer needs it.
    pub sin_local: Option<synTensor>,
    pub cos_local: Option<synTensor>,
    /// Additive mask laid out like the score matrix, `[keys, queries, 1, 1]`
    /// (broadcast over heads), for the layers without a window.
    pub mask: Option<synTensor>,
    /// The same with the sliding window applied, for the layers with one.
    pub mask_window: Option<synTensor>,
    /// KV-cache capacity in positions. When set, every layer gets cache
    /// input and output tensors (see [`cache_names`]) of `capacity + 1`
    /// positions (the last is a trash slot padded rows are written to),
    /// which is also the key axis of the scores.
    pub cache: Option<usize>,
    /// The int32 scatter indices input, `[3, tokens * groups]` device sizes:
    /// update `r + tokens * g` (row r of KV head g) goes to ONNX index
    /// `(g, 0, position)` of the `[hd, keys, 1, groups]` cache.
    pub kidx: Option<synTensor>,
    /// Whether the cache update writes in place (output aliasing input).
    /// In-place ScatterND runs its rows serially, so a block of many rows
    /// (prefill) is written into a separate output buffer that the caller
    /// copies back; a one-row decode step stays in place.
    pub inplace: bool,
    /// Whether attention is one fused `sdpa_recomp_fwd_bf16` node per
    /// layer instead of the qk gemm, mask add, softmax and av gemm
    /// (`RENG_SDPA`, read once when the graph is built, or the recipe's
    /// default; see [`fused_sdpa`]). The kernel reads the engine's own
    /// tensors: it broadcasts the size-1 K/V heads dim over the query
    /// heads of a group and takes the additive mask as it is.
    pub sdpa: bool,
}

/// Persistent tensor names of layer `li`'s cache state:
/// `(k_cache_in, v_cache_in, k_cache_out, v_cache_out)`, each
/// `[head_dim, capacity + 1, 1, n_kv_heads]` (keys after RoPE); the "out"
/// tensors alias the "in" ones (in-place scatter).
#[must_use]
pub fn cache_names(li: usize) -> (String, String, String, String) {
    (
        format!("l{li}_kci"),
        format!("l{li}_vci"),
        format!("l{li}_kco"),
        format!("l{li}_vco"),
    )
}

/// `synTransposeParams`: `permutation[i]` is the source dim of output dim
/// `i` (five entries), then the tensor's dim count.
#[repr(C)]
struct TransposeParams {
    permutation: [u32; 5],
    tensor_dim: u32,
}

/// `ns_ScatterNdUpdateKernel::Params`: 0 = non-deterministic on duplicate
/// indices (the padded rows all hit the trash slot, whose value is unused).
#[repr(C)]
struct ScatterNdUpdateParams {
    mode: i32,
}

/// Append one decoder layer reading `x` and return its output tensor. When
/// `out` is given (a persistent tensor) the layer writes into it instead of a
/// graph-internal tensor.
///
/// Attention is batched over heads: with `hpg = n_heads / n_kv_heads` query
/// heads per KV group, queries live in `[hd, tokens, hpg, groups]` and keys
/// and values in `[hd, keys, 1, groups]`, so one `batch_gemm` per step
/// serves every head (the size-1 dim broadcasts a KV head over its group).
/// Projections are `batch_gemm`s of the normalised input (broadcast) against
/// per-head weight blocks, RoPE runs on the batched tensors with one 2-D
/// table, and the per-head outputs go through one transpose back to
/// `[hidden, tokens]` for the output projection.
///
/// With `w.post_norm` the two layer norms move from the branch inputs to
/// the branch outputs (`h = x + rms(attn(x))`, `y = h + rms(mlp(h))`), and
/// full-width q/k norm gains (see [`LayerWeights::qn`]) are applied to the
/// flat `[width, tokens]` projections before the head reshape.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn build_layer<'a>(
    gb: &mut Gb<'a>,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'a>,
    sh: &Shared,
    tokens: usize,
    hidden: usize,
    inter: usize,
    out: Option<synTensor>,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && w.head_dim >= 1 && nkv >= 1 && nh % nkv == 0);
    let hd_us = w.head_dim;
    let hpg_us = nh / nkv;
    // Width of the query projection (all heads); `hidden` unless the
    // config decouples `head_dim` from `hidden / n_heads`.
    let qw_us = nh * hd_us;
    let (t, h, i, hd, hpg, groups, qw) = (
        tokens as u64,
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        qw_us as u64,
    );
    // Key axis of the score matrices: the block alone, or the whole cache
    // (its usable positions plus the trash slot).
    let keys = sh.cache.map_or(t, |c| c as u64 + 1);
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");

    // Weights are the checkpoint's bf16 `[out, in]` matrices, borrowed:
    // as the gemms' transposed B operand (`[K = in, N = out]`) they need
    // no copy; the per-head form is a free reshape of the same matrix.
    // Every MME node runs at the same rate whatever its N, except that a
    // batch_gemm over per-head blocks runs each head as an N = hd gemm at
    // a fraction of the rate (12% of a 1.7B prefill), so the plain form is
    // the default and diagnostic `RENG_HEAD_BLOCKS` keeps the per-head one.
    let post_norm = w.post_norm;
    // Full-width q/k norms run on the flat 2-D projections, which the
    // per-head projection form does not produce.
    let wide_qk = wide_qk_norm(w);
    let head_blocks = env_on("RENG_HEAD_BLOCKS") && !wide_qk;
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    // The layer's RoPE table and mask: the model's second table for a
    // local-rope layer, the windowed mask for a windowed layer.
    let (t_sin, t_cos) = if w.local_rope {
        (
            sh.sin_local.expect("local RoPE table"),
            sh.cos_local.expect("local RoPE table"),
        )
    } else {
        (sh.sin, sh.cos)
    };
    let mask = if w.window.is_some() {
        sh.mask_window
    } else {
        sh.mask
    };
    // With a q norm the scale cannot ride on `wq` (the norm would divide
    // it out again); it goes on the norm gain instead.
    let scale = score_scale(w);
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    // Query, key and value heads of the fused projection (see `QkvProj`).
    let heads_all = hpg * groups + 2 * groups;
    let fused = fused_qkv(w, head_blocks);
    let proj = if fused {
        QkvProj::Fused(gb.input_bf16(
            &p("wqkv"),
            &[h, hd * heads_all],
            std::borrow::Cow::Owned(qkv_weight(w)),
        )?)
    } else if head_blocks {
        QkvProj::Separate(
            wq_input(gb, &p("wq"), &[h, hd, hpg, groups], w)?,
            gb.input_bf16(
                &p("wk"),
                &[h, hd, 1, groups],
                std::borrow::Cow::Borrowed(w.wk),
            )?,
            gb.input_bf16(
                &p("wv"),
                &[h, hd, 1, groups],
                std::borrow::Cow::Borrowed(w.wv),
            )?,
        )
    } else {
        QkvProj::Separate(
            wq_input(gb, &p("wq2"), &[h, qw], w)?,
            gb.input_bf16(
                &p("wk2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wk),
            )?,
            gb.input_bf16(
                &p("wv2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wv),
            )?,
        )
    };
    let t_wo = col_input(
        gb,
        &p("wo"),
        &[qw, h],
        w.wo,
        hidden,
        qw as usize,
        w.wo_pitch,
    )?;
    let t_wg = gb.input_bf16(&p("wg"), &[h, i], std::borrow::Cow::Borrowed(w.wg))?;
    let t_wu = gb.input_bf16(&p("wu"), &[h, i], std::borrow::Cow::Borrowed(w.wu))?;
    let t_wd = col_input(gb, &p("wd"), &[i, h], w.wd, hidden, inter, w.wd_pitch)?;

    // The pre-norm tensors; a post-norm layer feeds the block input to the
    // projections and normalises the branch outputs instead (below).
    let t_n1 = (!post_norm)
        .then(|| gb.mid(&p("n1"), &[h, t], bf))
        .transpose()?;
    let t_inv1 = (!post_norm)
        .then(|| gb.mid(&p("inv1"), &[1, t], SYN_TYPE_F32))
        .transpose()?;
    let t_n1_4 = gb.mid(&p("n1_4"), &[h, t, 1, 1], bf)?;
    let t_q = gb.mid(&p("q"), &[hd, t, hpg, groups], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, t, 1, groups], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, t, 1, groups], bf)?;
    let t_at = gb.mid(&p("at"), &[hd, t, hpg, groups], bf)?;
    let t_at3 = gb.mid(&p("at3"), &[hd, t, hpg * groups], bf)?;
    let t_att = gb.mid(&p("att"), &[hd, hpg * groups, t], bf)?;
    let t_attn = gb.mid(&p("attn"), &[qw, t], bf)?;
    let t_o = gb.mid(&p("o"), &[h, t], bf)?;
    let t_h = gb.mid(&p("h"), &[h, t], bf)?;
    let t_n2 = (!post_norm)
        .then(|| gb.mid(&p("n2"), &[h, t], bf))
        .transpose()?;
    let t_inv2 = (!post_norm)
        .then(|| gb.mid(&p("inv2"), &[1, t], SYN_TYPE_F32))
        .transpose()?;
    let t_gate = gb.mid(&p("gate"), &[i, t], bf)?;
    let t_up = gb.mid(&p("up"), &[i, t], bf)?;
    let t_act = gb.mid(&p("act"), &[i, t], bf)?;
    let t_gated = gb.mid(&p("gated"), &[i, t], bf)?;
    let t_down = gb.mid(&p("down"), &[h, t], bf)?;
    let t_out = match out {
        Some(o) => o,
        None => gb.mid(&p("out"), &[h, t], bf)?,
    };

    let rms = RmsNormParams::new(w.eps);
    let rope = RopeParams { offset: 0, mode: 0 };
    let gemm = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let sm = synSoftmaxParams { dim: 0 };
    let tr = TransposeParams {
        permutation: [0, 2, 1, 0, 0],
        tensor_dim: 3,
    };
    let prm = (
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    );
    let pr = (
        (&raw const rope).cast::<c_void>(),
        core::mem::size_of::<RopeParams>() as u32,
    );
    let pg = (
        (&raw const gemm).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let pgt = (
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let ps = (
        (&raw const sm).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
    );
    let ptr_ = (
        (&raw const tr).cast::<c_void>(),
        core::mem::size_of::<TransposeParams>() as u32,
    );
    let scatter = ScatterNdUpdateParams { mode: 0 };
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let none = (core::ptr::null::<c_void>(), 0u32);

    // What the projections read: the normalised block input, or the block
    // input itself when the norms sit on the branch outputs.
    let t_ain = match (t_n1, t_inv1) {
        (Some(n1), Some(inv1)) => {
            gb.node(
                "rms_norm_fwd_bf16",
                &p("norm1"),
                &[x, t_g1],
                &[n1, inv1],
                prm.0,
                prm.1,
            )?;
            n1
        }
        _ => x,
    };
    // `[features, t]` is `[hd, heads, t]`; the head layout wants the token
    // dim inside the heads, so the flat projections end in a transpose of
    // their two outer dims (the inverse of `heads_last`).
    let tr_in = TransposeParams {
        permutation: [0, 2, 1, 0, 0],
        tensor_dim: 3,
    };
    let ptr_in = (
        (&raw const tr_in).cast::<c_void>(),
        core::mem::size_of::<TransposeParams>() as u32,
    );
    let split_heads = AxisParams { axis: 2 };
    let psplit = (
        (&raw const split_heads).cast::<c_void>(),
        core::mem::size_of::<AxisParams>() as u32,
    );
    match proj {
        QkvProj::Fused(t_wqkv) => {
            // One gemm; its `[hd, heads_all, t]` view is transposed once
            // and split along its outermost dim (a contiguous split, so
            // no copy) into the q, k and v heads.
            let flat = gb.mid(&p("qkv2"), &[hd * heads_all, t], bf)?;
            let by_head = gb.mid(&p("qkv_heads"), &[hd, heads_all, t], bf)?;
            let tokens_in = gb.mid(&p("qkv_tokens"), &[hd, t, heads_all], bf)?;
            let q_tokens = gb.mid(&p("q_tokens"), &[hd, t, hpg * groups], bf)?;
            let k_tokens = gb.mid(&p("k_tokens"), &[hd, t, groups], bf)?;
            let v_tokens = gb.mid(&p("v_tokens"), &[hd, t, groups], bf)?;
            gb.node(
                "gemm",
                &p("qkv_proj"),
                &[t_ain, t_wqkv],
                &[flat],
                pgt.0,
                pgt.1,
            )?;
            gb.node("reshape", &p("qkv_3d"), &[flat], &[by_head], none.0, none.1)?;
            gb.node(
                "transpose",
                &p("qkv_tokens_in"),
                &[by_head],
                &[tokens_in],
                ptr_in.0,
                ptr_in.1,
            )?;
            gb.node(
                "split",
                &p("qkv_split"),
                &[tokens_in],
                &[q_tokens, k_tokens, v_tokens],
                psplit.0,
                psplit.1,
            )?;
            gb.node("reshape", &p("q_4d"), &[q_tokens], &[t_q], none.0, none.1)?;
            gb.node("reshape", &p("k_4d"), &[k_tokens], &[t_k], none.0, none.1)?;
            gb.node("reshape", &p("v_4d"), &[v_tokens], &[t_v], none.0, none.1)?;
        }
        QkvProj::Separate(t_wq, t_wk, t_wv) if head_blocks => {
            gb.node("reshape", &p("n1_4d"), &[t_ain], &[t_n1_4], none.0, none.1)?;
            gb.node(
                "batch_gemm",
                &p("q_proj"),
                &[t_n1_4, t_wq],
                &[t_q],
                pgt.0,
                pgt.1,
            )?;
            gb.node(
                "batch_gemm",
                &p("k_proj"),
                &[t_n1_4, t_wk],
                &[t_k],
                pgt.0,
                pgt.1,
            )?;
            gb.node(
                "batch_gemm",
                &p("v_proj"),
                &[t_n1_4, t_wv],
                &[t_v],
                pgt.0,
                pgt.1,
            )?;
        }
        QkvProj::Separate(t_wq, t_wk, t_wv) => {
            // Full-width q/k norm gains (OLMo-2), applied to the flat
            // projections before the head reshape; the scale rides on the q
            // gain, as in the per-head form.
            let (t_qn_w, t_kn_w) = if wide_qk {
                let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
                (
                    Some(gb.input(&p("qn"), &[qw], &qn_scaled)?),
                    Some(gb.input(&p("kn"), &[hd * groups], w.kn)?),
                )
            } else {
                (None, None)
            };
            for (name, wt, n_out, heads, out, gain) in [
                ("q", t_wq, qw, hpg * groups, t_q, t_qn_w),
                ("k", t_wk, hd * groups, groups, t_k, t_kn_w),
                ("v", t_wv, hd * groups, groups, t_v, None),
            ] {
                let flat = gb.mid(&p(&format!("{name}2")), &[n_out, t], bf)?;
                let by_head = gb.mid(&p(&format!("{name}_heads")), &[hd, heads, t], bf)?;
                let tokens_in = gb.mid(&p(&format!("{name}_tokens")), &[hd, t, heads], bf)?;
                gb.node(
                    "gemm",
                    &p(&format!("{name}_proj")),
                    &[t_ain, wt],
                    &[flat],
                    pgt.0,
                    pgt.1,
                )?;
                let flat = match gain {
                    Some(g) => {
                        let normed = gb.mid(&p(&format!("{name}2n")), &[n_out, t], bf)?;
                        let inv = gb.mid(&p(&format!("{name}2inv")), &[1, t], SYN_TYPE_F32)?;
                        gb.node(
                            "rms_norm_fwd_bf16",
                            &p(&format!("{name}_norm")),
                            &[flat, g],
                            &[normed, inv],
                            prm.0,
                            prm.1,
                        )?;
                        normed
                    }
                    None => flat,
                };
                gb.node(
                    "reshape",
                    &p(&format!("{name}_3d")),
                    &[flat],
                    &[by_head],
                    none.0,
                    none.1,
                )?;
                gb.node(
                    "transpose",
                    &p(&format!("{name}_tokens_in")),
                    &[by_head],
                    &[tokens_in],
                    ptr_in.0,
                    ptr_in.1,
                )?;
                gb.node(
                    "reshape",
                    &p(&format!("{name}_4d")),
                    &[tokens_in],
                    &[out],
                    none.0,
                    none.1,
                )?;
            }
        }
    }
    // Attention biases (when present) broadcast over the token dim.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input(&p("bq"), &[hd, 1, hpg, groups], &bq_scaled)?;
        let t_bk = gb.input(&p("bk"), &[hd, 1, 1, groups], w.bk)?;
        let t_bv = gb.input(&p("bv"), &[hd, 1, 1, groups], w.bv)?;
        let qb = gb.mid(&p("qb"), &[hd, t, hpg, groups], bf)?;
        let kb = gb.mid(&p("kb"), &[hd, t, 1, groups], bf)?;
        let vb = gb.mid(&p("vb"), &[hd, t, 1, groups], bf)?;
        gb.node(
            "add_fwd_bf16",
            &p("q_bias"),
            &[t_q, t_bq],
            &[qb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("k_bias"),
            &[t_k, t_bk],
            &[kb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("v_bias"),
            &[t_v, t_bv],
            &[vb],
            none.0,
            none.1,
        )?;
        (qb, kb, vb)
    };
    // Qwen3 q/k norms (when present): an RMSNorm over the head dim (the
    // FCD) of every query and key head, the gain `[hd]` broadcast over the
    // outer dims, exactly like the layer norms over `[hidden, t]`.
    let (t_q, t_k) = if w.qn.is_empty() || wide_qk {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
        let t_qn = gb.input(&p("qn"), &[hd], &qn_scaled)?;
        let t_kn = gb.input(&p("kn"), &[hd], w.kn)?;
        let qn = gb.mid(&p("qn_out"), &[hd, t, hpg, groups], bf)?;
        let qn_inv = gb.mid(&p("qn_inv"), &[1, t, hpg, groups], SYN_TYPE_F32)?;
        let kn = gb.mid(&p("kn_out"), &[hd, t, 1, groups], bf)?;
        let kn_inv = gb.mid(&p("kn_inv"), &[1, t, 1, groups], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("q_norm"),
            &[t_q, t_qn],
            &[qn, qn_inv],
            prm.0,
            prm.1,
        )?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("k_norm"),
            &[t_k, t_kn],
            &[kn, kn_inv],
            prm.0,
            prm.1,
        )?;
        (qn, kn)
    };
    // RoPE on q and k, or neither for a NoPE layer.
    let (t_qr, t_kr) = if w.use_rope {
        let t_qr = gb.mid(&p("qr"), &[hd, t, hpg, groups], bf)?;
        let t_kr = gb.mid(&p("kr"), &[hd, t, 1, groups], bf)?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_q"),
            &[t_q, t_sin, t_cos],
            &[t_qr],
            pr.0,
            pr.1,
        )?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_k"),
            &[t_k, t_sin, t_cos],
            &[t_kr],
            pr.0,
            pr.1,
        )?;
        (t_qr, t_kr)
    } else {
        (t_q, t_k)
    };

    // Keys and values attention reads: the block's own, or the cache updated
    // in place with the block: an ONNX ScatterND update writes each real row
    // of the rotated keys and of the values at its position (padded rows go
    // to the trash slot); the "out" tensor aliases the "in" one, so only the
    // written rows move (see cached.rs).
    let (k_full, v_full) = if let Some(kidx) = sh.kidx {
        let (n_kci, n_vci, n_kco, n_vco) = cache_names(li);
        let kci = gb.scratch(&n_kci, &[hd, keys, 1, groups])?;
        let vci = gb.scratch(&n_vci, &[hd, keys, 1, groups])?;
        let (kco, vco) = if sh.inplace {
            (
                gb.scratch_alias(&n_kco, &[hd, keys, 1, groups], &n_kci)?,
                gb.scratch_alias(&n_vco, &[hd, keys, 1, groups], &n_vci)?,
            )
        } else {
            (
                gb.scratch(&n_kco, &[hd, keys, 1, groups])?,
                gb.scratch(&n_vco, &[hd, keys, 1, groups])?,
            )
        };
        let kru = gb.mid(&p("kru"), &[hd, t * groups], bf)?;
        let vu = gb.mid(&p("vu"), &[hd, t * groups], bf)?;
        gb.node("reshape", &p("kr_updates"), &[t_kr], &[kru], none.0, none.1)?;
        gb.node("reshape", &p("v_updates"), &[t_v], &[vu], none.0, none.1)?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("k_scatter"),
            &[kci, kidx, kru],
            &[kco],
            psc.0,
            psc.1,
        )?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("v_scatter"),
            &[vci, kidx, vu],
            &[vco],
            psc.0,
            psc.1,
        )?;
        (kco, vco)
    } else {
        (t_kr, t_v)
    };
    if sh.sdpa && w.attn_softcap.is_none() {
        // Fused attention: one node over the same tensors (the scale is
        // already on q). A softcapped layer (Gemma-2) keeps the four
        // nodes below: its `tanh` sits between the scores and the mask.
        let sdpa = SdpaParams::inference(1.0);
        let mut ins = vec![t_qr, k_full, v_full];
        ins.extend(mask);
        gb.node(
            "sdpa_recomp_fwd_bf16",
            &p("sdpa"),
            &ins,
            &[t_at],
            (&raw const sdpa).cast::<c_void>(),
            core::mem::size_of::<SdpaParams>() as u32,
        )?;
    } else {
        let t_sc = gb.mid(&p("scores"), &[keys, t, hpg, groups], bf)?;
        let t_pr = gb.mid(&p("probs"), &[keys, t, hpg, groups], bf)?;
        // scores[key, query] per head = q @ k^T with K in its natural layout.
        gb.node(
            "batch_gemm",
            &p("qk"),
            &[t_qr, k_full],
            &[t_sc],
            pgt.0,
            pgt.1,
        )?;
        // Gemma-2: softcap the scores (which carry `1 / cap`) before the mask.
        let t_sc = match w.attn_softcap {
            Some(cap) => softcap_nodes(gb, &p, t_sc, &[keys, t, hpg, groups], cap)?,
            None => t_sc,
        };
        let sm_in = if let Some(mask) = mask {
            let masked = gb.mid(&p("masked"), &[keys, t, hpg, groups], bf)?;
            gb.node(
                "add_fwd_bf16",
                &p("mask"),
                &[t_sc, mask],
                &[masked],
                none.0,
                none.1,
            )?;
            masked
        } else {
            t_sc
        };
        gb.node(
            "softmax_fwd_bf16",
            &p("softmax"),
            &[sm_in],
            &[t_pr],
            ps.0,
            ps.1,
        )?;
        gb.node("batch_gemm", &p("av"), &[t_pr, v_full], &[t_at], pg.0, pg.1)?;
    }
    // [hd, t, hpg, groups] -> [hd, t, heads] -> [hd, heads, t] = [hidden, t].
    gb.node("reshape", &p("at_3d"), &[t_at], &[t_at3], none.0, none.1)?;
    gb.node(
        "transpose",
        &p("heads_last"),
        &[t_at3],
        &[t_att],
        ptr_.0,
        ptr_.1,
    )?;
    gb.node(
        "reshape",
        &p("attn_2d"),
        &[t_att],
        &[t_attn],
        none.0,
        none.1,
    )?;
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
    // Post-norm: the attention output is normalised (g1) before it joins
    // the residual stream.
    let t_res1 = if post_norm {
        let t_on = gb.mid(&p("on"), &[h, t], bf)?;
        let t_oinv = gb.mid(&p("oinv"), &[1, t], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("norm1"),
            &[t_o, t_g1],
            &[t_on, t_oinv],
            prm.0,
            prm.1,
        )?;
        t_on
    } else {
        t_o
    };
    // Gemma: the attention branch is normalised (g_post_attn) before its
    // residual add, on top of the pre-norm.
    let t_res1 = if w.g_post_attn.is_empty() {
        t_res1
    } else {
        post_norm_node(
            gb,
            &p,
            "post_attn_norm",
            w.g_post_attn,
            t_res1,
            &[h, t],
            &rms,
        )?
    };
    gb.node(
        "add_fwd_bf16",
        &p("res1"),
        &[x, t_res1],
        &[t_h],
        none.0,
        none.1,
    )?;
    // What the MLP reads: the normalised residual, or the residual itself
    // when g2 normalises the MLP output instead (below).
    let t_min = match (t_n2, t_inv2) {
        (Some(n2), Some(inv2)) => {
            gb.node(
                "rms_norm_fwd_bf16",
                &p("norm2"),
                &[t_h, t_g2],
                &[n2, inv2],
                prm.0,
                prm.1,
            )?;
            n2
        }
        _ => t_h,
    };
    gb.node(
        "gemm",
        &p("gate_proj"),
        &[t_min, t_wg],
        &[t_gate],
        pgt.0,
        pgt.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_min, t_wu], &[t_up], pgt.0, pgt.1)?;
    activation_nodes(gb, &p, w.act, t_gate, &[i, t], t_act)?;
    gb.node(
        "mult_fwd_bf16",
        &p("gate_x_up"),
        &[t_act, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    gb.node(
        "gemm",
        &p("down_proj"),
        &[t_gated, t_wd],
        &[t_down],
        pgt.0,
        pgt.1,
    )?;
    // Post-norm: the MLP output is normalised (g2) before the residual add.
    let t_res2 = if post_norm {
        let t_dn = gb.mid(&p("dn"), &[h, t], bf)?;
        let t_dinv = gb.mid(&p("dinv"), &[1, t], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("norm2"),
            &[t_down, t_g2],
            &[t_dn, t_dinv],
            prm.0,
            prm.1,
        )?;
        t_dn
    } else {
        t_down
    };
    // Gemma: the MLP branch is normalised (g_post_mlp) before its residual
    // add, on top of the pre-norm.
    let t_res2 = if w.g_post_mlp.is_empty() {
        t_res2
    } else {
        post_norm_node(gb, &p, "post_mlp_norm", w.g_post_mlp, t_res2, &[h, t], &rms)?
    };
    gb.node(
        "add_fwd_bf16",
        &p("res2"),
        &[t_h, t_res2],
        &[t_out],
        none.0,
        none.1,
    )?;
    Ok(t_out)
}

/// Per-graph tensors of the batched decode layer: one row per sequence,
/// everything per-sequence carried in the outermost (fifth) dimension.
pub(crate) struct SharedBatched {
    /// RoPE rows per sequence, `[hd, 1, 1, 1, B]`, from the model's first
    /// table, and from its second one for the layers with `local_rope`.
    pub sin: synTensor,
    pub cos: synTensor,
    pub sin_local: Option<synTensor>,
    pub cos_local: Option<synTensor>,
    /// Additive mask per sequence, `[keys, 1, 1, 1, B]`, for the layers
    /// without a window, and its windowed twin for the layers with one.
    pub mask: Option<synTensor>,
    pub mask_window: Option<synTensor>,
    /// Int32 scatter indices, `[4, groups * B]`: update `g + groups * b`
    /// goes to ONNX index `(b, g, 0, position_b)` of the
    /// `[hd, keys, 1, groups, B]` cache.
    pub kidx: synTensor,
    pub capacity: usize,
    pub batch: usize,
    /// Fused attention, see [`Shared::sdpa`]; the kernel takes the 5-D
    /// tensors with one mask row per sequence.
    pub sdpa: bool,
}

/// Append one decoder layer for `B` sequences of one token each, reading `x`
/// (`[hidden, B]`) and returning the layer output in the same shape. The
/// attention path is the 4-D batched one of [`build_layer`] with the
/// sequence batch as a fifth, outermost dimension: the projections are 2-D
/// gemms with `M = B` whose `[hidden, B]` outputs are free reshapes of the
/// head layout, every sequence has its own RoPE row, cache
/// slot, placement column and mask, and with one query row per sequence the
/// context `[hd, 1, hpg, groups, B]` already is `[hidden, B]` in memory.
/// Weights carry the same names and element counts as in [`build_layer`],
/// so a runtime can share them with a prefill recipe.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn build_layer_batched<'a>(
    gb: &mut Gb<'a>,
    li: usize,
    x: synTensor,
    w: &LayerWeights<'a>,
    sh: &SharedBatched,
    hidden: usize,
    inter: usize,
) -> Result<synTensor> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && w.head_dim >= 1 && nkv >= 1 && nh % nkv == 0);
    let hd_us = w.head_dim;
    let hpg_us = nh / nkv;
    let (h, i, hd, hpg, groups, keys, b, qw) = (
        hidden as u64,
        inter as u64,
        hd_us as u64,
        hpg_us as u64,
        nkv as u64,
        sh.capacity as u64 + 1,
        sh.batch as u64,
        (nh * hd_us) as u64,
    );
    let bf = SYN_TYPE_BF16;
    let p = |s: &str| format!("l{li}_{s}");
    // See `build_layer`: the scale rides on the q norm gain when there is one.
    let scale = score_scale(w);
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    let post_norm = w.post_norm;
    let wide_qk = wide_qk_norm(w);
    let t_g1 = gb.input(&p("g1"), &[h], w.g1)?;
    let t_g2 = gb.input(&p("g2"), &[h], w.g2)?;
    let (t_sin, t_cos) = if w.local_rope {
        (
            sh.sin_local.expect("local RoPE rows"),
            sh.cos_local.expect("local RoPE rows"),
        )
    } else {
        (sh.sin, sh.cos)
    };
    let t_mask = if w.window.is_some() {
        sh.mask_window.expect("windowed mask")
    } else {
        sh.mask.expect("causal mask")
    };
    // With one row per sequence the projections are plain gemms with
    // M = B over the natural `[in, out]` weights: `[hidden, B]` is already
    // `[hd, 1, hpg, groups, B]` in memory (head-major features, sequence
    // outermost), so the head layout is a free reshape. These weights are
    // laid out differently from the wide recipe's per-head blocks, so they
    // get their own names and buffers.
    // With attention biases the projections are the fused gemm of
    // `build_layer` (see `QkvProj`), over the same concatenated weight.
    let heads_all = hpg * groups + 2 * groups;
    let fused = fused_qkv(w, false);
    let proj = if fused {
        QkvProj::Fused(gb.input_bf16(
            &p("wqkv"),
            &[h, hd * heads_all],
            std::borrow::Cow::Owned(qkv_weight(w)),
        )?)
    } else {
        QkvProj::Separate(
            wq_input(gb, &p("wq2"), &[h, qw], w)?,
            gb.input_bf16(
                &p("wk2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wk),
            )?,
            gb.input_bf16(
                &p("wv2"),
                &[h, hd * groups],
                std::borrow::Cow::Borrowed(w.wv),
            )?,
        )
    };
    let t_wo = col_input(
        gb,
        &p("wo"),
        &[qw, h],
        w.wo,
        hidden,
        qw as usize,
        w.wo_pitch,
    )?;
    let t_wg = gb.input_bf16(&p("wg"), &[h, i], std::borrow::Cow::Borrowed(w.wg))?;
    let t_wu = gb.input_bf16(&p("wu"), &[h, i], std::borrow::Cow::Borrowed(w.wu))?;
    let t_wd = col_input(gb, &p("wd"), &[i, h], w.wd, hidden, inter, w.wd_pitch)?;

    // Pre-norm tensors (see `build_layer`).
    let t_n1 = (!post_norm)
        .then(|| gb.mid(&p("n1"), &[h, b], bf))
        .transpose()?;
    let t_inv1 = (!post_norm)
        .then(|| gb.mid(&p("inv1"), &[1, b], SYN_TYPE_F32))
        .transpose()?;
    // The flat projections: the three, or the fused one.
    let t_flat = if fused {
        (None, Some(gb.mid(&p("qkv2"), &[hd * heads_all, b], bf)?))
    } else {
        (
            Some((
                gb.mid(&p("q2"), &[qw, b], bf)?,
                gb.mid(&p("k2"), &[hd * groups, b], bf)?,
                gb.mid(&p("v2"), &[hd * groups, b], bf)?,
            )),
            None,
        )
    };
    let t_q = gb.mid(&p("q"), &[hd, 1, hpg, groups, b], bf)?;
    let t_k = gb.mid(&p("k"), &[hd, 1, 1, groups, b], bf)?;
    let t_v = gb.mid(&p("v"), &[hd, 1, 1, groups, b], bf)?;
    let (n_kci, n_vci, n_kco, n_vco) = cache_names(li);
    let kci = gb.scratch(&n_kci, &[hd, keys, 1, groups, b])?;
    let vci = gb.scratch(&n_vci, &[hd, keys, 1, groups, b])?;
    let kco = gb.scratch_alias(&n_kco, &[hd, keys, 1, groups, b], &n_kci)?;
    let vco = gb.scratch_alias(&n_vco, &[hd, keys, 1, groups, b], &n_vci)?;
    let t_kru = gb.mid(&p("kru"), &[hd, groups * b], bf)?;
    let t_vu = gb.mid(&p("vu"), &[hd, groups * b], bf)?;
    let t_at = gb.mid(&p("at"), &[hd, 1, hpg, groups, b], bf)?;
    let t_attn = gb.mid(&p("attn"), &[qw, b], bf)?;
    let t_o = gb.mid(&p("o"), &[h, b], bf)?;
    let t_h = gb.mid(&p("h"), &[h, b], bf)?;
    let t_n2 = (!post_norm)
        .then(|| gb.mid(&p("n2"), &[h, b], bf))
        .transpose()?;
    let t_inv2 = (!post_norm)
        .then(|| gb.mid(&p("inv2"), &[1, b], SYN_TYPE_F32))
        .transpose()?;
    let t_gate = gb.mid(&p("gate"), &[i, b], bf)?;
    let t_up = gb.mid(&p("up"), &[i, b], bf)?;
    let t_act = gb.mid(&p("act"), &[i, b], bf)?;
    let t_gated = gb.mid(&p("gated"), &[i, b], bf)?;
    let t_down = gb.mid(&p("down"), &[h, b], bf)?;
    let t_out = gb.mid(&p("out"), &[h, b], bf)?;

    let rms = RmsNormParams::new(w.eps);
    let rope = RopeParams { offset: 0, mode: 0 };
    let gemm = synGEMMParams {
        transpose_a: false,
        transpose_b: false,
    };
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    let sm = synSoftmaxParams { dim: 0 };
    let prm = (
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    );
    let pr = (
        (&raw const rope).cast::<c_void>(),
        core::mem::size_of::<RopeParams>() as u32,
    );
    let pg = (
        (&raw const gemm).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let pgt = (
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    );
    let ps = (
        (&raw const sm).cast::<c_void>(),
        core::mem::size_of::<synSoftmaxParams>() as u32,
    );
    let scatter = ScatterNdUpdateParams { mode: 0 };
    let psc = (
        (&raw const scatter).cast::<c_void>(),
        core::mem::size_of::<ScatterNdUpdateParams>() as u32,
    );
    let none = (core::ptr::null::<c_void>(), 0u32);

    let t_ain = match (t_n1, t_inv1) {
        (Some(n1), Some(inv1)) => {
            gb.node(
                "rms_norm_fwd_bf16",
                &p("norm1"),
                &[x, t_g1],
                &[n1, inv1],
                prm.0,
                prm.1,
            )?;
            n1
        }
        _ => x,
    };
    match (proj, t_flat) {
        (QkvProj::Fused(t_wqkv), (_, Some(t_qkv2))) => {
            // One gemm; `[hd * heads_all, B]` is `[hd, heads_all, B]`,
            // split along the heads into the q, k and v heads of every
            // sequence (sequences outermost, as the 5-D views want them).
            let by_head = gb.mid(&p("qkv_heads"), &[hd, heads_all, b], bf)?;
            let q_heads = gb.mid(&p("q_heads"), &[hd, hpg * groups, b], bf)?;
            let k_heads = gb.mid(&p("k_heads"), &[hd, groups, b], bf)?;
            let v_heads = gb.mid(&p("v_heads"), &[hd, groups, b], bf)?;
            let split_heads = AxisParams { axis: 1 };
            gb.node(
                "gemm",
                &p("qkv_proj"),
                &[t_ain, t_wqkv],
                &[t_qkv2],
                pgt.0,
                pgt.1,
            )?;
            gb.node(
                "reshape",
                &p("qkv_3d"),
                &[t_qkv2],
                &[by_head],
                none.0,
                none.1,
            )?;
            gb.node(
                "split",
                &p("qkv_split"),
                &[by_head],
                &[q_heads, k_heads, v_heads],
                (&raw const split_heads).cast::<c_void>(),
                core::mem::size_of::<AxisParams>() as u32,
            )?;
            gb.node("reshape", &p("q_5d"), &[q_heads], &[t_q], none.0, none.1)?;
            gb.node("reshape", &p("k_5d"), &[k_heads], &[t_k], none.0, none.1)?;
            gb.node("reshape", &p("v_5d"), &[v_heads], &[t_v], none.0, none.1)?;
        }
        (QkvProj::Separate(t_wq, t_wk, t_wv), (Some((t_q2, t_k2, t_v2)), _)) => {
            gb.node("gemm", &p("q_proj"), &[t_ain, t_wq], &[t_q2], pgt.0, pgt.1)?;
            gb.node("gemm", &p("k_proj"), &[t_ain, t_wk], &[t_k2], pgt.0, pgt.1)?;
            gb.node("gemm", &p("v_proj"), &[t_ain, t_wv], &[t_v2], pgt.0, pgt.1)?;
            // Full-width q/k norms (OLMo-2) on the flat projections, as in
            // `build_layer`; the inputs carry the same names and sizes.
            let (t_q2, t_k2) = if wide_qk {
                let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
                let t_qn = gb.input(&p("qn"), &[qw], &qn_scaled)?;
                let t_kn = gb.input(&p("kn"), &[hd * groups], w.kn)?;
                let q2n = gb.mid(&p("q2n"), &[qw, b], bf)?;
                let q2inv = gb.mid(&p("q2inv"), &[1, b], SYN_TYPE_F32)?;
                let k2n = gb.mid(&p("k2n"), &[hd * groups, b], bf)?;
                let k2inv = gb.mid(&p("k2inv"), &[1, b], SYN_TYPE_F32)?;
                gb.node(
                    "rms_norm_fwd_bf16",
                    &p("q_norm"),
                    &[t_q2, t_qn],
                    &[q2n, q2inv],
                    prm.0,
                    prm.1,
                )?;
                gb.node(
                    "rms_norm_fwd_bf16",
                    &p("k_norm"),
                    &[t_k2, t_kn],
                    &[k2n, k2inv],
                    prm.0,
                    prm.1,
                )?;
                (q2n, k2n)
            } else {
                (t_q2, t_k2)
            };
            gb.node("reshape", &p("q_5d"), &[t_q2], &[t_q], none.0, none.1)?;
            gb.node("reshape", &p("k_5d"), &[t_k2], &[t_k], none.0, none.1)?;
            gb.node("reshape", &p("v_5d"), &[t_v2], &[t_v], none.0, none.1)?;
        }
        _ => unreachable!("projection form and its flat tensors agree"),
    }
    // Attention biases (when present) broadcast over the sequence batch.
    let (t_q, t_k, t_v) = if w.bq.is_empty() {
        (t_q, t_k, t_v)
    } else {
        let bq_scaled: Vec<f32> = w.bq.iter().map(|v| v * q_scale).collect();
        let t_bq = gb.input(&p("bq"), &[hd, 1, hpg, groups, 1], &bq_scaled)?;
        let t_bk = gb.input(&p("bk"), &[hd, 1, 1, groups, 1], w.bk)?;
        let t_bv = gb.input(&p("bv"), &[hd, 1, 1, groups, 1], w.bv)?;
        let qb = gb.mid(&p("qb"), &[hd, 1, hpg, groups, b], bf)?;
        let kb = gb.mid(&p("kb"), &[hd, 1, 1, groups, b], bf)?;
        let vb = gb.mid(&p("vb"), &[hd, 1, 1, groups, b], bf)?;
        gb.node(
            "add_fwd_bf16",
            &p("q_bias"),
            &[t_q, t_bq],
            &[qb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("k_bias"),
            &[t_k, t_bk],
            &[kb],
            none.0,
            none.1,
        )?;
        gb.node(
            "add_fwd_bf16",
            &p("v_bias"),
            &[t_v, t_bv],
            &[vb],
            none.0,
            none.1,
        )?;
        (qb, kb, vb)
    };
    // Qwen3 q/k norms (when present), as in `build_layer`.
    let (t_q, t_k) = if w.qn.is_empty() || wide_qk {
        (t_q, t_k)
    } else {
        let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
        let t_qn = gb.input(&p("qn"), &[hd], &qn_scaled)?;
        let t_kn = gb.input(&p("kn"), &[hd], w.kn)?;
        let qn = gb.mid(&p("qn_out"), &[hd, 1, hpg, groups, b], bf)?;
        let qn_inv = gb.mid(&p("qn_inv"), &[1, 1, hpg, groups, b], SYN_TYPE_F32)?;
        let kn = gb.mid(&p("kn_out"), &[hd, 1, 1, groups, b], bf)?;
        let kn_inv = gb.mid(&p("kn_inv"), &[1, 1, 1, groups, b], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("q_norm"),
            &[t_q, t_qn],
            &[qn, qn_inv],
            prm.0,
            prm.1,
        )?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("k_norm"),
            &[t_k, t_kn],
            &[kn, kn_inv],
            prm.0,
            prm.1,
        )?;
        (qn, kn)
    };
    let (t_qr, t_kr) = if w.use_rope {
        let t_qr = gb.mid(&p("qr"), &[hd, 1, hpg, groups, b], bf)?;
        let t_kr = gb.mid(&p("kr"), &[hd, 1, 1, groups, b], bf)?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_q"),
            &[t_q, t_sin, t_cos],
            &[t_qr],
            pr.0,
            pr.1,
        )?;
        gb.node(
            "rope_st2_fwd_bf16",
            &p("rope_k"),
            &[t_k, t_sin, t_cos],
            &[t_kr],
            pr.0,
            pr.1,
        )?;
        (t_qr, t_kr)
    } else {
        (t_q, t_k)
    };
    gb.node(
        "reshape",
        &p("kr_updates"),
        &[t_kr],
        &[t_kru],
        none.0,
        none.1,
    )?;
    gb.node("reshape", &p("v_updates"), &[t_v], &[t_vu], none.0, none.1)?;
    // Diagnostic `RENG_NO_SCATTER`: leave the cache stale (wrong results) to
    // time the step without the two ScatterND nodes.
    let (kco, vco) = if env_on("RENG_NO_SCATTER") {
        (kci, vci)
    } else {
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("k_scatter"),
            &[kci, sh.kidx, t_kru],
            &[kco],
            psc.0,
            psc.1,
        )?;
        gb.node(
            "scatter_nd_update_fwd_bf16",
            &p("v_scatter"),
            &[vci, sh.kidx, t_vu],
            &[vco],
            psc.0,
            psc.1,
        )?;
        (kco, vco)
    };
    if sh.sdpa && w.attn_softcap.is_none() {
        // Fused attention over the 5-D tensors (see `build_layer`).
        let sdpa = SdpaParams::inference(1.0);
        gb.node(
            "sdpa_recomp_fwd_bf16",
            &p("sdpa"),
            &[t_qr, kco, vco, t_mask],
            &[t_at],
            (&raw const sdpa).cast::<c_void>(),
            core::mem::size_of::<SdpaParams>() as u32,
        )?;
    } else {
        let t_sc = gb.mid(&p("scores"), &[keys, 1, hpg, groups, b], bf)?;
        let t_masked = gb.mid(&p("masked"), &[keys, 1, hpg, groups, b], bf)?;
        let t_pr = gb.mid(&p("probs"), &[keys, 1, hpg, groups, b], bf)?;
        gb.node("batch_gemm", &p("qk"), &[t_qr, kco], &[t_sc], pgt.0, pgt.1)?;
        let t_sc = match w.attn_softcap {
            Some(cap) => softcap_nodes(gb, &p, t_sc, &[keys, 1, hpg, groups, b], cap)?,
            None => t_sc,
        };
        gb.node(
            "add_fwd_bf16",
            &p("mask"),
            &[t_sc, t_mask],
            &[t_masked],
            none.0,
            none.1,
        )?;
        gb.node(
            "softmax_fwd_bf16",
            &p("softmax"),
            &[t_masked],
            &[t_pr],
            ps.0,
            ps.1,
        )?;
        gb.node("batch_gemm", &p("av"), &[t_pr, vco], &[t_at], pg.0, pg.1)?;
    }
    gb.node("reshape", &p("attn_2d"), &[t_at], &[t_attn], none.0, none.1)?;
    gb.node("gemm", &p("o_proj"), &[t_attn, t_wo], &[t_o], pgt.0, pgt.1)?;
    // Post-norm branches, as in `build_layer`.
    let t_res1 = if post_norm {
        let t_on = gb.mid(&p("on"), &[h, b], bf)?;
        let t_oinv = gb.mid(&p("oinv"), &[1, b], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("norm1"),
            &[t_o, t_g1],
            &[t_on, t_oinv],
            prm.0,
            prm.1,
        )?;
        t_on
    } else {
        t_o
    };
    // Gemma: the attention branch is normalised (g_post_attn) before its
    // residual add, on top of the pre-norm.
    let t_res1 = if w.g_post_attn.is_empty() {
        t_res1
    } else {
        post_norm_node(
            gb,
            &p,
            "post_attn_norm",
            w.g_post_attn,
            t_res1,
            &[h, b],
            &rms,
        )?
    };
    gb.node(
        "add_fwd_bf16",
        &p("res1"),
        &[x, t_res1],
        &[t_h],
        none.0,
        none.1,
    )?;
    let t_min = match (t_n2, t_inv2) {
        (Some(n2), Some(inv2)) => {
            gb.node(
                "rms_norm_fwd_bf16",
                &p("norm2"),
                &[t_h, t_g2],
                &[n2, inv2],
                prm.0,
                prm.1,
            )?;
            n2
        }
        _ => t_h,
    };
    gb.node(
        "gemm",
        &p("gate_proj"),
        &[t_min, t_wg],
        &[t_gate],
        pgt.0,
        pgt.1,
    )?;
    gb.node("gemm", &p("up_proj"), &[t_min, t_wu], &[t_up], pgt.0, pgt.1)?;
    activation_nodes(gb, &p, w.act, t_gate, &[i, b], t_act)?;
    gb.node(
        "mult_fwd_bf16",
        &p("gate_x_up"),
        &[t_act, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    gb.node(
        "gemm",
        &p("down_proj"),
        &[t_gated, t_wd],
        &[t_down],
        pgt.0,
        pgt.1,
    )?;
    let t_res2 = if post_norm {
        let t_dn = gb.mid(&p("dn"), &[h, b], bf)?;
        let t_dinv = gb.mid(&p("dinv"), &[1, b], SYN_TYPE_F32)?;
        gb.node(
            "rms_norm_fwd_bf16",
            &p("norm2"),
            &[t_down, t_g2],
            &[t_dn, t_dinv],
            prm.0,
            prm.1,
        )?;
        t_dn
    } else {
        t_down
    };
    // Gemma: the MLP branch is normalised (g_post_mlp) before its residual
    // add, on top of the pre-norm.
    let t_res2 = if w.g_post_mlp.is_empty() {
        t_res2
    } else {
        post_norm_node(gb, &p, "post_mlp_norm", w.g_post_mlp, t_res2, &[h, b], &rms)?
    };
    gb.node(
        "add_fwd_bf16",
        &p("res2"),
        &[t_h, t_res2],
        &[t_out],
        none.0,
        none.1,
    )?;
    Ok(t_out)
}

/// `ns_Reduction::Params` for the argmax over the FCD.
#[repr(C)]
struct ReductionParams {
    reduction_dimension: u32,
}

/// Append the final RMSNorm and LM head reading `cur`. Without `ids_out`
/// the logits `[vocab, tokens]` are the graph's read-back tensor; with it
/// the logits stay device-resident (the persistent scratch tensor
/// `LOGITS`, readable on demand) and an argmax over the vocabulary produces
/// the read-back tensor `IDS`, int32 `[1, tokens]`, so a decode step moves
/// four bytes per token over the bus. `lm` is the head's weight tensor
/// when the caller created it already (the device decode loop gathers
/// tied embeddings from it); else it is created here as input `LM`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_head<'a>(
    gb: &mut Gb<'a>,
    cur: synTensor,
    m: &ModelWeights<'a>,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    ids_out: bool,
    lm: Option<synTensor>,
) -> Result<Out> {
    let (t, h, v) = (tokens as u64, hidden as u64, vocab as u64);
    let bf = SYN_TYPE_BF16;
    let t_gf = gb.input("GF", &[h], m.final_gamma)?;
    let t_lm = match lm {
        Some(t) => t,
        None => lm_head_input(gb, m, hidden, vocab)?,
    };
    let t_nf = gb.mid("nf", &[h, t], bf)?;
    let t_invf = gb.mid("invf", &[1, t], SYN_TYPE_F32)?;
    let (t_logits, n_logits) = if ids_out {
        let t = gb.scratch("LOGITS", &[v, t])?;
        (t, CString::new("LOGITS").unwrap())
    } else {
        gb.output("LOGITS", &[v, t], bf)?
    };
    // Diagnostic `RENG_TPC_OUT`: route the logits through a TPC identity
    // (add 0) so a TPC kernel, not the MME, is the output's last writer.
    let tpc_out = env_on("RENG_TPC_OUT");
    let t_lg = if tpc_out {
        gb.mid("lg", &[v, t], bf)?
    } else {
        t_logits
    };
    // Gemma-2: the LM head gemm produces `logits / cap` (the final norm
    // gain carries `1 / cap`); `tanh` and a multiply by `cap` follow.
    let t_raw = match m.final_softcap {
        Some(_) => gb.mid("logits_raw", &[v, t], bf)?,
        None => t_lg,
    };
    let rms = RmsNormParams::new(m.layers[0].eps);
    gb.node(
        "rms_norm_fwd_bf16",
        "final_norm",
        &[cur, t_gf],
        &[t_nf, t_invf],
        (&raw const rms).cast::<c_void>(),
        core::mem::size_of::<RmsNormParams>() as u32,
    )?;
    let gemm_bt = synGEMMParams {
        transpose_a: false,
        transpose_b: true,
    };
    gb.node(
        "gemm",
        "lm_head",
        &[t_nf, t_lm],
        &[t_raw],
        (&raw const gemm_bt).cast::<c_void>(),
        core::mem::size_of::<synGEMMParams>() as u32,
    )?;
    if let Some(cap) = m.final_softcap {
        let none = (core::ptr::null::<c_void>(), 0u32);
        let t_cap = gb.input("FINAL_SOFTCAP", &[1, 1], &[cap])?;
        let t_th = gb.mid("logits_tanh", &[v, t], bf)?;
        gb.node(
            "tanh_fwd_bf16",
            "final_tanh",
            &[t_raw],
            &[t_th],
            none.0,
            none.1,
        )?;
        gb.node(
            "mult_fwd_bf16",
            "final_cap",
            &[t_th, t_cap],
            &[t_lg],
            none.0,
            none.1,
        )?;
    }
    if tpc_out {
        let t_zero = gb.input("ZERO", &[v, t], &vec![0.0; vocab * tokens])?;
        gb.node(
            "add_fwd_bf16",
            "logits_out",
            &[t_lg, t_zero],
            &[t_logits],
            core::ptr::null(),
            0,
        )?;
    }
    if !ids_out {
        return Ok(Out {
            name: n_logits,
            sizes: vec![v, t],
            kind: OutKind::Bf16,
        });
    }
    let (t_ids, n_ids) = gb.output("IDS", &[1, t], SYN_TYPE_INT32)?;
    let red = ReductionParams {
        reduction_dimension: 0,
    };
    // `argmax_fwd_bf16` is wrong for a single-row input (the decode shape)
    // whenever the row's maximum is small or negative: it returns 0 or an
    // index past the end (Phi-3 at 200 to 300 tokens; reng-argmax-test).
    // The f32 kernel is right in every case, so cast first.
    let t_lf32 = gb.mid("logits_f32", &[v, t], SYN_TYPE_F32)?;
    gb.node(
        "cast_bf16_to_f32",
        "logits_cast",
        &[t_logits],
        &[t_lf32],
        core::ptr::null(),
        0,
    )?;
    gb.node(
        "argmax_fwd_f32",
        "argmax",
        &[t_lf32],
        &[t_ids],
        (&raw const red).cast::<c_void>(),
        core::mem::size_of::<ReductionParams>() as u32,
    )?;
    Ok(Out {
        name: n_ids,
        sizes: vec![1, t],
        kind: OutKind::I32,
    })
}

/// The LM head weights as the graph input `LM`, bf16 `[hidden, vocab]`
/// (the checkpoint's `[vocab, hidden]` rows, borrowed).
pub(crate) fn lm_head_input<'a>(
    gb: &mut Gb<'a>,
    m: &ModelWeights<'a>,
    hidden: usize,
    vocab: usize,
) -> Result<synTensor> {
    gb.input_bf16(
        "LM",
        &[hidden as u64, vocab as u64],
        std::borrow::Cow::Borrowed(m.lm_head),
    )
}

/// The token embedding table for the device decode loop: bf16
/// `[vocab, hidden]` rows as the checkpoint stores them (the same slice as
/// [`ModelWeights::lm_head`] when the embeddings are tied, in which case
/// the loop gathers from the head's device copy), and the factor the
/// gathered row is multiplied by in f32 before its bf16 rounding (Gemma's
/// `sqrt(hidden)`, Granite's `embedding_multiplier`; 1 for the rest).
#[derive(Clone, Copy)]
pub struct EmbedTable<'a> {
    pub rows: &'a [u16],
    pub scale: f32,
}

/// RoPE tables `[positions, head_dim]` of a model for the cached and
/// batched decoders: the pair every layer reads, and the second pair the
/// layers with `local_rope` read (empty when no layer does).
#[derive(Clone, Copy)]
pub struct RopeTables<'a> {
    pub sin: &'a [f32],
    pub cos: &'a [f32],
    pub sin_local: &'a [f32],
    pub cos_local: &'a [f32],
}

impl<'a> RopeTables<'a> {
    /// One table pair, no local one.
    #[must_use]
    pub fn single(sin: &'a [f32], cos: &'a [f32]) -> Self {
        Self {
            sin,
            cos,
            sin_local: &[],
            cos_local: &[],
        }
    }
}

/// Whole-model weights. All layers share `layers[0]`'s head counts and
/// `eps`; a layer's RoPE caches (`sin`/`cos`, `[tokens, head_dim]`) are
/// one of at most two tables (the second for the `local_rope` layers), and
/// the windowed layers share one window.
#[derive(Clone)]
pub struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    /// Final RMSNorm gain, length `hidden`; with `final_softcap` it
    /// already carries the `1 / cap`.
    pub final_gamma: &'a [f32],
    /// LM head, bf16 `[vocab, hidden]` (the checkpoint's layout; tied
    /// embeddings as they are).
    pub lm_head: &'a [u16],
    /// Gemma-2 final logit softcap: the logits are `tanh(head / cap) *
    /// cap`, with the `1 / cap` folded into `final_gamma` by the caller.
    pub final_softcap: Option<f32>,
}

/// Build the shared inputs (activations, RoPE caches, causal masks) and the
/// decoder stack up to and including layer `upto`, returning the graph and the
/// last layer's output tensor. `probe_out` makes layer `upto` write into that
/// persistent tensor. The first RoPE table is the one of the first layer
/// without `local_rope`, the second that of the first layer with it; the
/// windowed layers' common window builds the second mask.
#[allow(clippy::too_many_arguments)]
fn build_stack<'a>(
    x: &[f32],
    m: &ModelWeights<'a>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
    probe_out: Option<synTensor>,
) -> Result<(Gb<'a>, synTensor)> {
    let l0 = &m.layers[0];
    let hd = l0.head_dim;
    let (t, h, hd64) = (tokens as u64, hidden as u64, hd as u64);
    let mut gb = Gb::new()?;
    let t_x = gb.input("X", &[h, t], x)?;
    let global = m.layers.iter().find(|l| !l.local_rope).unwrap_or(l0);
    assert_eq!(global.sin.len(), tokens * hd);
    let t_sin = gb.input("SIN", &[hd64, t], global.sin)?;
    let t_cos = gb.input("COS", &[hd64, t], global.cos)?;
    let (t_sin_local, t_cos_local) = match m.layers.iter().find(|l| l.local_rope) {
        Some(l) => {
            assert_eq!(l.sin.len(), tokens * hd);
            (
                Some(gb.input("SINL", &[hd64, t], l.sin)?),
                Some(gb.input("COSL", &[hd64, t], l.cos)?),
            )
        }
        None => (None, None),
    };
    // Causal masks laid out like the score matrix: [key (FCD), query]; a
    // key is visible when it is at or before the query and, for the
    // windowed layers, within the window.
    let mask_host = |window: Option<usize>| -> Vec<f32> {
        (0..tokens * tokens)
            .map(|idx| {
                let (q, k) = (idx / tokens, idx % tokens);
                if k <= q && window.is_none_or(|w| q - k < w) {
                    0.0
                } else {
                    MASK_NEG
                }
            })
            .collect()
    };
    let window = common_window(&m.layers);
    let t_mask = if causal && uses_full_mask(&m.layers) {
        Some(gb.input("MASK", &[t, t, 1, 1], &mask_host(None))?)
    } else {
        None
    };
    let t_mask_window = if causal && window.is_some() {
        Some(gb.input("MASKW", &[t, t, 1, 1], &mask_host(window))?)
    } else {
        None
    };
    let sh = Shared {
        sin: t_sin,
        cos: t_cos,
        sin_local: t_sin_local,
        cos_local: t_cos_local,
        mask: t_mask,
        mask_window: t_mask_window,
        cache: None,
        kidx: None,
        inplace: true,
        sdpa: fused_sdpa(false),
    };
    let persist = env_on("RENG_PERSIST_LAYERS");
    let mut cur = t_x;
    for (li, lw) in m.layers.iter().enumerate().take(upto + 1) {
        let out = if li == upto && probe_out.is_some() {
            probe_out
        } else if persist {
            Some(gb.scratch(&format!("l{li}_res"), &[h, t])?)
        } else {
            None
        };
        cur = build_layer(&mut gb, li, cur, lw, &sh, tokens, hidden, inter, out)?;
    }
    Ok((gb, cur))
}

/// Run the full forward pass on `x` (`[tokens, hidden]` embeddings, row-major)
/// and return logits `[tokens, vocab]` as f32. With `causal`, key positions
/// after the query are masked out in every layer (and, in the layers with a
/// window, the positions further back than it). `hidden`, `inter`, and
/// `vocab` should be at least 128 (per-head sizes may be smaller).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails or the output never completes.
///
/// # Panics
///
/// Panics if `layers` is empty or any buffer length disagrees with the sizes.
pub fn model_forward_bf16(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    assert!(!m.layers.is_empty());
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(m.final_gamma.len(), hidden);
    assert_eq!(m.lm_head.len(), hidden * vocab);
    let last = m.layers.len() - 1;
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, last, None)?;
    let out = build_head(&mut gb, cur, m, tokens, hidden, vocab, false, None)?;
    Runtime::new(gb, out)?.launch_and_read(tokens)
}

/// Diagnostic: build layers `0..=upto` and read back the residual stream after
/// layer `upto` (`[tokens, hidden]`, f32).
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails or the output never completes.
///
/// # Panics
///
/// Panics if `upto >= layers.len()` or a buffer length disagrees with the sizes.
pub fn model_probe_bf16(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
) -> Result<Vec<f32>> {
    assert!(upto < m.layers.len());
    assert_eq!(x.len(), tokens * hidden);
    let (t, h) = (tokens as u64, hidden as u64);
    // Build the stack, then expose the last layer's output through a TPC
    // identity into a dedicated persistent tensor that is read back.
    let (mut gb, cur) = build_stack(x, m, tokens, hidden, inter, causal, upto, None)?;
    let (t_probe, n_probe) = gb.output("PROBE", &[h, t], SYN_TYPE_BF16)?;
    let t_zero = gb.input("ZERO", &[h, t], &vec![0.0; hidden * tokens])?;
    gb.node(
        "add_fwd_bf16",
        "probe_out",
        &[cur, t_zero],
        &[t_probe],
        core::ptr::null(),
        0,
    )?;
    let out = Out {
        name: n_probe,
        sizes: vec![h, t],
        kind: OutKind::Bf16,
    };
    Runtime::new(gb, out)?.launch_and_read(tokens)
}

/// One decoder layer on the CPU (f32), with optional causal masking (and
/// the layer's sliding window, if any) and GQA.
#[must_use]
pub fn layer_cpu(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
) -> Vec<f32> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    let hd = w.head_dim;
    let qw = nh * hd;
    let kvd = nkv * hd;
    let n_rep = nh / nkv;
    let rmsnorm = |src: &[f32], g: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; tokens * hidden];
        for tk in 0..tokens {
            let b = tk * hidden;
            let ms = src[b..b + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + w.eps).sqrt();
            for f in 0..hidden {
                o[b + f] = src[b + f] * inv * g[f];
            }
        }
        o
    };
    // a[tokens, kin] @ W^T for W stored [kout, kin] (bf16) -> [tokens, kout]
    let matmul = |a: &[f32], mtx: &[u16], kin: usize, kout: usize| -> Vec<f32> {
        let wf: Vec<f32> = mtx.iter().map(|&b| bf16_to_f32(b)).collect();
        let mut o = vec![0.0f32; tokens * kout];
        for tk in 0..tokens {
            for c in 0..kout {
                let row = &wf[c * kin..(c + 1) * kin];
                o[tk * kout + c] = a[tk * kin..(tk + 1) * kin]
                    .iter()
                    .zip(row)
                    .map(|(x, y)| x * y)
                    .sum();
            }
        }
        o
    };
    // Rotate-half RoPE on head `head` of a [tokens, stride] tensor, written
    // into `out` (same layout).
    let rope_head = |src: &[f32], stride: usize, head: usize, out: &mut [f32]| {
        let half = hd / 2;
        for tk in 0..tokens {
            let b = tk * stride + head * hd;
            let c = tk * hd;
            for d in 0..hd {
                let rot = if d < half {
                    -src[b + d + half]
                } else {
                    src[b + d - half]
                };
                out[b + d] = src[b + d] * w.cos[c + d] + rot * w.sin[c + d];
            }
        }
    };

    // q/k RMSNorm of a `[tokens, stride]` tensor, in place: over every
    // `hd`-wide head with a `[hd]` gain (Qwen3), or over the whole row with
    // a `[stride]` gain (OLMo-2); a no-op without a gain.
    let head_norm = |m: &mut [f32], stride: usize, g: &[f32]| {
        if g.is_empty() {
            return;
        }
        let width = g.len();
        for row in m.chunks_exact_mut(stride) {
            for part in row.chunks_exact_mut(width) {
                let ms = part.iter().map(|v| v * v).sum::<f32>() / width as f32;
                let inv = 1.0 / (ms + w.eps).sqrt();
                for (v, gain) in part.iter_mut().zip(g) {
                    *v *= inv * gain;
                }
            }
        }
    };
    // Pre-norm feeds the normalised input to attention; post-norm (OLMo-2)
    // feeds the input itself and normalises each branch's output instead.
    let n1 = if w.post_norm {
        x.to_vec()
    } else {
        rmsnorm(x, w.g1)
    };
    // The scale rides on `wq` (and `bq`), or on the q norm gain when the
    // model has one (the norm would divide it out of `wq`).
    let scale = score_scale(w);
    let q_scale = if w.qn.is_empty() { scale } else { 1.0 };
    let wq_scaled = scaled_wq(w);
    let qn_scaled: Vec<f32> = w.qn.iter().map(|v| v * scale).collect();
    let bias = |mut m: Vec<f32>, b: &[f32], scale: f32| -> Vec<f32> {
        if !b.is_empty() {
            let cols = b.len();
            for (i, v) in m.iter_mut().enumerate() {
                *v += b[i % cols] * scale;
            }
        }
        m
    };
    let mut q = bias(matmul(&n1, &wq_scaled, hidden, qw), w.bq, q_scale);
    let mut k = bias(matmul(&n1, w.wk, hidden, kvd), w.bk, 1.0);
    let v = bias(matmul(&n1, w.wv, hidden, kvd), w.bv, 1.0);
    head_norm(&mut q, qw, &qn_scaled);
    head_norm(&mut k, kvd, w.kn);
    let mut qr = vec![0.0f32; tokens * qw];
    let mut kr = vec![0.0f32; tokens * kvd];
    if w.use_rope {
        for head in 0..nh {
            rope_head(&q, qw, head, &mut qr);
        }
        for g in 0..nkv {
            rope_head(&k, kvd, g, &mut kr);
        }
    } else {
        qr.copy_from_slice(&q);
        kr.copy_from_slice(&k);
    }
    let mut attn = vec![0.0f32; tokens * qw];
    let mut scores = vec![0.0f32; tokens];
    for head in 0..nh {
        let g = head / n_rep;
        let qoff = head * hd;
        let koff = g * hd;
        for qi in 0..tokens {
            // Keys `first..limit` are visible: up to the query (causal)
            // and, with a window, not further back than it.
            let (first, limit) = if causal {
                (
                    w.window.map_or(0, |win| (qi + 1).saturating_sub(win)),
                    qi + 1,
                )
            } else {
                (0, tokens)
            };
            for (ki, s) in scores.iter_mut().enumerate().take(limit).skip(first) {
                *s = (0..hd)
                    .map(|d| qr[qi * qw + qoff + d] * kr[ki * kvd + koff + d])
                    .sum();
                // Gemma-2 softcap; the scores already carry `1 / cap`.
                if let Some(cap) = w.attn_softcap {
                    *s = s.tanh() * cap;
                }
            }
            let mx = scores[first..limit]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores[first..limit] {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for (ki, s) in scores[..limit].iter().enumerate().skip(first) {
                let pr = s / sum;
                for d in 0..hd {
                    attn[qi * qw + qoff + d] += pr * v[ki * kvd + koff + d];
                }
            }
        }
    }
    let o = matmul(&attn, &col_view(w.wo, hidden, qw, w.wo_pitch), qw, hidden);
    let o = if w.post_norm { rmsnorm(&o, w.g1) } else { o };
    let o = if w.g_post_attn.is_empty() {
        o
    } else {
        rmsnorm(&o, w.g_post_attn)
    };
    let hres: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
    let n2 = if w.post_norm {
        hres.clone()
    } else {
        rmsnorm(&hres, w.g2)
    };
    let gate = matmul(&n2, w.wg, hidden, inter);
    let up = matmul(&n2, w.wu, hidden, inter);
    let act = |g: f32| match w.act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        Activation::GeluTanh => {
            0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh())
        }
    };
    let gated: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| act(*g) * u).collect();
    let down = matmul(
        &gated,
        &col_view(w.wd, hidden, inter, w.wd_pitch),
        inter,
        hidden,
    );
    let down = if w.post_norm {
        rmsnorm(&down, w.g2)
    } else {
        down
    };
    let down = if w.g_post_mlp.is_empty() {
        down
    } else {
        rmsnorm(&down, w.g_post_mlp)
    };
    hres.iter().zip(&down).map(|(a, b)| a + b).collect()
}

/// CPU reference for [`model_forward_bf16`]: logits `[tokens, vocab]`.
#[must_use]
pub fn model_forward_cpu(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    causal: bool,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for lw in &m.layers {
        cur = layer_cpu(&cur, lw, tokens, hidden, inter, causal);
    }
    let eps = m.layers[0].eps;
    let mut logits = vec![0.0f32; tokens * vocab];
    for tk in 0..tokens {
        let b = tk * hidden;
        let ms = cur[b..b + hidden].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for p in 0..hidden {
            let nv = cur[b + p] * inv * m.final_gamma[p];
            for c in 0..vocab {
                logits[tk * vocab + c] += nv * bf16_to_f32(m.lm_head[c * hidden + p]);
            }
        }
    }
    if let Some(cap) = m.final_softcap {
        for l in &mut logits {
            *l = l.tanh() * cap;
        }
    }
    logits
}

/// CPU reference for [`model_probe_bf16`]: residual stream after layer `upto`.
#[must_use]
pub fn model_probe_cpu(
    x: &[f32],
    m: &ModelWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
    causal: bool,
    upto: usize,
) -> Vec<f32> {
    let mut cur = x.to_vec();
    for lw in m.layers.iter().take(upto + 1) {
        cur = layer_cpu(&cur, lw, tokens, hidden, inter, causal);
    }
    cur
}
