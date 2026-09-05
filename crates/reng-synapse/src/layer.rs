//! A full pre-norm transformer decoder layer as ONE fused SynapseAI recipe:
//!
//! ```text
//! n1  = rmsnorm(x, g1)
//! q,k,v = n1 @ Wq (pre-scaled), n1 @ Wk, n1 @ Wv      (k, v have n_kv heads)
//! per query head j, with kv head g = j / (n_heads / n_kv):
//!   qr_j, kr_g = rope(q_j), rope(k_g)
//!   attn_j = softmax(qr_j @ kr_g^T) @ v_g
//! attn = concat_j(attn_j)
//! h    = x + attn @ Wo
//! n2   = rmsnorm(h, g2)
//! out  = h + down( silu(n2 @ Wg) * (n2 @ Wu) )
//! ```
//!
//! One graph, one launch. Activations never leave HBM; only the
//! `[tokens, hidden]` output is read back. Heads are split with the compiler's
//! `split` node and merged with `concat` (axis 0 = the feature FCD), so every
//! matmul stays a verified 2D gemm. Grouped-query attention (fewer K/V heads
//! than query heads) maps consecutive query heads onto one K/V head, matching
//! the HF `repeat_kv` convention. Attention is non-causal here; a causal mask
//! is one additive node away (see `model.rs`).

use crate::ffi::*;
use crate::{bf16_to_f32, f32_to_bf16};
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

/// Weights and inputs of one decoder layer, all row-major f32.
pub struct LayerWeights<'a> {
    /// Number of query heads; `hidden % n_heads == 0`.
    pub n_heads: usize,
    /// Number of key/value heads (GQA); `n_heads % n_kv_heads == 0`.
    pub n_kv_heads: usize,
    /// RMSNorm gains, each length `hidden`.
    pub g1: &'a [f32],
    pub g2: &'a [f32],
    /// Projections stored `[in, out]`: `wq`, `wo` are `hidden x hidden`;
    /// `wk`, `wv` are `hidden x (n_kv_heads * head_dim)`.
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    /// MLP: `wg`, `wu` are `[hidden, inter]`; `wd` is `[inter, hidden]`.
    pub wg: &'a [f32],
    pub wu: &'a [f32],
    pub wd: &'a [f32],
    /// RoPE caches `[tokens, head_dim]` (head_dim contiguous), shared by heads.
    pub sin: &'a [f32],
    pub cos: &'a [f32],
    /// Attention scale (normally `1/sqrt(head_dim)`), folded into `wq`.
    pub scale: f32,
    pub eps: f32,
}

fn tensor(
    graph: synGraphHandle,
    name: &str,
    sizes: &[u64],
    dtype: core::ffi::c_int,
    persistent: bool,
) -> Result<(synTensor, CString)> {
    let cname = CString::new(name).unwrap();
    let mut t: synTensor = core::ptr::null_mut();
    syn!(synTensorHandleCreate(
        &mut t,
        graph,
        SYN_TENSOR_DATA,
        cname.as_ptr()
    ));
    if persistent {
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        syn!(synTensorAssignToSection(t, sec, 0));
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

fn node(
    graph: synGraphHandle,
    guid: &str,
    name: &str,
    ins: &[synTensor],
    outs: &[synTensor],
    params: *const c_void,
    params_size: u32,
) -> Result<()> {
    let g = CString::new(guid).unwrap();
    let n = CString::new(name).unwrap();
    syn!(synNodeCreate(
        graph,
        ins.as_ptr(),
        outs.as_ptr(),
        ins.len() as u32,
        outs.len() as u32,
        params,
        params_size,
        g.as_ptr(),
        n.as_ptr(),
        core::ptr::null(),
        core::ptr::null(),
    ));
    Ok(())
}

#[repr(C)]
struct RmsNormParams {
    epsilon: f32,
    fused_gamma_beta: bool,
    use_stages: bool,
    bwd_mode: i32,
}

#[repr(C)]
struct RopeParams {
    offset: u32,
    mode: i32,
}

#[repr(C)]
struct AxisParams {
    axis: u32,
}

/// Run one fused decoder layer on `x` (`[tokens, hidden]`, row-major) and
/// return `out` (`[tokens, hidden]`) as f32. `hidden`, `inter`, and `tokens`
/// should be at least 128 (see [`crate::Device`]); per-head sizes may be
/// smaller because they never leave the recipe.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
///
/// # Panics
///
/// Panics if any buffer length disagrees with the sizes, if `hidden` is not a
/// multiple of `n_heads`, or if `n_heads` is not a multiple of `n_kv_heads`.
#[allow(clippy::too_many_lines)]
pub fn decoder_layer_bf16(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let (nh, nkv) = (w.n_heads, w.n_kv_heads);
    assert!(nh >= 1 && hidden % nh == 0);
    assert!(nkv >= 1 && nh % nkv == 0);
    let head_dim = hidden / nh;
    let kvd_us = nkv * head_dim;
    let n_rep = nh / nkv;
    let (t, h, i, hd, kvd) = (
        tokens as u64,
        hidden as u64,
        inter as u64,
        head_dim as u64,
        kvd_us as u64,
    );
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(w.wq.len(), hidden * hidden);
    assert_eq!(w.wo.len(), hidden * hidden);
    assert_eq!(w.wk.len(), hidden * kvd_us);
    assert_eq!(w.wv.len(), hidden * kvd_us);
    assert_eq!(w.wg.len(), hidden * inter);
    assert_eq!(w.wu.len(), hidden * inter);
    assert_eq!(w.wd.len(), inter * hidden);
    for v in [w.g1, w.g2] {
        assert_eq!(v.len(), hidden);
    }
    assert_eq!(w.sin.len(), tokens * head_dim);
    assert_eq!(w.cos.len(), tokens * head_dim);
    let wq_scaled: Vec<f32> = w.wq.iter().map(|v| v * w.scale).collect();

    syn!(synInitialize());
    let mut graph: synGraphHandle = core::ptr::null_mut();
    syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

    // Persistent I/O (FCD-first). Activations are [feature, tokens]; a gemm
    // weight stored [in, out] on the host is [out, in] on the device.
    let bf = SYN_TYPE_BF16;
    let io: [(&str, Vec<u64>, &[f32]); 12] = [
        ("X", vec![h, t], x),
        ("G1", vec![h], w.g1),
        ("G2", vec![h], w.g2),
        ("WQ", vec![h, h], &wq_scaled),
        ("WK", vec![kvd, h], w.wk),
        ("WV", vec![kvd, h], w.wv),
        ("WO", vec![h, h], w.wo),
        ("SIN", vec![hd, t], w.sin),
        ("COS", vec![hd, t], w.cos),
        ("WG", vec![i, h], w.wg),
        ("WU", vec![i, h], w.wu),
        ("WD", vec![h, i], w.wd),
    ];
    let mut in_t: Vec<synTensor> = Vec::with_capacity(io.len());
    let mut in_names: Vec<CString> = Vec::with_capacity(io.len());
    for (name, sizes, _) in &io {
        let (tt, cn) = tensor(graph, name, sizes, bf, true)?;
        in_t.push(tt);
        in_names.push(cn);
    }
    let (t_out, n_out) = tensor(graph, "OUT", &[h, t], bf, true)?;
    let [
        t_x,
        t_g1,
        t_g2,
        t_wq,
        t_wk,
        t_wv,
        t_wo,
        t_sin,
        t_cos,
        t_wg,
        t_wu,
        t_wd,
    ]: [synTensor; 12] = in_t.clone().try_into().unwrap();

    // Graph-internal intermediates.
    let mid = |name: &str, sizes: &[u64]| -> Result<synTensor> {
        Ok(tensor(graph, name, sizes, bf, false)?.0)
    };
    let t_n1 = mid("n1", &[h, t])?;
    let t_q = mid("q", &[h, t])?;
    let t_k = mid("k", &[kvd, t])?;
    let t_v = mid("v", &[kvd, t])?;
    let t_attn = mid("attn", &[h, t])?;
    let t_o = mid("o", &[h, t])?;
    let t_h = mid("h", &[h, t])?;
    let t_n2 = mid("n2", &[h, t])?;
    let t_gate = mid("gate", &[i, t])?;
    let t_up = mid("up", &[i, t])?;
    let t_sg = mid("sg", &[i, t])?;
    let t_silu = mid("silu", &[i, t])?;
    let t_gated = mid("gated", &[i, t])?;
    let t_down = mid("down", &[h, t])?;
    // RMSNorm's second output (inverse RMS) must be f32.
    let t_inv1 = tensor(graph, "inv1", &[1, t], SYN_TYPE_F32, false)?.0;
    let t_inv2 = tensor(graph, "inv2", &[1, t], SYN_TYPE_F32, false)?.0;
    let heads = |prefix: &str, count: usize, sizes: &[u64]| -> Result<Vec<synTensor>> {
        (0..count)
            .map(|j| mid(&format!("{prefix}{j}"), sizes))
            .collect()
    };
    let qr_h = heads("qr", nh, &[hd, t])?;
    let kr_h = heads("kr", nkv, &[hd, t])?;
    let sc_h = heads("scores", nh, &[t, t])?;
    let pr_h = heads("probs", nh, &[t, t])?;
    let at_h = heads("attn", nh, &[hd, t])?;

    let rms = RmsNormParams {
        epsilon: w.eps,
        fused_gamma_beta: false,
        use_stages: false,
        bwd_mode: 0,
    };
    let rope = RopeParams { offset: 0, mode: 0 };
    let axis0 = AxisParams { axis: 0 };
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
    let pax = (
        (&raw const axis0).cast::<c_void>(),
        core::mem::size_of::<AxisParams>() as u32,
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
    let none = (core::ptr::null::<c_void>(), 0u32);

    // Attention block.
    node(
        graph,
        "rms_norm_fwd_bf16",
        "norm1",
        &[t_x, t_g1],
        &[t_n1, t_inv1],
        prm.0,
        prm.1,
    )?;
    node(graph, "gemm", "q_proj", &[t_n1, t_wq], &[t_q], pg.0, pg.1)?;
    node(graph, "gemm", "k_proj", &[t_n1, t_wk], &[t_k], pg.0, pg.1)?;
    node(graph, "gemm", "v_proj", &[t_n1, t_wv], &[t_v], pg.0, pg.1)?;
    let qs = if nh > 1 {
        let q_h = heads("q", nh, &[hd, t])?;
        node(graph, "split", "split_q", &[t_q], &q_h, pax.0, pax.1)?;
        q_h
    } else {
        vec![t_q]
    };
    let (ks, vs) = if nkv > 1 {
        let k_h = heads("k", nkv, &[hd, t])?;
        let v_h = heads("v", nkv, &[hd, t])?;
        node(graph, "split", "split_k", &[t_k], &k_h, pax.0, pax.1)?;
        node(graph, "split", "split_v", &[t_v], &v_h, pax.0, pax.1)?;
        (k_h, v_h)
    } else {
        (vec![t_k], vec![t_v])
    };
    for g in 0..nkv {
        node(
            graph,
            "rope_st2_fwd_bf16",
            &format!("rope_k{g}"),
            &[ks[g], t_sin, t_cos],
            &[kr_h[g]],
            pr.0,
            pr.1,
        )?;
    }
    for j in 0..nh {
        let g = j / n_rep;
        node(
            graph,
            "rope_st2_fwd_bf16",
            &format!("rope_q{j}"),
            &[qs[j], t_sin, t_cos],
            &[qr_h[j]],
            pr.0,
            pr.1,
        )?;
        // scores[query, key] = qr @ kr^T: K in its natural [seq, head_dim] layout.
        node(
            graph,
            "gemm",
            &format!("qk{j}"),
            &[qr_h[j], kr_h[g]],
            &[sc_h[j]],
            pgt.0,
            pgt.1,
        )?;
        node(
            graph,
            "softmax_fwd_bf16",
            &format!("softmax{j}"),
            &[sc_h[j]],
            &[pr_h[j]],
            ps.0,
            ps.1,
        )?;
        node(
            graph,
            "gemm",
            &format!("av{j}"),
            &[pr_h[j], vs[g]],
            &[at_h[j]],
            pg.0,
            pg.1,
        )?;
    }
    let attn_full = if nh > 1 {
        node(
            graph,
            "concat",
            "merge_heads",
            &at_h,
            &[t_attn],
            pax.0,
            pax.1,
        )?;
        t_attn
    } else {
        at_h[0]
    };
    node(
        graph,
        "gemm",
        "o_proj",
        &[attn_full, t_wo],
        &[t_o],
        pg.0,
        pg.1,
    )?;
    node(
        graph,
        "add_fwd_bf16",
        "res1",
        &[t_x, t_o],
        &[t_h],
        none.0,
        none.1,
    )?;
    // MLP block.
    node(
        graph,
        "rms_norm_fwd_bf16",
        "norm2",
        &[t_h, t_g2],
        &[t_n2, t_inv2],
        prm.0,
        prm.1,
    )?;
    node(
        graph,
        "gemm",
        "gate_proj",
        &[t_n2, t_wg],
        &[t_gate],
        pg.0,
        pg.1,
    )?;
    node(graph, "gemm", "up_proj", &[t_n2, t_wu], &[t_up], pg.0, pg.1)?;
    node(
        graph,
        "sigmoid_fwd_bf16",
        "sig",
        &[t_gate],
        &[t_sg],
        none.0,
        none.1,
    )?;
    node(
        graph,
        "mult_fwd_bf16",
        "silu",
        &[t_gate, t_sg],
        &[t_silu],
        none.0,
        none.1,
    )?;
    node(
        graph,
        "mult_fwd_bf16",
        "gate_x_up",
        &[t_silu, t_up],
        &[t_gated],
        none.0,
        none.1,
    )?;
    node(
        graph,
        "gemm",
        "down_proj",
        &[t_gated, t_wd],
        &[t_down],
        pg.0,
        pg.1,
    )?;
    node(
        graph,
        "add_fwd_bf16",
        "res2",
        &[t_h, t_down],
        &[t_out],
        none.0,
        none.1,
    )?;

    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    syn!(synGraphCompile(
        &mut recipe,
        graph,
        CString::new("decoder_layer").unwrap().as_ptr(),
        core::ptr::null()
    ));

    let mut name_ptrs: Vec<*const core::ffi::c_char> =
        in_names.iter().map(|n| n.as_ptr()).collect();
    name_ptrs.push(n_out.as_ptr());
    let mut ids = vec![0u64; name_ptrs.len()];
    syn!(synTensorRetrieveIds(
        recipe,
        name_ptrs.as_ptr(),
        ids.as_mut_ptr(),
        name_ptrs.len() as u32
    ));

    let mut dev: synDeviceId = 0;
    syn!(synDeviceAcquireByDeviceType(&mut dev, SYN_DEVICE_GAUDI2));
    let mut stream: synStreamHandle = core::ptr::null_mut();
    syn!(synStreamCreateGeneric(&mut stream, dev, 0));

    // Device + pinned host buffers for every input, upload, and launch infos.
    let mut dev_bufs: Vec<u64> = Vec::with_capacity(io.len() + 1);
    let mut host_bufs: Vec<*mut c_void> = Vec::with_capacity(io.len() + 1);
    let mut infos: Vec<synLaunchTensorInfo> = Vec::with_capacity(io.len() + 1);
    for (idx, (_, sizes, data)) in io.iter().enumerate() {
        let bytes = (data.len() * 2) as u64;
        let mut d = 0u64;
        syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
        let mut hb: *mut c_void = core::ptr::null_mut();
        syn!(synHostMalloc(dev, bytes, 0, &mut hb));
        // SAFETY: hb holds data.len() bf16 elements.
        unsafe {
            let pb = hb.cast::<u16>();
            for (j, &v) in data.iter().enumerate() {
                *pb.add(j) = f32_to_bf16(v);
            }
        }
        syn!(synMemCopyAsync(
            stream,
            hb as u64,
            bytes,
            d,
            SYN_HOST_TO_DRAM
        ));
        let mut ti = synLaunchTensorInfo {
            tensor_name: in_names[idx].as_ptr(),
            tensor_address: d,
            tensor_type: SYN_TENSOR_DATA,
            tensor_size: [0; HABANA_DIM_MAX],
            tensor_id: ids[idx],
        };
        ti.tensor_size[..sizes.len()].copy_from_slice(sizes);
        infos.push(ti);
        dev_bufs.push(d);
        host_bufs.push(hb);
    }
    let out_bytes = (tokens * hidden * 2) as u64;
    let mut d_out = 0u64;
    syn!(synDeviceMalloc(dev, out_bytes, 0, 0, &mut d_out));
    let mut h_out: *mut c_void = core::ptr::null_mut();
    syn!(synHostMalloc(dev, out_bytes, 0, &mut h_out));
    let mut ti = synLaunchTensorInfo {
        tensor_name: n_out.as_ptr(),
        tensor_address: d_out,
        tensor_type: SYN_TENSOR_DATA,
        tensor_size: [0; HABANA_DIM_MAX],
        tensor_id: ids[io.len()],
    };
    ti.tensor_size[0] = h;
    ti.tensor_size[1] = t;
    infos.push(ti);

    let mut ws = 0u64;
    syn!(synWorkspaceGetSize(&mut ws, recipe));
    let mut dws = 0u64;
    if ws > 0 {
        syn!(synDeviceMalloc(dev, ws, 0, 0, &mut dws));
    }
    syn!(synStreamSynchronize(stream));

    syn!(synLaunch(
        stream,
        infos.as_ptr(),
        infos.len() as u32,
        dws,
        recipe,
        0
    ));
    syn!(synStreamSynchronize(stream));
    syn!(synMemCopyAsync(
        stream,
        d_out,
        out_bytes,
        h_out as u64,
        SYN_DRAM_TO_HOST
    ));
    syn!(synStreamSynchronize(stream));

    let mut out = vec![0.0f32; tokens * hidden];
    // SAFETY: h_out holds tokens*hidden bf16 elements just copied back.
    unsafe {
        let po = h_out.cast::<u16>();
        for (j, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(*po.add(j));
        }
    }

    unsafe {
        for hb in host_bufs {
            synHostFree(dev, hb, 0);
        }
        synHostFree(dev, h_out, 0);
        for d in dev_bufs {
            synDeviceFree(dev, d, 0);
        }
        synDeviceFree(dev, d_out, 0);
        if dws != 0 {
            synDeviceFree(dev, dws, 0);
        }
        synStreamDestroy(stream);
        synRecipeDestroy(recipe);
        synGraphDestroy(graph);
        synDeviceRelease(dev);
        synDestroy();
    }
    Ok(out)
}

/// CPU reference for [`decoder_layer_bf16`] (f32, same layouts, non-causal).
#[must_use]
pub fn decoder_layer_cpu(
    x: &[f32],
    w: &LayerWeights<'_>,
    tokens: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    crate::layer_cpu(x, w, tokens, hidden, inter, false)
}
