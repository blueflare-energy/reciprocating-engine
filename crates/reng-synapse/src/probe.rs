//! Run a single vendor node as its own recipe and read its output back with
//! the full readback protocol. This is the tool for pinning down a kernel's
//! contract (tensor ranks, layouts, broadcasting, parameter structs) before
//! it goes into a model graph; every `reng-*-test` for a new guid uses it.

use crate::ffi::SYN_TYPE_BF16;
use crate::model::{Gb, make_tensor};
use crate::runtime::{Out, OutKind, Runtime};
use core::ffi::c_void;
use reng_core::Result;

/// One input of [`run_node`]: FCD-first device sizes and row-major host data
/// (bf16 on the device), or raw bytes of another dtype via `raw`.
pub struct NodeInput<'a> {
    pub name: &'a str,
    pub sizes: &'a [u64],
    pub data: &'a [f32],
    /// `(dtype, bytes)` for a non-bf16 input; `data` is then ignored.
    pub raw: Option<(core::ffi::c_int, &'a [u8])>,
}

/// Synapse dtype code for int32 (`syn_type_int32`).
pub use crate::ffi::SYN_TYPE_INT32;

/// Build a graph with the single node `guid` over `ins`, producing one bf16
/// output of `out_sizes`, run it once, and return the output as f32.
///
/// # Errors
///
/// Returns an error if graph construction, compilation or the run fails
/// (an unregistered guid or a rejected contract fails at node creation or
/// compilation).
///
/// # Panics
///
/// Panics if an input's data length disagrees with its sizes.
pub fn run_node(
    guid: &str,
    ins: &[NodeInput<'_>],
    out_sizes: &[u64],
    params: *const c_void,
    params_size: u32,
) -> Result<Vec<f32>> {
    let mut rt = build_probe(
        guid,
        ins,
        out_sizes,
        crate::ffi::SYN_TYPE_BF16,
        OutKind::Bf16,
        params,
        params_size,
    )?;
    let rows = *out_sizes.last().unwrap_or(&1) as usize;
    rt.launch_and_read(rows)
}

/// Like [`run_node`] but with an int32 output.
///
/// # Errors
///
/// As [`run_node`].
pub fn run_node_i32(
    guid: &str,
    ins: &[NodeInput<'_>],
    out_sizes: &[u64],
    params: *const c_void,
    params_size: u32,
) -> Result<Vec<i32>> {
    let mut rt = build_probe(
        guid,
        ins,
        out_sizes,
        SYN_TYPE_INT32,
        OutKind::I32,
        params,
        params_size,
    )?;
    let rows = *out_sizes.last().unwrap_or(&1) as usize;
    rt.launch_and_read_i32(0, rows)
}

/// Time one node: compile it once, launch it `iters` times back to back,
/// and return the seconds per launch together with the first output row
/// (for a spot check). The final launch is read back, which also waits for
/// the earlier ones.
///
/// # Errors
///
/// Returns an error if any SynapseAI call fails.
pub fn bench_node(
    guid: &str,
    ins: &[NodeInput<'_>],
    out_sizes: &[u64],
    params: *const c_void,
    params_size: u32,
    iters: usize,
) -> Result<(f64, Vec<f32>)> {
    let mut rt = build_probe(
        guid,
        ins,
        out_sizes,
        SYN_TYPE_BF16,
        OutKind::Bf16,
        params,
        params_size,
    )?;
    for _ in 0..3 {
        rt.launch_and_read(1)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        rt.launch_only()?;
    }
    let row = rt.launch_and_read(1)?;
    let secs = t0.elapsed().as_secs_f64() / (iters as f64 + 1.0);
    Ok((secs, row))
}

#[allow(clippy::too_many_arguments)]
fn build_probe(
    guid: &str,
    ins: &[NodeInput<'_>],
    out_sizes: &[u64],
    out_dtype: core::ffi::c_int,
    kind: OutKind,
    params: *const c_void,
    params_size: u32,
) -> Result<Runtime> {
    let mut gb = Gb::new()?;
    let mut tensors = Vec::with_capacity(ins.len());
    for i in ins {
        if let Some((dtype, bytes)) = i.raw {
            tensors.push(gb.input_raw(i.name, i.sizes, dtype, bytes)?);
        } else {
            assert_eq!(i.sizes.iter().product::<u64>() as usize, i.data.len());
            tensors.push(gb.input(i.name, i.sizes, i.data)?);
        }
    }
    let (t_out, n_out) = make_tensor(gb.graph, "OUT", out_sizes, out_dtype, true)?;
    gb.node(guid, "probe", &tensors, &[t_out], params, params_size)?;
    let out = Out {
        name: n_out,
        sizes: out_sizes.to_vec(),
        kind,
    };
    Runtime::new(gb, out)
}
