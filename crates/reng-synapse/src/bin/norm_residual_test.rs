//! Probe the residual-fused RMSNorm guids: `rms_norm_fwd_residual_bf16`
//! with inputs (x, gamma, residual) and `ns_LayerNormKernel::ParamsRmsNormV3`
//! (`hasResidual`, `addOneToWeight`), against a host `rmsnorm(x + r) * g`.
//! A working form would fold each residual add into the following norm.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-norm-residual-test`

use core::ffi::c_void;
use reng_synapse::{NodeInput, run_node_extra_typed, run_node_pick};

/// `ns_LayerNormKernel::ParamsRmsNormV3` (C layout with padding).
#[repr(C)]
struct ParamsV3 {
    eps_valid: u8,
    _pad0: [u8; 3],
    eps: f32,
    norm_axis_bmp: i32,
    param_axis_bmp: i32,
    normalized_shape_dims: u32,
    fast_math: u8,
    _pad1: [u8; 3],
    retain_inv_rms_mode: u32,
    cl_aligned_pack_size: u32,
    has_residual: u8,
    add_one_to_weight: u8,
    _pad2: [u8; 2],
}

fn host(x: &[f32], r: Option<&[f32]>, g: &[f32], f: usize, eps: f32, plus_one: bool) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (t, row) in x.chunks(f).enumerate() {
        let v: Vec<f32> = row
            .iter()
            .enumerate()
            .map(|(i, a)| a + r.map_or(0.0, |r| r[t * f + i]))
            .collect();
        let ms = v.iter().map(|a| a * a).sum::<f32>() / f as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..f {
            let gain = if plus_one { 1.0 + g[i] } else { g[i] };
            out[t * f + i] = v[i] * inv * gain;
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

fn main() -> reng_core::Result<()> {
    let (f, t) = (256usize, 8usize);
    let x: Vec<f32> = (0..f * t)
        .map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) / 7.0)
        .collect();
    let r: Vec<f32> = (0..f * t)
        .map(|i| (((i * 11 + 5) % 23) as f32 - 11.0) / 9.0)
        .collect();
    let g: Vec<f32> = (0..f).map(|i| 0.5 + ((i % 11) as f32) / 10.0).collect();
    let eps = 1e-6f32;
    let xs = [f as u64, t as u64];
    let gs = [f as u64];
    // (guid, has_residual, add_one, with residual input)
    let cases: [(&str, u8, u8, bool); 4] = [
        ("rms_norm_fwd_residual_bf16", 1, 0, true),
        ("rms_norm_fwd_bf16", 1, 0, true),
        ("rms_norm_fwd_bf16", 0, 1, false),
        ("rms_norm_fwd_residual_bf16", 1, 1, true),
    ];
    for (guid, has_residual, add_one, with_r) in cases {
        let p = ParamsV3 {
            eps_valid: 1,
            _pad0: [0; 3],
            eps,
            norm_axis_bmp: 1,
            param_axis_bmp: 1,
            normalized_shape_dims: 1,
            fast_math: 0,
            _pad1: [0; 3],
            retain_inv_rms_mode: 0,
            cl_aligned_pack_size: 0,
            has_residual,
            add_one_to_weight: add_one,
            _pad2: [0; 2],
        };
        let mut ins = vec![
            NodeInput {
                name: "X",
                sizes: &xs,
                data: &x,
                raw: None,
            },
            NodeInput {
                name: "G",
                sizes: &gs,
                data: &g,
                raw: None,
            },
        ];
        if with_r {
            ins.push(NodeInput {
                name: "R",
                sizes: &xs,
                data: &r,
                raw: None,
            });
        }
        let extra: [(&str, &[u64], core::ffi::c_int); 1] = [("INV", &[1, t as u64], 1 << 2)];
        match run_node_extra_typed(
            guid,
            &ins,
            &xs,
            &extra,
            (&raw const p).cast::<c_void>(),
            core::mem::size_of::<ParamsV3>() as u32,
        ) {
            Ok(got) => {
                let want_r = host(&x, Some(&r), &g, f, eps, add_one == 1);
                let want_x = host(&x, None, &g, f, eps, add_one == 1);
                let want_plain = host(&x, None, &g, f, eps, false);
                println!(
                    "{guid} has_residual {has_residual} add_one {add_one} residual input {with_r}: rel_L2 vs norm(x+r) {:.4}, vs norm(x) {:.4}, vs plain norm(x) {:.4}",
                    rel_l2(&got, &want_r),
                    rel_l2(&got, &want_x),
                    rel_l2(&got, &want_plain)
                );
            }
            Err(e) => println!(
                "{guid} has_residual {has_residual} add_one {add_one} residual input {with_r}: rejected ({})",
                e.to_string().chars().take(60).collect::<String>()
            ),
        }
    }
    // Does the residual form also emit the sum x + r as a third output?
    let p = ParamsV3 {
        eps_valid: 1,
        _pad0: [0; 3],
        eps,
        norm_axis_bmp: 1,
        param_axis_bmp: 1,
        normalized_shape_dims: 1,
        fast_math: 0,
        _pad1: [0; 3],
        retain_inv_rms_mode: 0,
        cl_aligned_pack_size: 0,
        has_residual: 1,
        add_one_to_weight: 0,
        _pad2: [0; 2],
    };
    let ins = [
        NodeInput {
            name: "X",
            sizes: &xs,
            data: &x,
            raw: None,
        },
        NodeInput {
            name: "G",
            sizes: &gs,
            data: &g,
            raw: None,
        },
        NodeInput {
            name: "R",
            sizes: &xs,
            data: &r,
            raw: None,
        },
    ];
    let outs: [(&str, &[u64], core::ffi::c_int); 3] = [
        ("Y", &xs, 1 << 1),
        ("INV", &[1, t as u64], 1 << 2),
        ("SUM", &xs, 1 << 1),
    ];
    match run_node_pick(
        "rms_norm_fwd_bf16",
        &ins,
        &outs,
        2,
        (&raw const p).cast::<c_void>(),
        core::mem::size_of::<ParamsV3>() as u32,
    ) {
        Ok(got) => {
            let want: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
            println!(
                "rms_norm_fwd_bf16 has_residual with a third output: rel_L2 vs x + r {:.4} (first {:?} want {:?})",
                rel_l2(&got, &want),
                &got[..3],
                &want[..3]
            );
        }
        Err(e) => println!(
            "rms_norm_fwd_bf16 has_residual with a third output: rejected ({})",
            e.to_string().chars().take(70).collect::<String>()
        ),
    }
    Ok(())
}
