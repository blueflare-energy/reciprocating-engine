//! Pin the semantics of the `gemm` transpose flags by checking all four
//! `(transpose_a, transpose_b)` combinations against a CPU reference. A fused
//! attention graph needs `Q @ K^T` without a host-side transpose of an
//! intermediate, so the flag behaviour must be known exactly.
//!
//! `cargo run -p reng-synapse --features link-synapse --bin reng-gemm-t-test`.

use reng_synapse::Device;

/// `C[m,n] = op(A) @ op(B)` with `a` stored `[m,k]` (or `[k,m]` if `ta`) and
/// `b` stored `[k,n]` (or `[n,k]` if `tb`), all row-major.
fn cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                let av = if ta { a[p * m + i] } else { a[i * k + p] };
                let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                s += av * bv;
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn rel(h: &[f32], c: &[f32]) -> f32 {
    let num: f64 = h
        .iter()
        .zip(c)
        .map(|(x, y)| f64::from(*x - *y).powi(2))
        .sum();
    let den: f64 = c.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn main() -> reng_core::Result<()> {
    let (m, k, n) = (256usize, 384usize, 512usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| (((i * 5 + 1) % 17) as f32 - 8.0) / 8.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (((i * 3 + 2) % 19) as f32 - 9.0) / 9.0)
        .collect();

    // Optional arg selects one combo (0..3) so it runs as the only launch in
    // this process; several launches on one device race on this stack.
    let combos = [(false, false), (false, true), (true, false), (true, true)];
    let selected: Vec<(bool, bool)> = match std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(i) if i < combos.len() => vec![combos[i]],
        _ => combos.to_vec(),
    };

    let dev = Device::acquire()?;
    let mut worst = 0.0f32;
    for (ta, tb) in selected {
        let hpu = dev.gemm_ex(&a, &b, m, k, n, ta, tb)?;
        let r = rel(&hpu, &cpu(&a, &b, m, k, n, ta, tb));
        worst = worst.max(r);
        println!("transpose_a={ta} transpose_b={tb}: rel_L2={r:.4}");
    }
    if worst < 0.05 {
        println!("PASS");
        Ok(())
    } else {
        Err(reng_core::Error::Other(format!(
            "gemm transpose semantics mismatch, worst rel_L2 {worst}"
        )))
    }
}
