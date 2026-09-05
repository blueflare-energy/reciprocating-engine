//! A compiled recipe with its device buffers: the launch and readback side of
//! the fused-model graphs in `model.rs`. One `Runtime` is compiled once and
//! can be launched many times (the KV-cache decode loop); the one-shot paths
//! build one, launch it once, and drop it.
//!
//! Readback protocol. On this stack `synStreamSynchronize` returns before
//! the work it covers has landed: a deep recipe can still be writing its
//! output, and a device-to-host copy can still be in flight, so a plain
//! read of the pinned host buffer shows whatever was there before (zeros on
//! a fresh buffer, the previous step's data on a reused one). Two sentinels
//! make the read exact without any timed wait:
//!
//! 1. the device output is pre-filled with one NaN pattern before launch,
//!    so an element that still shows it was not written by the recipe;
//! 2. the host buffer is filled with a different NaN pattern before every
//!    copy, so an element that still shows it was not written by the DMA.
//!
//! After the copy the host spins until no host sentinel remains, then
//! repeats the copy while any device sentinel remains. Every element of the
//! output is written exactly once by the recipe (its last writer is an MME
//! gemm or a TPC kernel) and exactly once by the copy, so "no sentinel of
//! either kind" is completion. `RENG_STABILITY_MS` adds a diagnostic
//! re-read-until-stable pass on top.

use crate::ffi::*;
use crate::model::Gb;
use crate::{bf16_to_f32, f32_to_bf16};
use core::ffi::c_void;
use reng_core::{Error, Result};
use std::collections::HashMap;
use std::ffi::CString;
use std::time::{Duration, Instant};

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

/// bf16 quiet-NaN pattern the DEVICE output buffer is pre-filled with before
/// launch. The recipe writes every output element exactly once and never
/// produces this exact NaN, so "none left" means the recipe has written the
/// whole output.
const SENTINEL_BF16: u16 = 0x7FC1;
const SENTINEL_D32: u32 = 0x7FC1_7FC1;
/// A second quiet-NaN pattern the HOST buffer is filled with before every
/// device-to-host copy; "none left" means the copy has landed. Distinct from
/// the device sentinel so the two conditions stay separable.
const HOST_SENTINEL_BF16: u16 = 0x7FC2;
/// Upper bound on waiting for a recipe's output to complete.
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Diagnostic re-read window (`RENG_STABILITY_MS`, fractional): when set,
/// after both sentinels are gone the output is re-read until two reads
/// spaced by this many milliseconds are byte-identical. Used to prove that
/// the sentinel protocol alone is complete.
fn stability_window() -> Option<Duration> {
    std::env::var("RENG_STABILITY_MS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(Duration::from_secs_f64)
        .map(|d| d / 1000)
}

fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok()
}

/// The single persistent tensor a graph run reads back, `[fcd, rows]`.
pub(crate) struct Out {
    pub name: CString,
    pub sizes: Vec<u64>,
}

impl Out {
    fn elems(&self) -> usize {
        self.sizes.iter().product::<u64>() as usize
    }

    /// Elements per outermost row.
    fn row_elems(&self) -> usize {
        self.elems() / *self.sizes.last().unwrap_or(&1) as usize
    }
}

fn launch_info(name: &CString, addr: u64, id: u64, sizes: &[u64]) -> synLaunchTensorInfo {
    let mut ti = synLaunchTensorInfo {
        tensor_name: name.as_ptr(),
        tensor_address: addr,
        tensor_type: SYN_TENSOR_DATA,
        tensor_size: [0; HABANA_DIM_MAX],
        tensor_id: id,
    };
    ti.tensor_size[..sizes.len()].copy_from_slice(sizes);
    // A 1-D tensor (RMSNorm gain) is launched as [n, 1], never [n, 0].
    for d in sizes.len()..2 {
        ti.tensor_size[d] = 1;
    }
    ti
}

/// A compiled recipe bound to device buffers for all of its persistent
/// tensors. Inputs are uploaded once at construction; [`Runtime::upload`]
/// replaces one input's contents between launches, and
/// [`Runtime::copy_d2d`] moves bytes between device-resident tensors.
pub(crate) struct Runtime {
    gb: Gb,
    dev: synDeviceId,
    stream: synStreamHandle,
    recipe: synRecipeHandle,
    infos: Vec<synLaunchTensorInfo>,
    /// Index into `infos` per persistent tensor, by name.
    info_index: HashMap<String, usize>,
    /// Device buffer per persistent tensor, by name.
    addrs: HashMap<String, u64>,
    /// Pinned host staging buffer per uploaded input, by input index.
    host_bufs: Vec<*mut c_void>,
    dev_bufs: Vec<u64>,
    d_out: u64,
    h_out: *mut c_void,
    /// Pinned buffer for [`Runtime::fence_uploads`], grown on demand.
    h_fence: *mut c_void,
    fence_bytes: u64,
    out: Out,
    dws: u64,
}

impl Runtime {
    /// Compile `gb`, acquire a device, allocate every persistent tensor, and
    /// upload the inputs' host data.
    #[allow(clippy::too_many_lines)]
    pub fn new(gb: Gb, out: Out) -> Result<Self> {
        gb.serialize_if_requested()?;
        let mut recipe: synRecipeHandle = core::ptr::null_mut();
        syn!(synGraphCompile(
            &mut recipe,
            gb.graph,
            CString::new("model").unwrap().as_ptr(),
            core::ptr::null()
        ));

        // Tensor ids: inputs, then scratch, then the output.
        let mut name_ptrs: Vec<*const core::ffi::c_char> =
            gb.names.iter().map(|n| n.as_ptr()).collect();
        name_ptrs.extend(gb.scratch_names.iter().map(|n| n.as_ptr()));
        name_ptrs.push(out.name.as_ptr());
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

        let n_in = gb.names.len();
        let n_scratch = gb.scratch_names.len();
        let mut dev_bufs: Vec<u64> = Vec::with_capacity(n_in + n_scratch);
        let mut host_bufs: Vec<*mut c_void> = Vec::with_capacity(n_in);
        let mut infos: Vec<synLaunchTensorInfo> = Vec::with_capacity(n_in + n_scratch + 1);
        let mut addrs = HashMap::with_capacity(n_in + n_scratch + 1);
        let mut info_index = HashMap::with_capacity(n_in + n_scratch + 1);
        for (idx, data) in gb.data.iter().enumerate() {
            let bytes = (data.len() * 2) as u64;
            let mut d = 0u64;
            syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
            let mut hb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(dev, bytes, 0, &mut hb));
            // SAFETY: hb holds data.len() bf16 elements.
            unsafe {
                let pb = hb.cast::<u16>();
                for (j, &val) in data.iter().enumerate() {
                    *pb.add(j) = f32_to_bf16(val);
                }
            }
            syn!(synMemCopyAsync(
                stream,
                hb as u64,
                bytes,
                d,
                SYN_HOST_TO_DRAM
            ));
            info_index.insert(gb.names[idx].to_str().unwrap().to_owned(), infos.len());
            infos.push(launch_info(&gb.names[idx], d, ids[idx], &gb.sizes[idx]));
            addrs.insert(gb.names[idx].to_str().unwrap().to_owned(), d);
            dev_bufs.push(d);
            host_bufs.push(hb);
        }
        for (k, sizes) in gb.scratch_sizes.iter().enumerate() {
            let bytes = sizes.iter().product::<u64>() * if gb.scratch_f32[k] { 4 } else { 2 };
            let mut d = 0u64;
            syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
            // Device-resident state starts as zeros (finite, so a masked-out
            // stale cache row can never poison a softmax).
            syn!(synMemsetD32Async(d, 0, (bytes / 4) as usize, stream));
            info_index.insert(
                gb.scratch_names[k].to_str().unwrap().to_owned(),
                infos.len(),
            );
            infos.push(launch_info(&gb.scratch_names[k], d, ids[n_in + k], sizes));
            addrs.insert(gb.scratch_names[k].to_str().unwrap().to_owned(), d);
            dev_bufs.push(d);
        }
        let out_bytes = (out.elems() * 2) as u64;
        let mut d_out = 0u64;
        syn!(synDeviceMalloc(dev, out_bytes, 0, 0, &mut d_out));
        let mut h_out: *mut c_void = core::ptr::null_mut();
        syn!(synHostMalloc(dev, out_bytes, 0, &mut h_out));
        info_index.insert(out.name.to_str().unwrap().to_owned(), infos.len());
        infos.push(launch_info(
            &out.name,
            d_out,
            ids[n_in + n_scratch],
            &out.sizes,
        ));
        addrs.insert(out.name.to_str().unwrap().to_owned(), d_out);

        let mut ws = 0u64;
        syn!(synWorkspaceGetSize(&mut ws, recipe));
        // Diagnostic: `RENG_WS_SLACK_MB` over-allocates the workspace.
        let slack = std::env::var("RENG_WS_SLACK_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
            << 20;
        let mut dws = 0u64;
        if ws + slack > 0 {
            syn!(synDeviceMalloc(dev, ws + slack, 0, 0, &mut dws));
        }
        syn!(synStreamSynchronize(stream));
        Ok(Self {
            gb,
            dev,
            stream,
            recipe,
            infos,
            info_index,
            addrs,
            host_bufs,
            dev_bufs,
            d_out,
            h_out,
            h_fence: core::ptr::null_mut(),
            fence_bytes: 0,
            out,
            dws,
        })
    }

    /// Device address of a persistent tensor.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a persistent tensor of this graph.
    pub fn addr(&self, name: &str) -> u64 {
        self.addrs[name]
    }

    /// Replace input `idx`'s contents (bf16-converted, `data.len()` must equal
    /// the input's element count) ahead of the next launch.
    pub fn upload(&mut self, idx: usize, data: &[f32]) -> Result<()> {
        assert_eq!(data.len(), self.gb.data[idx].len());
        let hb = self.host_bufs[idx];
        // SAFETY: hb holds data.len() bf16 elements.
        unsafe {
            let pb = hb.cast::<u16>();
            for (j, &val) in data.iter().enumerate() {
                *pb.add(j) = f32_to_bf16(val);
            }
        }
        syn!(synMemCopyAsync(
            self.stream,
            hb as u64,
            (data.len() * 2) as u64,
            self.dev_bufs[idx],
            SYN_HOST_TO_DRAM
        ));
        Ok(())
    }

    /// Wait until input `idx` (the last one uploaded) is visible on the
    /// device: copy it back with the host sentinel and spin until the copy
    /// has landed and equals the staged data. Host-to-device copies on the
    /// stream are executed in order, so once the last one is visible all of
    /// them are. Needed because the stream sync returns before DMA copies
    /// have landed on this stack, and a launch issued right after them would
    /// otherwise compute on the previous step's inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the data never appears.
    pub fn fence_uploads(&mut self, idx: usize) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        let n = self.gb.data[idx].len();
        let bytes = (n * 2) as u64;
        if self.h_fence.is_null() || self.fence_bytes < bytes {
            if !self.h_fence.is_null() {
                unsafe { synHostFree(self.dev, self.h_fence, 0) };
            }
            let mut vb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(self.dev, bytes, 0, &mut vb));
            self.h_fence = vb;
            self.fence_bytes = bytes;
        }
        let vb = self.h_fence;
        let started = Instant::now();
        let mut copies = 0u32;
        loop {
            // SAFETY: vb holds at least n bf16 elements.
            unsafe {
                for j in 0..n {
                    *vb.cast::<u16>().add(j) = HOST_SENTINEL_BF16;
                }
            }
            syn!(synMemCopyAsync(
                self.stream,
                self.dev_bufs[idx],
                bytes,
                vb as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(self.stream));
            copies += 1;
            // Wait for this copy to land, then compare.
            loop {
                // SAFETY: as above.
                let pending = unsafe {
                    let p = vb.cast::<u16>();
                    (0..n).any(|j| core::ptr::read_volatile(p.add(j)) == HOST_SENTINEL_BF16)
                };
                if !pending {
                    break;
                }
                if started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "fence: copy of input {idx} did not land within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::hint::spin_loop();
            }
            // SAFETY: both buffers hold n bf16 elements.
            let differs = unsafe {
                let (p, q) = (vb.cast::<u16>(), self.host_bufs[idx].cast::<u16>());
                (0..n).filter(|&j| *p.add(j) != *q.add(j)).count()
            };
            if differs == 0 {
                break;
            }
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "fence: upload of input {idx} never became visible ({differs} elements differ)"
                )));
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        if copies > 1 && env_on("RENG_READBACK_TRACE") {
            eprintln!(
                "fence: input {idx} visible after {copies} copies ({:?})",
                started.elapsed()
            );
        }
        Ok(())
    }

    /// Index of the input named `name`.
    ///
    /// # Panics
    ///
    /// Panics if there is no such input.
    pub fn input_index(&self, name: &str) -> usize {
        self.gb
            .names
            .iter()
            .position(|n| n.to_str().unwrap() == name)
            .unwrap_or_else(|| panic!("no input named {name}"))
    }

    /// Zero `bytes` of device memory at `addr` (a DMA write; call
    /// [`Runtime::settle`] before the next launch reads it).
    pub fn zero(&self, addr: u64, bytes: u64) -> Result<()> {
        syn!(synMemsetD32Async(
            addr,
            0,
            (bytes / 4) as usize,
            self.stream
        ));
        Ok(())
    }

    /// Wait for the stream and then a few milliseconds more, for the rare
    /// paths where a DMA write must be visible to the next launch.
    pub fn settle(&self) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        std::thread::sleep(Duration::from_millis(20));
        Ok(())
    }

    /// Bind persistent tensor `name` to device address `addr` for the next
    /// launches (the KV cache swaps its read and write buffers).
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a persistent tensor of this graph.
    pub fn rebind(&mut self, name: &str, addr: u64) {
        let idx = self.info_index[name];
        self.infos[idx].tensor_address = addr;
    }

    /// Launch the recipe and read back the first `rows` outermost rows of the
    /// output as f32 (see the module docs for the two-sentinel protocol).
    #[allow(clippy::too_many_lines)]
    pub fn launch_and_read(&mut self, rows: usize) -> Result<Vec<f32>> {
        let n_out = rows * self.out.row_elems();
        assert!(n_out <= self.out.elems());
        let out_bytes = (n_out * 2) as u64;
        let (stream, dev, d_out, h_out) = (self.stream, self.dev, self.d_out, self.h_out);
        let trace = env_on("RENG_STEP_TRACE");
        let t0 = Instant::now();
        // Pre-fill the device output with the recipe-completion sentinel.
        syn!(synMemsetD32Async(
            d_out,
            SENTINEL_D32,
            self.out.elems() / 2,
            stream
        ));
        syn!(synStreamSynchronize(stream));
        let t_memset = t0.elapsed();

        syn!(synLaunch(
            stream,
            self.infos.as_ptr(),
            self.infos.len() as u32,
            self.dws,
            self.recipe,
            0
        ));
        syn!(synStreamSynchronize(stream));
        let t_launch = t0.elapsed() - t_memset;
        if env_on("RENG_DEVSYNC") {
            syn!(synDeviceSynchronize(dev));
        }
        if let Some(ms) = std::env::var("RENG_SETTLE_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            std::thread::sleep(Duration::from_millis(ms));
        }

        // One complete copy: fill the host buffer with the copy sentinel,
        // copy, then wait until the DMA has replaced every element (the
        // stream sync returns before the copy has landed on this stack).
        let started = Instant::now();
        let read_once = |s: synStreamHandle| -> Result<()> {
            // SAFETY: h_out holds at least n_out bf16 elements.
            unsafe {
                let p = h_out.cast::<u16>();
                for j in 0..n_out {
                    *p.add(j) = HOST_SENTINEL_BF16;
                }
            }
            syn!(synMemCopyAsync(
                s,
                d_out,
                out_bytes,
                h_out as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(s));
            loop {
                // SAFETY: as above; the DMA writes each element exactly once.
                let pending = unsafe {
                    let p = h_out.cast::<u16>();
                    (0..n_out).any(|j| core::ptr::read_volatile(p.add(j)) == HOST_SENTINEL_BF16)
                };
                if !pending {
                    return Ok(());
                }
                if started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "device-to-host copy did not complete within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::hint::spin_loop();
            }
        };
        // SAFETY (closure body): h_out holds n_out bf16 elements for the whole call.
        let incomplete = || -> usize {
            let p = h_out.cast::<u16>();
            (0..n_out)
                .filter(|&j| unsafe { *p.add(j) } == SENTINEL_BF16)
                .count()
        };
        if env_on("RENG_EVBRIDGE") {
            let mut ev: synEventHandle = core::ptr::null_mut();
            syn!(synEventCreate(&mut ev, dev, 0));
            syn!(synEventRecord(ev, stream));
            let mut d2h: synStreamHandle = core::ptr::null_mut();
            syn!(synStreamCreateGeneric(&mut d2h, dev, 0));
            syn!(synStreamWaitEvent(d2h, ev, 0));
            read_once(d2h)?;
            unsafe {
                synStreamDestroy(d2h);
                synEventDestroy(ev);
            }
        } else {
            read_once(stream)?;
        }
        let mut polls = 0u32;
        let mut left = incomplete();
        while left > 0 {
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "recipe output incomplete after {READBACK_TIMEOUT:?}: {left} of {n_out} elements unwritten"
                )));
            }
            std::thread::sleep(Duration::from_micros(200));
            read_once(stream)?;
            polls += 1;
            left = incomplete();
        }
        // Diagnostic stability check (see `stability_window`).
        let snapshot = |buf: *mut c_void, n: usize| -> Vec<u16> {
            // SAFETY: buf holds n bf16 elements.
            unsafe { std::slice::from_raw_parts(buf.cast::<u16>(), n).to_vec() }
        };
        let mut stable_polls = 0u32;
        if let Some(window) = stability_window() {
            let mut prev = snapshot(h_out, n_out);
            loop {
                if started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "recipe output did not stabilise within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::thread::sleep(window);
                read_once(stream)?;
                let cur = snapshot(h_out, n_out);
                stable_polls += 1;
                if cur == prev {
                    break;
                }
                prev = cur;
            }
        }
        if (polls > 0 || stable_polls > 1) && env_on("RENG_READBACK_TRACE") {
            eprintln!(
                "readback: {polls} sentinel polls, {stable_polls} stability reads ({:?})",
                started.elapsed()
            );
        }
        if trace {
            eprintln!(
                "step trace: memset {:.2} ms, launch+sync {:.2} ms, readback {:.2} ms ({polls} polls, {stable_polls} stability reads, {} KiB)",
                t_memset.as_secs_f64() * 1e3,
                t_launch.as_secs_f64() * 1e3,
                started.elapsed().as_secs_f64() * 1e3,
                out_bytes / 1024
            );
        }
        if env_on("RENG_DUMP_SCRATCH") {
            self.dump_scratch()?;
        }
        let mut result = vec![0.0f32; n_out];
        // SAFETY: h_out holds n_out bf16 elements just copied back.
        unsafe {
            let po = h_out.cast::<u16>();
            for (j, o) in result.iter_mut().enumerate() {
                *o = bf16_to_f32(*po.add(j));
            }
        }
        Ok(result)
    }
}

impl Runtime {
    /// Diagnostic: after a long settle, copy every scratch tensor down and
    /// print its zero fraction and largest magnitude, in creation order.
    fn dump_scratch(&self) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        std::thread::sleep(Duration::from_millis(100));
        let n_in = self.gb.names.len();
        for (k, sizes) in self.gb.scratch_sizes.iter().enumerate() {
            let elems = sizes.iter().product::<u64>() as usize;
            let f32s = self.gb.scratch_f32[k];
            let bytes = (elems * if f32s { 4 } else { 2 }) as u64;
            let mut hb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(self.dev, bytes, 0, &mut hb));
            syn!(synMemCopyAsync(
                self.stream,
                self.dev_bufs[n_in + k],
                bytes,
                hb as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(self.stream));
            let (mut zeros, mut max) = (0usize, 0.0f32);
            // SAFETY: hb holds `elems` elements of the recorded dtype.
            unsafe {
                for j in 0..elems {
                    let v = if f32s {
                        *hb.cast::<f32>().add(j)
                    } else {
                        bf16_to_f32(*hb.cast::<u16>().add(j))
                    };
                    if v == 0.0 {
                        zeros += 1;
                    }
                    max = max.max(v.abs());
                }
                synHostFree(self.dev, hb, 0);
            }
            eprintln!(
                "dump {:<14} {:>9} elems  zeros {:>5.1}%  max {max:.3e}",
                self.gb.scratch_names[k].to_str().unwrap(),
                elems,
                100.0 * zeros as f64 / elems as f64
            );
        }
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            synStreamSynchronize(self.stream);
            for &hb in &self.host_bufs {
                synHostFree(self.dev, hb, 0);
            }
            synHostFree(self.dev, self.h_out, 0);
            if !self.h_fence.is_null() {
                synHostFree(self.dev, self.h_fence, 0);
            }
            for &d in &self.dev_bufs {
                synDeviceFree(self.dev, d, 0);
            }
            synDeviceFree(self.dev, self.d_out, 0);
            if self.dws != 0 {
                synDeviceFree(self.dev, self.dws, 0);
            }
            synStreamDestroy(self.stream);
            synRecipeDestroy(self.recipe);
            synGraphDestroy(self.gb.graph);
            synDeviceRelease(self.dev);
            synDestroy();
        }
    }
}
