//! A compiled recipe with its device buffers: the launch and readback side of
//! the fused-model graphs in `model.rs`. One `Runtime` is compiled once and
//! can be launched many times (the KV-cache decode loop); the one-shot paths
//! build one, launch it once, and drop it.
//!
//! Readback: on this stack the stream and device syncs return before a deep
//! recipe has finished writing its output, so the output is pre-filled with a
//! NaN sentinel and copied down until no sentinel remains (exact completion,
//! since every output element is written exactly once), then re-read until
//! two reads spaced by a short window agree.

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

/// bf16 quiet-NaN pattern the output buffer is pre-filled with before launch.
/// The recipe writes every output element exactly once and never produces this
/// exact NaN, so "no sentinel left" is an exact completion test for the
/// readback (the stream and device syncs return before deep recipes finish).
const SENTINEL_BF16: u16 = 0x7FC1;
const SENTINEL_D32: u32 = 0x7FC1_7FC1;
/// Upper bound on waiting for a recipe's output to complete.
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);
/// Spacing between the consecutive readbacks that must agree before the
/// output is trusted. Measured: a 5 ms settle after the stream sync was always
/// enough for a 4-layer graph whose plain readback was wrong 4 times in 4.
const STABILITY_WINDOW: Duration = Duration::from_millis(5);

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
    /// Device buffer per persistent tensor, by name.
    addrs: HashMap<String, u64>,
    /// Pinned host staging buffer per uploaded input, by input index.
    host_bufs: Vec<*mut c_void>,
    dev_bufs: Vec<u64>,
    d_out: u64,
    h_out: *mut c_void,
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
            infos.push(launch_info(&gb.names[idx], d, ids[idx], &gb.sizes[idx]));
            addrs.insert(gb.names[idx].to_str().unwrap().to_owned(), d);
            dev_bufs.push(d);
            host_bufs.push(hb);
        }
        for (k, sizes) in gb.scratch_sizes.iter().enumerate() {
            let bytes = sizes.iter().product::<u64>() * 2;
            let mut d = 0u64;
            syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
            // Device-resident state starts as zeros (finite, so a masked-out
            // stale cache row can never poison a softmax).
            syn!(synMemsetD32Async(d, 0, (bytes / 4) as usize, stream));
            infos.push(launch_info(&gb.scratch_names[k], d, ids[n_in + k], sizes));
            addrs.insert(gb.scratch_names[k].to_str().unwrap().to_owned(), d);
            dev_bufs.push(d);
        }
        let out_bytes = (out.elems() * 2) as u64;
        let mut d_out = 0u64;
        syn!(synDeviceMalloc(dev, out_bytes, 0, 0, &mut d_out));
        let mut h_out: *mut c_void = core::ptr::null_mut();
        syn!(synHostMalloc(dev, out_bytes, 0, &mut h_out));
        infos.push(launch_info(
            &out.name,
            d_out,
            ids[n_in + n_scratch],
            &out.sizes,
        ));
        addrs.insert(out.name.to_str().unwrap().to_owned(), d_out);

        let mut ws = 0u64;
        syn!(synWorkspaceGetSize(&mut ws, recipe));
        let mut dws = 0u64;
        if ws > 0 {
            syn!(synDeviceMalloc(dev, ws, 0, 0, &mut dws));
        }
        syn!(synStreamSynchronize(stream));
        Ok(Self {
            gb,
            dev,
            stream,
            recipe,
            infos,
            addrs,
            host_bufs,
            dev_bufs,
            d_out,
            h_out,
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

    /// Enqueue a device-to-device copy of `bytes` from `src + src_off` to
    /// `dst + dst_off` (byte offsets) on the launch stream.
    pub fn copy_d2d(
        &self,
        src: u64,
        src_off: u64,
        dst: u64,
        dst_off: u64,
        bytes: u64,
    ) -> Result<()> {
        syn!(synMemCopyAsync(
            self.stream,
            src + src_off,
            bytes,
            dst + dst_off,
            SYN_DRAM_TO_DRAM
        ));
        Ok(())
    }

    /// Wait for everything enqueued on the launch stream.
    pub fn sync(&self) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        Ok(())
    }

    /// Launch the recipe and read back the first `rows` outermost rows of the
    /// output as f32, using the sentinel + stability protocol on that range.
    #[allow(clippy::too_many_lines)]
    pub fn launch_and_read(&mut self, rows: usize) -> Result<Vec<f32>> {
        let n_out = rows * self.out.row_elems();
        assert!(n_out <= self.out.elems());
        let out_bytes = (n_out * 2) as u64;
        let (stream, dev, d_out, h_out) = (self.stream, self.dev, self.d_out, self.h_out);
        // Pre-fill the output with the completion sentinel (see SENTINEL_BF16).
        syn!(synMemsetD32Async(
            d_out,
            SENTINEL_D32,
            self.out.elems() / 2,
            stream
        ));
        syn!(synStreamSynchronize(stream));

        syn!(synLaunch(
            stream,
            self.infos.as_ptr(),
            self.infos.len() as u32,
            self.dws,
            self.recipe,
            0
        ));
        syn!(synStreamSynchronize(stream));
        if env_on("RENG_DEVSYNC") {
            syn!(synDeviceSynchronize(dev));
        }
        if let Some(ms) = std::env::var("RENG_SETTLE_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            std::thread::sleep(Duration::from_millis(ms));
        }

        // Readback with completion polling.
        let read_once = |s: synStreamHandle| -> Result<()> {
            syn!(synMemCopyAsync(
                s,
                d_out,
                out_bytes,
                h_out as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(s));
            Ok(())
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
        let started = Instant::now();
        let mut polls = 0u32;
        let mut left = incomplete();
        while left > 0 {
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "recipe output incomplete after {READBACK_TIMEOUT:?}: {left} of {n_out} elements unwritten"
                )));
            }
            std::thread::sleep(Duration::from_millis(2));
            read_once(stream)?;
            polls += 1;
            left = incomplete();
        }
        // Stability: the recipe's final writes keep landing for a few
        // milliseconds after the sync returns, and an intermediate write can have
        // already replaced the sentinel, so also require two consecutive reads
        // spaced by the measured window to be byte-identical.
        let snapshot = |buf: *mut c_void, n: usize| -> Vec<u16> {
            // SAFETY: buf holds n bf16 elements.
            unsafe { std::slice::from_raw_parts(buf.cast::<u16>(), n).to_vec() }
        };
        let mut prev = snapshot(h_out, n_out);
        let mut stable_polls = 0u32;
        loop {
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "recipe output did not stabilise within {READBACK_TIMEOUT:?}"
                )));
            }
            std::thread::sleep(STABILITY_WINDOW);
            read_once(stream)?;
            let cur = snapshot(h_out, n_out);
            stable_polls += 1;
            if cur == prev {
                break;
            }
            prev = cur;
        }
        if (polls > 0 || stable_polls > 1) && env_on("RENG_READBACK_TRACE") {
            eprintln!(
                "readback: {polls} sentinel polls, {stable_polls} stability reads ({:?})",
                started.elapsed()
            );
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

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            synStreamSynchronize(self.stream);
            for &hb in &self.host_bufs {
                synHostFree(self.dev, hb, 0);
            }
            synHostFree(self.dev, self.h_out, 0);
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
