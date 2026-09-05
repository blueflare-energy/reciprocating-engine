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
const HOST_SENTINEL_D32: u32 = 0x7FC2_7FC2;
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

/// Element type of a read-back tensor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutKind {
    Bf16,
    I32,
}

impl OutKind {
    fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::I32 => 4,
        }
    }
}

/// The single persistent tensor a graph run reads back, `[fcd, .., rows]`.
pub(crate) struct Out {
    pub name: CString,
    pub sizes: Vec<u64>,
    pub kind: OutKind,
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

/// Fill `n` elements of `elem_bytes` each at `buf` with the host sentinel.
///
/// # Safety
///
/// `buf` must hold `n * elem_bytes` bytes.
unsafe fn fill_host_sentinel(buf: *mut c_void, n: usize, elem_bytes: usize) {
    if elem_bytes == 2 {
        let p = buf.cast::<u16>();
        for j in 0..n {
            unsafe { *p.add(j) = HOST_SENTINEL_BF16 };
        }
    } else {
        let p = buf.cast::<u32>();
        for j in 0..n {
            unsafe { *p.add(j) = HOST_SENTINEL_D32 };
        }
    }
}

/// Whether any of `n` elements at `buf` still shows `pattern16` (2-byte
/// elements) or `pattern32` (4-byte elements).
///
/// # Safety
///
/// `buf` must hold `n * elem_bytes` bytes.
unsafe fn any_sentinel(
    buf: *mut c_void,
    n: usize,
    elem_bytes: usize,
    pattern16: u16,
    pattern32: u32,
) -> bool {
    if elem_bytes == 2 {
        let p = buf.cast::<u16>();
        (0..n).any(|j| unsafe { core::ptr::read_volatile(p.add(j)) } == pattern16)
    } else {
        let p = buf.cast::<u32>();
        (0..n).any(|j| unsafe { core::ptr::read_volatile(p.add(j)) } == pattern32)
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
    /// Shape per persistent tensor, by name (sharing requires equal element
    /// counts).
    shapes: HashMap<String, Vec<u64>>,
    /// Whether this runtime acquired the device (else it borrows a parent's).
    owns_device: bool,
    /// Pinned host staging buffer per uploaded input, by input index (null
    /// for inputs bound to a parent's buffer).
    host_bufs: Vec<*mut c_void>,
    /// Device buffer per persistent tensor, by input then scratch index.
    dev_bufs: Vec<u64>,
    /// The device buffers this runtime allocated (freed on drop).
    owned: Vec<u64>,
    d_out: u64,
    h_out: *mut c_void,
    /// The small fence buffer of [`Runtime::fence`] (device, host in, host
    /// out) and the last pattern written to it.
    fence_dev: u64,
    fence_in: *mut c_void,
    fence_out: *mut c_void,
    fence_seq: u32,
    /// Pinned buffer for [`Runtime::read_bf16_range`], grown on demand.
    h_aux: *mut c_void,
    aux_bytes: u64,
    out: Out,
    dws: u64,
}

impl Runtime {
    /// Compile `gb`, acquire a device, allocate every persistent tensor, and
    /// upload the inputs' host data.
    pub fn new(gb: Gb, out: Out) -> Result<Self> {
        Self::new_with(gb, out, None)
    }

    /// Like [`Runtime::new`], but sharing `parent`'s device and stream, and
    /// binding every persistent tensor whose name `parent` also has to the
    /// parent's buffer instead of allocating (and uploading) its own. A
    /// second recipe over the same weights and KV cache costs only its own
    /// per-step inputs and output. The child must be dropped before the
    /// parent.
    #[allow(clippy::too_many_lines)]
    pub fn new_with(mut gb: Gb, out: Out, parent: Option<&Runtime>) -> Result<Self> {
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

        let (dev, stream) = match parent {
            Some(p) => (p.dev, p.stream),
            None => {
                let dev = crate::device::acquire_device()?;
                let mut stream: synStreamHandle = core::ptr::null_mut();
                syn!(synStreamCreateGeneric(&mut stream, dev, 0));
                (dev, stream)
            }
        };
        // A tensor is shared with the parent when it has the same name AND
        // the same element count (a weight may be declared 4-D in one graph
        // and 5-D with a trailing 1 in another; the per-step inputs of a
        // narrower recipe have the same names but different counts).
        let shared = |name: &CString, sizes: &[u64]| -> Option<u64> {
            let key = name.to_str().unwrap();
            let elems = sizes.iter().product::<u64>();
            parent.and_then(|p| match (p.addrs.get(key), p.shapes.get(key)) {
                (Some(&d), Some(sh)) if sh.iter().product::<u64>() == elems => Some(d),
                _ => None,
            })
        };

        let n_in = gb.names.len();
        let n_scratch = gb.scratch_names.len();
        let mut dev_bufs: Vec<u64> = Vec::with_capacity(n_in + n_scratch);
        let mut owned: Vec<u64> = Vec::with_capacity(n_in + n_scratch);
        let mut host_bufs: Vec<*mut c_void> = Vec::with_capacity(n_in);
        let mut infos: Vec<synLaunchTensorInfo> = Vec::with_capacity(n_in + n_scratch + 1);
        let mut addrs = HashMap::with_capacity(n_in + n_scratch + 1);
        let mut info_index = HashMap::with_capacity(n_in + n_scratch + 1);
        let mut shapes = HashMap::with_capacity(n_in + n_scratch + 1);
        for (idx, data) in gb.data.iter().enumerate() {
            let raw = gb.raw[idx].as_deref();
            let bytes = raw.map_or((data.len() * 2) as u64, |r| r.len() as u64);
            let mut hb: *mut c_void = core::ptr::null_mut();
            let d = if let Some(d) = shared(&gb.names[idx], &gb.sizes[idx]) {
                d
            } else {
                let mut d = 0u64;
                syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
                owned.push(d);
                syn!(synHostMalloc(dev, bytes, 0, &mut hb));
                // SAFETY: hb holds `bytes` bytes: the raw bytes, or
                // data.len() bf16 elements.
                unsafe {
                    if let Some(r) = raw {
                        core::ptr::copy_nonoverlapping(r.as_ptr(), hb.cast::<u8>(), r.len());
                    } else {
                        let pb = hb.cast::<u16>();
                        for (j, &val) in data.iter().enumerate() {
                            *pb.add(j) = f32_to_bf16(val);
                        }
                    }
                }
                syn!(synMemCopyAsync(
                    stream,
                    hb as u64,
                    bytes,
                    d,
                    SYN_HOST_TO_DRAM
                ));
                d
            };
            info_index.insert(gb.names[idx].to_str().unwrap().to_owned(), infos.len());
            infos.push(launch_info(&gb.names[idx], d, ids[idx], &gb.sizes[idx]));
            addrs.insert(gb.names[idx].to_str().unwrap().to_owned(), d);
            shapes.insert(
                gb.names[idx].to_str().unwrap().to_owned(),
                gb.sizes[idx].clone(),
            );
            dev_bufs.push(d);
            host_bufs.push(hb);
        }
        // The f32 host copies of the inputs (the weights, mostly) are not
        // needed once they are in the pinned staging buffers or bound to a
        // parent: uploads are validated against the tensor sizes.
        for d in &mut gb.data {
            *d = Vec::new();
        }
        for (k, sizes) in gb.scratch_sizes.iter().enumerate() {
            let bytes = sizes.iter().product::<u64>() * if gb.scratch_f32[k] { 4 } else { 2 };
            let d = if let Some(of) = &gb.scratch_alias[k] {
                // The output side of an in-place update: same memory.
                addrs[of.as_str()]
            } else if let Some(d) = shared(&gb.scratch_names[k], sizes) {
                d
            } else {
                let mut d = 0u64;
                syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
                owned.push(d);
                // Device-resident state starts as zeros (finite, so a
                // masked-out stale cache row can never poison a softmax).
                syn!(synMemsetD32Async(d, 0, (bytes / 4) as usize, stream));
                d
            };
            info_index.insert(
                gb.scratch_names[k].to_str().unwrap().to_owned(),
                infos.len(),
            );
            infos.push(launch_info(&gb.scratch_names[k], d, ids[n_in + k], sizes));
            addrs.insert(gb.scratch_names[k].to_str().unwrap().to_owned(), d);
            shapes.insert(
                gb.scratch_names[k].to_str().unwrap().to_owned(),
                sizes.clone(),
            );
            dev_bufs.push(d);
        }
        let out_bytes = (out.elems() * out.kind.bytes()) as u64;
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
            owns_device: parent.is_none(),
            recipe,
            infos,
            info_index,
            addrs,
            shapes,
            host_bufs,
            dev_bufs,
            owned,
            d_out,
            h_out,
            fence_dev: 0,
            fence_in: core::ptr::null_mut(),
            fence_out: core::ptr::null_mut(),
            fence_seq: 0,
            h_aux: core::ptr::null_mut(),
            aux_bytes: 0,
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

    /// Element count of input `idx`.
    fn input_elems(&self, idx: usize) -> usize {
        self.gb.sizes[idx].iter().product::<u64>() as usize
    }

    /// Copy `(src, dst, bytes)` ranges between device buffers and wait until
    /// they have landed (stream sync plus a [`Runtime::fence`], since the
    /// sync alone returns early on this stack).
    ///
    /// # Errors
    ///
    /// Returns an error if a copy cannot be enqueued or the fence times out.
    pub fn copy_d2d(&mut self, copies: &[(u64, u64, u64)]) -> Result<()> {
        // Chunks keep the descriptor arrays alive until the sync below.
        const CHUNK: usize = 4096;
        let chunks: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = copies
            .chunks(CHUNK)
            .map(|c| {
                (
                    c.iter().map(|x| x.0).collect(),
                    c.iter().map(|x| x.2).collect(),
                    c.iter().map(|x| x.1).collect(),
                )
            })
            .collect();
        for (src, size, dst) in &chunks {
            syn!(synMemCopyAsyncMultiple(
                self.stream,
                src.as_ptr(),
                size.as_ptr(),
                dst.as_ptr(),
                SYN_DRAM_TO_DRAM,
                src.len() as u64
            ));
        }
        syn!(synStreamSynchronize(self.stream));
        self.fence()
    }

    /// Replace input `idx`'s contents (bf16-converted, `data.len()` must equal
    /// the input's element count) ahead of the next launch.
    pub fn upload(&mut self, idx: usize, data: &[f32]) -> Result<()> {
        assert_eq!(data.len(), self.input_elems(idx));
        let hb = self.host_bufs[idx];
        assert!(!hb.is_null(), "input {idx} is bound to a shared buffer");
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

    /// Replace a raw (non-bf16) input's bytes (index tensors).
    pub fn upload_raw(&mut self, idx: usize, bytes: &[u8]) -> Result<()> {
        let expect = self.gb.raw[idx].as_ref().map_or(0, Vec::len);
        assert_eq!(bytes.len(), expect, "raw input {idx} size");
        let hb = self.host_bufs[idx];
        assert!(!hb.is_null(), "input {idx} is bound to a shared buffer");
        // SAFETY: hb holds `expect` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), hb.cast::<u8>(), bytes.len());
        }
        syn!(synMemCopyAsync(
            self.stream,
            hb as u64,
            bytes.len() as u64,
            self.dev_bufs[idx],
            SYN_HOST_TO_DRAM
        ));
        Ok(())
    }

    /// Replace input `idx`'s contents with bf16 data already in device format
    /// (for the large mask patterns that need no conversion).
    pub fn upload_bf16(&mut self, idx: usize, data: &[u16]) -> Result<()> {
        assert_eq!(data.len(), self.input_elems(idx));
        let hb = self.host_bufs[idx];
        assert!(!hb.is_null(), "input {idx} is bound to a shared buffer");
        // SAFETY: hb holds data.len() bf16 elements.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), hb.cast::<u16>(), data.len());
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

    /// Wait until every upload enqueued so far is visible on the device: a
    /// small fence buffer gets a fresh pattern uploaded after them, and is
    /// read back until that pattern shows. Host-to-device copies on the
    /// stream execute in order, so once the fence is visible all of them are.
    /// Needed because the stream sync returns before DMA copies have landed
    /// on this stack, and a launch issued right after them would otherwise
    /// compute on the previous step's inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern never appears.
    pub fn fence(&mut self) -> Result<()> {
        const WORDS: usize = 1024;
        if self.fence_dev == 0 {
            let bytes = (WORDS * 4) as u64;
            syn!(synDeviceMalloc(self.dev, bytes, 0, 0, &mut self.fence_dev));
            syn!(synHostMalloc(self.dev, bytes, 0, &mut self.fence_in));
            syn!(synHostMalloc(self.dev, bytes, 0, &mut self.fence_out));
            self.owned.push(self.fence_dev);
        }
        self.fence_seq = self.fence_seq.wrapping_add(1);
        let pattern = self.fence_seq | 0x5A00_0000;
        // SAFETY: both fence buffers hold WORDS u32 values.
        unsafe {
            for j in 0..WORDS {
                *self.fence_in.cast::<u32>().add(j) = pattern;
            }
        }
        syn!(synMemCopyAsync(
            self.stream,
            self.fence_in as u64,
            (WORDS * 4) as u64,
            self.fence_dev,
            SYN_HOST_TO_DRAM
        ));
        syn!(synStreamSynchronize(self.stream));
        let started = Instant::now();
        loop {
            // SAFETY: as above.
            unsafe {
                for j in 0..WORDS {
                    *self.fence_out.cast::<u32>().add(j) = HOST_SENTINEL_D32;
                }
            }
            syn!(synMemCopyAsync(
                self.stream,
                self.fence_dev,
                (WORDS * 4) as u64,
                self.fence_out as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(self.stream));
            let mut pending = true;
            let mut matched = false;
            while pending {
                // SAFETY: as above; the DMA writes each word exactly once.
                let (p, m) = unsafe {
                    let q = self.fence_out.cast::<u32>();
                    let mut p = false;
                    let mut m = true;
                    for j in 0..WORDS {
                        let v = core::ptr::read_volatile(q.add(j));
                        if v == HOST_SENTINEL_D32 {
                            p = true;
                        } else if v != pattern {
                            m = false;
                        }
                    }
                    (p, m)
                };
                pending = p;
                matched = m;
                if pending && started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "fence: copy did not land within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::hint::spin_loop();
            }
            if matched {
                return Ok(());
            }
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "fence: uploads never became visible within {READBACK_TIMEOUT:?}"
                )));
            }
            std::thread::sleep(Duration::from_micros(50));
        }
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

    /// Enqueue one launch without reading anything back (launches on the
    /// stream run in order; a later [`Runtime::launch_and_read`] completes
    /// after all of them). For throughput measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if the launch is rejected.
    pub fn launch_only(&mut self) -> Result<()> {
        syn!(synLaunch(
            self.stream,
            self.infos.as_ptr(),
            self.infos.len() as u32,
            self.dws,
            self.recipe,
            0
        ));
        Ok(())
    }

    /// Launch the recipe and read back the first `rows` outermost rows of the
    /// output as f32 (see the module docs for the two-sentinel protocol).
    pub fn launch_and_read(&mut self, rows: usize) -> Result<Vec<f32>> {
        self.launch_and_read_range(0, rows)
    }

    /// Launch the recipe and read back `rows` outermost rows of a bf16
    /// output starting at row `first`, as f32.
    pub fn launch_and_read_range(&mut self, first: usize, rows: usize) -> Result<Vec<f32>> {
        assert!(self.out.kind == OutKind::Bf16, "output is not bf16");
        let bytes = self.launch_and_read_bytes(first, rows)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    }

    /// Launch the recipe and read back `rows` outermost rows of an int32
    /// output starting at row `first`.
    pub fn launch_and_read_i32(&mut self, first: usize, rows: usize) -> Result<Vec<i32>> {
        assert!(self.out.kind == OutKind::I32, "output is not int32");
        let bytes = self.launch_and_read_bytes(first, rows)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Launch the recipe and read back a row range of the output as raw bytes.
    #[allow(clippy::too_many_lines)]
    fn launch_and_read_bytes(&mut self, first: usize, rows: usize) -> Result<Vec<u8>> {
        let row_elems = self.out.row_elems();
        let eb = self.out.kind.bytes();
        let n_out = rows * row_elems;
        assert!((first + rows) * row_elems <= self.out.elems());
        let out_bytes = (n_out * eb) as u64;
        let (stream, dev, h_out) = (self.stream, self.dev, self.h_out);
        let d_out = self.d_out + (first * row_elems * eb) as u64;
        let trace = env_on("RENG_STEP_TRACE");
        let t0 = Instant::now();
        // Pre-fill the device output with the recipe-completion sentinel.
        syn!(synMemsetD32Async(
            self.d_out,
            SENTINEL_D32,
            self.out.elems() * eb / 4,
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
            // SAFETY: h_out holds at least n_out elements of eb bytes.
            unsafe { fill_host_sentinel(h_out, n_out, eb) };
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
                    any_sentinel(h_out, n_out, eb, HOST_SENTINEL_BF16, HOST_SENTINEL_D32)
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
        // SAFETY (closure body): h_out holds n_out elements for the whole call.
        let incomplete =
            || -> bool { unsafe { any_sentinel(h_out, n_out, eb, SENTINEL_BF16, SENTINEL_D32) } };
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
        while incomplete() {
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "recipe output incomplete after {READBACK_TIMEOUT:?}"
                )));
            }
            std::thread::sleep(Duration::from_micros(200));
            read_once(stream)?;
            polls += 1;
        }
        // Diagnostic stability check (see `stability_window`).
        let snapshot = |buf: *mut c_void, n: usize| -> Vec<u8> {
            // SAFETY: buf holds n bytes.
            unsafe { std::slice::from_raw_parts(buf.cast::<u8>(), n).to_vec() }
        };
        let mut stable_polls = 0u32;
        if let Some(window) = stability_window() {
            let mut prev = snapshot(h_out, n_out * eb);
            loop {
                if started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "recipe output did not stabilise within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::thread::sleep(window);
                read_once(stream)?;
                let cur = snapshot(h_out, n_out * eb);
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
        Ok(snapshot(h_out, n_out * eb))
    }

    /// After a launch has completed, read back `rows` outermost rows of the
    /// persistent tensor `name` starting at row `first`, as f32 (bf16 data).
    /// Only the copy-landed sentinel is used: the recipe is known complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy never lands.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a persistent tensor of this graph.
    pub fn read_bf16_range(&mut self, name: &str, first: usize, rows: usize) -> Result<Vec<f32>> {
        let shape = self.shapes[name].clone();
        let row_elems = (shape.iter().product::<u64>() / *shape.last().unwrap_or(&1)) as usize;
        let n = rows * row_elems;
        assert!((first + rows) * row_elems <= shape.iter().product::<u64>() as usize);
        let bytes = (n * 2) as u64;
        if self.aux_bytes < bytes {
            if !self.h_aux.is_null() {
                unsafe { synHostFree(self.dev, self.h_aux, 0) };
            }
            let mut hb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(self.dev, bytes, 0, &mut hb));
            self.h_aux = hb;
            self.aux_bytes = bytes;
        }
        let src = self.addrs[name] + (first * row_elems * 2) as u64;
        let started = Instant::now();
        // SAFETY: h_aux holds at least n bf16 elements.
        unsafe { fill_host_sentinel(self.h_aux, n, 2) };
        syn!(synMemCopyAsync(
            self.stream,
            src,
            bytes,
            self.h_aux as u64,
            SYN_DRAM_TO_HOST
        ));
        syn!(synStreamSynchronize(self.stream));
        // SAFETY: as above.
        while unsafe { any_sentinel(self.h_aux, n, 2, HOST_SENTINEL_BF16, HOST_SENTINEL_D32) } {
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "read of {name}: copy did not land within {READBACK_TIMEOUT:?}"
                )));
            }
            std::hint::spin_loop();
        }
        // SAFETY: h_aux holds n bf16 elements just copied.
        let words = unsafe { std::slice::from_raw_parts(self.h_aux.cast::<u16>(), n) };
        Ok(words.iter().map(|&w| bf16_to_f32(w)).collect())
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
                if !hb.is_null() {
                    synHostFree(self.dev, hb, 0);
                }
            }
            synHostFree(self.dev, self.h_out, 0);
            if !self.fence_in.is_null() {
                synHostFree(self.dev, self.fence_in, 0);
                synHostFree(self.dev, self.fence_out, 0);
            }
            if !self.h_aux.is_null() {
                synHostFree(self.dev, self.h_aux, 0);
            }
            for &d in &self.owned {
                synDeviceFree(self.dev, d, 0);
            }
            synDeviceFree(self.dev, self.d_out, 0);
            if self.dws != 0 {
                synDeviceFree(self.dev, self.dws, 0);
            }
            synRecipeDestroy(self.recipe);
            synGraphDestroy(self.gb.graph);
            if self.owns_device {
                synStreamDestroy(self.stream);
                synDeviceRelease(self.dev);
                synDestroy();
            }
        }
    }
}
