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
use crate::{Stride, bf16_to_f32, f32_to_bf16};
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
pub(crate) const SENTINEL_D32: u32 = 0x7FC1_7FC1;
/// A second quiet-NaN pattern the HOST buffer is filled with before every
/// device-to-host copy; "none left" means the copy has landed. Distinct from
/// the device sentinel so the two conditions stay separable.
const HOST_SENTINEL_BF16: u16 = 0x7FC2;
pub(crate) const HOST_SENTINEL_D32: u32 = 0x7FC2_7FC2;
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

/// The buffers of [`Runtime::fence`]: a device buffer a fresh pattern is
/// copied into after the uploads, and the pinned host buffers the pattern
/// goes out from and comes back into. Allocated on first use.
struct Fence {
    dev_buf: u64,
    host_in: *mut c_void,
    host_out: *mut c_void,
    /// The last pattern written.
    seq: u32,
}

impl Fence {
    const WORDS: usize = 1024;

    fn none() -> Self {
        Self {
            dev_buf: 0,
            host_in: core::ptr::null_mut(),
            host_out: core::ptr::null_mut(),
            seq: 0,
        }
    }

    /// Wait until every host-to-device copy enqueued on `stream` so far is
    /// visible on the device (see [`Runtime::fence`]).
    fn wait(&mut self, dev: synDeviceId, stream: synStreamHandle) -> Result<()> {
        const WORDS: usize = Fence::WORDS;
        if self.dev_buf == 0 {
            let bytes = (WORDS * 4) as u64;
            syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut self.dev_buf));
            syn!(synHostMalloc(dev, bytes, 0, &mut self.host_in));
            syn!(synHostMalloc(dev, bytes, 0, &mut self.host_out));
        }
        self.seq = self.seq.wrapping_add(1);
        let pattern = self.seq | 0x5A00_0000;
        // SAFETY: both fence buffers hold WORDS u32 values.
        unsafe {
            for j in 0..WORDS {
                *self.host_in.cast::<u32>().add(j) = pattern;
            }
        }
        syn!(synMemCopyAsync(
            stream,
            self.host_in as u64,
            (WORDS * 4) as u64,
            self.dev_buf,
            SYN_HOST_TO_DRAM
        ));
        syn!(synStreamSynchronize(stream));
        let started = Instant::now();
        loop {
            // SAFETY: as above.
            unsafe {
                for j in 0..WORDS {
                    *self.host_out.cast::<u32>().add(j) = HOST_SENTINEL_D32;
                }
            }
            syn!(synMemCopyAsync(
                stream,
                self.dev_buf,
                (WORDS * 4) as u64,
                self.host_out as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(stream));
            let mut pending = true;
            let mut matched = false;
            while pending {
                // SAFETY: as above; the DMA writes each word exactly once.
                let (p, m) = unsafe {
                    let q = self.host_out.cast::<u32>();
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

    /// Release the buffers.
    ///
    /// # Safety
    ///
    /// No copy may be in flight on them.
    unsafe fn free(&mut self, dev: synDeviceId) {
        if self.dev_buf != 0 {
            // SAFETY: allocated by `wait`, used by no pending copy.
            unsafe {
                synHostFree(dev, self.host_in, 0);
                synHostFree(dev, self.host_out, 0);
                synDeviceFree(dev, self.dev_buf, 0);
            }
            self.dev_buf = 0;
        }
    }
}

/// A ring of pinned host buffers the construction-time uploads are staged
/// through: each input is copied into the next slot (in slot-sized pieces
/// when larger) and DMA'd from there; reusing a slot first waits, through
/// the fence, for every copy issued so far to land. Bounds the pinned
/// memory of an upload to a few hundred megabytes whatever the model size,
/// and replaces one pinned allocation per tensor with a handful.
struct Staging {
    slots: Vec<*mut c_void>,
    slot_bytes: usize,
    /// The next slot to use; equal to the slot count when the ring is full.
    next: usize,
}

impl Staging {
    /// Slot size cap and slot count cap: at most 1 GiB pinned.
    const SLOT_BYTES: usize = 256 << 20;
    const SLOTS: usize = 4;

    /// A ring for `total` bytes of inputs the largest of which has `largest`
    /// bytes (no ring when there is nothing to upload).
    fn new(dev: synDeviceId, total: usize, largest: usize) -> Result<Self> {
        let slot_bytes = largest.clamp(1, Self::SLOT_BYTES);
        let n = if total == 0 {
            0
        } else {
            total.div_ceil(slot_bytes).min(Self::SLOTS)
        };
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            let mut hb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(dev, slot_bytes as u64, 0, &mut hb));
            slots.push(hb);
        }
        Ok(Self {
            slots,
            slot_bytes,
            next: 0,
        })
    }

    /// Copy `src` to device address `dst` through the ring. With a
    /// `stride`, `src` is the base of a row-major matrix of which only a
    /// column window is wanted: `rows` runs of `cols` bf16 elements,
    /// `pitch` elements apart, gathered row by row into the slot so that
    /// the device gets the contiguous `[rows, cols]` matrix (a tensor-
    /// parallel shard of an `o_proj` or `down_proj` weight straight from
    /// the mapped checkpoint).
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        dev: synDeviceId,
        stream: synStreamHandle,
        fence: &mut Fence,
        src: &[u8],
        scale: Option<f32>,
        stride: Option<Stride>,
        dst: u64,
    ) -> Result<()> {
        let Some(st) = stride else {
            for (i, piece) in src.chunks(self.slot_bytes).enumerate() {
                let hb = self.slot(dev, stream, fence)?;
                // SAFETY: the slot holds `slot_bytes` bytes and no copy
                // reads it (every copy issued before the last fence has
                // landed).
                unsafe {
                    match scale {
                        Some(f) => copy_scaled_parallel(piece, f, hb.cast::<u8>()),
                        None => copy_parallel(piece, hb.cast::<u8>()),
                    }
                }
                syn!(synMemCopyAsync(
                    stream,
                    hb as u64,
                    piece.len() as u64,
                    dst + (i * self.slot_bytes) as u64,
                    SYN_HOST_TO_DRAM
                ));
            }
            return Ok(());
        };
        let row_bytes = st.cols * 2;
        let pitch_bytes = st.pitch * 2;
        assert!(
            row_bytes <= self.slot_bytes,
            "a strided row exceeds a staging slot"
        );
        assert!(
            (st.rows - 1) * pitch_bytes + row_bytes <= src.len(),
            "strided source too short"
        );
        let per_slot = self.slot_bytes / row_bytes;
        let mut row0 = 0usize;
        while row0 < st.rows {
            let n = per_slot.min(st.rows - row0);
            let hb = self.slot(dev, stream, fence)?;
            // SAFETY: as above; `n * row_bytes <= slot_bytes`.
            unsafe {
                gather_rows_parallel(
                    &src[row0 * pitch_bytes..],
                    n,
                    row_bytes,
                    pitch_bytes,
                    scale,
                    hb.cast::<u8>(),
                );
            }
            syn!(synMemCopyAsync(
                stream,
                hb as u64,
                (n * row_bytes) as u64,
                dst + (row0 * row_bytes) as u64,
                SYN_HOST_TO_DRAM
            ));
            row0 += n;
        }
        Ok(())
    }

    /// The next free slot, waiting for every copy in flight when the ring
    /// has been used up.
    fn slot(
        &mut self,
        dev: synDeviceId,
        stream: synStreamHandle,
        fence: &mut Fence,
    ) -> Result<*mut c_void> {
        if self.next == self.slots.len() {
            fence.wait(dev, stream)?;
            self.next = 0;
        }
        let hb = self.slots[self.next];
        self.next += 1;
        Ok(hb)
    }

    /// Release the slots.
    ///
    /// # Safety
    ///
    /// No copy may be in flight from them.
    unsafe fn free(&mut self, dev: synDeviceId) {
        for hb in self.slots.drain(..) {
            // SAFETY: allocated by `new`, used by no pending copy.
            unsafe { synHostFree(dev, hb, 0) };
        }
    }
}

/// `memcpy` of `src` to `dst`, split over up to eight threads for sources
/// of more than a few MB (one core moves a few GB/s, less when the source
/// is a mapped file whose pages fault in; a model is tens of GB).
///
/// # Safety
///
/// `dst` must be valid for `src.len()` bytes of writes and not overlap `src`.
unsafe fn copy_parallel(src: &[u8], dst: *mut u8) {
    const PER_THREAD: usize = 8 << 20;
    const THREADS: usize = 8;
    let threads = (src.len() / PER_THREAD).clamp(1, THREADS);
    if threads == 1 {
        // SAFETY: the caller's contract.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let chunk = src.len().div_ceil(threads);
    let dst = dst as usize;
    std::thread::scope(|s| {
        for (i, part) in src.chunks(chunk).enumerate() {
            s.spawn(move || {
                // SAFETY: the parts are disjoint ranges of the caller's
                // destination, offset like the source chunks.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        part.as_ptr(),
                        (dst + i * chunk) as *mut u8,
                        part.len(),
                    );
                }
            });
        }
    });
}

/// Like [`copy_parallel`] for bf16 data that is multiplied by `scale` on
/// the way (f32 product, rounded to bf16 as [`crate::scale_bf16`] does);
/// `src` is the little-endian bf16 bytes.
///
/// # Safety
///
/// `dst` must be writable for `src.len()` bytes and must not overlap `src`.
unsafe fn copy_scaled_parallel(src: &[u8], scale: f32, dst: *mut u8) {
    const PER_THREAD: usize = 8 << 20;
    const THREADS: usize = 8;
    let threads = (src.len() / PER_THREAD).clamp(1, THREADS);
    let convert = |part: &[u8], out: *mut u8| {
        for (j, b) in part.chunks_exact(2).enumerate() {
            let v =
                crate::f32_to_bf16(crate::bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) * scale);
            let le = v.to_le_bytes();
            // SAFETY: the caller's contract; `j * 2 + 1 < part.len()`.
            unsafe {
                *out.add(j * 2) = le[0];
                *out.add(j * 2 + 1) = le[1];
            }
        }
    };
    if threads == 1 {
        convert(src, dst);
        return;
    }
    // Chunks hold whole bf16 elements.
    let chunk = src.len().div_ceil(threads).div_ceil(2) * 2;
    let dst = dst as usize;
    std::thread::scope(|s| {
        for (i, part) in src.chunks(chunk).enumerate() {
            s.spawn(move || convert(part, (dst + i * chunk) as *mut u8));
        }
    });
}

/// Gather `rows` runs of `row_bytes` bytes, `pitch_bytes` apart in `src`,
/// into consecutive rows at `dst` (bf16 elements, scaled on the way when
/// `scale` is given), the rows split over up to eight threads.
///
/// # Safety
///
/// `dst` must be writable for `rows * row_bytes` bytes and must not
/// overlap `src`, which must hold `(rows - 1) * pitch_bytes + row_bytes`.
unsafe fn gather_rows_parallel(
    src: &[u8],
    rows: usize,
    row_bytes: usize,
    pitch_bytes: usize,
    scale: Option<f32>,
    dst: *mut u8,
) {
    const PER_THREAD: usize = 8 << 20;
    const THREADS: usize = 8;
    let threads = ((rows * row_bytes) / PER_THREAD).clamp(1, THREADS);
    let chunk = rows.div_ceil(threads);
    let dst = dst as usize;
    let run = |r0: usize, r1: usize| {
        for r in r0..r1 {
            let row = &src[r * pitch_bytes..r * pitch_bytes + row_bytes];
            let out = (dst + r * row_bytes) as *mut u8;
            // SAFETY: the caller's contract; rows are disjoint.
            unsafe {
                match scale {
                    Some(f) => copy_scaled_parallel(row, f, out),
                    None => core::ptr::copy_nonoverlapping(row.as_ptr(), out, row_bytes),
                }
            }
        }
    };
    if threads == 1 {
        run(0, rows);
        return;
    }
    std::thread::scope(|s| {
        for i in 0..threads {
            let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(rows));
            if r0 < r1 {
                s.spawn(move || run(r0, r1));
            }
        }
    });
}

/// Device buffers of persistent tensors that other runtimes own, keyed by
/// tensor name and element count, so that a recipe built with
/// [`Runtime::new_bound`] binds to them instead of allocating its own (the
/// generalisation of [`Runtime::new_with`] to several parents: the
/// tensor-parallel decoder shares its residual stream, partial sums, KV
/// cache and weights across half a dozen recipes). The first runtime
/// added under a key owns the buffer; a later one with the same name and
/// count is ignored.
#[derive(Default)]
pub(crate) struct Bindings {
    map: HashMap<(String, u64), u64>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add every persistent tensor of `rt` (inputs, scratch and the
    /// read-back tensor) that is not bound yet.
    pub fn add(&mut self, rt: &Runtime<'_>) {
        for (name, shape) in &rt.shapes {
            let elems = shape.iter().product::<u64>();
            self.map
                .entry((name.clone(), elems))
                .or_insert(rt.addrs[name]);
        }
    }

    fn get(&self, name: &str, elems: u64) -> Option<u64> {
        self.map.get(&(name.to_owned(), elems)).copied()
    }
}

/// Device buffers allocated and filled outside any recipe: the weights of
/// the layers a tensor-parallel runtime binds per launch (the recipe
/// itself owns one layer's), and zeroed state buffers. Uploads go through
/// the same staging ring as a runtime's inputs; [`Store::finish`] waits
/// for the last of them and releases the ring. Everything is freed on
/// drop.
pub(crate) struct Store {
    dev: synDeviceId,
    stream: synStreamHandle,
    owned: Vec<u64>,
    staging: Option<Staging>,
    fence: Fence,
    /// Bytes uploaded and allocated, for reports.
    pub bytes: u64,
}

impl Store {
    pub fn new(dev: synDeviceId, stream: synStreamHandle) -> Self {
        Self {
            dev,
            stream,
            owned: Vec::new(),
            staging: None,
            fence: Fence::none(),
            bytes: 0,
        }
    }

    /// A new device buffer holding `src` (bf16 bytes, scaled while staged
    /// when `scale` is given, gathered from a column window when `stride`
    /// is given; see `Staging::upload`).
    pub fn upload(
        &mut self,
        src: &[u8],
        scale: Option<f32>,
        stride: Option<Stride>,
    ) -> Result<u64> {
        let bytes = stride.map_or(src.len(), |s| s.rows * s.cols * 2);
        let mut d = 0u64;
        syn!(synDeviceMalloc(self.dev, bytes as u64, 0, 0, &mut d));
        self.owned.push(d);
        self.bytes += bytes as u64;
        if self.staging.is_none() {
            self.staging = Some(Staging::new(
                self.dev,
                Staging::SLOT_BYTES * Staging::SLOTS,
                Staging::SLOT_BYTES,
            )?);
        }
        let st = self.staging.as_mut().expect("staging ring");
        st.upload(
            self.dev,
            self.stream,
            &mut self.fence,
            src,
            scale,
            stride,
            d,
        )?;
        Ok(d)
    }

    /// A new zeroed device buffer of `bytes` bytes (rounded up to a whole
    /// 4-byte word, which is the granularity `synMemsetD32Async` zeroes at).
    pub fn alloc_zeroed(&mut self, bytes: u64) -> Result<u64> {
        let bytes = bytes.next_multiple_of(4);
        let mut d = 0u64;
        syn!(synDeviceMalloc(self.dev, bytes, 0, 0, &mut d));
        self.owned.push(d);
        self.bytes += bytes;
        syn!(synMemsetD32Async(d, 0, (bytes / 4) as usize, self.stream));
        Ok(d)
    }

    /// Wait until every upload so far has landed and release the staging
    /// ring (the next upload makes a new one).
    pub fn finish(&mut self) -> Result<()> {
        if let Some(mut st) = self.staging.take() {
            self.fence.wait(self.dev, self.stream)?;
            // SAFETY: every copy from the ring has landed.
            unsafe { st.free(self.dev) };
        }
        syn!(synStreamSynchronize(self.stream));
        Ok(())
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        // SAFETY: owned buffers, no copy in flight after the sync.
        unsafe {
            synStreamSynchronize(self.stream);
            if let Some(mut st) = self.staging.take() {
                st.free(self.dev);
            }
            self.fence.free(self.dev);
            for &d in &self.owned {
                synDeviceFree(self.dev, d, 0);
            }
        }
    }
}

/// A compiled recipe bound to device buffers for all of its persistent
/// tensors. Inputs are uploaded once at construction; [`Runtime::upload`]
/// replaces one input's contents between launches, and
/// [`Runtime::copy_d2d`] moves bytes between device-resident tensors.
pub(crate) struct Runtime<'a> {
    gb: Gb<'a>,
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
    /// Whether the read-back tensor's buffer is this runtime's own (else it
    /// is bound to another runtime's through [`Runtime::new_bound`]).
    owns_out: bool,
    /// Pinned host buffer per input for the per-step re-uploads
    /// ([`Runtime::upload`] and friends), allocated on first use; the
    /// construction-time upload goes through a small staging ring instead,
    /// so the weights never hold pinned memory.
    host_bufs: Vec<*mut c_void>,
    /// Whether each input has its own device buffer (else it is bound to
    /// the parent's and cannot be re-uploaded here).
    own_input: Vec<bool>,
    /// Device buffer per persistent tensor, by input then scratch index.
    dev_bufs: Vec<u64>,
    /// The device buffers this runtime allocated (freed on drop).
    owned: Vec<u64>,
    d_out: u64,
    h_out: *mut c_void,
    /// The buffers of [`Runtime::fence`].
    fence: Fence,
    /// Pinned buffer for [`Runtime::read_bf16_range`], grown on demand and
    /// held for the runtime's life. It is per runtime, not per family, so
    /// every key-bucket recipe that serves one block of a `feed_all` grows
    /// its own: `rows * vocab * 2` bytes of `synHostMalloc` each, 65.7 MB
    /// for Llama-3.1-8B at 256 rows, about 0.6 GB over the nine buckets a
    /// 32768-token cached prefill touches. Pinned memory is scarcer than
    /// pageable, so a path that reads all the logits from many buckets
    /// pays a host cost the device-byte accounting does not show.
    h_aux: *mut c_void,
    aux_bytes: u64,
    /// Pinned buffers for [`Runtime::upload_at`] and
    /// [`Runtime::read_i32_strided`], grown on demand.
    h_up: *mut c_void,
    up_bytes: u64,
    h_ring: *mut c_void,
    ring_bytes: u64,
    out: Out,
    dws: u64,
    /// Bytes of the workspace at `dws`, and whether this runtime allocated
    /// it. Recipes over one model differ only in their block and key
    /// shapes, so a child whose workspace fits the parent's borrows it:
    /// only one recipe of a runtime family is ever in flight (every launch
    /// is followed by a readback or a stream sync), and the workspace holds
    /// nothing between launches.
    ws_bytes: u64,
    owns_ws: bool,
}

impl<'a> Runtime<'a> {
    /// Compile `gb`, acquire a device, allocate every persistent tensor, and
    /// upload the inputs' host data.
    pub fn new(gb: Gb<'a>, out: Out) -> Result<Self> {
        Self::new_with(gb, out, None)
    }

    /// Like [`Runtime::new`], but sharing `parent`'s device and stream, and
    /// binding every persistent tensor whose name `parent` also has to the
    /// parent's buffer instead of allocating (and uploading) its own. A
    /// second recipe over the same weights and KV cache costs only its own
    /// per-step inputs and output. The child must be dropped before the
    /// parent.
    pub fn new_with(gb: Gb<'a>, out: Out, parent: Option<&Runtime<'_>>) -> Result<Self> {
        let borrowed = parent.map(|p| (p.dev, p.stream));
        let parent_ws = parent.map(|p| (p.dws, p.ws_bytes));
        // A tensor is shared with the parent when it has the same name AND
        // the same element count (a weight may be declared 4-D in one graph
        // and 5-D with a trailing 1 in another; the per-step inputs of a
        // narrower recipe have the same names but different counts).
        let lookup = |name: &str, elems: u64| -> Option<u64> {
            parent.and_then(|p| match (p.addrs.get(name), p.shapes.get(name)) {
                (Some(&d), Some(sh)) if sh.iter().product::<u64>() == elems => Some(d),
                _ => None,
            })
        };
        Self::build(gb, out, &lookup, borrowed, parent_ws)
    }

    /// Like [`Runtime::new`], but on a device and stream the caller has
    /// already acquired and owns (the multi-card probe, whose process holds
    /// exactly one device for its HCCL rank). Nothing is shared with another
    /// runtime, and drop leaves the device and stream alone.
    pub fn new_on(gb: Gb<'a>, out: Out, dev: synDeviceId, stream: synStreamHandle) -> Result<Self> {
        Self::build(gb, out, &|_, _| None, Some((dev, stream)), None)
    }

    /// Like [`Runtime::new_on`], binding every persistent tensor (the
    /// read-back tensor included) whose name and element count `bind` has
    /// to that buffer instead of allocating its own. The runtimes bound to
    /// must outlive this one.
    pub fn new_bound(
        gb: Gb<'a>,
        out: Out,
        dev: synDeviceId,
        stream: synStreamHandle,
        bind: &Bindings,
    ) -> Result<Self> {
        Self::build(gb, out, &|n, e| bind.get(n, e), Some((dev, stream)), None)
    }

    /// Persistent tensors `lookup` knows (by name and element count) bind
    /// to those buffers; a `borrowed` device and stream are used instead
    /// of acquiring one.
    #[allow(clippy::too_many_lines)]
    fn build(
        mut gb: Gb<'a>,
        out: Out,
        lookup: &dyn Fn(&str, u64) -> Option<u64>,
        borrowed: Option<(synDeviceId, synStreamHandle)>,
        parent_ws: Option<(u64, u64)>,
    ) -> Result<Self> {
        gb.serialize_if_requested()?;
        let trace = env_on("RENG_RECIPE_TRACE");
        let t0 = Instant::now();
        let recipe = compile_cached(&gb)?;
        let t_compile = t0.elapsed();

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

        let (dev, stream) = match borrowed {
            Some(ds) => ds,
            None => {
                let dev = crate::device::acquire_device()?;
                let mut stream: synStreamHandle = core::ptr::null_mut();
                syn!(synStreamCreateGeneric(&mut stream, dev, 0));
                (dev, stream)
            }
        };
        let t_device = t0.elapsed() - t_compile;
        let shared = |name: &CString, sizes: &[u64]| -> Option<u64> {
            lookup(name.to_str().unwrap(), sizes.iter().product::<u64>())
        };

        let n_in = gb.names.len();
        let n_scratch = gb.scratch_names.len();
        let mut dev_bufs: Vec<u64> = Vec::with_capacity(n_in + n_scratch);
        let mut owned: Vec<u64> = Vec::with_capacity(n_in + n_scratch);
        let mut own_input: Vec<bool> = Vec::with_capacity(n_in);
        let mut infos: Vec<synLaunchTensorInfo> = Vec::with_capacity(n_in + n_scratch + 1);
        let mut addrs = HashMap::with_capacity(n_in + n_scratch + 1);
        let mut info_index = HashMap::with_capacity(n_in + n_scratch + 1);
        let mut shapes = HashMap::with_capacity(n_in + n_scratch + 1);
        // The bytes of input `idx` as the device wants them: the raw bytes,
        // or the bf16 elements as is.
        let input_bytes = |idx: usize| -> &[u8] {
            match gb.raw[idx].as_deref() {
                Some(r) => r,
                None => {
                    let data: &[u16] = &gb.data[idx];
                    // SAFETY: a u16 slice is readable as twice as many bytes.
                    unsafe {
                        core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2)
                    }
                }
            }
        };
        // The bytes input `idx` occupies on the device: its host bytes, or
        // the column window of a strided source.
        let device_bytes = |idx: usize| -> usize {
            gb.strides[idx].map_or(input_bytes(idx).len(), |s| s.rows * s.cols * 2)
        };
        // Every input with its own device buffer goes through a ring of a
        // few pinned staging buffers sized for the largest of them, so the
        // pinned memory of an upload stays bounded whatever the model size.
        let (mut total, mut largest) = (0usize, 0usize);
        for idx in 0..n_in {
            if shared(&gb.names[idx], &gb.sizes[idx]).is_none() {
                let n = device_bytes(idx);
                total += n;
                largest = largest.max(n);
            }
        }
        let mut staging = Staging::new(dev, total, largest)?;
        let mut fence = Fence::none();
        // Device bytes this runtime allocates itself (shared buffers are the
        // parent's), reported by `RENG_RECIPE_TRACE`.
        let (mut in_bytes, mut scratch_bytes) = (0u64, 0u64);
        for (idx, &id) in ids.iter().take(n_in).enumerate() {
            let d = if let Some(d) = shared(&gb.names[idx], &gb.sizes[idx]) {
                own_input.push(false);
                d
            } else {
                let src = input_bytes(idx);
                let bytes = device_bytes(idx);
                let mut d = 0u64;
                syn!(synDeviceMalloc(dev, bytes as u64, 0, 0, &mut d));
                in_bytes += bytes as u64;
                owned.push(d);
                staging.upload(
                    dev,
                    stream,
                    &mut fence,
                    src,
                    gb.scales[idx],
                    gb.strides[idx],
                    d,
                )?;
                own_input.push(true);
                d
            };
            info_index.insert(gb.names[idx].to_str().unwrap().to_owned(), infos.len());
            infos.push(launch_info(&gb.names[idx], d, id, &gb.sizes[idx]));
            addrs.insert(gb.names[idx].to_str().unwrap().to_owned(), d);
            shapes.insert(
                gb.names[idx].to_str().unwrap().to_owned(),
                gb.sizes[idx].clone(),
            );
            dev_bufs.push(d);
        }
        // Every staged copy must have landed before its slot is freed (the
        // stream sync alone does not guarantee that on this stack).
        if total > 0 {
            fence.wait(dev, stream)?;
        }
        // SAFETY: no copy reads the ring any more.
        unsafe { staging.free(dev) };
        let host_bufs: Vec<*mut c_void> = vec![core::ptr::null_mut(); n_in];
        // The host copies of the inputs (the weights, mostly) are not needed
        // once they are on the device or bound to a parent: uploads are
        // validated against the tensor sizes.
        for d in &mut gb.data {
            *d = std::borrow::Cow::Owned(Vec::new());
        }
        for (k, sizes) in gb.scratch_sizes.iter().enumerate() {
            // Rounded up to a whole `synMemsetD32Async` word, so the zero
            // fill below covers the last element of a buffer whose byte
            // count is not a multiple of 4 (an fp8 or bf16 tensor with a
            // ragged element count).
            let bytes = sizes.iter().product::<u64>() * gb.scratch_elem[k] as u64;
            let bytes = bytes.next_multiple_of(4);
            let d = if let Some(of) = &gb.scratch_alias[k] {
                // The output side of an in-place update: same memory.
                addrs[of.as_str()]
            } else if let Some(d) = shared(&gb.scratch_names[k], sizes) {
                d
            } else {
                let mut d = 0u64;
                syn!(synDeviceMalloc(dev, bytes, 0, 0, &mut d));
                scratch_bytes += bytes;
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
        // The read-back tensor: a buffer of its own, or one another runtime
        // owns (a recipe whose product is state the next recipe reads).
        let (d_out, owns_out) = match shared(&out.name, &out.sizes) {
            Some(d) => (d, false),
            None => {
                let mut d = 0u64;
                syn!(synDeviceMalloc(dev, out_bytes, 0, 0, &mut d));
                (d, true)
            }
        };
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
        shapes.insert(out.name.to_str().unwrap().to_owned(), out.sizes.clone());

        let mut ws = 0u64;
        syn!(synWorkspaceGetSize(&mut ws, recipe));
        // Diagnostic: `RENG_WS_SLACK_MB` over-allocates the workspace.
        let slack = std::env::var("RENG_WS_SLACK_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
            << 20;
        let need = ws + slack;
        // The parent's workspace when it is big enough (see the field docs).
        // Sharing it is safe only while the two runtimes launch in order on
        // one stream, which they do because a child borrows the parent's.
        debug_assert!(
            parent_ws.is_none() || borrowed.is_some(),
            "a runtime offered a workspace must also run on the lender's stream"
        );
        let (mut dws, ws_bytes, owns_ws) = match parent_ws {
            Some((addr, bytes)) if addr != 0 && bytes >= need => (addr, bytes, false),
            _ => (0, need, true),
        };
        if owns_ws && need > 0 {
            syn!(synDeviceMalloc(dev, need, 0, 0, &mut dws));
        }
        syn!(synStreamSynchronize(stream));
        if trace {
            let to_gb = |b: u64| b as f64 / 1e9;
            eprintln!(
                "runtime: compile {:.2} s, device {:.2} s, buffers + uploads {:.2} s ({} inputs, {} shared); device bytes {:.2} GB (inputs {:.2}, scratch {:.2}, output {:.2}, workspace {:.2})",
                t_compile.as_secs_f64(),
                t_device.as_secs_f64(),
                (t0.elapsed() - t_compile - t_device).as_secs_f64(),
                n_in,
                own_input.iter().filter(|o| !**o).count(),
                to_gb(in_bytes + scratch_bytes + out_bytes + if owns_ws { need } else { 0 }),
                to_gb(in_bytes),
                to_gb(scratch_bytes),
                to_gb(out_bytes),
                to_gb(if owns_ws { need } else { 0 })
            );
        }
        Ok(Self {
            gb,
            dev,
            stream,
            owns_device: borrowed.is_none(),
            owns_out,
            recipe,
            infos,
            info_index,
            addrs,
            shapes,
            host_bufs,
            own_input,
            dev_bufs,
            owned,
            d_out,
            h_out,
            fence,
            h_aux: core::ptr::null_mut(),
            aux_bytes: 0,
            h_up: core::ptr::null_mut(),
            up_bytes: 0,
            h_ring: core::ptr::null_mut(),
            ring_bytes: 0,
            out,
            dws,
            ws_bytes,
            owns_ws,
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

    /// The pinned host buffer of `bytes` bytes input `idx` is re-uploaded
    /// from, allocated on the first call.
    ///
    /// # Panics
    ///
    /// Panics if the input is bound to a parent's buffer.
    fn host_buf(&mut self, idx: usize, bytes: usize) -> Result<*mut c_void> {
        assert!(
            self.own_input[idx],
            "input {idx} is bound to a shared buffer"
        );
        if self.host_bufs[idx].is_null() {
            syn!(synHostMalloc(
                self.dev,
                bytes as u64,
                0,
                &mut self.host_bufs[idx]
            ));
        }
        Ok(self.host_bufs[idx])
    }

    /// Replace input `idx`'s contents (bf16-converted, `data.len()` must equal
    /// the input's element count) ahead of the next launch.
    pub fn upload(&mut self, idx: usize, data: &[f32]) -> Result<()> {
        assert_eq!(data.len(), self.input_elems(idx));
        let hb = self.host_buf(idx, data.len() * 2)?;
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
        let expect = self.gb.raw[idx].as_ref().map_or(0, |b| b.len());
        assert_eq!(bytes.len(), expect, "raw input {idx} size");
        let hb = self.host_buf(idx, expect)?;
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
        let hb = self.host_buf(idx, data.len() * 2)?;
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
        let (dev, stream) = (self.dev, self.stream);
        self.fence.wait(dev, stream)
    }

    /// Index of the input named `name`.
    ///
    /// # Panics
    ///
    /// Panics if there is no such input.
    pub fn input_index(&self, name: &str) -> usize {
        self.find_input(name)
            .unwrap_or_else(|| panic!("no input named {name}"))
    }

    /// The index of input `name`, if the graph has one.
    pub fn find_input(&self, name: &str) -> Option<usize> {
        self.gb
            .names
            .iter()
            .position(|n| n.to_str().unwrap() == name)
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

    /// The launch-table index of persistent tensor `name`, for
    /// [`Runtime::rebind_at`] (a rebind without the name lookup, for the
    /// hundreds of rebinds per token of the tensor-parallel decoder).
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a persistent tensor of this graph.
    pub fn bind_index(&self, name: &str) -> usize {
        self.info_index[name]
    }

    /// [`Runtime::rebind`] by launch-table index.
    pub fn rebind_at(&mut self, idx: usize, addr: u64) {
        self.infos[idx].tensor_address = addr;
    }

    /// Enqueue a fill of `words` 32-bit words at `addr` with `value` on
    /// the stream (ordered with the launches around it).
    ///
    /// # Errors
    ///
    /// Returns an error if the fill cannot be enqueued.
    pub fn memset_d32(&mut self, addr: u64, value: u32, words: usize) -> Result<()> {
        syn!(synMemsetD32Async(addr, value, words, self.stream));
        Ok(())
    }

    /// Enqueue one launch without reading anything back (launches on the
    /// stream run in order; a later [`Runtime::launch_and_read`] completes
    /// after all of them). For throughput measurements.
    ///
    /// The workspace at `dws` may be borrowed from another runtime of the
    /// family (see the `ws_bytes` field), and a launch owns it until it
    /// completes. That is safe only because every runtime of a family
    /// shares one stream and launches on it run in order; anything that
    /// gives a family a second stream, or launches two of its recipes to
    /// be read later, has to stop sharing the workspace first.
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
    pub(crate) fn launch_and_read_bytes(&mut self, first: usize, rows: usize) -> Result<Vec<u8>> {
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

    /// A zeroed device buffer of `bytes` bytes owned by this runtime (freed
    /// on drop), for state the recipe's tensors are bound into per launch
    /// (see [`Runtime::rebind`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation or the clearing fails.
    pub fn alloc(&mut self, bytes: u64) -> Result<u64> {
        // Rounded up to a whole memset word (see `alloc_zeroed`).
        let bytes = bytes.next_multiple_of(4);
        let mut d = 0u64;
        syn!(synDeviceMalloc(self.dev, bytes, 0, 0, &mut d));
        self.owned.push(d);
        syn!(synMemsetD32Async(d, 0, (bytes / 4) as usize, self.stream));
        syn!(synStreamSynchronize(self.stream));
        Ok(d)
    }

    /// Grow a pinned buffer to at least `bytes`.
    fn grow_pinned(
        dev: synDeviceId,
        buf: &mut *mut c_void,
        have: &mut u64,
        bytes: u64,
    ) -> Result<()> {
        if *have < bytes {
            if !buf.is_null() {
                // SAFETY: allocated below on an earlier call, no copy in flight.
                unsafe { synHostFree(dev, *buf, 0) };
            }
            let mut hb: *mut c_void = core::ptr::null_mut();
            syn!(synHostMalloc(dev, bytes, 0, &mut hb));
            *buf = hb;
            *have = bytes;
        }
        Ok(())
    }

    /// Enqueue a copy of `bytes` to device address `addr` (through a pinned
    /// buffer of this runtime); [`Runtime::fence`] waits until it is
    /// visible. One call at a time: the next call reuses the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy cannot be enqueued.
    pub fn upload_at(&mut self, addr: u64, bytes: &[u8]) -> Result<()> {
        self.upload_at_multi(&[(addr, bytes)])
    }

    /// Enqueue one copy per `(addr, bytes)` part through the same pinned
    /// buffer (each part at its own 128-byte aligned offset), so that one
    /// [`Runtime::fence`] covers them all. One call at a time, as
    /// [`Runtime::upload_at`]; empty parts are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if a copy cannot be enqueued.
    pub fn upload_at_multi(&mut self, parts: &[(u64, &[u8])]) -> Result<()> {
        const ALIGN: usize = 128;
        let total: usize = parts
            .iter()
            .map(|p| p.1.len().div_ceil(ALIGN) * ALIGN)
            .sum();
        if total == 0 {
            return Ok(());
        }
        Self::grow_pinned(self.dev, &mut self.h_up, &mut self.up_bytes, total as u64)?;
        let mut off = 0usize;
        for &(addr, bytes) in parts {
            if bytes.is_empty() {
                continue;
            }
            // SAFETY: h_up holds `total` bytes, of which `off..off +
            // bytes.len()` belong to this part, and no copy reads it (the
            // caller fenced the previous call).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.h_up.cast::<u8>().add(off),
                    bytes.len(),
                );
            }
            syn!(synMemCopyAsync(
                self.stream,
                self.h_up as u64 + off as u64,
                bytes.len() as u64,
                addr,
                SYN_HOST_TO_DRAM
            ));
            off += bytes.len().div_ceil(ALIGN) * ALIGN;
        }
        Ok(())
    }

    /// Pre-fill `words` 32-bit words at `addr` with the recipe-completion
    /// sentinel, ahead of launches that will overwrite them (see
    /// [`Runtime::read_i32_strided`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the fill cannot be enqueued.
    pub fn fill_sentinel_d32(&mut self, addr: u64, words: usize) -> Result<()> {
        syn!(synMemsetD32Async(addr, SENTINEL_D32, words, self.stream));
        syn!(synStreamSynchronize(self.stream));
        Ok(())
    }

    /// Read `n` int32 values `stride` bytes apart starting at device
    /// address `addr`: the first word of each of `n` slots that were
    /// pre-filled by [`Runtime::fill_sentinel_d32`] and are each written
    /// once by a launch enqueued since. The two-sentinel protocol of the
    /// module docs applies to those words: the copy is repeated until none
    /// shows the device sentinel (every launch has written its slot) and
    /// the host waits until none shows the host sentinel (the copy has
    /// landed). The slots' padding words are copied too but never checked.
    ///
    /// # Errors
    ///
    /// Returns an error if a copy fails or the slots never complete.
    ///
    /// # Panics
    ///
    /// Panics if `stride` is not a positive multiple of 4 or `n` is 0.
    pub fn read_i32_strided(&mut self, addr: u64, stride: usize, n: usize) -> Result<Vec<i32>> {
        self.read_i32_rows(addr, stride, n, 1)
    }

    /// Like [`Runtime::read_i32_strided`] for rows of `words` int32 values
    /// each: the first `words` words of each of `n` slots of `stride`
    /// bytes, every one pre-filled with the sentinel and written once by a
    /// launch; the result is the `n * words` values row by row.
    ///
    /// # Errors
    ///
    /// Returns an error if a copy fails or the slots never complete.
    ///
    /// # Panics
    ///
    /// Panics if `stride` is not a positive multiple of 4, `n` or `words`
    /// is 0, or a row's words do not fit its slot.
    pub fn read_i32_rows(
        &mut self,
        addr: u64,
        stride: usize,
        n: usize,
        words: usize,
    ) -> Result<Vec<i32>> {
        assert!(stride >= 4 && stride % 4 == 0 && n >= 1 && words >= 1);
        assert!(
            words * 4 <= stride,
            "{words} words do not fit a {stride}-byte slot"
        );
        let bytes = (stride * n) as u64;
        Self::grow_pinned(self.dev, &mut self.h_ring, &mut self.ring_bytes, bytes)?;
        let (stream, h) = (self.stream, self.h_ring.cast::<u32>());
        let step = stride / 4;
        // SAFETY (all blocks below): h_ring holds `bytes` bytes, so word
        // `j * step + w` is inside it for every j < n and w < words; the
        // DMA writes each word exactly once per copy.
        let picked = |pattern: u32| -> bool {
            (0..n).any(|j| {
                (0..words)
                    .any(|w| unsafe { core::ptr::read_volatile(h.add(j * step + w)) } == pattern)
            })
        };
        let started = Instant::now();
        let mut polls = 0u32;
        loop {
            unsafe {
                for j in 0..n {
                    for w in 0..words {
                        *h.add(j * step + w) = HOST_SENTINEL_D32;
                    }
                }
            }
            syn!(synMemCopyAsync(
                stream,
                addr,
                bytes,
                self.h_ring as u64,
                SYN_DRAM_TO_HOST
            ));
            syn!(synStreamSynchronize(stream));
            while picked(HOST_SENTINEL_D32) {
                if started.elapsed() > READBACK_TIMEOUT {
                    return Err(Error::Other(format!(
                        "device-to-host copy did not complete within {READBACK_TIMEOUT:?}"
                    )));
                }
                std::hint::spin_loop();
            }
            if !picked(SENTINEL_D32) {
                break;
            }
            if started.elapsed() > READBACK_TIMEOUT {
                return Err(Error::Other(format!(
                    "{n} launches did not complete within {READBACK_TIMEOUT:?}"
                )));
            }
            std::thread::sleep(Duration::from_micros(200));
            polls += 1;
        }
        if env_on("RENG_STEP_TRACE") {
            eprintln!(
                "loop trace: readback of {n} x {words} ids {:.2} ms ({polls} polls)",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok((0..n)
            .flat_map(|j| {
                (0..words)
                    .map(move |w| unsafe { core::ptr::read_volatile(h.add(j * step + w)) } as i32)
            })
            .collect())
    }
}

impl<'a> Runtime<'a> {
    /// Diagnostic: after a long settle, copy every scratch tensor down and
    /// print its zero fraction and largest magnitude, in creation order.
    fn dump_scratch(&self) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        std::thread::sleep(Duration::from_millis(100));
        let n_in = self.gb.names.len();
        for (k, sizes) in self.gb.scratch_sizes.iter().enumerate() {
            let elems = sizes.iter().product::<u64>() as usize;
            let elem = self.gb.scratch_elem[k];
            let bytes = (elems * elem) as u64;
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
            // SAFETY: hb holds `elems` elements of `elem` bytes each, and
            // each arm below reads exactly that width (1 for the fp8
            // formats, 2 for bf16, 4 for f32 and int32).
            unsafe {
                for j in 0..elems {
                    let v = match elem {
                        4 => *hb.cast::<f32>().add(j),
                        2 => bf16_to_f32(*hb.cast::<u16>().add(j)),
                        // One byte: an fp8 code. Its magnitude needs the
                        // format, which the scratch record does not carry,
                        // so report the code as the number it is.
                        _ => f32::from(*hb.cast::<u8>().add(j)),
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

/// Bump when the cache file format or the key's meaning changes.
const RECIPE_CACHE_VERSION: &str = "1";

/// The recipe cache directory: `RENG_RECIPE_CACHE` (a path, or `0`/`off`
/// to disable), else `$HOME/.cache/reng/recipes`.
fn recipe_cache_dir() -> Option<std::path::PathBuf> {
    match std::env::var("RENG_RECIPE_CACHE") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.is_empty() => None,
        Ok(v) => Some(std::path::PathBuf::from(v)),
        Err(_) => std::env::var("HOME")
            .ok()
            .map(|h| std::path::PathBuf::from(h).join(".cache/reng/recipes")),
    }
}

/// Everything besides the graph that changes what the compiler emits:
/// the SynapseAI version and the compiler's environment knobs.
fn compiler_salt() -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    RECIPE_CACHE_VERSION.hash(&mut h);
    let mut ver = [0u8; 128];
    // SAFETY: the buffer holds 128 bytes and the call writes at most that.
    let st = unsafe { synDriverGetVersion(ver.as_mut_ptr().cast::<core::ffi::c_char>(), 128) };
    if st == SYN_SUCCESS {
        ver.hash(&mut h);
    }
    let mut knobs: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            [
                "ENABLE_",
                "DISABLE_",
                "GC_",
                "HABANA_",
                "SRAM_",
                "PIPELINE_",
                "DEFAULT_PIPELINE",
                "COMMON_DIM",
                "TPC_",
                "MME_",
                "RUN_TPC",
                "HL_",
                "SDPA_",
            ]
            .iter()
            .any(|p| k.starts_with(p))
        })
        .collect();
    knobs.sort();
    knobs.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Compile `gb`, or load the recipe a graph of the same structure compiled
/// to before from the cache; a fresh compile is stored for next time. A
/// cache file that fails to load is compiled over. `RENG_RECIPE_TRACE`
/// reports hits and misses.
fn compile_cached(gb: &Gb) -> Result<synRecipeHandle> {
    let trace = env_on("RENG_RECIPE_TRACE");
    let mut recipe: synRecipeHandle = core::ptr::null_mut();
    let path = recipe_cache_dir()
        .map(|d| d.join(format!("{}-{}.recipe", gb.cache_key(), compiler_salt())));
    if let Some(p) = &path {
        if p.is_file() {
            let c = CString::new(p.to_string_lossy().as_bytes()).unwrap();
            // SAFETY: valid pointers; a failure leaves `recipe` null.
            let st = unsafe { synRecipeDeSerialize(&mut recipe, c.as_ptr()) };
            if st == SYN_SUCCESS && !recipe.is_null() {
                if trace {
                    eprintln!("recipe cache: hit {}", p.display());
                }
                return Ok(recipe);
            }
            eprintln!(
                "recipe cache: {} did not load (synStatus {st}), compiling",
                p.display()
            );
            recipe = core::ptr::null_mut();
        }
    }
    syn!(synGraphCompile(
        &mut recipe,
        gb.graph,
        CString::new("model").unwrap().as_ptr(),
        core::ptr::null()
    ));
    if let Some(p) = &path {
        if let Some(dir) = p.parent() {
            if std::fs::create_dir_all(dir).is_ok() {
                let tmp = p.with_extension(format!("tmp{}", std::process::id()));
                let c = CString::new(tmp.to_string_lossy().as_bytes()).unwrap();
                // SAFETY: valid recipe and path.
                let st = unsafe { synRecipeSerialize(recipe, c.as_ptr()) };
                if st == SYN_SUCCESS && std::fs::rename(&tmp, p).is_ok() {
                    if trace {
                        eprintln!("recipe cache: stored {}", p.display());
                    }
                } else {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!(
                        "recipe cache: could not store {} (synStatus {st})",
                        p.display()
                    );
                }
            }
        }
    }
    Ok(recipe)
}

impl Drop for Runtime<'_> {
    fn drop(&mut self) {
        // `RENG_RECIPE_TRACE` also reports where the teardown time goes.
        let trace = env_on("RENG_RECIPE_TRACE");
        let t0 = Instant::now();
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let mut mark = |what: &'static str, since: &mut Instant| {
            if trace {
                marks.push((what, since.elapsed().as_secs_f64()));
                *since = Instant::now();
            }
        };
        let mut since = t0;
        unsafe {
            synStreamSynchronize(self.stream);
            for &hb in &self.host_bufs {
                if !hb.is_null() {
                    synHostFree(self.dev, hb, 0);
                }
            }
            synHostFree(self.dev, self.h_out, 0);
            self.fence.free(self.dev);
            for hb in [self.h_aux, self.h_up, self.h_ring] {
                if !hb.is_null() {
                    synHostFree(self.dev, hb, 0);
                }
            }
            mark("host frees", &mut since);
            for &d in &self.owned {
                synDeviceFree(self.dev, d, 0);
            }
            if self.owns_out {
                synDeviceFree(self.dev, self.d_out, 0);
            }
            if self.dws != 0 && self.owns_ws {
                synDeviceFree(self.dev, self.dws, 0);
            }
            mark("device frees", &mut since);
            synRecipeDestroy(self.recipe);
            synGraphDestroy(self.gb.graph);
            mark("recipe + graph destroy", &mut since);
            if self.owns_device {
                synStreamDestroy(self.stream);
                synDeviceRelease(self.dev);
                mark("stream destroy + device release", &mut since);
                synDestroy();
                mark("synDestroy", &mut since);
            }
        }
        if trace {
            let parts: Vec<String> = marks.iter().map(|(w, t)| format!("{w} {t:.2} s")).collect();
            eprintln!(
                "runtime teardown: {} (total {:.2} s)",
                parts.join(", "),
                t0.elapsed().as_secs_f64()
            );
        }
    }
}
