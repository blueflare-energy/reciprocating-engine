//! HCCL collectives from Rust, and the worker side of the two-card probe
//! `reng-hccl-test` (`bin/hccl_test.rs` is the coordinator).
//!
//! The process model is one card per process: HCCL supports a single device
//! per process and uses the device the process acquired first, so a rank is
//! a process that acquires one card by module id ([`Card`]), creates one
//! generic stream, and joins a communicator ([`Comm`]) whose 1032-byte
//! unique id ([`UniqueId`]) rank 0 created and the coordinator carried to
//! the others. Collectives are stream operations (not graph nodes): they
//! are enqueued on a `synStreamHandle` after or before recipe launches and
//! DMA copies on the same stream.
//!
//! The probe ([`run_worker`]) checks a summed all-reduce against an exact
//! integer expectation, times all-reduces over a size sweep
//! ([`SWEEP_BYTES`], 8 KB to 64 MB), and settles the ordering question
//! the tensor-parallel design depends on:
//! whether a recipe launch, an all-reduce and another launch on one stream
//! execute in order without host synchronisation (and, for comparison, the
//! same with the collective on a second stream bridged by events).

use crate::ffi::*;
use crate::model::Gb;
use crate::runtime::{HOST_SENTINEL_D32, Out, OutKind, Runtime, SENTINEL_D32};
use core::ffi::{CStr, c_int, c_void};
use reng_core::{Error, Result};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// The text of an HCCL status plus the library's last error message.
fn hccl_error(what: &str, st: hcclResult_t) -> Error {
    // SAFETY: both functions return a static or library-owned C string,
    // or null.
    let (name, last) = unsafe {
        let p = hcclGetErrorString(st);
        let name = if p.is_null() {
            String::from("?")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        let q = hcclGetLastErrorMessage();
        let last = if q.is_null() {
            String::new()
        } else {
            CStr::from_ptr(q).to_string_lossy().into_owned()
        };
        (name, last)
    };
    Error::Other(format!(
        "{what} -> hcclResult {st} ({name}){}{last}",
        if last.is_empty() { "" } else { ": " }
    ))
}

macro_rules! hccl {
    ($call:expr) => {{
        let st = unsafe { $call };
        if st != hcclSuccess {
            return Err(hccl_error(stringify!($call), st));
        }
    }};
}

/// The HCCL library version (`hcclGetVersion`, 2604 for HCCL 2.6.4).
///
/// # Errors
///
/// Returns an error if the call fails.
pub fn version() -> Result<i32> {
    let mut v: c_int = 0;
    hccl!(hcclGetVersion(&mut v));
    Ok(v)
}

/// An `hcclUniqueId`: 1024 bytes of opaque content plus its length, 1032
/// bytes in all, serialised as the raw struct (content, then the length as
/// eight little-endian bytes) so a coordinator can carry it between
/// processes. Rank 0 creates one with [`UniqueId::create`]; with
/// `HCCL_COMM_ID=<IP:PORT>` set the other ranks may pass
/// [`UniqueId::zeroed`] instead (hccl_demo's non-MPI mode).
#[derive(Clone, Copy)]
pub struct UniqueId(hcclUniqueId);

impl UniqueId {
    /// The serialised size.
    pub const BYTES: usize = HCCL_UNIQUE_ID_MAX_BYTES + 8;

    /// An all-zero id (length 0).
    #[must_use]
    pub fn zeroed() -> Self {
        Self(hcclUniqueId {
            internal: [0; HCCL_UNIQUE_ID_MAX_BYTES],
            length: 0,
        })
    }

    /// `hcclGetUniqueId`: starts the communicator's coordinator in this
    /// process (rank 0) and returns its address.
    ///
    /// # Errors
    ///
    /// Returns an error if the call fails.
    pub fn create() -> Result<Self> {
        let mut id = Self::zeroed();
        hccl!(hcclGetUniqueId(&mut id.0));
        Ok(id)
    }

    /// The 1032 serialised bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::BYTES);
        b.extend_from_slice(&self.0.internal);
        b.extend_from_slice(&(self.0.length as u64).to_le_bytes());
        b
    }

    /// The inverse of [`UniqueId::to_bytes`].
    ///
    /// # Errors
    ///
    /// Returns an error unless `b` has exactly 1032 bytes.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() != Self::BYTES {
            return Err(Error::Other(format!(
                "unique id: {} bytes, expected {}",
                b.len(),
                Self::BYTES
            )));
        }
        let mut id = Self::zeroed();
        id.0.internal
            .copy_from_slice(&b[..HCCL_UNIQUE_ID_MAX_BYTES]);
        let mut len = [0u8; 8];
        len.copy_from_slice(&b[HCCL_UNIQUE_ID_MAX_BYTES..]);
        id.0.length = usize::try_from(u64::from_le_bytes(len))
            .map_err(|_| Error::Other("unique id: bad length".into()))?;
        Ok(id)
    }

    /// The length field.
    #[must_use]
    pub fn length(&self) -> usize {
        self.0.length
    }

    /// The printable prefix of the content (the coordinator's `IP:PORT`
    /// on this stack), for logs.
    #[must_use]
    pub fn text(&self) -> String {
        let n = self.0.length.min(HCCL_UNIQUE_ID_MAX_BYTES);
        self.0.internal[..n]
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| if c.is_ascii_graphic() { c as char } else { '.' })
            .collect()
    }
}

/// Element type of a collective.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
}

impl DType {
    fn code(self) -> hcclDataType_t {
        match self {
            Self::F32 => hcclFloat32,
            Self::Bf16 => hcclBfloat16,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// An acquired card and one generic stream on it: the device side of one
/// rank. Drop releases the device and tears the process's Synapse state
/// down, so a process holds exactly one.
pub struct Card {
    dev: synDeviceId,
    /// Null until [`Card::create_stream`].
    stream: synStreamHandle,
    module: u32,
}

impl Card {
    /// `synInitialize` (tolerating an earlier one) and acquire the card
    /// with module id `module`. The stream comes separately
    /// ([`Card::create_stream`]), after the communicator, which is the
    /// order hccl_demo uses.
    ///
    /// # Errors
    ///
    /// Returns an error if the acquire fails (the card is held by another
    /// process, out of service, or unbound).
    pub fn acquire(module: u32) -> Result<Self> {
        // SAFETY: plain library init.
        let st = unsafe { synInitialize() };
        if st != SYN_SUCCESS && st != SYN_OBJECT_ALREADY_INITIALIZED {
            return Err(Error::Other(format!("synInitialize -> synStatus {st}")));
        }
        let mut dev: synDeviceId = 0;
        // SAFETY: valid out-pointer.
        let st = unsafe { synDeviceAcquireByModuleId(&mut dev, module) };
        if st != SYN_SUCCESS {
            return Err(Error::Other(format!(
                "synDeviceAcquireByModuleId({module}) -> synStatus {st}"
            )));
        }
        Ok(Self {
            dev,
            stream: core::ptr::null_mut(),
            module,
        })
    }

    /// Create the card's generic stream (once).
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be made.
    pub fn create_stream(&mut self) -> Result<()> {
        assert!(self.stream.is_null(), "stream already created");
        syn!(synStreamCreateGeneric(&mut self.stream, self.dev, 0));
        Ok(())
    }

    /// The module id the card was acquired by.
    #[must_use]
    pub fn module(&self) -> u32 {
        self.module
    }

    /// The Synapse device id the acquire returned.
    #[must_use]
    pub fn device_id(&self) -> u32 {
        self.dev
    }

    /// The card's generic stream (null before [`Card::create_stream`]).
    #[must_use]
    pub fn stream_handle(&self) -> synStreamHandle {
        self.stream
    }

    /// Wait for the stream's queued work (as far as the stack's sync goes:
    /// it returns before DMA copies have landed, see `runtime.rs`).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        syn!(synStreamSynchronize(self.stream));
        Ok(())
    }

    /// A second generic stream on the card (freed by [`Card::destroy_stream`]).
    fn new_stream(&self) -> Result<synStreamHandle> {
        let mut s: synStreamHandle = core::ptr::null_mut();
        syn!(synStreamCreateGeneric(&mut s, self.dev, 0));
        Ok(s)
    }

    fn destroy_stream(s: synStreamHandle) {
        // SAFETY: a stream from `new_stream`, no longer used.
        unsafe { synStreamDestroy(s) };
    }

    fn dev_alloc(&self, bytes: u64) -> Result<u64> {
        let mut d = 0u64;
        syn!(synDeviceMalloc(self.dev, bytes, 0, 0, &mut d));
        Ok(d)
    }

    fn dev_free(&self, d: u64) {
        // SAFETY: a buffer from `dev_alloc`, no longer in use.
        unsafe { synDeviceFree(self.dev, d, 0) };
    }

    fn host_alloc(&self, bytes: u64) -> Result<*mut c_void> {
        let mut h: *mut c_void = core::ptr::null_mut();
        syn!(synHostMalloc(self.dev, bytes, 0, &mut h));
        Ok(h)
    }

    fn host_free(&self, h: *mut c_void) {
        // SAFETY: a buffer from `host_alloc`, no longer in use.
        unsafe { synHostFree(self.dev, h, 0) };
    }
}

impl Drop for Card {
    fn drop(&mut self) {
        // SAFETY: owned handles; the device is released once.
        unsafe {
            if !self.stream.is_null() {
                synStreamSynchronize(self.stream);
                synStreamDestroy(self.stream);
            }
            synDeviceRelease(self.dev);
            synDestroy();
        }
    }
}

/// Leave the process at once, skipping every destructor and atexit
/// handler. After an HCL failure the orderly path (`hcclCommDestroy`,
/// `synDeviceRelease`, `synDestroy`) fails inside libSynapse
/// ("Failed to destroy HCCL device", then `std::unexpected` in
/// `~DeviceCommon`) and the process hangs in the failure-analysis dump for
/// minutes while still holding the card; the kernel driver reclaims the
/// device on exit anyway.
pub fn die(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` never returns.
    unsafe { _exit(code) }
}

/// How long a rank asked to stop is given to abort its communicator and
/// leave before the coordinator kills it, and how long
/// [`abort_and_die`] waits for `hcclCommAbort` before leaving anyway.
pub const ABORT_GRACE: Duration = Duration::from_secs(10);

/// Exit code a worker leaves with when the coordinator asked the group to
/// stop (or went away) while the rank was running.
pub const EXIT_ABORTED: i32 = 76;

/// Abort this process's communicator, if it has one, and leave at once
/// through [`die`], so a rank that is stuck in a collective releases its
/// card instead of being killed inside HCL (Multi-Card risk 10: a process
/// killed mid-collective can leave the card needing a reboot). A second
/// thread holds a hard deadline of [`ABORT_GRACE`] in case
/// `hcclCommAbort` itself blocks.
pub fn abort_and_die(code: i32) -> ! {
    use std::io::Write;
    std::thread::spawn(move || {
        std::thread::sleep(ABORT_GRACE);
        let _ = writeln!(
            std::io::stderr(),
            "hcclCommAbort did not return in {} s, leaving anyway",
            ABORT_GRACE.as_secs()
        );
        die(code);
    });
    let t = Instant::now();
    Comm::abort_current();
    let _ = writeln!(
        std::io::stderr(),
        "hcclCommAbort returned in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    die(code)
}

/// The live communicator handle of this process (0 when there is none),
/// so [`abort_and_die`] can reach it from another thread. One card and one
/// communicator per process, so one slot is enough.
static ABORT_COMM: AtomicUsize = AtomicUsize::new(0);

/// An HCCL communicator this process is one rank of.
pub struct Comm {
    h: hcclComm_t,
}

impl Comm {
    /// `hcclCommInitRank`: join the communicator `id` names as `rank` of
    /// `world`. Blocks until every rank has joined. Must be called after the
    /// process acquired its card.
    ///
    /// # Errors
    ///
    /// Returns an error if the init fails (`hcclPortDown`: a scale-up link
    /// between the cards is down; `hcclSocketError`: the sideband TCP
    /// connection to rank 0's coordinator failed).
    pub fn init_rank(world: usize, id: &UniqueId, rank: usize) -> Result<Self> {
        let mut h: hcclComm_t = core::ptr::null_mut();
        let world = c_int::try_from(world).map_err(|_| Error::Other("world too large".into()))?;
        let rank = c_int::try_from(rank).map_err(|_| Error::Other("rank too large".into()))?;
        hccl!(hcclCommInitRank(&mut h, world, id.0, rank));
        ABORT_COMM.store(h as usize, Ordering::SeqCst);
        Ok(Self { h })
    }

    /// `hcclCommCount`.
    ///
    /// # Errors
    ///
    /// Returns an error if the call fails.
    pub fn count(&self) -> Result<i32> {
        let mut v: c_int = 0;
        hccl!(hcclCommCount(self.h, &mut v));
        Ok(v)
    }

    /// `hcclCommUserRank`.
    ///
    /// # Errors
    ///
    /// Returns an error if the call fails.
    pub fn user_rank(&self) -> Result<i32> {
        let mut v: c_int = 0;
        hccl!(hcclCommUserRank(self.h, &mut v));
        Ok(v)
    }

    /// `hcclCommSynDevice`.
    ///
    /// # Errors
    ///
    /// Returns an error if the call fails.
    pub fn syn_device(&self) -> Result<i32> {
        let mut v: c_int = 0;
        hccl!(hcclCommSynDevice(self.h, &mut v));
        Ok(v)
    }

    /// The communicator's asynchronous error state, `Ok(None)` when clean,
    /// else the code and the library's message.
    ///
    /// # Errors
    ///
    /// Returns an error if the query itself fails.
    pub fn async_error(&self) -> Result<Option<(i32, String)>> {
        let mut e: hcclResult_t = 0;
        hccl!(hcclCommGetAsyncError(self.h, &mut e));
        if e == hcclSuccess {
            return Ok(None);
        }
        // SAFETY: a library-owned C string or null.
        let msg = unsafe {
            let p = hcclCommGetAsyncErrorMessage(self.h);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Ok(Some((e, msg)))
    }

    /// Enqueue a summed all-reduce of `count` elements from device address
    /// `send` into device address `recv` (in place when equal) on `card`'s
    /// stream. Returns when the operation is enqueued, not when it is done.
    ///
    /// # Errors
    ///
    /// Returns an error if the enqueue is rejected.
    pub fn all_reduce_sum(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DType,
        card: &Card,
    ) -> Result<()> {
        self.all_reduce_on(send, recv, count, dtype, card.stream)
    }

    fn all_reduce_on(
        &self,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DType,
        stream: synStreamHandle,
    ) -> Result<()> {
        hccl!(hcclAllReduce(
            send as *const c_void,
            recv as *mut c_void,
            count,
            dtype.code(),
            hcclSum,
            self.h,
            stream
        ));
        Ok(())
    }

    /// `hcclCommAbort`: tear the communicator down without waiting for
    /// the peers (before killing a rank that is stuck in a collective).
    pub fn abort(&self) {
        // SAFETY: a live communicator handle.
        unsafe { hcclCommAbort(self.h) };
    }

    /// `hcclCommAbort` on this process's communicator, from any thread,
    /// or nothing when the process has none (world 1, or the communicator
    /// is already going away).
    pub fn abort_current() {
        let h = ABORT_COMM.swap(0, Ordering::SeqCst);
        if h != 0 {
            // SAFETY: the handle `init_rank` recorded, taken exactly once
            // (the swap), so no other thread can be destroying it.
            unsafe { hcclCommAbort(h as hcclComm_t) };
        }
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        let _ = ABORT_COMM.compare_exchange(self.h as usize, 0, Ordering::SeqCst, Ordering::SeqCst);
        // SAFETY: a live communicator handle, finalized then destroyed
        // once. `hcclCommFinalize` (the documented sequence, and the
        // symbol is exported by libSynapse.so) drains the communicator's
        // outstanding work while the device is still acquired; without it
        // `hcclCommDestroy` prints "device not initialized" at teardown.
        unsafe {
            hcclCommFinalize(self.h);
            hcclCommDestroy(self.h);
        }
    }
}

/// The device side of one rank of a model run: its card (with the stream
/// the recipes and collectives go on) and, with more than one rank, the
/// communicator.
pub struct Rank {
    /// `None` when the world has one rank (no collectives are needed).
    /// Declared before `card`: fields drop in declaration order and
    /// `hcclCommFinalize` / `hcclCommDestroy` need the device the card
    /// releases.
    pub comm: Option<Comm>,
    pub card: Card,
    pub rank: usize,
    pub world: usize,
}

impl Rank {
    /// Acquire card `module`, hand-shake with the coordinator through
    /// `dir` (as the probe does: `rank<r>.acquired`, then `go` or
    /// `abort`), exchange the unique id through `dir/id.bin`, join the
    /// communicator and create the stream. With `world == 1` there is no
    /// communicator; the hand-shake still runs so one coordinator serves
    /// both cases.
    ///
    /// # Errors
    ///
    /// Returns an error whose message starts with `acquire:` when the
    /// card cannot be acquired (the coordinator relaunches the group),
    /// and other errors for a failed hand-shake or communicator init.
    pub fn join(rank: usize, world: usize, module: u32, dir: &Path) -> Result<Self> {
        assert!(world >= 1 && rank < world, "rank {rank} of {world}");
        let t0 = Instant::now();
        let card = match Card::acquire(module) {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::write(
                    dir.join(format!("rank{rank}.status")),
                    format!("acquire-failed {e}"),
                );
                return Err(Error::Other(format!("acquire: {e}")));
            }
        };
        println!(
            "rank {rank}: module {module} -> synDeviceId {} in {:.2} s",
            card.device_id(),
            t0.elapsed().as_secs_f64()
        );
        std::fs::write(dir.join(format!("rank{rank}.acquired")), b"ok")?;
        let ppid = std::os::unix::process::parent_id();
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            if dir.join("go").exists() {
                break;
            }
            if dir.join("abort").exists() {
                return Err(Error::Other("acquire: aborted by coordinator".into()));
            }
            if std::os::unix::process::parent_id() != ppid {
                return Err(Error::Other("the coordinator went away before go".into()));
            }
            if Instant::now() > deadline {
                return Err(Error::Other("no go from the coordinator in 180 s".into()));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let comm = if world > 1 {
            let id_path = dir.join("id.bin");
            let id = if rank == 0 {
                let id = UniqueId::create()?;
                let tmp = dir.join("id.tmp");
                std::fs::write(&tmp, id.to_bytes())?;
                std::fs::rename(&tmp, &id_path)?;
                id
            } else {
                let deadline = Instant::now() + Duration::from_secs(120);
                loop {
                    if let Ok(b) = std::fs::read(&id_path) {
                        if b.len() == UniqueId::BYTES {
                            break UniqueId::from_bytes(&b)?;
                        }
                    }
                    if Instant::now() > deadline {
                        return Err(Error::Other("no unique id from rank 0 in 120 s".into()));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            };
            let t1 = Instant::now();
            let comm = Comm::init_rank(world, &id, rank)?;
            println!(
                "rank {rank}: communicator of {world} joined in {:.2} s (HCCL {}, id {:?})",
                t1.elapsed().as_secs_f64(),
                version()?,
                id.text()
            );
            Some(comm)
        } else {
            None
        };
        let mut card = card;
        card.create_stream()?;
        watch_coordinator(dir, rank);
        Ok(Self {
            comm,
            card,
            rank,
            world,
        })
    }

    /// Enqueue a summed all-reduce of `count` f32 elements in place at
    /// device address `addr` on the card's stream (a no-op in a world of
    /// one).
    ///
    /// # Errors
    ///
    /// Returns an error if the enqueue is rejected.
    pub fn all_reduce_f32(&self, addr: u64, count: usize) -> Result<()> {
        match &self.comm {
            Some(c) => c.all_reduce_sum(addr, addr, count, DType::F32, &self.card),
            None => Ok(()),
        }
    }
}

/// Watch the coordinator from a background thread: once a rank is
/// running, nothing else notices that the coordinator asked the group to
/// stop (`dir/abort`) or died (Ctrl-C, a kill), and a worker that keeps
/// its card until it finishes on its own is exactly the "already
/// acquired by PID" condition on a shared box. On either the rank aborts
/// its communicator and leaves through [`abort_and_die`].
fn watch_coordinator(dir: &Path, rank: usize) {
    use std::io::Write;
    let abort = dir.join("abort");
    let ppid = std::os::unix::process::parent_id();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(100));
            let orphan = std::os::unix::process::parent_id() != ppid;
            if !orphan && !abort.exists() {
                continue;
            }
            // Not `println!`: when the coordinator is gone the pipe is
            // closed, and `println!` panics on a broken pipe, which would
            // kill this thread and leave the rank holding its card.
            let _ = writeln!(
                std::io::stderr(),
                "rank {rank}: {}, aborting the communicator and leaving",
                if orphan {
                    "the coordinator went away"
                } else {
                    "the coordinator asked the group to stop"
                }
            );
            abort_and_die(EXIT_ABORTED);
        }
    });
}

/// The coordinator side of a rank group: one child process per module id,
/// their stdout and stderr echoed line by line under `[r<rank>]` prefixes.
pub struct Group {
    ranks: Vec<Child>,
    pub dir: std::path::PathBuf,
    started: Instant,
}

/// One spawned rank.
pub struct Child {
    pub rank: usize,
    pub module: u32,
    child: std::process::Child,
    readers: Vec<std::thread::JoinHandle<()>>,
    pub status: Option<i32>,
}

impl Child {
    fn poll(&mut self) {
        if self.status.is_none() {
            if let Ok(Some(st)) = self.child.try_wait() {
                self.status = Some(st.code().unwrap_or(-1));
            }
        }
    }
}

/// Exit code a worker leaves with when its acquire failed (the
/// coordinator may relaunch the group after a pause).
pub const EXIT_ACQUIRE: i32 = 75;

impl Group {
    /// Spawn `exe` once per module id (in rank order) with `--rank r
    /// --world n --module m --dir DIR` followed by `args`, `RENG_MODULE_ID`
    /// set to the module, `HABANA_LOGS` under `dir`, and (with `numa`)
    /// under `numactl` on the card's NUMA node.
    ///
    /// # Errors
    ///
    /// Returns an error if `dir` cannot be made or a child cannot spawn.
    pub fn spawn(
        exe: &Path,
        modules: &[u32],
        args: &[String],
        dir: &Path,
        numa: bool,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let numactl = numa
            && std::process::Command::new("numactl")
                .arg("--hardware")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
        let mut ranks = Vec::with_capacity(modules.len());
        for (rank, &module) in modules.iter().enumerate() {
            let node = if numactl { numa_node_of(module) } else { None };
            let mut cmd = match node {
                Some(n) => {
                    let mut c = std::process::Command::new("numactl");
                    c.arg(format!("--cpunodebind={n}"))
                        .arg(format!("--membind={n}"))
                        .arg(exe);
                    c
                }
                None => std::process::Command::new(exe),
            };
            cmd.arg("--rank")
                .arg(rank.to_string())
                .arg("--world")
                .arg(modules.len().to_string())
                .arg("--module")
                .arg(module.to_string())
                .arg("--dir")
                .arg(dir)
                .args(args)
                .env("RENG_MODULE_ID", module.to_string())
                .env("HABANA_LOGS", dir.join(format!("logs-r{rank}")))
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = cmd
                .spawn()
                .map_err(|e| Error::Other(format!("cannot spawn rank {rank}: {e}")))?;
            println!(
                "coordinator: rank {rank} = module {module}, pid {}{}",
                child.id(),
                node.map_or(String::new(), |n| format!(", numa node {n}"))
            );
            let mut readers = Vec::new();
            if let Some(out) = child.stdout.take() {
                readers.push(echo(out, format!("[r{rank}]")));
            }
            if let Some(err) = child.stderr.take() {
                readers.push(echo(err, format!("[r{rank} err]")));
            }
            ranks.push(Child {
                rank,
                module,
                child,
                readers,
                status: None,
            });
        }
        Ok(Self {
            ranks,
            dir: dir.to_path_buf(),
            started: Instant::now(),
        })
    }

    /// Wait until every rank has written `rank<r>.acquired`, then write
    /// `go`; on a failed acquire (a status file or an early exit) or the
    /// deadline, write `abort` and return false.
    pub fn wait_acquired(&mut self, deadline: Instant) -> bool {
        loop {
            for r in &mut self.ranks {
                r.poll();
            }
            let acquired = self
                .ranks
                .iter()
                .filter(|r| self.dir.join(format!("rank{}.acquired", r.rank)).exists())
                .count();
            let failed = self
                .ranks
                .iter()
                .filter(|r| {
                    self.dir.join(format!("rank{}.status", r.rank)).exists() || r.status.is_some()
                })
                .count();
            if acquired == self.ranks.len() {
                let _ = std::fs::write(self.dir.join("go"), b"go");
                println!(
                    "coordinator: all {} ranks acquired their cards in {:.1} s, go",
                    self.ranks.len(),
                    self.started.elapsed().as_secs_f64()
                );
                return true;
            }
            if failed > 0 || Instant::now() > deadline {
                let _ = std::fs::write(self.dir.join("abort"), b"abort");
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Wait until every rank has exited or `deadline` passes (then kill
    /// the rest by pid). A rank that outlives a failed peer by more than
    /// 30 s is killed too: with its peer gone it never leaves its
    /// collective. Returns each rank's exit status in rank order.
    pub fn wait_all(&mut self, deadline: Instant) -> Vec<i32> {
        let mut deadline = deadline;
        let mut peer_failed = false;
        loop {
            for r in &mut self.ranks {
                r.poll();
            }
            if self.ranks.iter().all(|r| r.status.is_some()) {
                break;
            }
            if !peer_failed {
                if let Some(f) = self.ranks.iter().find(|r| r.status.is_some_and(|c| c != 0)) {
                    peer_failed = true;
                    let grace = Instant::now() + Duration::from_secs(30);
                    if grace < deadline {
                        println!(
                            "coordinator: rank {} exited with code {}; the other ranks get 30 s",
                            f.rank,
                            f.status.unwrap_or(-1)
                        );
                        deadline = grace;
                    }
                }
            }
            if Instant::now() > deadline {
                println!(
                    "coordinator: {}, asking the remaining ranks to abort their communicators",
                    if peer_failed {
                        "peer failed"
                    } else {
                        "TIMEOUT"
                    }
                );
                self.abort_and_reap();
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        for r in &mut self.ranks {
            for h in r.readers.drain(..) {
                let _ = h.join();
            }
        }
        self.ranks.iter().map(|r| r.status.unwrap_or(-1)).collect()
    }

    /// Whether every rank's exit status is one of `codes`.
    pub fn all_exited_with(&self, codes: &[i32]) -> bool {
        self.ranks
            .iter()
            .all(|r| r.status.is_some_and(|c| codes.contains(&c)))
    }

    /// Seconds since the group was spawned.
    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Write `abort` (every running rank's watchdog polls it and leaves
    /// through `hcclCommAbort`), give the group [`ABORT_GRACE`] to go, and
    /// only then kill by pid and reap. Multi-Card risk 10: a rank killed
    /// inside a collective can leave its card in the state that needed a
    /// reboot, so the kill is the last resort, never the first move.
    fn abort_and_reap(&mut self) {
        let _ = std::fs::write(self.dir.join("abort"), b"abort");
        let kill_at = Instant::now() + ABORT_GRACE;
        while Instant::now() < kill_at {
            for r in &mut self.ranks {
                r.poll();
            }
            if self.ranks.iter().all(|r| r.status.is_some()) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        for r in self.ranks.iter_mut().filter(|r| r.status.is_none()) {
            println!(
                "coordinator: killing rank {} (module {}, pid {}) after the abort grace",
                r.rank,
                r.module,
                r.child.id()
            );
            let _ = r.child.kill();
            let _ = r.child.wait();
            r.status = Some(-9);
        }
    }
}

impl Drop for Group {
    /// `std::process::Child` does not kill on drop, so without this an
    /// early return, a `?` or a panic in the coordinator would leave the
    /// workers holding their cards until they finished on their own.
    fn drop(&mut self) {
        for r in &mut self.ranks {
            r.poll();
        }
        if !self.ranks.iter().all(|r| r.status.is_some()) {
            println!("coordinator: teardown with ranks still running");
            self.abort_and_reap();
        }
        for r in &mut self.ranks {
            for h in r.readers.drain(..) {
                let _ = h.join();
            }
        }
    }
}

/// Echo a child's pipe line by line under a prefix.
fn echo<R: std::io::Read + Send + 'static>(
    reader: R,
    prefix: String,
) -> std::thread::JoinHandle<()> {
    use std::io::BufRead;
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(reader)
            .lines()
            .map_while(std::result::Result::ok)
        {
            println!("{prefix} {line}");
        }
    })
}

/// Upper bound on waiting for a copy or a collective to land.
const LAND_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a wrong value is re-read before it counts as wrong rather than
/// late.
const LATE_LIMIT: Duration = Duration::from_secs(3);

/// The elements of the probe's vectors.
const N: usize = 4096;

/// The all-reduce message sizes the timing sweep covers, in bytes: from a
/// per-layer tensor-parallel message (8 KB is one batch-1 f32 row of a
/// 2048-wide model) to a size where the links, not the launch, decide.
/// Every entry must be at most [`BIG_BYTES`] and a multiple of four.
const SWEEP_BYTES: [usize; 6] = [8 << 10, 32 << 10, 128 << 10, 1 << 20, 16 << 20, 64 << 20];

/// What the probe worker is asked to do.
pub struct WorkerArgs<'a> {
    pub rank: usize,
    pub world: usize,
    pub module: u32,
    /// Where the coordinator's files live (`rank<r>.acquired`, `go`,
    /// `abort`, `id.bin`).
    pub dir: &'a Path,
    /// `true`: rank 0 writes `id.bin` and the others read it; `false`: the
    /// others pass a zeroed id and rely on `HCCL_COMM_ID`.
    pub id_file: bool,
    pub iters: usize,
    /// `false`: only the recipe-free all-reduce checks (the hccl_demo
    /// sequence); `true`: everything.
    pub full: bool,
}

/// One rank of the probe: acquire, hand-shake with the coordinator, join
/// the communicator, run every check and timing, print one line each, and
/// return the number of failed checks. A hard error (a rejected call, a
/// copy that never lands) after the communicator exists is printed as an
/// `ERROR:` line and the process leaves through [`die`] with code 2,
/// skipping the Synapse teardown (see there).
///
/// # Errors
///
/// Returns an error when a call fails before the communicator exists or
/// a wait for the coordinator times out. An acquire failure is reported
/// as an error whose message starts with `acquire:` so the coordinator can
/// retry the whole group.
#[allow(clippy::too_many_lines)]
pub fn run_worker(a: &WorkerArgs<'_>) -> Result<usize> {
    let r = a.rank;
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "<unset>".into());
    println!(
        "env: RENG_MODULE_ID={} HCCL_COMM_ID={} HCCL_SOCKET_IFNAME={} pid={}",
        env("RENG_MODULE_ID"),
        env("HCCL_COMM_ID"),
        env("HCCL_SOCKET_IFNAME"),
        std::process::id()
    );
    let t0 = Instant::now();
    let card = match Card::acquire(a.module) {
        Ok(c) => c,
        Err(e) => {
            std::fs::write(
                a.dir.join(format!("rank{r}.status")),
                format!("acquire-failed {e}"),
            )?;
            return Err(Error::Other(format!("acquire: {e}")));
        }
    };
    println!(
        "acquire: module {} -> synDeviceId {} in {:.2} s",
        card.module(),
        card.device_id(),
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(a.dir.join(format!("rank{r}.acquired")), b"ok")?;
    // The coordinator says go once every rank holds its card, or abort when
    // one could not (that rank exits and the group is relaunched).
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if a.dir.join("go").exists() {
            break;
        }
        if a.dir.join("abort").exists() {
            return Err(Error::Other("acquire: aborted by coordinator".into()));
        }
        if Instant::now() > deadline {
            return Err(Error::Other("no go from the coordinator in 180 s".into()));
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    // Unique id: made by rank 0 after its acquire, carried by file.
    let id_path = a.dir.join("id.bin");
    let id = if r == 0 {
        let id = UniqueId::create()?;
        let tmp = a.dir.join("id.tmp");
        std::fs::write(&tmp, id.to_bytes())?;
        std::fs::rename(&tmp, &id_path)?;
        println!(
            "unique-id: created by rank 0, length {}, {:?}",
            id.length(),
            id.text()
        );
        id
    } else if a.id_file {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Ok(b) = std::fs::read(&id_path) {
                if b.len() == UniqueId::BYTES {
                    break UniqueId::from_bytes(&b)?;
                }
            }
            if Instant::now() > deadline {
                return Err(Error::Other("no unique id from rank 0 in 120 s".into()));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    } else {
        println!("unique-id: zeroed (env mode, HCCL_COMM_ID must be set)");
        UniqueId::zeroed()
    };
    if r > 0 && a.id_file {
        println!(
            "unique-id: read from file, length {}, {:?}",
            id.length(),
            id.text()
        );
    }

    let t1 = Instant::now();
    let comm = Comm::init_rank(a.world, &id, r)?;
    println!(
        "comm: hcclCommInitRank(world {}, rank {r}) ok in {:.2} s; version {}, count {}, user rank {}, syn device {}",
        a.world,
        t1.elapsed().as_secs_f64(),
        version()?,
        comm.count()?,
        comm.user_rank()?,
        comm.syn_device()?
    );

    let mut card = card;
    if let Err(e) = card.create_stream() {
        println!("ERROR: {e}");
        abort_and_die(2);
    }
    println!("stream: synStreamCreateGeneric after the communicator init, ok");

    let mut w = match Worker::new(card, comm, a) {
        Ok(w) => w,
        Err(e) => {
            println!("ERROR: {e}");
            abort_and_die(2);
        }
    };
    let failures = match checks(&mut w, a) {
        Ok(n) => n,
        Err(e) => {
            println!("ERROR: {e}");
            if let Ok(Some((code, msg))) = w.comm.async_error() {
                println!("async-error: {code} {msg}");
            }
            abort_and_die(2);
        }
    };
    let t2 = Instant::now();
    drop(w);
    println!(
        "teardown: ok in {:.2} s ({:.1} s total)",
        t2.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );
    Ok(failures)
}

/// Every check and timing, in an order that isolates causes: all-reduces
/// on raw buffers first (out of place, then in place), then the ones that
/// follow a recipe launch, then the timings and the ordering tests.
fn checks(w: &mut Worker<'_>, a: &WorkerArgs<'_>) -> Result<usize> {
    let mut failures = 0usize;
    failures += usize::from(!w.plain(false)?);
    failures += usize::from(!w.plain(true)?);
    if !a.full {
        return Ok(failures);
    }
    w.build_recipes(a)?;
    failures += usize::from(!w.correctness()?);
    failures += usize::from(!w.correctness_large()?);
    w.timing_pipelined("timing-A", N, DType::F32, a.iters)?;
    // The sweep runs before the serial timing: the 200 host round trips of
    // `timing_serial` leave the host slow enough to add several us to the
    // next pipelined size, which would land on whichever size came first.
    for bytes in SWEEP_BYTES {
        w.timing_pipelined(
            &format!("timing-S{}", bytes),
            bytes / DType::F32.bytes(),
            DType::F32,
            a.iters,
        )?;
    }
    w.timing_pipelined("timing-C2", 1 << 23, DType::Bf16, a.iters)?;
    w.timing_serial("timing-B", N, 200.min(a.iters.max(1)))?;
    failures += usize::from(!w.ordering_single(100.min(a.iters.max(1)))?);
    failures += usize::from(!w.ordering_chain(a.iters)?);
    failures += usize::from(!w.ordering_events(a.iters)?);
    match w.comm.async_error()? {
        None => println!("async-error: none"),
        Some((e, m)) => {
            println!("async-error: {e} {m}");
            failures += 1;
        }
    }
    Ok(failures)
}

/// A rank's device state for the checks: the card, the communicator, the
/// canonical `P` buffer the three recipes are bound to, timing buffers and
/// a pinned host buffer for reads.
struct Worker<'a> {
    /// Declared before `card`: the communicator's finalize and destroy
    /// need the device the card releases.
    comm: Comm,
    card: Card,
    rank: usize,
    world: usize,
    /// `X_r[i] = (r + 1) * (1 + i % 7)`, the rank's summand.
    x: Vec<f32>,
    /// `S[i] = sum_r X_r[i]`.
    s: Vec<f32>,
    /// The canonical 16 KB f32 vector every recipe reads or writes.
    p: u64,
    /// [`BIG_BYTES`] scratch buffers for the timings (send, recv).
    big_a: u64,
    big_b: u64,
    /// Pinned host buffer of [`BIG_BYTES`] for reads.
    h_read: *mut c_void,
    /// The three recipes, once [`Worker::build_recipes`] ran: W `P = X + 0`
    /// (writes P), R `Q = P + 0` (reads P; Q is read back), D
    /// `P = P / world + X` in place plus `Q = P + 0`.
    recipes: Option<Recipes<'a>>,
}

struct Recipes<'a> {
    w: Runtime<'a>,
    r: Runtime<'a>,
    d: Runtime<'a>,
}

const BIG_BYTES: u64 = 64 << 20;

impl<'a> Worker<'a> {
    fn new(card: Card, comm: Comm, a: &WorkerArgs<'_>) -> Result<Self> {
        let x: Vec<f32> = (0..N)
            .map(|i| ((a.rank + 1) * (1 + i % 7)) as f32)
            .collect();
        let s: Vec<f32> = (0..N)
            .map(|i| ((a.world * (a.world + 1) / 2) * (1 + i % 7)) as f32)
            .collect();
        let p = card.dev_alloc((N * 4) as u64)?;
        let big_a = card.dev_alloc(BIG_BYTES)?;
        let big_b = card.dev_alloc(BIG_BYTES)?;
        let h_read = card.host_alloc(BIG_BYTES)?;
        Ok(Self {
            comm,
            card,
            rank: a.rank,
            world: a.world,
            x,
            s,
            p,
            big_a,
            big_b,
            h_read,
            recipes: None,
        })
    }

    /// Compile the three recipes on the card and bind their `P` to the
    /// canonical buffer.
    fn build_recipes(&mut self, a: &WorkerArgs<'_>) -> Result<()> {
        let t0 = Instant::now();
        let zeros = vec![0f32; N];
        // `1 / world` (a power of two here, so exact in f32) and not 0.5:
        // the all-reduce that follows D sums `world` copies of `P / world`,
        // so P grows by exactly S each pass at any world size. With 0.5 the
        // chain converges only at world 2 and overflows at 4 and 8.
        let inv = vec![1.0 / self.world as f32; N];
        let mut rec = Recipes {
            w: recipe_w(&self.card, &self.x, &zeros)?,
            r: recipe_r(&self.card, &zeros)?,
            d: recipe_d(&self.card, &self.x, &zeros, &inv)?,
        };
        rec.w.rebind("P", self.p);
        rec.r.rebind("P", self.p);
        rec.d.rebind("P", self.p);
        rec.d.rebind("P_out", self.p);
        self.recipes = Some(rec);
        println!(
            "recipes: W (P = X + 0), R (Q = P + 0), D (P = P / {} + X; Q = P + 0) ready in {:.2} s (iters {})",
            self.world,
            t0.elapsed().as_secs_f64(),
            a.iters
        );
        Ok(())
    }

    fn rec(&mut self) -> &mut Recipes<'a> {
        self.recipes.as_mut().expect("recipes built")
    }

    /// The hccl_demo sequence with no recipe involved: X_r uploaded to a
    /// raw buffer, one summed all-reduce (out of place into a sentinel-
    /// filled buffer, or in place), stream sync, exact comparison with S.
    fn plain(&mut self, in_place: bool) -> Result<bool> {
        let data: Vec<u32> = self.x.iter().map(|v| v.to_bits()).collect();
        self.upload(self.big_a, &data)?;
        let recv = if in_place {
            self.big_a
        } else {
            self.memset(self.big_b, SENTINEL_D32, N)?;
            self.big_b
        };
        self.comm
            .all_reduce_sum(self.big_a, recv, N, DType::F32, &self.card)?;
        self.card.sync()?;
        let s = self.s.clone();
        let res = self.read_until(recv, N, &|i| s[i])?;
        Ok(report(
            if in_place { "plain-inplace" } else { "plain" },
            &format!(
                "f32[{N}] sum {} on a raw buffer, no recipe (rank {} of {})",
                if in_place { "in place" } else { "out of place" },
                self.rank,
                self.world
            ),
            res,
        ))
    }

    /// Fill `words` u32 at device address `d` with `v` on the stream.
    fn memset(&self, d: u64, v: u32, words: usize) -> Result<()> {
        syn!(synMemsetD32Async(d, v, words, self.card.stream));
        Ok(())
    }

    /// Upload `data` (as raw bytes) to device address `d` and wait until
    /// the copy is visible (read back until it matches).
    fn upload(&self, d: u64, data: &[u32]) -> Result<()> {
        let bytes = data.len() * 4;
        // SAFETY: h_read holds BIG_BYTES bytes and data is at most that.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), self.h_read.cast::<u32>(), data.len());
        }
        syn!(synMemCopyAsync(
            self.card.stream,
            self.h_read as u64,
            bytes as u64,
            d,
            SYN_HOST_TO_DRAM
        ));
        self.card.sync()?;
        let t0 = Instant::now();
        loop {
            let back = self.read(d, data.len())?;
            if back == data {
                return Ok(());
            }
            if t0.elapsed() > LAND_TIMEOUT {
                return Err(Error::Other("upload never became visible".into()));
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// One complete device-to-host copy of `words` u32 from `d`: the host
    /// buffer is filled with the host sentinel first and the read spins
    /// until no sentinel is left (the stream sync returns before the copy
    /// has landed on this stack).
    fn read(&self, d: u64, words: usize) -> Result<Vec<u32>> {
        assert!(words * 4 <= BIG_BYTES as usize);
        let h = self.h_read.cast::<u32>();
        // SAFETY: h_read holds at least `words` u32.
        unsafe {
            for j in 0..words {
                *h.add(j) = HOST_SENTINEL_D32;
            }
        }
        syn!(synMemCopyAsync(
            self.card.stream,
            d,
            (words * 4) as u64,
            self.h_read as u64,
            SYN_DRAM_TO_HOST
        ));
        self.card.sync()?;
        let t0 = Instant::now();
        loop {
            // SAFETY: as above; every word is written exactly once by the DMA.
            let pending = (0..words)
                .any(|j| unsafe { core::ptr::read_volatile(h.add(j)) } == HOST_SENTINEL_D32);
            if !pending {
                break;
            }
            if t0.elapsed() > LAND_TIMEOUT {
                return Err(Error::Other(format!(
                    "device-to-host copy did not land within {LAND_TIMEOUT:?}"
                )));
            }
            std::hint::spin_loop();
        }
        // SAFETY: the copy has landed.
        Ok(unsafe { std::slice::from_raw_parts(h, words) }.to_vec())
    }

    /// Read `words` f32 from `d` until every element equals `expect(i)`,
    /// re-reading for up to `LATE_LIMIT`. Returns the number of re-reads
    /// it took (0 = right at the first read after the stream sync) or the
    /// first mismatch `(i, got, expected)` after the limit.
    fn read_until(
        &self,
        d: u64,
        words: usize,
        expect: &dyn Fn(usize) -> f32,
    ) -> Result<std::result::Result<usize, (usize, f32, f32, usize)>> {
        let t0 = Instant::now();
        let mut polls = 0usize;
        loop {
            let got = self.read(d, words)?;
            let bad = (0..words).find(|&i| f32::from_bits(got[i]).to_bits() != expect(i).to_bits());
            match bad {
                None => return Ok(Ok(polls)),
                Some(i) if t0.elapsed() > LATE_LIMIT => {
                    let n_bad = (0..words)
                        .filter(|&i| f32::from_bits(got[i]).to_bits() != expect(i).to_bits())
                        .count();
                    return Ok(Err((i, f32::from_bits(got[i]), expect(i), n_bad)));
                }
                Some(_) => {
                    polls += 1;
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        }
    }

    /// The summed all-reduce of `P = X_r` (written by recipe W) in place
    /// on the recipe's stream, read straight from `P` and compared with
    /// `S` exactly.
    fn correctness(&mut self) -> Result<bool> {
        self.memset(self.p, 0, N)?;
        self.rec().w.launch_only()?;
        self.comm
            .all_reduce_sum(self.p, self.p, N, DType::F32, &self.card)?;
        self.card.sync()?;
        let s = self.s.clone();
        let res = self.read_until(self.p, N, &|i| s[i])?;
        Ok(report(
            "correctness",
            &format!(
                "f32[{N}] sum in place after recipe W on one stream (rank {} of {})",
                self.rank, self.world
            ),
            res,
        ))
    }

    /// A 16 MB out-of-place summed all-reduce with the receive buffer
    /// pre-filled with the device sentinel: every element must show the
    /// exact sum.
    fn correctness_large(&mut self) -> Result<bool> {
        let words = (BIG_BYTES / 4) as usize;
        let data: Vec<u32> = (0..words)
            .map(|i| (((self.rank + 1) * (1 + i % 7)) as f32).to_bits())
            .collect();
        self.upload(self.big_a, &data)?;
        self.memset(self.big_b, SENTINEL_D32, words)?;
        self.comm
            .all_reduce_sum(self.big_a, self.big_b, words, DType::F32, &self.card)?;
        self.card.sync()?;
        let k = (self.world * (self.world + 1) / 2) as f32;
        let res = self.read_until(self.big_b, words, &|i| k * (1 + i % 7) as f32)?;
        Ok(report(
            "correctness-16MB",
            &format!("f32[{words}] sum out of place into a sentinel-filled buffer"),
            res,
        ))
    }

    /// `iters` in-place all-reduces of `count` elements back to back on the
    /// stream, one sync at the end: the queued cost per collective and the
    /// bandwidth. Data: +1 on rank 0, -1 on rank 1, 0 elsewhere, so the sum
    /// is 0 from the first op on (verified at the end). Device-side time
    /// from a pair of timestamped events when the stack provides it.
    fn timing_pipelined(
        &mut self,
        tag: &str,
        count: usize,
        dtype: DType,
        iters: usize,
    ) -> Result<()> {
        let bytes = count * dtype.bytes();
        assert!(bytes as u64 <= BIG_BYTES);
        let v: u32 = match (self.rank, dtype) {
            (0, DType::F32) => 1f32.to_bits(),
            (1, DType::F32) => (-1f32).to_bits(),
            (0, DType::Bf16) => 0x3F80_3F80,
            (1, DType::Bf16) => 0xBF80_BF80,
            _ => 0,
        };
        let words = bytes / 4;
        self.upload(self.big_a, &vec![v; words])?;
        // Warm-up.
        for _ in 0..3 {
            self.comm
                .all_reduce_sum(self.big_a, self.big_a, count, dtype, &self.card)?;
        }
        self.card.sync()?;
        let (e0, e1) = (self.timed_event(), self.timed_event());
        let t0 = Instant::now();
        if let Some(e) = e0 {
            syn!(synEventRecord(e, self.card.stream));
        }
        for _ in 0..iters {
            self.comm
                .all_reduce_sum(self.big_a, self.big_a, count, dtype, &self.card)?;
        }
        let t_enqueue = t0.elapsed();
        if let Some(e) = e1 {
            syn!(synEventRecord(e, self.card.stream));
        }
        self.card.sync()?;
        let t_sync = t0.elapsed();
        let back = self.read(self.big_a, words)?;
        let wall = t0.elapsed();
        let nonzero = back.iter().filter(|&&w| w != 0).count();
        let device = match (e0, e1) {
            (Some(a), Some(b)) => {
                let mut ns = 0u64;
                // SAFETY: two recorded events.
                let st = unsafe { synEventElapsedTime(&mut ns, a, b) };
                // SAFETY: created by `timed_event`.
                unsafe {
                    synEventDestroy(a);
                    synEventDestroy(b);
                }
                if st == SYN_SUCCESS {
                    format!("device {:.1} us/op", ns as f64 / 1e3 / iters as f64)
                } else {
                    format!("device time n/a (synEventElapsedTime -> {st})")
                }
            }
            _ => "device time n/a (timestamped events unavailable)".into(),
        };
        let per = wall.as_secs_f64() / iters as f64;
        let algo = bytes as f64 / per / 1e9;
        let nw = algo * 2.0 * (self.world as f64 - 1.0) / self.world as f64;
        let size = if bytes >= 1 << 20 {
            format!("{} MB", bytes >> 20)
        } else {
            format!("{} KB", bytes >> 10)
        };
        println!(
            "{tag}: {size} {} in place x{iters} pipelined: {:.1} us/op wall ({:.1} us/op enqueue, sync at {:.1} us/op), {device}, {algo:.2} GB/s algo, {nw:.2} GB/s NW; final buffer {}",
            match dtype {
                DType::F32 => "f32",
                DType::Bf16 => "bf16",
            },
            per * 1e6,
            t_enqueue.as_secs_f64() / iters as f64 * 1e6,
            t_sync.as_secs_f64() / iters as f64 * 1e6,
            if nonzero == 0 {
                "all zero as expected".to_string()
            } else {
                format!("WRONG: {nonzero} non-zero words")
            }
        );
        Ok(())
    }

    /// An event with `EVENT_COLLECT_TIME`, or `None` if the stack refuses.
    fn timed_event(&self) -> Option<synEventHandle> {
        let mut e: synEventHandle = core::ptr::null_mut();
        // SAFETY: valid out-pointer and device.
        let st = unsafe { synEventCreate(&mut e, self.card.dev, EVENT_COLLECT_TIME) };
        (st == SYN_SUCCESS && !e.is_null()).then_some(e)
    }

    /// `iters` times: an out-of-place 16 KB all-reduce into a sentinel-filled
    /// buffer, stream sync, sentinel readback until the sum has landed. The
    /// one-shot latency as the engine would pay it at a recipe boundary
    /// with a host round trip, and whether the stream sync alone covered
    /// the collective (how often the first read still showed sentinels).
    fn timing_serial(&mut self, tag: &str, count: usize, iters: usize) -> Result<()> {
        let data: Vec<u32> = self.x.iter().map(|v| v.to_bits()).collect();
        self.upload(self.big_a, &data)?;
        let s = self.s.clone();
        let mut early = 0usize;
        let mut wrong = 0usize;
        let mut t_sync_total = Duration::ZERO;
        let t0 = Instant::now();
        for _ in 0..iters {
            self.memset(self.big_b, SENTINEL_D32, count)?;
            self.comm
                .all_reduce_sum(self.big_a, self.big_b, count, DType::F32, &self.card)?;
            let ts = Instant::now();
            self.card.sync()?;
            t_sync_total += ts.elapsed();
            let first = self.read(self.big_b, count)?;
            if first.contains(&SENTINEL_D32) {
                early += 1;
            }
            match self.read_until(self.big_b, count, &|i| s[i])? {
                Ok(_) => {}
                Err(_) => wrong += 1,
            }
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64;
        println!(
            "{tag}: {} KB f32 out of place x{iters} serial (memset, all-reduce, sync, sentinel readback): {:.1} us/op, sync wait {:.1} us/op; sync returned before the sum landed {early}/{iters} times; wrong {wrong}/{iters}",
            (count * 4) >> 10,
            per * 1e6,
            t_sync_total.as_secs_f64() / iters as f64 * 1e6
        );
        Ok(())
    }

    /// Recipe W, an in-place all-reduce, recipe R, all on one stream with no
    /// host sync in between, `iters` times: R's output must show `S` every
    /// time. P is filled with a different garbage pattern before each pass
    /// so a collective that ran before W would sum garbage.
    fn ordering_single(&mut self, iters: usize) -> Result<bool> {
        let s = self.s.clone();
        let mut bad = 0usize;
        let mut first_bad = None;
        let t0 = Instant::now();
        for k in 0..iters {
            self.memset(self.p, 0x4B00_0000 | ((k as u32) & 0xFFFF), N)?;
            self.rec().w.launch_only()?;
            self.comm
                .all_reduce_sum(self.p, self.p, N, DType::F32, &self.card)?;
            let q = self.rec().r.launch_and_read_i32(0, N)?;
            let q: Vec<f32> = q
                .iter()
                .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_ne_bytes())))
                .collect();
            if let Some(i) = (0..N).find(|&i| q[i].to_bits() != s[i].to_bits()) {
                bad += 1;
                if first_bad.is_none() {
                    first_bad = Some((k, i, q[i], s[i]));
                }
            }
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64;
        match first_bad {
            None => {
                println!(
                    "ordering-1: W, all-reduce, R on one stream x{iters}: PASS ({iters}/{iters} exact), {:.1} us/pass with readback",
                    per * 1e6
                );
                Ok(true)
            }
            Some((k, i, got, want)) => {
                println!(
                    "ordering-1: W, all-reduce, R on one stream x{iters}: FAIL ({bad}/{iters} wrong; first at pass {k} element {i}: got {got} expected {want})"
                );
                Ok(false)
            }
        }
    }

    /// Recipe D (`P = P / world + X_r`) then an in-place all-reduce of P, on one
    /// stream, `iters` times without any host sync: each pass adds exactly
    /// `S` to P, so P must equal `iters * S` at the end (read through R).
    /// Any reordering of a launch and a collective breaks the count.
    fn ordering_chain(&mut self, iters: usize) -> Result<bool> {
        self.memset(self.p, 0, N)?;
        self.card.sync()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            self.rec().d.launch_only()?;
            self.comm
                .all_reduce_sum(self.p, self.p, N, DType::F32, &self.card)?;
        }
        let t_enqueue = t0.elapsed();
        let q = self.rec().r.launch_and_read_i32(0, N)?;
        let wall = t0.elapsed();
        let s = self.s.clone();
        let q: Vec<f32> = q
            .iter()
            .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_ne_bytes())))
            .collect();
        let want = |i: usize| iters as f32 * s[i];
        let bad = (0..N)
            .filter(|&i| q[i].to_bits() != want(i).to_bits())
            .count();
        let per = wall.as_secs_f64() / iters as f64;
        if bad == 0 {
            println!(
                "ordering-chain: D + all-reduce x{iters} on one stream, no host sync: PASS (P == {iters} * S exactly), {:.1} us/iter ({:.1} us/iter enqueue)",
                per * 1e6,
                t_enqueue.as_secs_f64() / iters as f64 * 1e6
            );
            Ok(true)
        } else {
            let i = (0..N)
                .find(|&i| q[i].to_bits() != want(i).to_bits())
                .unwrap_or(0);
            println!(
                "ordering-chain: D + all-reduce x{iters} on one stream, no host sync: FAIL ({bad}/{N} wrong; element {i}: got {} expected {}), {:.1} us/iter",
                q[i],
                want(i),
                per * 1e6
            );
            Ok(false)
        }
    }

    /// As [`Worker::ordering_chain`] with the collective on a second stream
    /// bridged by events both ways (record on the recipe stream, wait on
    /// the collective stream; record after the collective, wait on the
    /// recipe stream), the documented cross-stream pattern.
    fn ordering_events(&mut self, iters: usize) -> Result<bool> {
        const RING: usize = 16;
        let s2 = self.card.new_stream()?;
        let mut ev_a = Vec::with_capacity(RING);
        let mut ev_b = Vec::with_capacity(RING);
        for _ in 0..RING {
            let mut e: synEventHandle = core::ptr::null_mut();
            syn!(synEventCreate(&mut e, self.card.dev, 0));
            ev_a.push(e);
            let mut e: synEventHandle = core::ptr::null_mut();
            syn!(synEventCreate(&mut e, self.card.dev, 0));
            ev_b.push(e);
        }
        self.memset(self.p, 0, N)?;
        self.card.sync()?;
        let s1 = self.card.stream;
        let t0 = Instant::now();
        for k in 0..iters {
            let (ea, eb) = (ev_a[k % RING], ev_b[k % RING]);
            self.rec().d.launch_only()?;
            syn!(synEventRecord(ea, s1));
            syn!(synStreamWaitEvent(s2, ea, 0));
            self.comm.all_reduce_on(self.p, self.p, N, DType::F32, s2)?;
            syn!(synEventRecord(eb, s2));
            syn!(synStreamWaitEvent(s1, eb, 0));
        }
        let t_enqueue = t0.elapsed();
        let q = self.rec().r.launch_and_read_i32(0, N)?;
        let wall = t0.elapsed();
        syn!(synStreamSynchronize(s2));
        for e in ev_a.into_iter().chain(ev_b) {
            // SAFETY: created above, no longer waited on.
            unsafe { synEventDestroy(e) };
        }
        Card::destroy_stream(s2);
        let s = self.s.clone();
        let q: Vec<f32> = q
            .iter()
            .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_ne_bytes())))
            .collect();
        let want = |i: usize| iters as f32 * s[i];
        let bad = (0..N)
            .filter(|&i| q[i].to_bits() != want(i).to_bits())
            .count();
        let per = wall.as_secs_f64() / iters as f64;
        if bad == 0 {
            println!(
                "ordering-events: D on stream 1, all-reduce on stream 2 bridged by events x{iters}: PASS (P == {iters} * S exactly), {:.1} us/iter ({:.1} us/iter enqueue)",
                per * 1e6,
                t_enqueue.as_secs_f64() / iters as f64 * 1e6
            );
            Ok(true)
        } else {
            let i = (0..N)
                .find(|&i| q[i].to_bits() != want(i).to_bits())
                .unwrap_or(0);
            println!(
                "ordering-events: D on stream 1, all-reduce on stream 2 bridged by events x{iters}: FAIL ({bad}/{N} wrong; element {i}: got {} expected {}), {:.1} us/iter",
                q[i],
                want(i),
                per * 1e6
            );
            Ok(false)
        }
    }
}

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        let _ = self.card.sync();
        // Recipes first (their drop syncs the stream and frees their
        // buffers on the still-acquired device), then the raw buffers; the
        // communicator and then the card drop after this body, in field
        // order.
        self.recipes = None;
        self.card.host_free(self.h_read);
        self.card.dev_free(self.p);
        self.card.dev_free(self.big_a);
        self.card.dev_free(self.big_b);
    }
}

/// Print one result line for a [`Worker::read_until`] outcome and return
/// whether it passed.
fn report(
    tag: &str,
    what: &str,
    res: std::result::Result<usize, (usize, f32, f32, usize)>,
) -> bool {
    match res {
        Ok(0) => {
            println!("{tag}: {what}: PASS (exact at the first read after the stream sync)");
            true
        }
        Ok(polls) => {
            println!("{tag}: {what}: PASS but late (exact after {polls} re-reads)");
            true
        }
        Err((i, got, want, n_bad)) => {
            println!("{tag}: {what}: FAIL ({n_bad} wrong; element {i}: got {got} expected {want})");
            false
        }
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Recipe W: `P = X + Z` (Z zeros), P a persistent f32 output written by a
/// TPC node; bound to the canonical P buffer by the caller.
fn recipe_w<'a>(card: &Card, x: &[f32], zeros: &[f32]) -> Result<Runtime<'a>> {
    let mut gb = Gb::new()?;
    let sizes = [N as u64];
    let tx = gb.input_raw("X", &sizes, SYN_TYPE_F32, &f32_bytes(x))?;
    let tz = gb.input_raw("Z", &sizes, SYN_TYPE_F32, &f32_bytes(zeros))?;
    let (tp, np) = gb.output("P", &sizes, SYN_TYPE_F32)?;
    gb.node(
        "add_fwd_f32",
        "w_add",
        &[tx, tz],
        &[tp],
        core::ptr::null(),
        0,
    )?;
    let out = Out {
        name: np,
        sizes: sizes.to_vec(),
        kind: OutKind::I32,
    };
    Runtime::new_on(gb, out, card.dev, card.stream)
}

/// Recipe R: `Q = P + Z`, P a persistent f32 scratch tensor (rebound to the
/// canonical buffer), Q the read-back output.
fn recipe_r<'a>(card: &Card, zeros: &[f32]) -> Result<Runtime<'a>> {
    let mut gb = Gb::new()?;
    let sizes = [N as u64];
    let tz = gb.input_raw("Z", &sizes, SYN_TYPE_F32, &f32_bytes(zeros))?;
    let tp = gb.scratch_typed("P", &sizes, SYN_TYPE_F32)?;
    let (tq, nq) = gb.output("Q", &sizes, SYN_TYPE_F32)?;
    gb.node(
        "add_fwd_f32",
        "r_add",
        &[tp, tz],
        &[tq],
        core::ptr::null(),
        0,
    )?;
    let out = Out {
        name: nq,
        sizes: sizes.to_vec(),
        kind: OutKind::I32,
    };
    Runtime::new_on(gb, out, card.dev, card.stream)
}

/// Recipe D: `T = P * H` (H = `1 / world`), `P_out = T + X` with `P_out`
/// in P's section (in place), `Q = P_out + Z` as the read-back output.
fn recipe_d<'a>(card: &Card, x: &[f32], zeros: &[f32], inv: &[f32]) -> Result<Runtime<'a>> {
    let mut gb = Gb::new()?;
    let sizes = [N as u64];
    let tx = gb.input_raw("X", &sizes, SYN_TYPE_F32, &f32_bytes(x))?;
    let tz = gb.input_raw("Z", &sizes, SYN_TYPE_F32, &f32_bytes(zeros))?;
    let th = gb.input_raw("H", &sizes, SYN_TYPE_F32, &f32_bytes(inv))?;
    let tp = gb.scratch_typed("P", &sizes, SYN_TYPE_F32)?;
    let tpo = gb.scratch_alias_typed("P_out", &sizes, "P", SYN_TYPE_F32)?;
    let tt = gb.mid("T", &sizes, SYN_TYPE_F32)?;
    let (tq, nq) = gb.output("Q", &sizes, SYN_TYPE_F32)?;
    gb.node(
        "mult_fwd_f32",
        "d_mult",
        &[tp, th],
        &[tt],
        core::ptr::null(),
        0,
    )?;
    gb.node(
        "add_fwd_f32",
        "d_add",
        &[tt, tx],
        &[tpo],
        core::ptr::null(),
        0,
    )?;
    gb.node(
        "add_fwd_f32",
        "d_copy",
        &[tpo, tz],
        &[tq],
        core::ptr::null(),
        0,
    )?;
    let out = Out {
        name: nq,
        sizes: sizes.to_vec(),
        kind: OutKind::I32,
    };
    Runtime::new_on(gb, out, card.dev, card.stream)
}

/// The NUMA node of the card with module id `module` (sysfs), if known.
#[must_use]
pub fn numa_node_of(module: u32) -> Option<u32> {
    let dir = std::fs::read_dir("/sys/class/accel").ok()?;
    for e in dir.flatten() {
        let dev = e.path().join("device");
        let Ok(m) = std::fs::read_to_string(dev.join("module_id")) else {
            continue;
        };
        if m.trim().parse::<u32>().ok() == Some(module) {
            return std::fs::read_to_string(dev.join("numa_node"))
                .ok()?
                .trim()
                .parse::<u32>()
                .ok();
        }
    }
    None
}
