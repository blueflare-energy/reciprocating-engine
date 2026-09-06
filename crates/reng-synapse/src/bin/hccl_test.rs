//! Multi-card probe: one HCCL communicator over one process per card, a
//! verified all-reduce, timings, and the same-stream ordering check the
//! tensor-parallel design depends on.
//!
//! Coordinator (spawns one worker per module id, in rank order):
//!
//! `reng-hccl-test 4 1 [--iters 1000] [--id-mode file|env] [--mode full|plain] [--timeout 600] [--no-numa] [--port 5555]`
//!
//! Worker (spawned by the coordinator; the same binary):
//!
//! `reng-hccl-test --rank r --world n --module m --dir DIR [--iters N] [--id-mode file|env] [--mode full|plain]`
//!
//! `--mode plain` runs only the recipe-free all-reduce checks (the
//! hccl_demo call sequence); `full` (the default) adds the recipe-adjacent
//! correctness check, the timings and the ordering tests. Each rank logs
//! to its own `HABANA_LOGS` directory under the hand-shake directory.
//!
//! The coordinator makes a directory the workers hand-shake through: each
//! worker writes `rank<r>.acquired` once it holds its card (or
//! `rank<r>.status` when the acquire failed); the coordinator writes `go`
//! when every rank holds one, else `abort`, waits a minute and relaunches
//! the group (a stray acquire by another process on a card lasts seconds).
//! Rank 0 then writes the 1032-byte unique id to `id.bin` (file mode; in
//! env mode the other ranks pass a zeroed id and `HCCL_COMM_ID=IP:PORT`,
//! set by the coordinator when unset, names rank 0's coordinator socket).
//! Every worker line is echoed with a `[r<rank>]` prefix; the verdict is
//! the last line and the exit code is non-zero on any failure. Workers are
//! killed by pid only after the overall timeout, or 30 s after a peer rank
//! died (a rank whose peer is gone never leaves its collective).

use reng_synapse::hccl::{WorkerArgs, numa_node_of, run_worker};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Exit code of a worker whose acquire failed (the coordinator retries).
const EXIT_ACQUIRE: i32 = 75;

#[derive(Clone, Copy, PartialEq, Eq)]
enum IdMode {
    File,
    Env,
}

struct Opts {
    modules: Vec<u32>,
    iters: usize,
    id_mode: IdMode,
    full: bool,
    timeout_s: u64,
    numa: bool,
    port: u16,
    rank: Option<usize>,
    world: usize,
    module: u32,
    dir: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: reng-hccl-test <module id>... [--iters N] [--id-mode file|env] [--mode full|plain] [--timeout S] [--no-numa] [--port P]"
    );
    std::process::exit(2)
}

fn parse() -> Opts {
    let mut o = Opts {
        modules: Vec::new(),
        iters: 1000,
        id_mode: IdMode::File,
        full: true,
        timeout_s: 600,
        numa: true,
        port: 5555,
        rank: None,
        world: 0,
        module: 0,
        dir: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let value = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| usage())
    };
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => o.iters = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--id-mode" => {
                o.id_mode = match value(&mut i).as_str() {
                    "file" | "pipe" => IdMode::File,
                    "env" => IdMode::Env,
                    _ => usage(),
                }
            }
            "--mode" => {
                o.full = match value(&mut i).as_str() {
                    "full" => true,
                    "plain" => false,
                    _ => usage(),
                }
            }
            "--timeout" => o.timeout_s = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--no-numa" => o.numa = false,
            "--port" => o.port = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--rank" => o.rank = Some(value(&mut i).parse().unwrap_or_else(|_| usage())),
            "--world" => o.world = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--module" => o.module = value(&mut i).parse().unwrap_or_else(|_| usage()),
            "--dir" => o.dir = Some(PathBuf::from(value(&mut i))),
            "--modules" => {
                for m in value(&mut i).split(',') {
                    o.modules.push(m.trim().parse().unwrap_or_else(|_| usage()));
                }
            }
            a if a.starts_with("--") => usage(),
            a => o.modules.push(a.parse().unwrap_or_else(|_| usage())),
        }
        i += 1;
    }
    o
}

fn main() {
    let o = parse();
    let code = match o.rank {
        Some(rank) => worker(&o, rank),
        None => coordinate(&o),
    };
    std::process::exit(code);
}

fn worker(o: &Opts, rank: usize) -> i32 {
    let Some(dir) = o.dir.as_deref() else { usage() };
    if o.world == 0 {
        usage();
    }
    let a = WorkerArgs {
        rank,
        world: o.world,
        module: o.module,
        dir,
        id_file: o.id_mode == IdMode::File,
        iters: o.iters,
        full: o.full,
    };
    match run_worker(&a) {
        Ok(0) => {
            println!("RESULT: rank {rank} PASS");
            0
        }
        Ok(n) => {
            println!("RESULT: rank {rank} FAIL ({n} failed checks)");
            1
        }
        Err(e) if e.to_string().starts_with("acquire:") => {
            println!("RESULT: rank {rank} ACQUIRE-FAILED: {e}");
            EXIT_ACQUIRE
        }
        Err(e) => {
            println!("RESULT: rank {rank} ERROR: {e}");
            2
        }
    }
}

/// Echo a child's pipe line by line under a prefix.
fn echo<R: std::io::Read + Send + 'static>(
    reader: R,
    prefix: String,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            println!("{prefix} {line}");
        }
    })
}

struct Rank {
    rank: usize,
    module: u32,
    child: Child,
    readers: Vec<std::thread::JoinHandle<()>>,
    status: Option<i32>,
}

impl Rank {
    fn poll(&mut self) {
        if self.status.is_none() {
            if let Ok(Some(st)) = self.child.try_wait() {
                self.status = Some(st.code().unwrap_or(-1));
            }
        }
    }
}

fn spawn_rank(o: &Opts, exe: &Path, dir: &Path, rank: usize, module: u32, numactl: bool) -> Rank {
    let numa = if numactl { numa_node_of(module) } else { None };
    let mut cmd = match numa {
        Some(n) => {
            let mut c = Command::new("numactl");
            c.arg(format!("--cpunodebind={n}"))
                .arg(format!("--membind={n}"))
                .arg(exe);
            c
        }
        None => Command::new(exe),
    };
    cmd.arg("--rank")
        .arg(rank.to_string())
        .arg("--world")
        .arg(o.modules.len().to_string())
        .arg("--module")
        .arg(module.to_string())
        .arg("--dir")
        .arg(dir)
        .arg("--iters")
        .arg(o.iters.to_string())
        .arg("--id-mode")
        .arg(match o.id_mode {
            IdMode::File => "file",
            IdMode::Env => "env",
        })
        .arg("--mode")
        .arg(if o.full { "full" } else { "plain" })
        .env("RENG_MODULE_ID", module.to_string())
        .env("HABANA_LOGS", dir.join(format!("logs-r{rank}")))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if o.id_mode == IdMode::Env && std::env::var_os("HCCL_COMM_ID").is_none() {
        cmd.env("HCCL_COMM_ID", format!("127.0.0.1:{}", o.port));
    }
    let mut child = cmd.spawn().unwrap_or_else(|e| {
        eprintln!("cannot spawn rank {rank}: {e}");
        std::process::exit(2)
    });
    println!(
        "coordinator: rank {rank} = module {module}, pid {}{}",
        child.id(),
        numa.map_or(String::new(), |n| format!(", numa node {n}"))
    );
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        readers.push(echo(out, format!("[r{rank}]")));
    }
    if let Some(err) = child.stderr.take() {
        readers.push(echo(err, format!("[r{rank} err]")));
    }
    Rank {
        rank,
        module,
        child,
        readers,
        status: None,
    }
}

/// Wait until every rank has exited or the deadline passes (then kill the
/// rest by pid). A rank that outlives a failed peer by more than 30 s is
/// killed too: with its peer gone it never leaves its collective.
fn wait_all(ranks: &mut [Rank], deadline: Instant) {
    let mut deadline = deadline;
    let mut peer_failed = false;
    loop {
        for r in ranks.iter_mut() {
            r.poll();
        }
        if ranks.iter().all(|r| r.status.is_some()) {
            break;
        }
        if !peer_failed {
            if let Some(f) = ranks.iter().find(|r| r.status.is_some_and(|c| c != 0)) {
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
            for r in ranks.iter_mut().filter(|r| r.status.is_none()) {
                println!(
                    "coordinator: {}, killing rank {} (module {}, pid {}); the card may need a reset if it was inside a collective",
                    if peer_failed {
                        "peer failed"
                    } else {
                        "TIMEOUT"
                    },
                    r.rank,
                    r.module,
                    r.child.id()
                );
                let _ = r.child.kill();
                let _ = r.child.wait();
                r.status = Some(-9);
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    for r in ranks.iter_mut() {
        for h in r.readers.drain(..) {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_lines)]
fn coordinate(o: &Opts) -> i32 {
    if o.modules.is_empty() {
        usage();
    }
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("current_exe: {e}");
        std::process::exit(2)
    });
    let numactl = o.numa
        && Command::new("numactl")
            .arg("--hardware")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "<unset>".into());
    println!(
        "coordinator: modules {:?} (rank order), iters {}, id-mode {}, mode {}, numactl {}, HCCL_COMM_ID={}, HCCL_SOCKET_IFNAME={}, timeout {} s",
        o.modules,
        o.iters,
        match o.id_mode {
            IdMode::File => "file",
            IdMode::Env => "env",
        },
        if o.full { "full" } else { "plain" },
        if numactl { "yes" } else { "no" },
        env("HCCL_COMM_ID"),
        env("HCCL_SOCKET_IFNAME"),
        o.timeout_s
    );
    let started = Instant::now();
    let deadline = started + Duration::from_secs(o.timeout_s);
    let max_attempts = 4;
    for attempt in 1..=max_attempts {
        let dir = std::env::temp_dir().join(format!("reng-hccl-{}-{attempt}", std::process::id()));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("cannot create {}: {e}", dir.display());
            return 2;
        }
        println!(
            "coordinator: attempt {attempt}/{max_attempts}, directory {}",
            dir.display()
        );
        let mut ranks: Vec<Rank> = o
            .modules
            .iter()
            .enumerate()
            .map(|(r, &m)| spawn_rank(o, &exe, &dir, r, m, numactl))
            .collect();
        // Phase 1: every rank acquires its card.
        let mut all_acquired = false;
        loop {
            for r in &mut ranks {
                r.poll();
            }
            let acquired = ranks
                .iter()
                .filter(|r| dir.join(format!("rank{}.acquired", r.rank)).exists())
                .count();
            let failed = ranks
                .iter()
                .filter(|r| {
                    dir.join(format!("rank{}.status", r.rank)).exists() || r.status.is_some()
                })
                .count();
            if acquired == ranks.len() {
                all_acquired = true;
                break;
            }
            if failed > 0 || Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if !all_acquired {
            let _ = std::fs::write(dir.join("abort"), b"abort");
            wait_all(&mut ranks, deadline);
            let codes: Vec<String> = ranks
                .iter()
                .map(|r| format!("rank {} (module {}) exit {:?}", r.rank, r.module, r.status))
                .collect();
            println!("coordinator: acquire phase failed: {}", codes.join(", "));
            let retry = ranks
                .iter()
                .all(|r| r.status == Some(EXIT_ACQUIRE) || r.status == Some(0));
            if retry
                && attempt < max_attempts
                && Instant::now() + Duration::from_secs(60) < deadline
            {
                println!("coordinator: waiting 60 s before relaunching the group");
                std::thread::sleep(Duration::from_secs(60));
                continue;
            }
            println!("VERDICT: FAIL (could not acquire every card)");
            return 1;
        }
        let _ = std::fs::write(dir.join("go"), b"go");
        println!(
            "coordinator: all {} ranks acquired their cards in {:.1} s, go",
            ranks.len(),
            started.elapsed().as_secs_f64()
        );
        // Phase 2: the probe runs; wait for every rank.
        wait_all(&mut ranks, deadline);
        let mut verdict = 0;
        let mut parts = Vec::new();
        for r in &ranks {
            let st = r.status.unwrap_or(-1);
            parts.push(format!("rank {} (module {}) exit {st}", r.rank, r.module));
            if st != 0 {
                verdict = 1;
            }
        }
        println!(
            "coordinator: {} in {:.1} s",
            parts.join(", "),
            started.elapsed().as_secs_f64()
        );
        if verdict == 0 {
            println!("VERDICT: PASS");
        } else {
            println!("VERDICT: FAIL");
        }
        println!(
            "coordinator: per-rank Synapse/HCL logs under {}",
            dir.display()
        );
        return verdict;
    }
    1
}
