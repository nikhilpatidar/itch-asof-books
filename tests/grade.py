"""Compile the submission and the baked-in starter, then score log speedup."""

from __future__ import annotations

import json
import math
import os
import pwd
import select
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from model import answer_books, generate, pinned_query_plan

FAMILIES = {
    "bulk": {
        "messages": 110000,
        "symbols": 8,
        "queries": 48,
        "snaps": 8,
        "backfill": 12000,
        "chunk": 0,
        "interleave": False,
        "seed": 26041901,
    },
    "serving": {
        "messages": 100000,
        "symbols": 8,
        "queries": 44,
        "snaps": 8,
        "backfill": 12000,
        "chunk": 32000,
        "interleave": True,
        "seed": 26041902,
    },
    "backfill": {
        # Heavier than the other families' message counts so the weakest
        # cell is not a ~2s backfill ratio that swings with host syscall noise.
        "messages": 140000,
        "symbols": 8,
        "queries": 44,
        "snaps": 8,
        "backfill": 12000,
        "chunk": 0,
        "interleave": False,
        "seed": 26041903,
    },
    "spill": {
        "messages": 130000,
        "symbols": 8,
        "queries": 44,
        "snaps": 8,
        "backfill": 12000,
        "chunk": 0,
        "interleave": False,
        "seed": 26041904,
    },
}

# Cap is peak summed VmRSS across the engine uid (one process). Grader RSS is not counted.
# Propagation starter peaks well under this; naive dense full-history snapshots do not.
# Peak summed VmRSS across the engine uid. Sized so the baked rebuild starter
# fits under this workload (~282 MiB on backfill); submissions above
# this score 0. Baseline/floor are not gated on this cap (they are infra).
RSS_CAP_KB = 393216
REPEATS = 5
READ_TIMEOUT_SEC = 20.0
QUERY_BATCH = 4
CXX = ["g++", "-O2", "-std=c++17", "-pthread"]
# Per-family wall for one timed session. Kept low enough that
# 4 families × (harvest + checker + REPEATS × (base+floor+sub)) cannot blow the
# 3600 s verifier timeout; a correct but slow submission scores 0 here instead.
FAMILY_WALL_SEC = 50.0
# Scored time is max(uid-wide /proc user+sys, wait4 user+sys). Starter is
# checkpointed rebuild (kCk=256) with ~12k rewrites; mid-span asks share one
# QUERY. REPEATS=5 with a trimmed CPU median. Score is
# ln(T_start/T)/ln(T_start/T_floor) where T_floor is the sealed reference
# engine. Matching the starter scores 0; matching the reference bound scores 1.0.



def _trimmed_median(values: list[float]) -> float:
    """Median after dropping one low and one high sample when len >= 5."""
    if len(values) >= 5:
        ordered = sorted(values)
        ordered = ordered[1:-1]
        return float(statistics.median(ordered))
    return float(statistics.median(values))


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate timed runs; CPU uses a trimmed median, other keys a plain median."""
    keys = ("wall", "cpu", "query", "snap")
    out: dict = {}
    for key in keys:
        samples = [float(row.get(key, 0.0)) for row in runs]
        out[key] = _trimmed_median(samples) if key == "cpu" else float(statistics.median(samples))
    return out


class EngineFail(Exception):
    """The engine crashed, hung, or broke the protocol. That is a floor score."""


def compile_one(
    src: Path,
    dest: Path,
    identity: tuple[int, int] | None = None,
    *,
    hide_tests: bool = False,
) -> None:
    """Compile src to dest.

    When identity is set, g++ runs as that uid with /tests mode 0700 so a
    submission cannot #include verifier-only sources such as /tests/floor/bookd.cpp.
    """
    import tempfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    compile_src = src
    out = dest
    scratch: Path | None = None
    if identity is not None:
        scratch = Path(tempfile.mkdtemp(prefix="itch-cc-"))
        os.chown(scratch, identity[0], identity[1])
        os.chmod(scratch, 0o700)
        compile_src = scratch / "bookd.cpp"
        compile_src.write_bytes(src.read_bytes())
        os.chown(compile_src, identity[0], identity[1])
        os.chmod(compile_src, 0o400)
        out = scratch / "out"

    def preexec() -> None:
        if hide_tests and os.geteuid() == 0:
            tests = Path("/tests")
            if tests.is_dir():
                os.chmod(tests, 0o700)
        if identity is not None:
            os.setgid(identity[1])
            os.setuid(identity[0])

    try:
        subprocess.run(
            [*CXX, "-o", str(out), str(compile_src)],
            check=True,
            preexec_fn=preexec if identity is not None else None,
        )
        if scratch is not None:
            dest.write_bytes(out.read_bytes())
        os.chmod(dest, 0o555)
    finally:
        if scratch is not None:
            for path in scratch.iterdir():
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                scratch.rmdir()
            except OSError:
                pass


def engine_identity() -> tuple[int, int] | None:
    if os.geteuid() != 0:
        return None
    user = pwd.getpwnam("nobody")
    return user.pw_uid, user.pw_gid


def reap_uid(identity: tuple[int, int] | None) -> None:
    """Kill every process of the engine uid, repeating so a fork chain cannot hide."""
    if identity is None:
        return
    uid = identity[0]
    me = os.getpid()
    quiet = 0
    for _ in range(40):
        victims: list[int] = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == me:
                continue
            try:
                text = Path(f"/proc/{pid}/status").read_text()
            except OSError:
                continue
            for line in text.splitlines():
                if line.startswith("Uid:") and int(line.split()[1]) == uid:
                    victims.append(pid)
                    break
        if not victims:
            quiet += 1
            if quiet >= 3:
                return
            time.sleep(0.02)
            continue
        quiet = 0
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.02)


def drop_ipc() -> None:
    """Remove System V and POSIX IPC a previous repeat could use as an answer cache."""
    if os.geteuid() != 0:
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    ipc_rmid = 0
    libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.shmctl.restype = ctypes.c_int
    libc.msgctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.msgctl.restype = ctypes.c_int
    libc.semctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libc.semctl.restype = ctypes.c_int
    if hasattr(libc, "mq_unlink"):
        libc.mq_unlink.argtypes = [ctypes.c_char_p]
        libc.mq_unlink.restype = ctypes.c_int

    def ids(path: str) -> list[int]:
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            return []
        found = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                found.append(int(parts[1]))
        return found

    for shmid in ids("/proc/sysvipc/shm"):
        libc.shmctl(shmid, ipc_rmid, None)
    for msqid in ids("/proc/sysvipc/msg"):
        libc.msgctl(msqid, ipc_rmid, None)
    for semid in ids("/proc/sysvipc/sem"):
        libc.semctl(semid, 0, ipc_rmid)
    queue_dir = Path("/dev/mqueue")
    if queue_dir.is_dir():
        for child in list(queue_dir.iterdir()):
            if hasattr(libc, "mq_unlink"):
                libc.mq_unlink(f"/{child.name}".encode())
            try:
                child.unlink(missing_ok=True)
            except OSError:
                pass


def wipe_shared(keep: Path) -> None:
    """Drop files and IPC a previous repeat could have used as a cache."""
    drop_ipc()
    if os.geteuid() != 0:
        return
    keep_resolved = keep.resolve()
    for root in ("/tmp", "/var/tmp", "/dev/shm", "/app/scratch", "/run/lock", "/var/lock"):
        base = Path(root)
        if not base.is_dir():
            continue
        for child in list(base.iterdir()):
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if resolved == keep_resolved or keep_resolved.is_relative_to(resolved) or resolved.is_relative_to(keep_resolved):
                continue
            try:
                if child.is_symlink() or not child.is_dir():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass


def _private_tmp() -> None:
    """Give this process empty writable dirs so a previous repeat's cache is gone."""
    import ctypes

    clone_newns = 0x00020000
    ms_rec = 1 << 14
    ms_private = 1 << 18
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(clone_newns) != 0:
        return
    libc.mount(b"none", b"/", None, ms_rec | ms_private, None)
    for path in (b"/tmp", b"/var/tmp", b"/dev/shm", b"/app/scratch"):
        os.makedirs(path.decode(), exist_ok=True)
        libc.mount(b"tmpfs", path, b"tmpfs", 0, b"mode=1777,size=512m")


def _preexec(identity: tuple[int, int] | None) -> None:
    import resource

    _private_tmp()
    if identity is not None:
        os.setgid(identity[1])
        os.setuid(identity[0])
        # One process only: busy helper threads cannot warm the sandbox or hide
        # rewrite work off the timed path. watch_engine still fails a second
        # live process under this uid if the limit is raised or bypassed.
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024))


def _clk_tck() -> float:
    return float(os.sysconf("SC_CLK_TCK"))


def _proc_cpu_ticks(pid: int) -> int | None:
    """User+sys ticks from /proc/<pid>/stat (fields 14 and 15).

    Parse after the last ')': the comm field is parenthesized and may contain
    spaces or digits, so whitespace/fixed-index splits (and spoofed comm names)
    are not used. Settles the /proc-parser open lead from cheat trials.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    cut = text.rfind(")")
    if cut < 0:
        return None
    fields = text[cut + 2 :].split()
    if len(fields) < 13:
        return None
    return int(fields[11]) + int(fields[12])


def _proc_vm_rss_kb(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _engine_pids(root_pid: int, uid: int | None) -> set[int]:
    """Pids charged to the engine: same uid, or the session rooted at root_pid."""
    pids = {root_pid}
    if uid is not None:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            other = int(name)
            try:
                status = Path(f"/proc/{other}/status").read_text()
            except OSError:
                continue
            for line in status.splitlines():
                if line.startswith("Uid:") and int(line.split()[1]) == uid:
                    pids.add(other)
                    break
        return pids
    # Trusted/local runs: include every process in the engine's process group.
    try:
        text = Path(f"/proc/{root_pid}/stat").read_text()
        cut = text.rfind(")")
        fields = text[cut + 2 :].split()
        pgrp = int(fields[2])
    except (OSError, IndexError, ValueError):
        return pids
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        other = int(name)
        try:
            text = Path(f"/proc/{other}/stat").read_text()
            cut = text.rfind(")")
            fields = text[cut + 2 :].split()
            if int(fields[2]) == pgrp:
                pids.add(other)
        except (OSError, IndexError, ValueError):
            continue
    return pids


def watch_engine(
    root_pid: int,
    stop: threading.Event,
    rss_slot: list[int],
    cpu_slot: list[float],
    multi_slot: list[bool],
    uid: int | None,
) -> None:
    """Peak RSS and accrued CPU across every engine process; flag multi-proc cheats.

    Scored CPU is summed /proc utime+sys ticks for the engine uid (or process
    group), including processes that exit mid-session. wait4 of the direct child
    alone is not used: a fork proxy or a patched parent wait4 can zero that.
    More than one live process under the submission uid is a hard failure.
    """
    clk = _clk_tck()
    prev_ticks: dict[int, int] = {}
    accrued = 0
    misses = 0
    while not stop.is_set():
        try:
            pids = _engine_pids(root_pid, uid)
        except OSError:
            pids = {root_pid}
        if uid is not None and len(pids) > 1:
            multi_slot[0] = True
        rss = 0
        current: dict[int, int] = {}
        alive = False
        for cur in pids:
            ticks = _proc_cpu_ticks(cur)
            if ticks is not None:
                current[cur] = ticks
                alive = True
            kb = _proc_vm_rss_kb(cur)
            if kb is not None:
                rss += kb
                alive = True
        for pid, ticks in prev_ticks.items():
            if pid not in current:
                accrued += ticks
        prev_ticks = current
        cpu_slot[0] = (accrued + sum(current.values())) / clk
        if alive:
            misses = 0
            rss_slot[0] = max(rss_slot[0], rss)
        else:
            misses += 1
            if misses > 50:
                break
        stop.wait(0.01)
    # Final sample so the last tick counts are not dropped when stop races exit.
    try:
        pids = _engine_pids(root_pid, uid)
    except OSError:
        pids = set()
    current = {}
    for cur in pids:
        ticks = _proc_cpu_ticks(cur)
        if ticks is not None:
            current[cur] = ticks
    for pid, ticks in prev_ticks.items():
        if pid not in current:
            accrued += ticks
    cpu_slot[0] = (accrued + sum(current.values())) / clk


class Session:
    def __init__(
        self,
        binary: Path,
        identity: tuple[int, int] | None,
        env: dict[str, str] | None = None,
    ):
        wipe_shared(binary.parent)
        scratch = Path("/app/scratch")
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            if identity is not None:
                os.chown(scratch, identity[0], identity[1])
                os.chmod(scratch, 0o700)
        except OSError:
            scratch = Path("/tmp")
        reap_uid(identity)
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(scratch if scratch.is_dir() else "/tmp"),
            env=run_env,
            preexec_fn=lambda: _preexec(identity),
        )
        self.identity = identity
        self._buf = b""
        self.rss = [0]
        self.cpu_acc = [0.0]
        self.multi_proc = [False]
        self.stop = threading.Event()
        uid = identity[0] if identity is not None else None
        self.watcher = threading.Thread(
            target=watch_engine,
            args=(self.proc.pid, self.stop, self.rss, self.cpu_acc, self.multi_proc, uid),
            daemon=True,
        )
        self.watcher.start()
        self.query_times: list[float] = []
        self.snap_times: list[float] = []
        self.cpu_sec = 0.0

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                self.proc.kill()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        # Orphans under the engine uid (fork-proxy workers) must die too.
        reap_uid(self.identity)

    def _write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineFail("engine closed stdin") from exc

    def _line(self) -> str:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            ready, _, _ = select.select([fd], [], [], READ_TIMEOUT_SEC)
            if not ready:
                self.kill()
                raise EngineFail("read timed out")
            try:
                chunk = os.read(fd, 1 << 20)
            except OSError as exc:
                raise EngineFail("engine stdout failed") from exc
            if not chunk:
                raise EngineFail("engine closed stdout")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            return line.decode()
        except UnicodeDecodeError as exc:
            raise EngineFail("non-utf8 stdout") from exc

    def apply(self, blob: bytes) -> None:
        self._write(f"APPLY {len(blob)}\n".encode() + blob)
        if self._line() != "READY":
            raise EngineFail("apply did not ack")

    def query(self, rows: list[tuple[str, int]]) -> list[str]:
        payload = "".join(f"{name} {seq}\n" for name, seq in rows)
        start = time.perf_counter()
        self._write(f"QUERY {len(rows)}\n{payload}".encode())
        lines = [self._line() for _ in rows]
        if self._line() != "END" or self._line() != "READY":
            raise EngineFail("bad query trailer")
        self.query_times.append(time.perf_counter() - start)
        return lines

    def snap(self, seq: int) -> list[str]:
        start = time.perf_counter()
        self._write(f"SNAP {seq}\n".encode())
        lines = []
        while True:
            line = self._line()
            if line == "END":
                break
            lines.append(line)
        if self._line() != "READY":
            raise EngineFail("bad snap trailer")
        self.snap_times.append(time.perf_counter() - start)
        return lines

    def _reap(self) -> None:
        reap_uid(self.identity)

    def close(self) -> tuple[int, float]:
        try:
            self._write(b"QUIT\n")
        except EngineFail:
            pass
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        deadline = time.monotonic() + 5
        got_exit = False
        hung = False
        wait4_cpu = 0.0
        while True:
            try:
                pid, status, usage = os.wait4(self.proc.pid, os.WNOHANG)
            except ChildProcessError:
                break
            if pid:
                # Take wait4 user+sys as a lower bound only. Scored CPU is
                # max(/proc accrual, wait4): forging parent wait4 or spoofing
                # /proc alone cannot zero the metric; a fork-proxy still shows
                # up in the uid-wide /proc sample (and fails multi_proc).
                wait4_cpu = float(usage.ru_utime + usage.ru_stime)
                self.proc.returncode = os.waitstatus_to_exitcode(status)
                got_exit = True
                break
            if time.monotonic() >= deadline:
                hung = True
                self.kill()
                try:
                    pid, status, usage = os.wait4(self.proc.pid, 0)
                except ChildProcessError:
                    pid = 0
                    usage = None
                if pid:
                    wait4_cpu = float(usage.ru_utime + usage.ru_stime)
                    self.proc.returncode = os.waitstatus_to_exitcode(status)
                    got_exit = True
                break
            time.sleep(0.01)
        self.stop.set()
        self.watcher.join(timeout=1)
        self.cpu_sec = max(float(self.cpu_acc[0]), wait4_cpu, 0.0)
        self._reap()
        if self.multi_proc[0]:
            raise EngineFail("engine spawned another process")
        # A hang on QUIT must not score: without this, a killed child can leave
        # cpu_sec at 0 and the family cell clips to 1. Sub-tick CPU (0.0) on a
        # clean exit is fine for the protocol probe; play() floors at 1e-6.
        if hung or not got_exit:
            raise EngineFail("engine hung on exit")
        return self.rss[0], self.cpu_sec


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def play(
    binary: Path,
    trace: dict,
    identity: tuple[int, int] | None,
    check: bool,
    env: dict[str, str] | None = None,
) -> tuple[dict, int, list[str]]:
    session = Session(binary, identity, env=env)
    mismatches: list[str] = []
    captured: list[tuple] = []
    # Wall gates the family budget; scored productive time is derived later.
    start = time.perf_counter()
    deadline = start + FAMILY_WALL_SEC
    try:
        for op in trace["ops"]:
            if time.perf_counter() > deadline:
                session.kill()
                raise EngineFail(
                    f"family wall budget {FAMILY_WALL_SEC:.0f}s exceeded during session"
                )
            kind, payload = op[0], op[1]
            if kind == "apply":
                session.apply(payload)
            elif kind == "query":
                captured.append(("query", payload, op[2], session.query(payload), op[3] if len(op) > 3 else None))
            else:
                captured.append(("snap", payload, op[2], session.snap(payload), None))
    except EngineFail:
        session.close()
        raise
    wall = time.perf_counter() - start
    rss, cpu = session.close()
    if wall > FAMILY_WALL_SEC:
        raise EngineFail(f"family wall budget {FAMILY_WALL_SEC:.0f}s exceeded ({wall:.1f}s)")
    if check:
        pending: dict[int, list] = {}
        for item in captured:
            kind, payload, phase, got, pinned = item
            if pinned is not None:
                for (name, seq), line, expect in zip(payload, got, pinned):
                    if line != expect:
                        mismatches.append(f"{name}@{seq}: {line!r} != {expect!r}")
                continue
            events = trace["events_before"] if phase == "before" else trace["events"]
            pending.setdefault(id(events), []).append((events, kind, payload, got))
        for rows in pending.values():
            events = rows[0][0]
            specs = []
            for _events, kind, payload, _got in rows:
                if kind == "query":
                    specs.extend((seq, name) for name, seq in payload)
                else:
                    specs.append((payload, None))
            answers = answer_books(events, specs)
            cursor = 0
            for _events, kind, payload, got in rows:
                if kind == "query":
                    for (name, seq), line in zip(payload, got):
                        expect = answers[cursor][0]
                        cursor += 1
                        if line != expect:
                            mismatches.append(f"{name}@{seq}: {line!r} != {expect!r}")
                else:
                    expect = answers[cursor]
                    cursor += 1
                    if got != expect:
                        mismatches.append(f"snap {payload}: {got[:3]!r} != {expect[:3]!r}")
    metrics = {
        "wall": wall,
        "cpu": max(cpu, 0.0),
        "query": p95(session.query_times),
        "snap": p95(session.snap_times),
    }
    return metrics, rss, mismatches


def io_overhead(row: dict) -> float:
    """Grader pipe I/O estimated as starter wall minus /proc CPU (diagnostics)."""
    return max(0.0, float(row.get("wall", 0.0)) - float(row.get("cpu", 0.0)))


def productive(row: dict, overhead: float) -> float:
    """Scored engine work: max(/proc, wait4) user+sys CPU (I/O is not scored).

    Floored at 1e-6 so a zeroed metric cannot make ln(T_start/T) diverge; a
    near-zero reading still scores near the ceiling only if the starter is
    comparatively huge, and multi-proc / forge paths are handled elsewhere.
    """
    del overhead
    return max(float(row.get("cpu", 0.0)), 1e-6)


def _rewrite_reads(frames: list[bytes], events: dict[int, bytes], end_seq: int) -> list[tuple]:
    """Apply rewritten adds in order. Each rewrite is followed by one mid-span QUERY.

    The QUERY carries three mid-span rows in a single round-trip. Expected lines
    start as None and are filled by fill_pinned_expects from the oracle binary.
    """
    del end_seq
    plan = pinned_query_plan(events, frames)
    ops = []
    for i, blob in enumerate(frames):
        ops.append(("apply", blob))
        rows = [(symbol, at) for at, symbol in plan[3 * i : 3 * i + 3]]
        ops.append(("query", rows, "pinned", None))
    return ops


def fill_pinned_expects(binary: Path, trace: dict, identity: tuple[int, int] | None) -> None:
    """Run the oracle once; stamp QUERY expects (pinned + tip) and SNAP replies."""
    session = Session(binary, identity)
    try:
        for index, op in enumerate(trace["ops"]):
            kind, payload = op[0], op[1]
            if kind == "apply":
                session.apply(payload)
            elif kind == "query":
                got = session.query(payload)
                phase = op[2] if len(op) > 2 else "after"
                # Keep pinned expects for grading; also store every QUERY for playback.
                trace["ops"][index] = ("query", payload, phase, list(got))
            else:
                got = session.snap(payload)
                phase = op[2] if len(op) > 2 else "after"
                trace["ops"][index] = ("snap", payload, phase, list(got))
    finally:
        session.close()
    missing = sum(
        1
        for op in trace["ops"]
        if op[0] == "query" and (len(op) < 4 or op[3] is None)
    )
    if missing:
        raise EngineFail(f"pinned expect harvest left {missing} queries unset")


def cross_check_expects_with_checker(
    checker: Path,
    trace: dict,
    identity: tuple[int, int] | None,
) -> None:
    """Independently validate harvested expects against the rebuild checker.

    The checker (checkpointed rebuild, kCk=96) is a different density from the
    rebuild harvest oracle (kCk=256) and from the order-diff reference.
    Tip/snapshot rows also still use the Python fold in play(). Infrastructure
    fails on any mismatch.
    """
    _metrics, _rss, bad = play(checker, trace, identity, check=True)
    if bad:
        raise EngineFail(f"checker disagrees with oracle harvest: {bad[:3]}")


def write_floor_expects(trace: dict, path: Path) -> None:
    """Serialize harvested QUERY/SNAP book lines for the correct-emit floor."""
    chunks: list[bytes] = []
    for op in trace["ops"]:
        kind = op[0]
        if kind == "query":
            if len(op) < 4 or op[3] is None:
                raise EngineFail("floor expects require harvested QUERY lines")
            for line in op[3]:
                payload = (line + "\n").encode()
                chunks.append(struct.pack("<I", len(payload)) + payload)
        elif kind == "snap":
            if len(op) < 4 or op[3] is None:
                raise EngineFail("floor expects require harvested SNAP lines")
            body = "".join(line + "\n" for line in op[3]).encode()
            chunks.append(struct.pack("<I", len(body)) + body)
    path.write_bytes(b"".join(chunks))


def build_ops(spec_name: str) -> dict:
    spec = FAMILIES[spec_name]
    trace = generate(spec["seed"], spec)

    def batches(rows: list) -> list[list]:
        return [rows[i : i + QUERY_BATCH] for i in range(0, len(rows), QUERY_BATCH)]

    ops = []
    if spec["interleave"]:
        queries = sorted(trace["queries"], key=lambda row: row[1])
        qi = 0
        covered = 0
        seqs = sorted(trace["events"])

        def frames(blob: bytes) -> int:
            i = 0
            count = 0
            while i + 6 <= len(blob):
                length = int.from_bytes(blob[i + 4 : i + 6], "big")
                i += 6 + length
                count += 1
            return count

        for blob in trace["blobs"]:
            ops.append(("apply", blob))
            covered += frames(blob)
            limit = seqs[min(covered, len(seqs)) - 1]
            bucket = []
            while qi < len(queries) and queries[qi][1] <= limit:
                bucket.append(queries[qi])
                qi += 1
            for batch in batches(bucket):
                ops.append(("query", batch, "before"))
            if spec["snaps"]:
                ops.append(("snap", limit, "before"))
        ops.extend(_rewrite_reads(trace["rewrite_frames"], trace["events_before"], max(trace["events_before"])))
        for batch in batches(queries[qi:]):
            ops.append(("query", batch, "after"))
    else:
        for blob in trace["blobs"]:
            ops.append(("apply", blob))
        ops.extend(_rewrite_reads(trace["rewrite_frames"], trace["events_before"], max(trace["events_before"])))
        for batch in batches(trace["queries"]):
            ops.append(("query", batch, "after"))
        for seq in trace["snaps"]:
            ops.append(("snap", seq, "after"))
    trace["ops"] = ops
    return trace


def median_run(binary: Path, trace: dict, identity: tuple[int, int] | None) -> tuple[dict, int, list[str]]:
    runs = []
    rss = 0
    bad: list[str] = []
    for _attempt in range(REPEATS):
        metrics, peak, mismatches = play(binary, trace, identity, check=True)
        rss = max(rss, peak)
        if mismatches:
            bad.extend(mismatches[:3])
            break
        runs.append(metrics)
    if not runs:
        return {"wall": 0.0, "cpu": 0.0, "query": 0.0, "snap": 0.0}, rss, bad
    return summarize_runs(runs), rss, bad


def score(
    base: dict[str, dict],
    sub: dict[str, dict],
    floor: dict[str, dict],
    failed: bool,
) -> float:
    """Worst family on ln(T_start / T) / ln(T_start / T_floor) using max(/proc, wait4) CPU.

    T_start is the baked checkpointed-rebuild starter. T_floor is the sealed
    reference engine for that family. Matching the starter scores 0; matching
    the reference bound (within timing noise) scores 1.
    """
    if failed:
        return 0.0
    cells = []
    for family in FAMILIES:
        oh = io_overhead(base[family])
        b = productive(base[family], oh)
        t = productive(sub[family], oh)
        f = productive(floor[family], oh)
        if b <= 0 or t <= 0 or f <= 0 or t >= b or f >= b:
            # Matching or slower than the starter is the zero of the scale.
            # Floor must stay strictly faster than the starter (infra).
            cell = 0.0
        elif t <= f * 1.08:
            # Same algorithm as the reference bound: treat timing noise as full marks.
            cell = 1.0
        else:
            cell = min(1.0, math.log(b / t) / math.log(b / f))
        cells.append(cell)
    value = min(cells) if cells else 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _frame(seq: int, body: bytes) -> bytes:
    return struct.pack(">IH", seq, len(body)) + body


def _add_body(oid: int, side: str, shares: int, stock: str, price: int) -> bytes:
    body = bytearray(36)
    body[0] = ord("A")
    body[11:19] = oid.to_bytes(8, "big")
    body[19] = ord(side)
    body[20:24] = shares.to_bytes(4, "big")
    body[24:32] = stock.ljust(8).encode()[:8]
    body[32:36] = price.to_bytes(4, "big")
    return bytes(body)


def probe(binary: Path, identity: tuple[int, int] | None) -> None:
    """Unknown types are ignored. A short frame must exit nonzero."""
    unknown = bytearray(8)
    unknown[0] = ord("Z")
    blob = _frame(1, bytes(unknown)) + _frame(2, _add_body(9, "B", 4, "XYZ", 10100))
    session = Session(binary, identity)
    try:
        session.apply(blob)
        got = session.query([("XYZ", 2)])
    finally:
        session.close()
    if got != ["XYZ B 10100:4:9 A -"]:
        raise EngineFail(f"unknown message type was not ignored: {got!r}")
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=lambda: _preexec(identity),
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(b"APPLY 40\n" + b"short")
        proc.stdin.close()
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise EngineFail("truncated frame hung") from exc
        if rc == 0:
            raise EngineFail("truncated frame exited 0")
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reap_uid(identity)


CHECK_NAMES = (
    "compile_submission",
    "probe_unknown_type",
    "probe_truncated_frame",
    *(f"{family}_{gate}" for family in FAMILIES for gate in ("book", "rss")),
)


def write_reward(reward: Path, value: float) -> None:
    reward.parent.mkdir(parents=True, exist_ok=True)
    if reward.parent == Path("/logs/verifier"):
        os.chmod(reward.parent, 0o700)
    reward.write_text(f"{value:.6f}\n")


def write_checks(reward: Path, checks: dict[str, str]) -> None:
    reward.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: checks.get(name, "skip") for name in CHECK_NAMES}
    (reward.parent / "checks.json").write_text(json.dumps(payload) + "\n")


def main() -> int:
    root = Path(__file__).resolve().parent
    app = Path(os.environ.get("APP_SRC", "/app/src/bookd.cpp"))
    baseline_src = Path(os.environ.get("BASELINE_SRC", root / "baseline" / "bookd.cpp"))
    checker_src = Path(os.environ.get("CHECKER_SRC", root / "checker" / "bookd.cpp"))
    floor_src = Path(os.environ.get("FLOOR_SRC", root / "floor" / "bookd.cpp"))
    oracle_src = Path(os.environ.get("ORACLE_SRC", root / "oracle" / "bookd.cpp"))
    reward = Path(os.environ.get("REWARD_PATH", "/logs/verifier/reward.txt"))
    if os.geteuid() == 0:
        tests = Path("/tests")
        if tests.is_dir():
            os.chmod(tests, 0o700)
        if reward.parent == Path("/logs/verifier"):
            reward.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(reward.parent, 0o700)
    checks = {name: "skip" for name in CHECK_NAMES}
    if app.is_symlink() or not app.is_file():
        write_reward(reward, 0.0)
        write_checks(reward, checks)
        print("missing submission", file=sys.stderr)
        return 0
    if "WORK_DIR" in os.environ:
        work = Path(os.environ["WORK_DIR"])
    elif os.geteuid() == 0:
        work = Path("/var/lib/itch-grade")
    else:
        work = Path("/tmp/itch-asof")
    work.mkdir(parents=True, exist_ok=True)
    # Submission binaries are the only engine artifacts the unprivileged engine
    # process may see. Starter/floor live in a root-only tree so a submission
    # cannot execv the verifier floor via /proc/self/exe sibling discovery.
    public = work / "run"
    private = work / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        os.chmod(work, 0o755)
        os.chmod(public, 0o755)
        os.chmod(private, 0o700)
    identity = engine_identity()
    # Time starter/floor as root so their binaries never need to be runnable by nobody.
    trusted_identity = None if os.geteuid() == 0 else identity
    # Starter/floor binaries are always built from verifier-held sources under
    # tests/ (or BASELINE_SRC/FLOOR_SRC), never from /app/bookd. Replacing the
    # agent-visible starter cannot inflate T_start — settles that cheat lead.
    try:
        compile_one(baseline_src, private / "baseline")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("baseline failed to compile", file=sys.stderr)
        return 2
    try:
        compile_one(checker_src, private / "checker")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("checker failed to compile", file=sys.stderr)
        return 2
    try:
        compile_one(floor_src, private / "floor")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("floor failed to compile", file=sys.stderr)
        return 2
    try:
        compile_one(oracle_src, private / "oracle")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("oracle failed to compile", file=sys.stderr)
        return 2
    # Seal verifier-only sources before the submission compile. The submission
    # compiler drops to nobody with /tests mode 0700, so #include "/tests/..." fails.
    if os.geteuid() == 0:
        tests = Path("/tests")
        if tests.is_dir():
            os.chmod(tests, 0o700)
        for sealed in (floor_src, baseline_src, checker_src, oracle_src, root / "model.py"):
            try:
                if sealed.is_file():
                    os.chmod(sealed, 0o000)
            except OSError:
                pass
        os.chmod(private, 0o700)
        for path in private.iterdir():
            os.chmod(path, 0o500)
    try:
        compile_one(app, public / "submission", identity, hide_tests=True)
    except FileNotFoundError:
        print("compiler missing", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError:
        checks["compile_submission"] = "fail"
        write_reward(reward, 0.0)
        write_checks(reward, checks)
        print("submission failed to compile", file=sys.stderr)
        return 0
    checks["compile_submission"] = "pass"
    if os.geteuid() == 0:
        os.chmod(private, 0o700)
        for path in private.iterdir():
            os.chmod(path, 0o500)

    try:
        probe(private / "baseline", trusted_identity)
    except EngineFail as exc:
        print(f"baseline failed the protocol probe: {exc}", file=sys.stderr)
        return 2
    try:
        probe(private / "oracle", trusted_identity)
    except EngineFail as exc:
        print(f"oracle failed the protocol probe: {exc}", file=sys.stderr)
        return 2
    try:
        probe(public / "submission", identity)
    except EngineFail as exc:
        text = str(exc)
        if "truncated" in text:
            checks["probe_unknown_type"] = "pass"
            checks["probe_truncated_frame"] = "fail"
        else:
            checks["probe_unknown_type"] = "fail"
        write_reward(reward, 0.0)
        write_checks(reward, checks)
        print(f"submission failed the protocol probe: {exc}", file=sys.stderr)
        return 0
    checks["probe_unknown_type"] = "pass"
    checks["probe_truncated_frame"] = "pass"

    traces = {name: build_ops(name) for name in FAMILIES}
    for name, trace in traces.items():
        print(f"harvest pinned expects {name}", flush=True)
        try:
            fill_pinned_expects(private / "oracle", trace, trusted_identity)
            print(f"checker cross-check expects {name}", flush=True)
            cross_check_expects_with_checker(
                private / "checker", trace, trusted_identity
            )
        except EngineFail as exc:
            print(f"oracle/checker expect validation failed {name}: {exc}", file=sys.stderr)
            return 2
    base_m: dict[str, dict] = {}
    sub_m: dict[str, dict] = {}
    floor_m: dict[str, dict] = {}
    failed = False
    for name, trace in traces.items():
        # Interleave baseline, floor, and submission so each is timed under the
        # same host conditions (the submission is normalized against the starter).
        print(f"baseline+floor+submission {name}", flush=True)
        base_runs: list[dict] = []
        floor_runs: list[dict] = []
        sub_runs: list[dict] = []
        base_rss = 0
        floor_rss = 0
        sub_rss = 0
        sub_bad: list[str] = []
        for _attempt in range(REPEATS):
            try:
                # check=False: expects already validated once against baseline above.
                metrics, rss, bad = play(
                    private / "baseline", trace, trusted_identity, check=False
                )
            except EngineFail as exc:
                print(f"baseline failed {name}: {exc}", file=sys.stderr)
                return 2
            # Baseline RSS is not gated on RSS_CAP_KB: the starter can
            # sit near the submission cap, and treating that as infra exit 2
            # leaves nop with no reward file. Wrong books still fail infra.
            if bad:
                print(f"baseline failed {name} {bad[:2]}", file=sys.stderr)
                return 2
            base_runs.append(metrics)
            base_rss = max(base_rss, rss)
            try:
                # Sealed reference engine bound (order-diff). check=True confirms
                # it matches the harvested full-depth books.
                metrics, rss, bad = play(
                    private / "floor",
                    trace,
                    trusted_identity,
                    check=True,
                )
            except EngineFail as exc:
                print(f"floor failed {name}: {exc}", file=sys.stderr)
                return 2
            if bad:
                print(f"floor failed {name} {bad[:2]}", file=sys.stderr)
                return 2
            floor_runs.append(metrics)
            floor_rss = max(floor_rss, rss)
            try:
                metrics, rss, bad = play(public / "submission", trace, identity, check=True)
            except EngineFail as exc:
                checks[f"{name}_book"] = "fail"
                checks[f"{name}_rss"] = "fail"
                write_reward(reward, 0.0)
                write_checks(reward, checks)
                print(f"submission failed {name}: {exc}", file=sys.stderr)
                return 0
            if bad or rss > RSS_CAP_KB:
                sub_bad = bad
                sub_rss = max(sub_rss, rss)
                checks[f"{name}_book"] = "fail" if bad else "pass"
                checks[f"{name}_rss"] = "fail" if rss > RSS_CAP_KB else "pass"
                print(f"submission failed {name} rss={rss} {bad[:2]}", file=sys.stderr)
                failed = True
                break
            sub_runs.append(metrics)
            sub_rss = max(sub_rss, rss)
        if failed:
            break
        base_m[name] = summarize_runs(base_runs)
        floor_m[name] = summarize_runs(floor_runs)
        sub_m[name] = summarize_runs(sub_runs)
        oh = io_overhead(base_m[name])
        base_p = productive(base_m[name], oh)
        floor_p = productive(floor_m[name], oh)
        if floor_p <= 0 or floor_p >= base_p:
            print(
                f"floor not faster than baseline on {name}: "
                f"base_p={base_p:.4f}s floor_p={floor_p:.4f}s "
                f"wall={base_m[name]['wall']:.4f}/{floor_m[name]['wall']:.4f} "
                f"cpu={base_m[name]['cpu']:.4f}/{floor_m[name]['cpu']:.4f} oh={oh:.4f}",
                file=sys.stderr,
            )
            return 2
        checks[f"{name}_book"] = "pass"
        checks[f"{name}_rss"] = "pass"
        print(
            f"{name}: base_p={base_p:.4f}s floor_p={floor_p:.4f}s "
            f"sub_p={productive(sub_m[name], oh):.4f}s oh={oh:.4f} "
            f"rss={max(base_rss, floor_rss, sub_rss)}",
            flush=True,
        )
    if failed:
        value = 0.0
    else:
        value = score(base_m, sub_m, floor_m, False)
    value = float(f"{value:.6f}")
    write_reward(reward, value)
    write_checks(reward, checks)
    detail = reward.with_name("reward.json")
    detail.write_text(json.dumps({"reward": value}) + "\n")
    print(f"reward {value:.6f}")
    return 0



if __name__ == "__main__":
    try:
        sys.exit(main())
    except EngineFail as exc:
        print(f"submission failed: {exc}", file=sys.stderr)
        reward = Path(os.environ.get("REWARD_PATH", "/logs/verifier/reward.txt"))
        try:
            write_reward(reward, 0.0)
        except OSError:
            sys.exit(2)
        sys.exit(0)
    except Exception as exc:
        print(f"grader infrastructure error: {exc}", file=sys.stderr)
        sys.exit(2)
