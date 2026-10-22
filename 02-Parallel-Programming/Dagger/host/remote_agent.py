#!/usr/bin/env python3
"""Login-side coordinator, owning the portal's one-node Slurm holding job.

stdin/stdout carry JSON over one SSH session (Odo allows one session/transport).
Only the Julia children run on compute nodes. EOF releases our allocation.
"""
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "@@SC26@@"
worker = None
mode = None
ready = threading.Event()
write_lock = threading.Lock()
operation = threading.Lock()
stopping = threading.Event()
job = None


def configure_stdio():
    # Julia's login-node preparation can leave inherited pipe descriptions in
    # nonblocking mode. Large timeline reports must wait for SSH backpressure.
    for fd in (0, 1, 2):
        os.set_blocking(fd, True)


def emit(kind, **fields):
    with write_lock:
        print(PREFIX + json.dumps(dict(kind=kind, **fields)), flush=True)


def relay(proc):
    for line in proc.stdout:
        if line.startswith(PREFIX):
            try:
                event = json.loads(line[len(PREFIX):])
                if event["kind"] == "worker_ready":
                    ready.set()
                emit(**event)
            except (ValueError, KeyError):
                emit("output", text=line)
        else:
            emit("output", text=line)
    rc = proc.wait()
    emit("worker_exit", returncode=rc, expected=getattr(proc, "closing", False))
    ready.set()  # Wake a pending command on early initialization failure.


def close_worker():
    global worker, mode
    proc, worker, mode = worker, None, None
    if proc is None:
        return
    proc.closing = True
    if proc.poll() is None:
        # Also handles a running expression that cannot consume a quit command.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    ready.clear()


def execute(command):
    global worker, mode
    if not operation.acquire(blocking=False):
        emit("done", run=command.get("id", ""), ok=False, error="A worker is still starting; wait or cancel it.")
        return
    try:
        wanted = command.get("mode", "single")
        if wanted not in ("single", "mpi"):
            raise ValueError("Unknown worker layout")
        if worker is None or worker.poll() is not None or mode != wanted:
            close_worker()
            if stopping.is_set():
                return
            mode = wanted
            env = dict(os.environ, SC26_NODES="1", SC26_GPUS_PER_TASK="2" if wanted == "mpi" else "4",
                       SC26_PORTAL="1", SC26_DEMO_LOAD_ONLY="1", JULIA_PKG_OFFLINE="true")
            args = ["bash", str(ROOT / "frontier/launch.sh"), "2" if wanted == "mpi" else "1",
                    "--startup-file=no", str(ROOT / "host/worker.jl")]
            if wanted == "mpi":
                args.append("--mpi")
            ready.clear()
            emit("status", message=f"Starting {'two MPI ranks' if wanted == 'mpi' else 'one Julia process'}…")
            worker = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      universal_newlines=True, bufsize=1, start_new_session=True)
            threading.Thread(target=relay, args=(worker,), daemon=True).start()
            if not ready.wait(timeout=300):
                raise TimeoutError("Julia startup exceeded five minutes; cancel and inspect the output")
        if worker is None or worker.poll() is not None:
            raise RuntimeError("Julia worker exited during startup")
        worker.stdin.write(json.dumps(command) + "\n")
        worker.stdin.flush()
    except Exception as exc:
        emit("done", run=command.get("id", ""), ok=False, error=str(exc))
    finally:
        operation.release()


def allocate():
    global job
    (ROOT / "results").mkdir(exist_ok=True)
    args = ["sbatch", "--parsable", "-A", os.environ.get("SC26_ACCOUNT", "trn047"),
            "-p", "batch", "-N", "1", "-n", "8", "--ntasks-per-node=8", "-c", "7",
            "--threads-per-core=1", "--gpus-per-node=8", "-t", os.environ.get("SC26_TIME", "02:00:00"),
            "-J", "sc26-dagger-portal", "--output", str(ROOT / "results/portal-%j.log"),
            "--wrap", "exec sleep infinity"]
    if os.environ.get("SC26_RESERVATION"):
        args.append("--reservation=" + os.environ["SC26_RESERVATION"])
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unexpected sbatch response: " + result.stdout)
    os.environ["SLURM_JOB_ID"] = job
    emit("status", message=f"Job {job} queued · waiting for one node…")


def watch_allocation():
    announced = False
    while not stopping.wait(2):
        result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T|%N"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if not result.stdout.strip():
            emit("error", error=f"Allocation {job} ended. Release the session and reconnect.")
            stopping.set()
            break
        state, nodes = result.stdout.strip().split("|", 1)
        if state == "RUNNING" and not announced:
            emit("allocated", job=job, nodes=nodes)
            announced = True


def main():
    try:
        allocate()
        threading.Thread(target=watch_allocation, daemon=True).start()
        pending = b""
        while not stopping.is_set():
            # Avoid TextIO read-ahead: a second buffered command (e.g. Cancel)
            # must not wait for another OS-level readiness notification.
            if b"\n" not in pending:
                readable, _, _ = select.select([sys.stdin], [], [], 1)
                if not readable:
                    continue
                chunk = os.read(0, 65536)
                if not chunk:
                    break
                pending += chunk
                continue
            line, pending = pending.split(b"\n", 1)
            line = line.decode("utf-8")
            command = json.loads(line)
            if command["op"] == "quit":
                break
            if command["op"] == "cancel":
                close_worker()
                ready.set()
                emit("cancelled")
            else:
                threading.Thread(target=execute, args=(command,), daemon=True).start()
    finally:
        stopping.set()
        close_worker()
        if job is not None:
            subprocess.run(["scancel", job], check=False)


if __name__ == "__main__":
    configure_stdio()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))
    main()
