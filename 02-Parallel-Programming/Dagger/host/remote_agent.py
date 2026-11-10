#!/usr/bin/env python3
"""Login-side coordinator: every run is its own Slurm job.

stdin/stdout carry JSON over one SSH session. Each demo, exercise or REPL
evaluation becomes one job (andromeda/launch.sh) with the CPUs and GPUs it
needs, and the job ends when the run does: nothing stays allocated between
runs. Cancel or EOF stops the run in progress, which ends its job.
"""
import json
import os
from pathlib import Path
import re
import signal
import select
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "@@SC26@@"
MAX_GPUS = 2   # andromeda's one node has two A30s
HEARTBEAT = 30   # seconds between heartbeats to the console
write_lock = threading.Lock()
lock = threading.Lock()
current = None        # the run in progress: dict(proc, id, name, done, cancelled)
compiling = None      # the package compile job while it runs
stopping = threading.Event()
QUEUED = re.compile(r"srun: job (\d+) queued and waiting for resources")
ALLOCATED = re.compile(r"srun: job (\d+) has been allocated resources")


def configure_stdio():
    # Julia's login-node preparation can leave inherited pipe descriptions in
    # nonblocking mode. Large timeline reports must wait for SSH backpressure.
    for fd in (0, 1, 2):
        os.set_blocking(fd, True)


def emit(kind, **fields):
    with write_lock:
        print(PREFIX + json.dumps(dict(kind=kind, **fields)), flush=True)


def heartbeat():
    """Write to the console regularly. A run can be quiet for minutes; if the
    SSH connection dies meanwhile, a write eventually fails and the
    coordinator shuts down instead of holding its run forever."""
    while not stopping.wait(HEARTBEAT):
        try:
            emit("heartbeat")
        except (OSError, ValueError):
            os.kill(os.getpid(), signal.SIGTERM)
            return


def pid_file():
    return ROOT / "results" / "agent.pid"


def replace_stale_agent(path=None):
    """Stop a coordinator left by an earlier connection to this workspace, so
    its run does not keep CPUs or GPUs; then record this one."""
    path = path or pid_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    def coordinator(pid):
        # An exited process (even a zombie) has an empty command line.
        try:
            return "remote_agent.py" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            return False
    try:
        old = int(path.read_text().strip())
    except (OSError, ValueError):
        old = None
    if old and old != os.getpid() and coordinator(old):
        emit("status", message="Stopping the coordinator of an earlier connection…")
        try:
            os.kill(old, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(40):   # it cancels its run first; give it 20 seconds
            if not coordinator(old):
                break
            time.sleep(0.5)
    path.write_text(str(os.getpid()))


def forget_pid(path=None):
    path = path or pid_file()
    try:
        if path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def resources(command):
    """(ranks, GPUs) for a command; app.py chooses them, this checks them."""
    ranks = 2 if command.get("mode") == "mpi" else 1
    gpus = command.get("gpus", 0)
    if not isinstance(gpus, int) or not 0 <= gpus <= MAX_GPUS or gpus % ranks:
        raise ValueError(f"Cannot give {ranks} process(es) {gpus!r} GPUs")
    return ranks, gpus


def describe(ranks, gpus):
    threads = 8 if ranks == 1 else 4
    who = "one process" if ranks == 1 else "two MPI ranks"
    cpus = f"{threads} CPUs" + ("" if ranks == 1 else " each")
    if gpus == 0:
        devices = "no GPU"
    elif ranks > 1:
        devices = f"{gpus // ranks} GPU each"
    else:
        devices = f"{gpus} GPU" + ("s" if gpus > 1 else "")
    return f"{who} · {cpus} · {devices}"


def stop(proc):
    """End a launch.sh process group; srun cancels its job on SIGTERM."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def precompile():
    """Compile the packages once per connection, in a Slurm job with a GPU."""
    global compiling
    if os.environ.get("SC26_PRECOMPILE", "1") == "0":
        emit("ready")
        return
    emit("status", message="Compiling packages in a Slurm job (16 CPUs, one GPU)…")
    with lock:
        if stopping.is_set():
            return
        compiling = subprocess.Popen(["bash", str(ROOT / "andromeda/precompile.sh")], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True,
                                     start_new_session=True)
    output, _ = compiling.communicate()
    rc = compiling.returncode
    compiling = None
    if stopping.is_set():
        return
    lines = (output or "").strip().splitlines()
    if rc == 0:
        emit("status", message=" · ".join(line for line in lines if line.startswith(("GPU:", "Dagger "))))
        emit("ready")
    else:
        tail = "\n".join(lines[-12:])
        emit("output", text=f"Package compile job failed (status {rc}):\n{tail}\n")
        emit("error", error="The package compile job failed: " + (lines[-1] if lines else f"status {rc}")
             + ". Runs will compile on first use; see the session output.")
        emit("ready", message="Ready · compile job failed, runs compile on first use")


def relay(run):
    global current
    proc = run["proc"]
    for line in proc.stdout:
        if line.startswith(PREFIX):
            try:
                event = json.loads(line[len(PREFIX):])
                if event["kind"] == "done":
                    run["done"] = True
                emit(**event)
            except (ValueError, KeyError):
                emit("output", text=line, run=run["id"])
            continue
        queued, allocated = QUEUED.search(line), ALLOCATED.search(line)
        if queued:
            emit("job", job=queued.group(1))
            emit("status", message=f"Job {queued.group(1)} queued · waiting for resources…")
        elif allocated:
            emit("job", job=allocated.group(1))
            emit("status", message=f"Job {allocated.group(1)} started · loading Julia…")
        # Tag the line with its run, so the console shows it in the panel
        # (demo, exercise or final exercise) that started the run.
        emit("output", text=line, run=run["id"])
    proc.stdout.close()
    rc = proc.wait()
    with lock:
        if current is run:
            current = None
    if not run["done"] and not run["cancelled"] and not stopping.is_set():
        emit("done", run=run["id"], ok=False,
             error=f"The Slurm job ended with status {rc} before the run finished; see the output.")
    emit("worker_exit", returncode=rc, expected=run["done"] or run["cancelled"])


def execute(command):
    """Submit one run as its own Slurm job and stream its output."""
    global current
    run_id = str(command.get("id", ""))
    with lock:
        if stopping.is_set():
            return
        if current is not None:
            emit("done", run=run_id, ok=False, error="A run is in progress; wait or cancel it.")
            return
        try:
            ranks, gpus = resources(command)
        except ValueError as exc:
            emit("done", run=run_id, ok=False, error=str(exc))
            return
        name = "sc26-" + re.sub(r"[^A-Za-z0-9]", "", run_id)[:16]
        args = ["bash", str(ROOT / "andromeda/launch.sh"), str(ranks), str(gpus),
                "--startup-file=no", str(ROOT / "host/worker.jl")]
        if ranks > 1:
            args.append("--mpi")
        if gpus:
            args.append("--gpu")
        env = dict(os.environ, SC26_PORTAL="1", JULIA_PKG_OFFLINE="true", SC26_JOB_NAME=name)
        emit("status", message=f"Submitting to Slurm: {describe(ranks, gpus)}…")
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1, start_new_session=True)
        run = dict(proc=proc, id=run_id, name=name, done=False, cancelled=False)
        current = run
    # The worker runs this one command, then reads quit and exits, which ends the job.
    try:
        proc.stdin.write(json.dumps(command) + "\n" + '{"op":"quit"}\n')
        proc.stdin.close()
    except (OSError, ValueError):
        pass
    threading.Thread(target=relay, args=(run,), daemon=True).start()


def cancel(final=False):
    with lock:
        run = current
    if run is None:
        if not final:
            emit("cancelled")
        return
    run["cancelled"] = True
    # Cancel the job through Slurm first, so it ends as CANCELLED and srun
    # exits by itself; then make sure the launcher is gone.
    if os.environ.get("SC26_LAUNCHER", "slurm") != "local":
        subprocess.run(["scancel", "--name", run["name"], "--user", os.environ.get("USER", "")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        try:
            run["proc"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    stop(run["proc"])
    if not final:
        emit("cancelled", run=run["id"])


def main():
    replace_stale_agent()
    threading.Thread(target=precompile, daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
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
            command = json.loads(line.decode("utf-8"))
            if command["op"] == "quit":
                break
            if command["op"] == "cancel":
                cancel()
            else:
                execute(command)
    finally:
        stopping.set()
        cancel(final=True)
        job = compiling
        if job is not None:
            stop(job)
        forget_pid()


if __name__ == "__main__":
    configure_stdio()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))
    main()
