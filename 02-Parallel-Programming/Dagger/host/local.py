"""Local CPU preparation and launch; no SSH, Slurm or platform shell tools."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys


def environment(root, cache, threads):
    env = dict(os.environ, SC26_ROOT=str(root), SC26_CACHE_ROOT=str(cache),
               SC26_LOCAL="1", SC26_BACKEND="cpu", SC26_THREADS=str(threads),
               JULIA_PROJECT=str(root / "cpu"), JULIA_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               JULIA_NUM_PRECOMPILE_TASKS="2", JULIA_PKG_PRECOMPILE_AUTO="0")
    # Respect the attendee's depot (and installed package/artifact caches).
    return env


def julia():
    binary = shutil.which("julia")
    if not binary:
        raise RuntimeError("Local mode requires Julia 1.11+ on PATH")
    return binary


def prepare():
    root = Path(os.environ["SC26_ROOT"])
    cache = Path(os.environ["SC26_CACHE_ROOT"])
    cache.mkdir(parents=True, exist_ok=True)
    project = root / "cpu"
    manifest = project / "Manifest.toml"
    seed = cache / "Manifest.toml"
    print("Preparing local Julia packages (first connection may take several minutes)…", flush=True)
    # macOS/Linux/WSL: no dependency on Linux-only flock or nproc commands.
    with (cache / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if seed.exists() and not manifest.exists():
            shutil.copyfile(seed, manifest)
        code = '''
            VERSION >= v"1.11" || error("Local mode requires Julia 1.11 or newer")
            using Pkg
            isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")
            Pkg.resolve()
            Pkg.instantiate(; allow_autoprecomp=false)
            using MPIPreferences
            MPIPreferences.use_jll_binary("MPICH_jll")
        '''
        command = [julia(), "--startup-file=no", "--project=" + str(project), "-e"]
        env = dict(os.environ, JULIA_PKG_OFFLINE="false")
        subprocess.run(command + [code], env=env, stdin=subprocess.DEVNULL, check=True)
        # MPI preferences must take effect in a fresh Julia process before
        # precompilation and loading MPI-dependent extensions.
        code = '''
            using Pkg
            Pkg.precompile()
            using Dagger, MPI, Highlights, UnicodePlots
            println("Local Julia ", VERSION, " · Dagger ", pkgversion(Dagger), " · ", Threads.nthreads(), " threads")
        '''
        subprocess.run(command + [code], env=env, stdin=subprocess.DEVNULL, check=True)
        shutil.copyfile(manifest, seed.with_suffix(".tmp"))
        os.replace(seed.with_suffix(".tmp"), seed)
    agent = root / "host/remote_agent.py"
    os.execv(sys.executable, [sys.executable, "-u", str(agent)])


def launch(ranks, arguments):
    project = os.environ["JULIA_PROJECT"]
    # The coordinator divides the CPU budget across MPI ranks.
    os.environ["JULIA_NUM_THREADS"] = os.environ["SC26_THREADS"]
    argv = [julia(), "--startup-file=no", "--project=" + project]
    if os.environ.get("SC26_CPU_MPI") == "1":
        argv += ["-e", '''using MPI
            ranks = parse(Int, popfirst!(ARGS))
            run(`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$(Base.active_project()) $ARGS`)
            ''', str(ranks)]
    os.execv(argv[0], argv + arguments)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        prepare()
    elif sys.argv[1] == "launch":
        launch(int(sys.argv[2]), sys.argv[3:])
    else:
        raise SystemExit("Expected serve or launch")
