#!/usr/bin/env python3
"""Local tutorial console. Python 3.9+, OpenSSH; no Python packages required."""
import argparse
import atexit
import codecs
import errno
import getpass
import io
import ipaddress
import json
import os
from pathlib import Path
import posixpath
import re
import secrets
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
try:
    import fcntl
    import pty
    import termios
except ImportError:  # Windows has no Unix PTYs.
    fcntl = pty = termios = None
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "@@SC26@@"
DEMOS = [
    ("00_julia", "Julia basics", "single"),
    ("01_tasks", "Tasks & dependencies", "single"),
    ("02_datadeps", "Mutable data & tiles", "single"),
    ("03_gpu", "Single GPU", "single"),
    ("05_multigpu", "Multiple GPUs", "single"),
    ("06_multinode", "Multi-node · CPU MPI", "mpi"),
    ("07_multinode_multigpu", "Multi-node + multi-GPU", "mpi"),
]
# Demos that share one page section. Everything else gets its own section.
SECTIONS = [
    ("julia", "Julia basics", ["00_julia"]),
    ("cpu", "CPU", ["01_tasks", "02_datadeps"]),
]
# One Exercise panel per section after Julia basics. Each one runs the shared
# Function editor (my_cholesky!) and then its own Setup and Use code, which
# chooses the hardware and calls the function. `target` and `mode` are the
# section's starting choices; the attendee can change both.
TARGETS = {"cpu": "CPU", "gpu": "GPU", "gpus": "All GPUs", "mixed": "Mixed"}
MODES = {"single": "One process", "mpi": "MPI ranks"}
# Every run is its own Slurm job. A single-process run that uses GPUs sees four
# of the node's eight (the Multiple GPUs demo asks for 2, 4 or 8). An MPI run
# puts each rank on its own Odo node, with all eight of its GPUs when the run
# uses GPUs. A CPU run asks for none.
GPUS_PER_TASK = {"single": 4, "mpi": 8}
MAX_RANKS = 32
DEFAULT_RANKS = 2


def mpi_ranks(value):
    """The rank count of an MPI run: 1 to 32, one rank per Odo node."""
    try:
        ranks = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Choose 1 to {MAX_RANKS} MPI ranks")
    if not 1 <= ranks <= MAX_RANKS:
        raise ValueError(f"Choose 1 to {MAX_RANKS} MPI ranks")
    return ranks


def layout_label(mode, ranks):
    if mode != "mpi":
        return MODES["single"]
    return f"{ranks} MPI rank" + ("s" if ranks != 1 else "")
GPU_DEMOS = {"03_gpu", "05_multigpu", "07_multinode_multigpu"}


def run_gpus(target, mode):
    """GPUs per process for an exercise or final-exercise run."""
    return 0 if target == "cpu" else GPUS_PER_TASK[mode]


def demo_gpus(demo, mode):
    """GPUs per process for a demo, or for a REPL evaluation in its section."""
    return GPUS_PER_TASK[mode] if demo in GPU_DEMOS else 0

EXERCISES = {
    "02_datadeps": dict(
        target="cpu", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: parallel tiled Cholesky (CPU)

            The Function editor starts from the serial, tile-by-tile Cholesky
            factorization in exercise/my_cholesky.jl. Make it parallel
            without changing the algorithm or the loop structure.

            1. Wrap the whole loop nest in Dagger.spawn_datadeps() do ... end.
            2. Put Dagger.@spawn in front of each of the four tile operations
               (factor_diagonal_tile!, solve_panel_tile!, update_diagonal_tile!,
               update_offdiagonal_tile! -- aliases of potrf!, trsm!, syrk!,
               gemm!).
            3. Annotate every tile argument:
               In(tile)    -- the call only reads the tile.
               InOut(tile) -- the call reads and writes the tile.
            4. Return T.

            Rules:
            - Spawn the BLAS/LAPACK call itself (the aliases are those calls),
              not a helper that calls it. Dagger replaces BLAS with
              rocBLAS/rocSOLVER on GPU tasks.
            - Do not use fetch or wait inside the Datadeps region.
            - Under MPI every rank must build the same graph; do not branch
              on rank.

            Run exercise runs the Function, then Setup and Use. Done when it
            prints PASS.
            Full handout: exercise/README.md""")),
    "03_gpu": dict(
        target="gpu", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: same code, on the GPU

            No changes to the Function. The my_cholesky! you wrote in
            exercise/my_cholesky.jl for the CPU must run unchanged on an AMD
            GPU. Setup and Use now uses the
            GPU scope, Dagger.scope(rocm_gpu=1); Dagger maps the spawned
            BLAS/LAPACK calls to rocBLAS/rocSOLVER when a task lands on the
            GPU.

            If the GPU run fails inside your own helper, spawn the BLAS/LAPACK
            call itself so Dagger can substitute the GPU implementation.

            Full handout: exercise/README.md""")),
    "05_multigpu": dict(
        target="gpus", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: every GPU, or CPU and GPU together

            No changes to the Function. Setup and Use now uses every visible
            GPU, Dagger.scope(rocm_gpus=:).
            - Run it as it is, then compare the timeline with one GPU.
            - Set Target to Mixed to let Dagger place tasks on both CPU and
              GPU. Is it faster? Why or why not?
            - Try other matrix and tile sizes (n and bs) in Setup and Use.
            - Look at solution/my_cholesky.jl only after you finish.

            Full handout: exercise/README.md""")),
    "06_multinode": dict(
        target="cpu", mode="mpi",
        prompt=textwrap.dedent("""\
            Exercise: same code, on MPI ranks (CPU)

            No changes to the Function. Processes is now MPI ranks, each on
            its own Odo node; choose how many with Ranks (1 to 32). Every
            rank runs the same Function and Setup and Use, and Dagger
            spreads the tiles and their tasks across the ranks. The test
            matrix is seeded, so every rank builds the same one, and only
            rank 0 prints.

            Your my_cholesky! from exercise/my_cholesky.jl is correct under
            MPI only if every In/InOut annotation is right: a tile written
            on another rank without InOut loses its update.

            Full handout: exercise/README.md""")),
    "07_multinode_multigpu": dict(
        target="gpu", mode="mpi",
        prompt=textwrap.dedent("""\
            Exercise: MPI ranks and GPUs together

            No changes to the Function. MPI ranks, each on its own Odo node
            with all of the node's GPUs: the same my_cholesky! from
            exercise/my_cholesky.jl, the same Setup and Use, with Target GPU
            (one GPU per rank) or All GPUs (every GPU of every rank) and
            Processes MPI ranks. Compare the timeline with the CPU run in the
            previous section, and try more ranks.

            Full handout: exercise/README.md""")),
}


def _imports(target, mode):
    lines = ["using Dagger, LinearAlgebra, Random"]
    if target != "cpu":
        lines.append("using AMDGPU")
    if mode == "mpi":
        lines += ["using MPI", "",
                  "rank = MPI.Comm_rank(MPI.COMM_WORLD)   # only rank 0 prints"]
    return lines


def _on_every_rank(mode):
    return ", on every MPI rank." if mode == "mpi" else "."


def _scope_lines(target, mode):
    """The scope line(s) for a target, as in harness/common.jl."""
    where = {"cpu": "the CPU threads", "gpu": "one GPU",
             "gpus": "every visible GPU", "mixed": "the CPU threads and one GPU"}[target]
    if mode == "mpi":
        where = where.replace("the CPU threads", "every rank's CPU threads")
        where += "" if target == "cpu" else " per rank"
    lines = [f"# (1) Where the tasks run: {where}."]
    if mode == "mpi" and target in ("cpu", "mixed"):
        lines.append("#     Constrain the MPI ranks to their CPUs; GPUs are opt-in.")
        cpu = "Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())"
    else:
        cpu = "Dagger.scope(worker=1)"
    gpu = "Dagger.scope(rocm_gpus=:)" if target == "gpus" else "Dagger.scope(rocm_gpu=1)"
    if target == "cpu":
        lines.append(f"scope = {cpu}")
    elif target in ("gpu", "gpus"):
        lines.append(f"scope = {gpu}")
    else:
        lines += [f"cpu = {cpu}", f"gpu = {gpu}", "scope = Dagger.UnionScope(cpu, gpu)"]
    return lines


def _say(mode, expression):
    """Print once: on rank 0 under MPI."""
    if mode == "mpi":
        return f"rank == 0 && println({expression})"
    return f"println({expression})"


def exercise_setup(target, mode):
    """Runnable Setup and Use for the Cholesky exercise."""
    n, bs = (1024, 256) if target == "cpu" else (2048, 512)
    lines = ["# Setup and Use: where the tasks run, the test matrix, the call to",
             "# my_cholesky!, and the check. Run exercise runs the Function first,",
             "# then this code" + _on_every_rank(mode),
             ""] + _imports(target, mode)
    lines += ["",
              "BLAS.set_num_threads(1)   # Dagger runs one tile operation per thread",
              ""]
    lines += _scope_lines(target, mode)
    lines += ["",
              "# (2) A symmetric positive-definite test matrix. It is seeded, so every",
              "#     MPI rank builds the same one.",
              f"n, bs = {n}, {bs}",
              "A = rand(Xoshiro(1234), n, n)",
              "A = A + A'",
              "A[diagind(A)] .+= 2n",
              "",
              "# (3) Tile A, factor the tiles in place, and gather the lower factor L.",
              "L = Dagger.with_options(scope=scope) do",
              "    DA = DArray(A, Blocks(bs, bs))",
              "    my_cholesky!(DA.chunks)",
              "    LowerTriangular(collect(DA))",
              "end",
              "",
              "# (4) Check A = L * L' with a scaled residual; it passes below 16.",
              "x = rand(Xoshiro(7), n)",
              "residual = norm(A * x - L * (L' * x), Inf) / (norm(A, Inf) * norm(x, Inf) * n * eps())",
              'status = residual < 16 ? "PASS" : "FAIL"',
              _say(mode, '"$status: n = $n, tiles of $bs, residual ", round(residual; sigdigits=3)'),
              'residual < 16 || error("my_cholesky! returned a wrong factor")']
    return "\n".join(lines) + "\n"


def final_setup(target, mode):
    """Runnable Setup and Use for the final exercise."""
    lines = ["# Setup and Use: where the tasks run, the parameters, the initial fields,",
             "# the call to simulate!, and the check. Run final exercise runs the",
             "# Simulation first, then this code" + _on_every_rank(mode),
             "# The Visualization panel then draws u and v.",
             ""] + _imports(target, mode)
    lines += [""] + _scope_lines(target, mode)
    lines += ["",
              "# (2) Parameters: grid size, time step, diffusion rates, feed and kill rates.",
              "n, dt = 256, 1.0",
              "Du, Dv = 0.16, 0.08",
              "F, k = 0.055, 0.062",
              "nsteps = 3600",
              "",
              "# (3) Initial fields: substrate u = 1 and a little activator v, plus 14",
              "#     seeded squares where the reaction starts. The seed makes every MPI",
              "#     rank start from the same fields.",
              "Random.seed!(42)",
              "u = ones(n, n)",
              "v = 0.01 .* rand(n, n)",
              "for _ in 1:14",
              "    cx, cy = rand(6:n-5, 2)",
              "    v[cx-3:cx+3, cy-3:cy+3] .= 1.0",
              "    u[cx-3:cx+3, cy-3:cy+3] .= 0.5",
              "end",
              "",
              "# (4) Run the simulation where the scope says; it updates u and v in place.",
              "Dagger.with_options(scope=scope) do",
              "    simulate!(u, v, (; n, dt, Du, Dv, F, k), nsteps)",
              "end",
              "",
              "# (5) Check and report: both fields must stay finite.",
              'all(isfinite, u) && all(isfinite, v) || error("the simulation blew up")',
              _say(mode, '"u range: ", extrema(u), " v range: ", extrema(v)')]
    return "\n".join(lines) + "\n"


def setup_templates(build):
    """Every Setup and Use variant, keyed "target/mode" for the console."""
    return {f"{target}/{mode}": build(target, mode) for target in TARGETS for mode in MODES}


FINAL_VISUALS = [
    dict(id="substrate", label="Substrate u"),
    dict(id="activator", label="Activator v"),
]
FINAL_VISUAL_IDS = tuple(item["id"] for item in FINAL_VISUALS)

FINAL_EXERCISE = dict(
    title="Final Exercise",
    prompt=textwrap.dedent("""\
        Final exercise: Gray-Scott reaction-diffusion

        Two chemicals diffuse and react on a periodic 2D grid:
            du/dt = Du laplacian(u) - u v^2 + F (1 - u)
            dv/dt = Dv laplacian(v) + u v^2 - (F + k) v

        The feed and kill rates F = 0.055 and k = 0.062 produce a
        labyrinth of spots and stripes. Use ADI splitting: each time
        step is two half steps. In the first, x is implicit and y is
        explicit; in the second, y is implicit and x is explicit.
        Each half step solves one independent 1D system per column
        or row, which is exactly the loop to parallelize with Dagger.
        For the first half step,
            rhs = u + dt/2 * (Du * (u * L) + reaction(u, v))
        then solve with Fu for every column j; repeat for v with
        Fv. The second half step uses L * u and solves rows i.

        The 1D periodic Laplacian L is symmetric. The two implicit
        operators
            A_u = I - (dt/2) Du L       A_v = I - (dt/2) Dv L
        are symmetric positive definite, so factor each once with
        cholesky(Symmetric(A)) and reuse the factors every step. The
        1D systems are small; the parallel work is the line solves.

        Fill in L, Fu, Fv, and the two half steps of simulate!,
        updating u and v in place. Setup and Use sets the parameters
        and the initial fields, calls simulate!(u, v, p, nsteps) where
        the scope says, and checks the result; Target and Processes
        choose the hardware and whether it runs on MPI ranks. The
        Visualization panel draws substrate u and activator v side by
        side and animates the sampled time steps; no plotting code
        belongs in your script.
    """),
    template=textwrap.dedent("""\
        # Final exercise: Gray-Scott reaction-diffusion on a periodic grid.
        # Fill in simulate! below. Setup and Use sets the parameters and the
        # initial fields, calls simulate!, and checks the result.

        using Dagger, LinearAlgebra

        # Advance the substrate u and the activator v by nsteps ADI time steps,
        # in place. p holds the parameters from Setup and Use: n, dt, Du, Dv, F, k.
        function simulate!(u, v, p, nsteps)
            (; n, dt, Du, Dv, F, k) = p

            # 1. Build the 1D periodic Laplacian L: diagonal -2, neighbour
            #    entries +1, and the first/last rows wrap around.
            L = ???

            # 2. Factor the implicit operators A_u = I - (dt/2) Du L and
            #    A_v = I - (dt/2) Dv L with Cholesky. They are SPD and
            #    constant, so one factorization serves every time step.
            Fu = ???
            Fv = ???

            # 3. Time loop with ADI splitting. Each column/row solve is
            #    independent; this is the loop to spawn as Dagger tasks.
            ru = similar(u)
            rv = similar(v)
            for step in 1:nsteps
                # First half step: y explicit (u * L), x implicit (Fu).
                # TODO

                # Second half step: x explicit (L * u), y implicit (Fv).
                # TODO
            end
            return u, v
        end
    """),
)


FINAL_ROOT = ROOT / "results" / "final-exercise"
FINAL_WORKSPACE = FINAL_ROOT
FINAL_FILE = FINAL_WORKSPACE / "final_exercise.jl"
FINAL_README = FINAL_WORKSPACE / "README.md"


def use_final_workspace(user):
    """Keep each Odo user's Simulation file apart, so consoles running side by
    side for different users do not overwrite each other's simulate!. The first
    time, start from the file an older console kept for everyone."""
    global FINAL_WORKSPACE, FINAL_FILE, FINAL_README
    workspace = FINAL_ROOT / re.sub(r"[^A-Za-z0-9_.-]", "-", user)
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("final_exercise.jl", "README.md"):
        shared = FINAL_ROOT / name
        if shared.is_file() and not (workspace / name).exists():
            shutil.copy2(shared, workspace / name)
    FINAL_WORKSPACE, FINAL_FILE, FINAL_README = (workspace, workspace / "final_exercise.jl",
                                                 workspace / "README.md")


def ensure_final_workspace():
    FINAL_WORKSPACE.mkdir(parents=True, exist_ok=True)
    # Older consoles kept the whole script, setup included, in this file. Setup
    # and Use is now separate, so keep that copy aside and start from simulate!.
    for path, old_layout in ((FINAL_FILE, "# Setup: parameters, fields, and the constant operators."),
                             (FINAL_README, "Fill in L, Fu, Fv, and the two half steps, then run.")):
        if path.exists() and old_layout in path.read_text():
            path.replace(path.with_name(path.stem + ".old" + path.suffix))
    if not FINAL_FILE.exists():
        FINAL_FILE.write_text(FINAL_EXERCISE["template"])
    if not FINAL_README.exists():
        FINAL_README.write_text(FINAL_EXERCISE["prompt"].rstrip()
                                + "\n\nEdit only final_exercise.jl.\n")


def final_file_state():
    ensure_final_workspace()
    return dict(code=FINAL_FILE.read_text(),
                mtime=FINAL_FILE.stat().st_mtime_ns)


def write_final_file(code):
    if not isinstance(code, str) or len(code) > 262144:
        raise ValueError("Invalid final exercise code")
    ensure_final_workspace()
    FINAL_FILE.write_text(code)
    return dict(ok=True, mtime=FINAL_FILE.stat().st_mtime_ns)


def exercises():
    """Exercise configuration keyed by demo id; unknown ids are a layout bug."""
    known = {demo for demo, _, _ in DEMOS}
    unknown = set(EXERCISES) - known
    if unknown:
        raise ValueError(f"Unknown demos in EXERCISES: {sorted(unknown)}")
    for demo, exercise in EXERCISES.items():
        if exercise["target"] not in TARGETS:
            raise ValueError(f"Unknown exercise target for {demo}")
        if exercise["mode"] not in MODES:
            raise ValueError(f"Unknown exercise layout for {demo}")
    return {demo: dict(exercise) for demo, exercise in EXERCISES.items()}


def exercise_template():
    """Starting code shown in the exercise editor."""
    return (ROOT / "exercise" / "my_cholesky.jl").read_text()


def section_layout():
    """Numbered page sections, each holding one or more demo consoles."""
    titles = {demo: title for demo, title, _ in DEMOS}
    grouped = {demo for _, _, demos in SECTIONS for demo in demos}
    missing = grouped - set(titles)
    if missing:
        raise ValueError(f"Unknown demos in SECTIONS: {sorted(missing)}")
    if len(grouped) != sum(len(demos) for _, _, demos in SECTIONS):
        raise ValueError("A demo appears in more than one section")
    layout = []
    for section_id, title, demos in SECTIONS:
        layout.append(dict(id=section_id, title=title, demos=list(demos)))
    for demo, title, _ in DEMOS:
        if demo not in grouped:
            layout.append(dict(id=f"demo-{demo}", title=title, demos=[demo]))
    for number, section in enumerate(layout, 1):
        section["number"] = f"{number:02d}"
    return layout


def source_excerpt(path):
    blocks, lines, active = [], [], False
    for line in path.read_text().splitlines():
        if line.strip() == "# source-begin":
            active, lines = True, []
        elif line.strip() == "# source-end":
            blocks.append(textwrap.dedent("\n".join(lines)))
            active = False
        elif active:
            lines.append(line)
    return "\n\n".join(blocks)


def demo_sources(demo):
    if demo not in {entry[0] for entry in DEMOS}:
        raise ValueError("Unknown demo")
    catalog = json.loads((ROOT / "host/demo-sources.json").read_text())
    entry = next(e for e in catalog if e["demo"] == demo)
    sources = []
    for cached in entry["sources"]:
        code = source_excerpt(ROOT / cached["path"])
        # Always show current source, even if a developer has not regenerated
        # the highlighted artifact after an edit. Never pair old ANSI with it.
        sources.append(dict(file=cached["file"], code=code,
                            ansi=cached["ansi"] if code == cached["code"] else None))
    return dict(demo=demo, sources=sources)


def snapshot():
    """Upload course code only; results, credentials and preferences stay put."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ("Project.toml", "demos", "harness", "frontier", "solution"):
            archive.add(ROOT / name, arcname=name,
                        filter=lambda item: None if "__pycache__" in item.name else item)
        for name in ("worker.jl", "remote_agent.py", "final_visuals.jl"):
            archive.add(ROOT / "host" / name, arcname="host/" + name)
    return stream.getvalue()


ODO_HOSTNAME = "login1.odo.olcf.ornl.gov"
# Older consoles kept one managed block for whichever user ran last; it is
# replaced by one block per alias, so consoles for several users can run at once.
MANAGED_BEGIN = "# >>> sc26-dagger console: managed host block >>>"
MANAGED_END = "# <<< sc26-dagger console: managed host block <<<"


def managed_markers(alias):
    return (f"# >>> sc26-dagger console: {alias} >>>", f"# <<< sc26-dagger console: {alias} <<<")


TERMINALS = {}
TERMINALS_LOCK = threading.Lock()


def _terminal_size(cols, rows):
    try:
        cols, rows = int(cols), int(rows)
    except (TypeError, ValueError):
        raise ValueError("Invalid terminal size")
    return max(20, min(cols, 300)), max(5, min(rows, 120))


def _agent_path():
    """PATH plus the usual per-user install locations."""
    home = Path.home()
    parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    for extra in (home / ".opencode/bin", home / ".local/bin",
                  home / ".local/share/npm/bin", home / ".npm-global/bin",
                  home / ".npm-packages/bin", home / ".bun/bin", home / ".volta/bin",
                  home / ".yarn/bin", home / ".local/share/pnpm",
                  home / ".codex/packages/standalone/current/bin",
                  Path("/opt/homebrew/bin"), Path("/home/linuxbrew/.linuxbrew/bin"),
                  Path("/usr/local/bin")):
        if str(extra) not in parts:
            parts.append(str(extra))
    return os.pathsep.join(parts)


def _agent_executable(agent):
    return shutil.which(agent, path=_agent_path())


def agent_status(agent):
    if agent not in ("opencode", "claude", "codex"):
        raise ValueError("Unknown agent")
    executable = _agent_executable(agent)
    return dict(installed=executable is not None, path=executable)


def install_agent(agent):
    if agent not in ("opencode", "claude", "codex"):
        raise ValueError("Unknown agent")
    env = dict(os.environ, PATH=_agent_path())
    if agent in ("opencode", "claude"):
        curl = shutil.which("curl", path=env["PATH"])
        bash = shutil.which("bash", path=env["PATH"])
        if curl is None or bash is None:
            raise ValueError(f"{agent} needs curl and bash to install")
        url = ("https://opencode.ai/install" if agent == "opencode"
               else "https://claude.ai/install.sh")
        command = [bash, "-lc", f"{shlex.quote(curl)} -fsSL {url} | {shlex.quote(bash)}"]
    else:
        npm = shutil.which("npm", path=env["PATH"])
        brew = shutil.which("brew", path=env["PATH"])
        if npm is not None:
            command = [npm, "install", "-g", "@openai/codex"]
        elif brew is not None:
            command = [brew, "install", "codex"]
        else:
            raise ValueError("Codex needs npm or Homebrew; install it manually")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Installation timed out") from exc
    output = re.sub(r"\x1b\[[0-9;]*m", "",
                    (result.stdout or "") + (result.stderr or ""))
    if result.returncode != 0:
        raise ValueError(output[-2000:] or f"Installer exited with {result.returncode}")
    return dict(ok=True, output=output[-2000:])


def open_agent_terminal(agent, cols=100, rows=30, cwd=None):
    """Start an agent in a local PTY and return its session id."""
    if agent not in ("opencode", "claude", "codex"):
        raise ValueError("Unknown agent")
    if pty is None:
        raise ValueError("Embedded terminal requires a Unix-like system")
    executable = _agent_executable(agent)
    if executable is None:
        raise ValueError(f"{agent} is not installed or not on PATH")
    cwd = os.path.abspath(os.path.expanduser(str(cwd or os.getcwd())))
    if not os.path.isdir(cwd):
        raise ValueError(f"Working directory does not exist: {cwd}")
    cols, rows = _terminal_size(cols, rows)
    env = dict(os.environ, TERM="xterm-256color", COLORTERM="truecolor")
    pid, master = pty.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            os.execvpe(executable, [executable], env)
        except OSError as exc:
            os.write(2, f"{agent}: {exc}\n".encode())
        finally:
            os._exit(127)
    try:
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        os.set_blocking(master, False)
    except OSError:
        os.close(master)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        raise
    session = secrets.token_hex(16)
    with TERMINALS_LOCK:
        TERMINALS[session] = dict(
            pid=pid, fd=master, buffer="", base=0, total=0, alive=True,
            flushed=False, agent=agent, cols=cols, rows=rows,
            decoder=codecs.getincrementaldecoder("utf-8")("replace"))
    return session


def terminal_output(session, after=0):
    """Return output after a character offset and whether the agent is alive."""
    with TERMINALS_LOCK:
        term = TERMINALS.get(session)
        if term is None:
            raise ValueError("Unknown terminal session")
        read_total = 0
        while term["alive"] and read_total < (1 << 18):
            try:
                chunk = os.read(term["fd"], 65536)
            except BlockingIOError:
                break
            except OSError:
                term["alive"] = False
                break
            if not chunk:
                term["alive"] = False
                break
            read_total += len(chunk)
            text = term["decoder"].decode(chunk)
            term["buffer"] += text
            term["total"] += len(text)
        if not term["alive"] and not term["flushed"]:
            term["flushed"] = True
            tail = term["decoder"].decode(b"", final=True)
            if tail:
                term["buffer"] += tail
                term["total"] += len(tail)
        if len(term["buffer"]) > (1 << 20):
            drop = len(term["buffer"]) - (1 << 20)
            term["buffer"] = term["buffer"][drop:]
            term["base"] += drop
        start = max(int(after), term["base"])
        return dict(data=term["buffer"][start - term["base"]:],
                    next=term["total"], alive=term["alive"])


def terminal_input(session, data):
    if not isinstance(data, str) or len(data) > 65536:
        raise ValueError("Invalid terminal input")
    with TERMINALS_LOCK:
        term = TERMINALS.get(session)
        if term is None:
            raise ValueError("Unknown terminal session")
        try:
            os.write(term["fd"], data.encode("utf-8"))
        except OSError as exc:
            raise ValueError(f"Terminal is closed: {exc}") from exc


def terminal_resize(session, cols, rows):
    with TERMINALS_LOCK:
        term = TERMINALS.get(session)
        if term is None:
            raise ValueError("Unknown terminal session")
        cols, rows = _terminal_size(cols, rows)
        try:
            fcntl.ioctl(term["fd"], termios.TIOCSWINSZ,
                        struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            return
        term["cols"], term["rows"] = cols, rows


def close_terminal_session(session):
    with TERMINALS_LOCK:
        term = TERMINALS.pop(session, None)
    if term is None:
        return
    pid, fd = term["pid"], term["fd"]
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            break
        if done:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    try:
        os.close(fd)
    except OSError:
        pass


def close_all_terminals():
    with TERMINALS_LOCK:
        sessions = list(TERMINALS)
    for session in sessions:
        close_terminal_session(session)


atexit.register(close_all_terminals)


def is_odo_hostname(hostname):
    name = (hostname or "").lower().rstrip(".")
    return name == ODO_HOSTNAME or name.endswith(".odo.olcf.ornl.gov")


def default_remote_root(account, user):
    """Where the console's sessions live; each Connect makes its own folder here."""
    return f"/gpfs/wolf2/olcf/{account}/scratch/{user}/sc26-dagger/sessions"


def new_session_name():
    """A unique name for one connection: its folder on Odo and its Slurm job name.

    Many consoles can share one set of SSH credentials, so the name combines
    the time with random bits, and the folder is created with a plain mkdir,
    which fails rather than reuse a folder that already exists."""
    return "sc26-" + time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def session_upload_script(base, name, account="trn047"):
    """Create this session's own folder and unpack the course code into it."""
    hint = (f"The console cannot create its sessions folder {base} on Odo. Your project "
            f"scratch directory must exist and be yours: check `groups` for {account} and "
            f"`ls -ld {posixpath.dirname(posixpath.dirname(base.rstrip('/')))}`. Use --account "
            "for another project, or --remote-root for another directory you can write.")
    base, name = shlex.quote(base), shlex.quote(name)
    return (f"set -eu; umask 077; mkdir -p {base} || {{ echo {shlex.quote(hint)} >&2; exit 1; }}; "
            f"cd {base}; mkdir {name}; cd {name}; touch .sc26-session; tar -xzf -")


def cache_root(remote_root):
    """The package depot and the shared manifest live next to the sessions."""
    return posixpath.dirname(remote_root.rstrip("/"))


def managed_alias(user):
    """A per-user alias that will not collide with a hand-written `Host odo`."""
    return "sc26-odo-" + re.sub(r"[^A-Za-z0-9_.-]", "-", user)


def ssh_config_option(config_path):
    """Pass -F only when the file exists; OpenSSH treats an explicit missing one as fatal."""
    path = Path(config_path).expanduser()
    return ["-F", str(path)] if path.is_file() else []


def ssh_config_values(alias, config_path=None):
    """Ask OpenSSH how it would resolve an alias: User, HostName, port, ..."""
    command = ["ssh", "-G"]
    if config_path is not None:
        path = Path(config_path).expanduser()
        command += ["-F", str(path if path.is_file() else os.devnull)]
    command.append(alias)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    values = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key and value:
            values[key] = value.strip()
    return values


def extract_managed_block(text, begin=MANAGED_BEGIN, end=MANAGED_END):
    """Return (block, remaining lines); refuse to guess at a damaged block."""
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == begin]
    ends = [index for index, line in enumerate(lines) if line.strip() == end]
    if not starts and not ends:
        return None, lines
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError("The SSH config has a damaged sc26-dagger managed block; "
                         f"remove or fix the lines between {begin!r} and {end!r}")
    return lines[starts[0]:ends[0] + 1], lines[:starts[0]] + lines[ends[0] + 1:]


def without_managed_blocks(lines):
    """The hand-written part of a config: every console-managed block removed."""
    kept, inside = [], False
    for line in lines:
        mark = line.strip()
        if mark.startswith("# >>> sc26-dagger console:"):
            inside = True
        elif mark.startswith("# <<< sc26-dagger console:"):
            inside = False
        elif not inside:
            kept.append(line)
    return kept


def ssh_aliases(lines):
    for line in lines:
        match = re.match(r"\s*host\s+(.+)$", line, re.IGNORECASE)
        if match:
            yield from match.group(1).split()


def write_managed_host(config_path, alias, user, server=ODO_HOSTNAME, alternative=None):
    """Insert or update this alias's own block; never touch hand-written Host
    sections or other users' blocks.

    Each alias has its own marked block, so consoles for several users can run
    at once. The block names no ControlPath: every console opens its own SSH
    connection and passes its socket explicitly. Blocks go first because
    OpenSSH keeps the first value it obtains for User, HostName and Port.
    """
    config_path = Path(config_path).expanduser()
    # Follow a dotfile-manager symlink instead of replacing it with a plain file.
    if config_path.is_symlink():
        config_path = config_path.resolve()
    text = config_path.read_text() if config_path.exists() else ""
    begin, end = managed_markers(alias)
    # The single block of older consoles is ours too; this replaces it.
    _, lines = extract_managed_block(text)
    own, remainder = extract_managed_block("\n".join(lines), begin, end)
    # Keep an existing block where it is, so starting one user's console does
    # not reorder (and rewrite) the others'; a new block goes first.
    position = lines.index(begin) if own else 0
    for pattern in ssh_aliases(without_managed_blocks(remainder)):
        if pattern.lower() == alias.lower():
            hint = "Remove or rename that entry"
            if alternative and alternative.lower() != alias.lower():
                hint += f", or let the console manage {alternative!r} instead"
            raise ValueError(f"SSH alias {alias!r} is already defined outside the console's "
                             f"managed block. {hint}.")
    block = [begin,
             f"Host {alias}",
             f"    HostName {server}",
             f"    User {user}",
             "    ServerAliveInterval 60",
             end]
    if own:
        updated = "\n".join(remainder[:position] + block + remainder[position:]) + "\n"
    else:
        rest = "\n".join(remainder).strip("\n")
        updated = "\n".join(block) + "\n" + ("\n" + rest + "\n" if rest else "")
    if updated == text:
        return "unchanged"
    config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existed = config_path.exists()
    # Write through a sibling temporary file so a crash cannot truncate the config.
    temporary = config_path.with_name(config_path.name + ".sc26-tmp")
    temporary.write_text(updated)
    temporary.chmod(config_path.stat().st_mode & 0o777 if existed else 0o600)
    os.replace(temporary, config_path)
    return "updated" if existed else "created"


def verify_managed_host(alias, user, config_path=None):
    values = ssh_config_values(alias, config_path)
    hostname = values.get("hostname", "").lower().rstrip(".")
    resolved = values.get("user", "")
    if hostname != ODO_HOSTNAME or resolved != user:
        raise RuntimeError(
            f"SSH alias {alias!r} resolves to {resolved or '?'}@{hostname or '?'}, not "
            f"{user}@{ODO_HOSTNAME}. An earlier User/HostName entry or an Include is overriding it")


def prompt_odo_user(default):
    if not sys.stdin.isatty():
        raise ValueError("No Odo username: pass --odo-user, set SC26_ODO_USER, "
                         "or run the console from an interactive terminal")
    while True:
        try:
            answer = input(f"OLCF/UCAMS username for Odo [{default}]: ").strip()
        except EOFError:
            raise ValueError("No Odo username provided")
        user = answer or default
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", user):
            return user
        print("Use letters, digits, dots, dashes or underscores.", flush=True)


def bootstrap_ssh(args):
    """Resolve the Odo identity, create a managed alias when needed, return (user, alias).

    A usable hand-written alias is never modified. Otherwise the console owns one
    explicitly marked block, so an attendee's `Host odo` keeps working unchanged.
    """
    local = getpass.getuser()
    config_path = Path(args.ssh_config).expanduser()
    explicit = (args.odo_user or os.environ.get("SC26_ODO_USER", "")).strip()
    if explicit and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", explicit):
        raise ValueError(f"Invalid Odo username: {explicit!r}")
    preferred = args.ssh_host or "odo"
    resolved = ssh_config_values(preferred, config_path)
    hostname_ready = is_odo_hostname(resolved.get("hostname", ""))
    resolved_user = resolved.get("user", "")
    if explicit:
        user = explicit
    elif hostname_ready and resolved_user and resolved_user != local:
        # An existing, working alias wins; do not rewrite the file.
        print(f"Using existing SSH alias {preferred!r} as {resolved_user}@{ODO_HOSTNAME}", flush=True)
        return resolved_user, preferred
    else:
        user = prompt_odo_user(resolved_user or local)
    if hostname_ready and resolved_user == user:
        print(f"Using existing SSH alias {preferred!r} as {user}@{ODO_HOSTNAME}", flush=True)
        return user, preferred
    alias = args.ssh_host or managed_alias(user)
    action = write_managed_host(config_path, alias, user,
                                alternative=None if args.ssh_host else managed_alias(user))
    verify_managed_host(alias, user, config_path)
    print(f"SSH alias {alias!r} {action} in {config_path}", flush=True)
    return user, alias


def console_socket(user):
    """A control socket for this console alone: consoles never share an SSH
    connection, so several can run at once, for one user or for several."""
    return Path.home() / ".ssh" / "control" / f"sc26-{user}-{secrets.token_hex(3)}"


def close_master(args):
    """Close the SSH connection this console opened."""
    subprocess.run(["ssh"] + ssh_config_option(args.ssh_config)
                   + ["-S", str(args.control_path), "-O", "exit", args.ssh_host],
                   capture_output=True, timeout=20)


def ensure_master(args):
    """Open the multiplexed connection once, interactively if a password is needed."""
    socket_path = args.control_path
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    option = ssh_config_option(args.ssh_config)
    check = subprocess.run(["ssh"] + option + ["-S", str(socket_path), "-O", "check", args.ssh_host],
                           capture_output=True)
    if check.returncode == 0:
        print(f"Reusing the open SSH connection to {args.ssh_host}", flush=True)
        return True
    command = ["ssh"] + option + ["-S", str(socket_path),
               "-o", "ControlMaster=auto", "-o", "ControlPersist=8h",
               "-o", "ConnectTimeout=20", "-o", "RemoteCommand=none"]
    if sys.stdin.isatty():
        command += ["-o", "BatchMode=no"]
    else:
        command += ["-o", "BatchMode=yes"]
    command += ["-MNf", args.ssh_host]
    print(f"Opening SSH connection to {args.ssh_host} (enter your OLCF password if asked)…",
          flush=True)
    if subprocess.run(command).returncode == 0:
        return True
    print("Could not open the SSH connection. Authenticate once, then reconnect the page:\n"
          + shlex.join(command), flush=True)
    return False


class Console:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.events = []
        self.seq = 0
        self.process = None
        self.connecting = False
        self.busy = False
        self.ready = False
        self.job = None
        self.mode = None
        self.status = "Disconnected"
        self.run_id = None
        self.session = None       # this connection's name: its folder and its job name
        self.session_dir = None
        self.token = secrets.token_urlsafe(32)
        self.stop_event = threading.Event()

    def emit(self, kind, **fields):
        with self.lock:
            self.seq += 1
            self.events.append(dict(seq=self.seq, kind=kind, **fields))
            if len(self.events) > 10000:
                del self.events[:1000]

    def state(self, after=0):
        with self.lock:
            return dict(status=self.status, job=self.job, mode=self.mode, ready=self.ready,
                        session=self.session, remote_dir=self.session_dir,
                        connecting=self.connecting, busy=self.busy,
                        connected=self.process is not None and self.process.poll() is None,
                        events=[event for event in self.events if event["seq"] > after], seq=self.seq)

    def ssh(self):
        return ["ssh"] + ssh_config_option(getattr(self.args, "ssh_config", Path.home()/".ssh/config")) + [
                "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "RemoteCommand=none",
                "-S", str(self.args.control_path), self.args.ssh_host]

    def connect(self):
        with self.lock:
            if self.connecting or self.process is not None:
                raise ValueError("Already connected or connecting")
            self.connecting = True
            self.stop_event.clear()
            self.status = "Connecting to Odo…"
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            socket = self.args.control_path
            socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            check = subprocess.run(["ssh"] + ssh_config_option(getattr(self.args, "ssh_config", Path.home()/".ssh/config"))
                                   + ["-S", str(socket), "-O", "check", self.args.ssh_host],
                                   capture_output=True)
            if check.returncode:
                helper = shutil.which("odo-connect") if self.args.ssh_host == "odo" else None
                cmd = ([helper, str(socket)] if helper else
                       ["ssh"] + ssh_config_option(getattr(self.args, "ssh_config", Path.home()/".ssh/config"))
                       + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto",
                          "-o", "ControlPersist=8h", "-S", str(socket), "-MNf", self.args.ssh_host])
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    raise RuntimeError("SSH authentication failed. In a local terminal run: " +
                                       shlex.join(["ssh", "-S", str(socket), "-MNf", self.args.ssh_host]) +
                                       "\n" + result.stderr)
            if self.stop_event.is_set():
                return
            # Every connection gets its own new folder: the code, its manifest,
            # its logs and its results live there, and nowhere else. Its name
            # is also the name of the job this session submits.
            self.status = "Uploading tutorial code…"
            payload = snapshot()
            for _ in range(3):
                name = new_session_name()
                result = subprocess.run(self.ssh() + [session_upload_script(self.args.remote_root, name,
                                                                            self.args.account)],
                                        input=payload, capture_output=True, timeout=120)
                if result.returncode == 0 or b"File exists" not in result.stderr:
                    break
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            with self.lock:
                self.session = name
                self.session_dir = self.args.remote_root.rstrip("/") + "/" + name
            self.emit("session", session=name, remote_dir=self.session_dir)
            root = shlex.quote(self.session_dir)
            if self.stop_event.is_set():
                return
            self.status = "Preparing packages on the login node…"
            # Every later run is its own Slurm job; nothing is held between runs.
            env = ["env", f"SC26_SESSION={name}", f"SC26_CACHE_ROOT={cache_root(self.args.remote_root)}",
                   "SC26_NODES=1", f"SC26_STEP_TIME={self.args.time}", f"SC26_ACCOUNT={self.args.account}"]
            if self.args.reservation:
                env.append(f"SC26_RESERVATION={self.args.reservation}")
            script = "source frontier/env.sh; bash frontier/prepare.sh </dev/null; exec python3 -u host/remote_agent.py"
            command = f"cd {root} && " + shlex.join(env + ["bash", "-ec", script])
            with self.lock:
                if self.stop_event.is_set():
                    return
                self.process = subprocess.Popen(self.ssh() + [command], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self._read(self.process)
        except Exception as exc:
            self.emit("error", error=str(exc))
            self.status = "Connection failed"
        finally:
            with self.lock:
                self.process = None
                self.job = None
                self.mode = None
                self.connecting = False
                self.ready = False
                self.busy = False
                if self.status != "Connection failed":
                    self.status = "Disconnected"

    def _read(self, proc):
        for line in proc.stdout:
            if line.startswith(PREFIX):
                try:
                    event = json.loads(line[len(PREFIX):])
                except ValueError:
                    self.emit("output", text=line, run=self.run_id)
                    continue
                if event.get("kind") == "heartbeat":
                    continue   # keeps the connection busy; nothing to show
                with self.lock:
                    kind = event["kind"]
                    # The coordinator relays the worker's lines as output events
                    # without a run; they belong to the run in progress, so the
                    # console shows them in the panel that started it.
                    if kind == "output" and not event.get("run"):
                        event["run"] = self.run_id
                    if kind == "ready":
                        self.ready = True
                        self.connecting = False
                        self.status = event.get("message") or "Ready · every run is its own Slurm job"
                    elif kind == "status":
                        self.status = event["message"]
                    elif kind == "job":
                        self.job = event.get("job")
                    elif kind == "worker_ready":
                        self.mode = event["mode"]
                        self.job = event.get("job") or self.job
                        self.status = f"Running · job {self.job}…" if self.job else "Running…"
                    elif kind in ("done", "cancelled"):
                        self.busy = False
                        self.job = None
                        self.status = ("Ready · every run is its own Slurm job" if event.get("ok", True)
                                       else "Run failed · see output")
                        if kind == "cancelled":
                            self.mode = None
                    elif kind == "worker_exit":
                        self.mode = None
                    self.emit(**event)
                    if kind in ("done", "cancelled"):
                        # Later coordinator chatter belongs to the next run.
                        self.run_id = None
            else:
                self.emit("output", text=line, run=self.run_id)
        rc = proc.wait()
        if rc:
            self.emit("error", error=f"Remote session exited with status {rc}")

    def send(self, command):
        with self.lock:
            if self.process is None or self.process.poll() is not None or not self.ready:
                raise ValueError("Connect to Odo first and wait until it is ready")
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()

    def run(self, command):
        with self.lock:
            if self.busy:
                raise ValueError("A run is in progress; wait or cancel it")
            if command.get("op") == "demo":
                demo = next((d for d in DEMOS if d[0] == command.get("demo")), None)
                if demo is None:
                    raise ValueError("Unknown demo")
                command["mode"] = demo[2]
                # The only demo option exposed by the console is GPU count.
                gpus = str(command.pop("gpus", "2"))
                if gpus not in ("2", "4", "8"):
                    raise ValueError("Choose 2, 4 or 8 GPUs")
                command["args"] = [f"--gpus={gpus}"] if demo[0] == "05_multigpu" else []
                # The Multiple GPUs demo's job asks for exactly the GPUs chosen;
                # an Odo node has eight (four MI250X, two GCDs each).
                command["gpus"] = int(gpus) if demo[0] == "05_multigpu" else demo_gpus(demo[0], demo[2])
                command["ranks"] = mpi_ranks(command.get("ranks", DEFAULT_RANKS)) if demo[2] == "mpi" else 1
                command["label"] = (f"{demo[1]} · {layout_label('mpi', command['ranks'])}"
                                    if demo[2] == "mpi" else demo[1])
            elif command.get("op") == "exercise":
                demo = next((d for d in DEMOS if d[0] == command.get("demo")), None)
                if demo is None:
                    raise ValueError("Unknown demo")
                exercise = EXERCISES.get(demo[0])
                if exercise is None:
                    raise ValueError("This demo has no exercise")
                target = str(command.get("target") or exercise["target"])
                if target not in TARGETS:
                    raise ValueError("Unknown exercise target")
                layout = str(command.get("mode") or exercise["mode"])
                if layout not in MODES:
                    raise ValueError("Unknown exercise layout")
                for field, what in (("code", "the function"), ("setup", "Setup and Use")):
                    if not isinstance(command.get(field), str) or not command[field].strip():
                        raise ValueError(f"Write Julia code for {what} first")
                command["mode"] = layout
                command["target"] = target
                command["gpus"] = run_gpus(target, layout)
                command["ranks"] = mpi_ranks(command.get("ranks", DEFAULT_RANKS)) if layout == "mpi" else 1
                command["label"] = f"Exercise · {TARGETS[target]} · {layout_label(layout, command['ranks'])}"
            elif command.get("op") == "eval":
                if command.get("mode") not in ("single", "mpi"):
                    raise ValueError("Unknown REPL layout")
                command["ranks"] = (mpi_ranks(command.get("ranks", DEFAULT_RANKS))
                                    if command["mode"] == "mpi" else 1)
                if not isinstance(command.get("code"), str) or not command["code"].strip():
                    raise ValueError("Enter Julia code")
                if command.get("demo") is not None and command["demo"] not in {d[0] for d in DEMOS} \
                        and command["demo"] != "final":
                    raise ValueError("Unknown demo")
                if command.get("demo") == "final":
                    setup = command.get("setup")
                    if not isinstance(setup, str) or not setup.strip():
                        raise ValueError("Write Julia code for Setup and Use first")
                    target = str(command.get("target") or "cpu")
                    if target not in TARGETS:
                        raise ValueError("Unknown exercise target")
                    command["target"] = target
                    command["gpus"] = run_gpus(target, command["mode"])
                    command["label"] = (f"Final Exercise · {TARGETS[target]} · "
                                        f"{layout_label(command['mode'], command['ranks'])}")
                    visuals = command.get("visuals", [])
                    if not isinstance(visuals, list) or any(v not in FINAL_VISUAL_IDS for v in visuals):
                        raise ValueError("Unknown visualization")
                    command["visuals"] = visuals
                else:
                    command["gpus"] = demo_gpus(command.get("demo"), command["mode"])
            else:
                raise ValueError("Unknown operation")
            command["id"] = secrets.token_hex(8)
            self.send(command)
            self.busy = True
            self.status = "Submitting to Slurm…"
            self.run_id = command["id"]
            self.emit("run", run=command["id"], demo=command.get("demo"),
                      label=command.get("label") or command.get("demo", "REPL"),
                      mode=command["mode"], ranks=command["ranks"], gpus=command["gpus"],
                      args=command.get("args", []),
                      exercise=command.get("op") == "exercise")

    def disconnect(self):
        self.stop_event.set()
        with self.lock:
            proc = self.process
            if proc is None:
                return
            try:
                proc.stdin.write('{"op":"quit"}\n')
                proc.stdin.flush()
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        # The coordinator's EOF/finally cancels the Slurm job of a run in
        # progress, by this session's unique name. Nothing else is allocated.


_JS, _CSS = "text/javascript; charset=utf-8", "text/css; charset=utf-8"
STATIC_FILES = {"/": "text/html; charset=utf-8", "/app.js": _JS, "/ansi.js": _JS,
                "/highlight.js": _JS, "/timeline.js": _JS, "/visuals.js": _JS,
                "/style.css": _CSS, "/xterm.js": _JS, "/xterm.css": _CSS, "/addon-fit.js": _JS}


def handler_for(console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, content_type="application/json", code=200):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.allowed_host():
                self.reply(dict(error="Use this console's IP address or an allowed hostname"), code=403)
                return
            url = urlsplit(self.path)
            if url.path == "/api/state":
                try:
                    self.reply(console.state(int(parse_qs(url.query).get("after", [0])[0])))
                except ValueError:
                    self.reply(dict(error="Invalid cursor"), code=400)
            elif url.path == "/api/config":
                self.reply(dict(token=console.token, demos=DEMOS, sections=section_layout(),
                                exercises=exercises(), exercise_template=exercise_template(),
                                exercise_setups=setup_templates(exercise_setup),
                                targets=TARGETS, modes=MODES, max_ranks=MAX_RANKS,
                                default_ranks=DEFAULT_RANKS,
                                final_exercise=dict(FINAL_EXERCISE, visuals=FINAL_VISUALS,
                                                    target="cpu", mode="single",
                                                    setups=setup_templates(final_setup)),
                                final_workspace=str(FINAL_WORKSPACE),
                                host=console.args.ssh_host,
                                remote_root=console.args.remote_root, time=console.args.time))
            elif url.path == "/api/source":
                try:
                    self.reply(demo_sources(parse_qs(url.query).get("demo", [""])[0]))
                except ValueError as exc:
                    self.reply(dict(error=str(exc)), code=400)
            elif url.path == "/api/final/file":
                if self.headers.get("X-Console-Token") != console.token:
                    self.reply(dict(error="Invalid console token"), code=403)
                    return
                try:
                    self.reply(final_file_state())
                except (OSError, ValueError) as exc:
                    self.reply(dict(error=str(exc)), code=400)
            elif url.path == "/api/terminal/output":
                if self.headers.get("X-Console-Token") != console.token:
                    self.reply(dict(error="Invalid console token"), code=403)
                    return
                query = parse_qs(url.query)
                try:
                    self.reply(terminal_output(query.get("session", [""])[0],
                                               int(query.get("after", ["0"])[0])))
                except (ValueError, TypeError) as exc:
                    self.reply(dict(error=str(exc)), code=400)
            elif url.path in STATIC_FILES:
                file = "index.html" if url.path == "/" else url.path[1:]
                self.reply((ROOT / "host" / file).read_bytes(), STATIC_FILES[url.path])
            else:
                self.reply(dict(error="Not found"), code=404)

        def allowed_host(self):
            try:
                address = urlsplit("//" + self.headers.get("Host", ""))
                if address.port != self.server.server_port:
                    return False
                name = (address.hostname or "").lower().rstrip(".")
                if name in getattr(self.server, "allowed_hosts", {"127.0.0.1", "localhost"}):
                    return True
                # Wildcard binding accepts literal LAN/Tailscale interface IPs.
                # Unrecognized DNS names still fail the rebinding check.
                if self.server.server_address[0] == "0.0.0.0":
                    ipaddress.ip_address(name)
                    return True
            except ValueError:
                pass
            return False

        def do_POST(self):
            # A recognized host and per-launch token prevent other websites
            # from submitting Julia/SSH commands to this console.
            if not self.allowed_host() or self.headers.get("X-Console-Token") != console.token:
                self.reply(dict(error="Invalid console token"), code=403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 262144:
                    raise ValueError("Invalid request size")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/connect":
                    console.connect()
                elif self.path == "/api/run":
                    console.run(command)
                elif self.path == "/api/cancel":
                    console.send(dict(op="cancel"))
                elif self.path == "/api/disconnect":
                    threading.Thread(target=console.disconnect, daemon=True).start()
                elif self.path == "/api/final/file":
                    self.reply(write_final_file(command.get("code")))
                    return
                elif self.path == "/api/terminal/check":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    self.reply(agent_status(command.get("agent")))
                    return
                elif self.path == "/api/terminal/install":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    self.reply(install_agent(command.get("agent")))
                    return
                elif self.path == "/api/terminal/open":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    session = open_agent_terminal(command.get("agent"),
                                                  command.get("cols", 100),
                                                  command.get("rows", 30),
                                                  command.get("cwd"))
                    self.reply(dict(session=session))
                    return
                elif self.path == "/api/terminal/input":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    terminal_input(command.get("session"), command.get("data"))
                elif self.path == "/api/terminal/resize":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    terminal_resize(command.get("session"), command.get("cols", 100),
                                    command.get("rows", 30))
                elif self.path == "/api/terminal/close":
                    if self.client_address[0] not in ("127.0.0.1", "::1"):
                        raise ValueError("Terminal is only available from localhost")
                    close_terminal_session(command.get("session"))
                else:
                    self.reply(dict(error="Not found"), code=404)
                    return
                self.reply(dict(ok=True))
            except (ValueError, OSError) as exc:
                self.reply(dict(error=str(exc)), code=400)
    return Handler


def bind_server(host, port, handler):
    """Serve on --port, or on the first free port from 8767 when none is given,
    so a second console starts next to the first."""
    for candidate in [port] if port else range(8767, 8800):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if port or exc.errno != errno.EADDRINUSE:
                raise
    raise OSError("No free port between 8767 and 8799; pass --port")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: all IPv4 interfaces)")
    parser.add_argument("--allow-host", action="append", default=[], help="Additional allowed browser hostname")
    parser.add_argument("--port", type=int, help="Local port (default: 8767, or the next free one)")
    parser.add_argument("--ssh-host", help="SSH alias; reuse `odo` when it already points at Odo, "
                                           "otherwise the console manages `sc26-odo-<user>`")
    parser.add_argument("--odo-user", help="OLCF/UCAMS username (skips the prompt; also SC26_ODO_USER)")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh/config",
                        help="SSH config to read and update (default: ~/.ssh/config)")
    parser.add_argument("--account", default=os.environ.get("SC26_ACCOUNT", "trn047"))
    parser.add_argument("--remote-root", help="Directory for the console's sessions on Odo; every "
                                              "Connect creates its own new folder inside it")
    parser.add_argument("--control-path", type=Path,
                        help="SSH control socket to use; by default each console opens its own")
    parser.add_argument("--time", default="00:20:00", help="Slurm time limit of each run (default 00:20:00)")
    parser.add_argument("--reservation", default="")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip the startup SSH login; Connect then needs an existing connection")
    parser.add_argument("--no-auto-connect", "--no-auto-allocate", dest="no_auto_connect",
                        action="store_true",
                        help="Wait for Connect instead of preparing Julia at startup")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.ssh_host and (not re.fullmatch(r"[\w@.-]+", args.ssh_host) or args.ssh_host.startswith("-")):
        parser.error("Invalid SSH host alias")
    args.ssh_config = args.ssh_config.expanduser()
    try:
        args.odo_user, args.ssh_host = bootstrap_ssh(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    own_socket = args.control_path is None
    args.control_path = (console_socket(args.odo_user) if own_socket
                         else args.control_path.expanduser().resolve())
    use_final_workspace(args.odo_user)
    ensure_final_workspace()
    args.remote_root = args.remote_root or default_remote_root(args.account, args.odo_user)
    if not args.remote_root.startswith("/"):
        parser.error("--remote-root must be an absolute path")
    authenticated = False if args.no_auth else ensure_master(args)
    console = Console(args)
    try:
        server = bind_server(args.host, args.port, handler_for(console))
    except OSError as exc:
        if own_socket:
            close_master(args)
        parser.error(f"Cannot serve the page: {exc}")
    port = server.server_port
    server.allowed_hosts = {"127.0.0.1", "localhost", socket.gethostname().lower(), socket.getfqdn().lower(),
                            *(name.lower().rstrip(".") for name in args.allow_host)}
    tailscale_ips = []
    if shutil.which("tailscale"):
        try:
            status = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=3)
            peer = json.loads(status.stdout).get("Self") or {}
            if peer.get("DNSName"):
                server.allowed_hosts.add(peer["DNSName"].lower().rstrip("."))
            tailscale_ips = [ip for ip in peer.get("TailscaleIPs", []) if ":" not in ip]
        except (ValueError, OSError, subprocess.TimeoutExpired):
            pass
    atexit.register(console.disconnect)
    if authenticated and not args.no_auto_connect:
        def _auto_connect():
            time.sleep(0.5)
            try:
                console.connect()
            except Exception:
                pass
        threading.Thread(target=_auto_connect, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    if authenticated:
        connection_note = ""
    elif args.no_auth:
        connection_note = "\nSSH login skipped (--no-auth); Connect needs an existing connection"
    else:
        connection_note = "\nSSH login failed; use Connect after fixing SSH"
    print(f"Tutorial console: {url}\nOdo user: {args.odo_user} via {args.ssh_host}"
          f"\nOdo sessions: {args.remote_root} (each Connect makes its own folder){connection_note}"
          + "\nEvery run is its own Slurm job; Ctrl-C cancels a run in progress.", flush=True)
    if authenticated and not args.no_auto_connect:
        print("Preparing Julia and compiling packages in a short job, in the background…", flush=True)
    if args.host == "0.0.0.0":
        for ip in tailscale_ips:
            print(f"Tailscale: http://{ip}:{port}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        console.disconnect()
        server.server_close()
        # This console's SSH connection is its own: close it with the console.
        if own_socket:
            close_master(args)


if __name__ == "__main__":
    main()
