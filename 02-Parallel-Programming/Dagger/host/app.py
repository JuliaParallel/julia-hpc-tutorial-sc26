#!/usr/bin/env python3
"""Local tutorial console. Python 3.10+, OpenSSH; no Python packages required."""
import argparse
import atexit
import codecs
import getpass
import io
import ipaddress
import json
import os
from pathlib import Path
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
    ("01_tasks", "Tasks & dependencies", "single"),
    ("02_datadeps", "Mutable data & tiles", "single"),
    ("03_gpu", "Single GPU", "single"),
    ("05_multigpu", "Multiple GPUs", "single"),
    ("06_multinode", "Multi-node · CPU MPI", "mpi"),
    ("07_multinode_multigpu", "Multi-node + multi-GPU", "mpi"),
    ("04_mpi", "MPI tile processing", "mpi"),
]
# Demos that share one page section. Everything else gets its own section.
SECTIONS = [
    ("cpu", "CPU", ["01_tasks", "02_datadeps"]),
]
# Proposed hands-on exercise for the demos that use it. The console shows the
# prompt and a shared code editor in each relevant demo section.
EXERCISES = {
    "02_datadeps": dict(
        target="cpu", strict=False, gpus="1",
        prompt=textwrap.dedent("""\
            Exercise: parallel tiled Cholesky (CPU)

            The editor starts from the serial, tile-by-tile Cholesky
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

            Run the exercise below. Done when every line says PASS.
            Full handout: exercise/README.md""")),
    "03_gpu": dict(
        target="gpu", strict=False, gpus="1",
        prompt=textwrap.dedent("""\
            Exercise: same code, on the GPU

            No code changes. The exercise/my_cholesky.jl function you wrote
            for the CPU must run unchanged on an AMD GPU. Dagger maps the
            spawned BLAS/LAPACK calls to rocBLAS/rocSOLVER when a task lands on
            the GPU.

            Keep the shared editor content, set Target to GPU, and run.
            If the GPU run fails inside your own helper, spawn the BLAS/LAPACK
            call itself so Dagger can substitute the GPU implementation.

            Full handout: exercise/README.md""")),
    "04_mpi": dict(
        target="cpu", strict=True, gpus="1",
        prompt=textwrap.dedent("""\
            Exercise: same code, across MPI ranks

            No code changes. Run the same exercise/my_cholesky.jl function on
            two MPI ranks on one node. This section uses the MPI worker layout;
            keep Target on CPU and Strict checked.

            --strict places tasks away from their data, so a missing InOut
            loses its writes and fails the check. Every rank must run
            identical Dagger code; do not branch on the rank.

            Full handout: exercise/README.md""")),
    "05_multigpu": dict(
        target="gpu", strict=False, gpus="all",
        prompt=textwrap.dedent("""\
            Exercise: stretch goals

            The same code can use every visible GPU or a mix of CPU and GPU.
            - Keep Target on GPU to use all visible GPUs in this section.
            - Switch Target to Mixed to let Dagger place tasks on both CPU and
              GPU. Is it faster? Why or why not?
            - Use the REPL or the terminal harness to sweep tile sizes and
              compare builtin and vendor implementations.
            - Look at solution/my_cholesky.jl only after you finish.

            Full handout: exercise/README.md""")),
}


EXERCISE_PRIMER = textwrap.dedent("""\
    # Walkthrough: create, distribute, and call the Cholesky exercise.
    # Fill in the missing pieces, then compare with the harness.

    using Dagger, LinearAlgebra
    import LinearAlgebra: BLAS, LAPACK

    n, bs = 1024, 256
    A = rand(n, n); A = A + A'; A[diagind(A)] .+= 2n

    # 1. Choose a scope for the selected target.
    #    CPU:   Dagger.scope(worker=1)
    #    GPU:   Dagger.scope(rocm_gpu=1)
    #    MPI:   Dagger.scope(mpi_ranks=:)
    #    Mixed: Dagger.UnionScope(cpu, gpu)
    scope = ???

    Dagger.with_options(scope=scope) do
        # 2. Tile A into a DArray with bs x bs blocks.
        DA = DArray(A, ???)

        # 3. Call my_cholesky! on the grid of tiles.
        my_cholesky!(???)

        # 4. Collect and check the factor.
        L = collect(???)
        println("collected ", size(L))
    end

    # The Run exercise button performs this setup in the harness.
""")



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

        Fill in L, Fu, Fv, and the two half steps, then run. The
        Visualization panel draws substrate u and activator v side by
        side and animates the sampled time steps; no plotting code
        belongs in your script.
    """),
    template=textwrap.dedent("""\
        # Final exercise: Gray-Scott reaction-diffusion on a periodic grid.
        # Fill in the missing pieces, then run. Select the fields to draw
        # in the Visualization panel; the console plots them after the run.

        using Dagger, LinearAlgebra, Random

        n = 256
        dt = 1.0
        Du, Dv = 0.16, 0.08
        F, k = 0.055, 0.062
        nsteps = 3000

        Random.seed!(42)
        u = ones(n, n)
        v = 0.01 .* rand(n, n)
        for _ in 1:14
            cx, cy = rand(6:n-5, 2)
            v[cx-3:cx+3, cy-3:cy+3] .= 1.0
            u[cx-3:cx+3, cy-3:cy+3] .= 0.5
        end

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

        println("u range: ", extrema(u), " v range: ", extrema(v))
    """),
)

FINAL_WORKSPACE = ROOT / "results" / "final-exercise"
FINAL_FILE = FINAL_WORKSPACE / "final_exercise.jl"
FINAL_README = FINAL_WORKSPACE / "README.md"


def ensure_final_workspace():
    FINAL_WORKSPACE.mkdir(parents=True, exist_ok=True)
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
        if exercise["target"] not in ("cpu", "gpu", "mixed"):
            raise ValueError(f"Unknown exercise target for {demo}")
        if exercise["gpus"] not in ("1", "all"):
            raise ValueError(f"Unknown exercise GPU selection for {demo}")
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
MANAGED_BEGIN = "# >>> sc26-dagger console: managed host block >>>"
MANAGED_END = "# <<< sc26-dagger console: managed host block <<<"


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
    return f"/gpfs/wolf2/olcf/{account}/scratch/{user}/sc26-dagger/portal"


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


def extract_managed_block(text):
    """Return (block, remaining lines); refuse to guess at a damaged block."""
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == MANAGED_BEGIN]
    ends = [index for index, line in enumerate(lines) if line.strip() == MANAGED_END]
    if not starts and not ends:
        return None, lines
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError("The SSH config has a damaged sc26-dagger managed block; "
                         f"remove or fix the lines between {MANAGED_BEGIN!r} and {MANAGED_END!r}")
    return lines[starts[0]:ends[0] + 1], lines[:starts[0]] + lines[ends[0] + 1:]


def ssh_aliases(lines):
    for line in lines:
        match = re.match(r"\s*host\s+(.+)$", line, re.IGNORECASE)
        if match:
            yield from match.group(1).split()


def write_managed_host(config_path, alias, user, control_path, server=ODO_HOSTNAME, alternative=None):
    """Insert or update our own block; never touch hand-written Host sections.

    The block goes first because OpenSSH keeps the first value it obtains for
    User, HostName, Port and ControlPath, regardless of block specificity.
    """
    config_path = Path(config_path).expanduser()
    # Follow a dotfile-manager symlink instead of replacing it with a plain file.
    if config_path.is_symlink():
        config_path = config_path.resolve()
    text = config_path.read_text() if config_path.exists() else ""
    _, remainder = extract_managed_block(text)
    for pattern in ssh_aliases(remainder):
        if pattern.lower() == alias.lower():
            hint = "Remove or rename that entry"
            if alternative and alternative.lower() != alias.lower():
                hint += f", or let the console manage {alternative!r} instead"
            raise ValueError(f"SSH alias {alias!r} is already defined outside the console's "
                             f"managed block. {hint}.")
    socket = str(control_path)
    if any(character.isspace() for character in socket):
        socket = '"' + socket.replace('"', '\\"') + '"'
    block = [MANAGED_BEGIN,
             f"Host {alias}",
             f"    HostName {server}",
             f"    User {user}",
             "    ServerAliveInterval 60",
             "    ControlMaster auto",
             f"    ControlPath {socket}",
             "    ControlPersist 8h",
             MANAGED_END]
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
    action = write_managed_host(config_path, alias, user, args.control_path,
                                alternative=None if args.ssh_host else managed_alias(user))
    verify_managed_host(alias, user, config_path)
    print(f"SSH alias {alias!r} {action} in {config_path}", flush=True)
    return user, alias


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
        self.job = None
        self.mode = None
        self.status = "Disconnected"
        self.run_id = None
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
            return dict(status=self.status, job=self.job, mode=self.mode,
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
            # A dedicated, marked directory protects an attendee's existing checkout.
            root = shlex.quote(self.args.remote_root)
            script = (f"set -eu; mkdir -p {root}; cd {root}; "
                      'if [ ! -f .sc26-portal ]; then '
                      '[ -z "$(ls -A)" ] || { echo "Refusing to upload into a nonempty, unmarked directory" >&2; exit 1; }; '
                      'touch .sc26-portal; fi; tar -xzf -')
            self.status = "Uploading tutorial code…"
            result = subprocess.run(self.ssh() + [script], input=snapshot(), capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            if self.stop_event.is_set():
                return
            self.status = "Preparing packages on the login node, then requesting one node…"
            env = ["env", "SC26_NODES=1", f"SC26_TIME={self.args.time}", f"SC26_ACCOUNT={self.args.account}"]
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
                self.busy = False
                if self.status != "Connection failed":
                    self.status = "Disconnected · allocation released"

    def _read(self, proc):
        for line in proc.stdout:
            if line.startswith(PREFIX):
                try:
                    event = json.loads(line[len(PREFIX):])
                except ValueError:
                    self.emit("output", text=line, run=self.run_id)
                    continue
                with self.lock:
                    kind = event["kind"]
                    if kind == "allocated":
                        self.job = event["job"]
                        self.connecting = False
                        self.status = f"Ready · job {self.job} · {event['nodes']}"
                    elif kind == "status":
                        self.status = event["message"]
                    elif kind == "worker_ready":
                        self.mode = event["mode"]
                        self.status = "Running…"
                    elif kind in ("done", "cancelled"):
                        self.busy = False
                        self.status = f"Ready · job {self.job}" if event.get("ok", True) else "Run failed · see output"
                        if kind == "cancelled":
                            self.mode = None
                    elif kind == "worker_exit":
                        self.mode = None
                        # A nonzero exit can be cancellation or an MPI abort.
                        if event["returncode"] != 0 and not event.get("expected"):
                            self.busy = False
                            self.status = "Worker stopped · next run starts a new process"
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
            if self.process is None or self.process.poll() is not None or not self.job:
                raise ValueError("Allocate a node first")
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
                if gpus not in ("2", "4"):
                    raise ValueError("Choose two or four GPUs")
                command["args"] = [f"--gpus={gpus}"] if demo[0] == "05_multigpu" else []
            elif command.get("op") == "exercise":
                demo = next((d for d in DEMOS if d[0] == command.get("demo")), None)
                if demo is None:
                    raise ValueError("Unknown demo")
                exercise = EXERCISES.get(demo[0])
                if exercise is None:
                    raise ValueError("This demo has no exercise")
                target = str(command.get("target") or exercise["target"])
                if target not in ("cpu", "gpu", "mixed"):
                    raise ValueError("Unknown exercise target")
                if not isinstance(command.get("code"), str) or not command["code"].strip():
                    raise ValueError("Write Julia code for the exercise first")
                command["mode"] = demo[2]
                command["target"] = target
                command["strict"] = bool(command.get("strict", exercise["strict"]))
                command["gpus"] = exercise["gpus"]
                command["sizes"] = {"cpu": "1024", "gpu": "2048", "mixed": "2048"}[target]
                command["bs"] = 256 if target == "cpu" else 512
                command["trials"] = 1
                command["label"] = f"Exercise · {target.upper()}"
            elif command.get("op") == "eval":
                if command.get("mode") not in ("single", "mpi"):
                    raise ValueError("Unknown REPL layout")
                if not isinstance(command.get("code"), str) or not command["code"].strip():
                    raise ValueError("Enter Julia code")
                if command.get("demo") is not None and command["demo"] not in {d[0] for d in DEMOS} \
                        and command["demo"] != "final":
                    raise ValueError("Unknown demo")
                if command.get("demo") == "final":
                    command["label"] = "Final Exercise"
                    visuals = command.get("visuals", [])
                    if not isinstance(visuals, list) or any(v not in FINAL_VISUAL_IDS for v in visuals):
                        raise ValueError("Unknown visualization")
                    command["visuals"] = visuals
            else:
                raise ValueError("Unknown operation")
            command["id"] = secrets.token_hex(8)
            self.send(command)
            self.busy = True
            self.status = "Running…"
            self.run_id = command["id"]
            self.emit("run", run=command["id"], demo=command.get("demo"),
                      label=command.get("label") or command.get("demo", "REPL"),
                      mode=command["mode"], args=command.get("args", []),
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
        # The remote agent's EOF/finally terminates its step and cancels only
        # the holding job it submitted. Other terminal allocations stay intact.


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
                                exercise_primer=EXERCISE_PRIMER,
                                final_exercise=dict(FINAL_EXERCISE, visuals=FINAL_VISUALS),
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
            elif url.path in ("/", "/app.js", "/ansi.js", "/timeline.js", "/visuals.js", "/style.css",
                              "/xterm.js", "/xterm.css", "/addon-fit.js"):
                file = {"/":"index.html", "/app.js":"app.js", "/ansi.js":"ansi.js", "/timeline.js":"timeline.js",
                        "/visuals.js":"visuals.js",
                        "/style.css":"style.css", "/xterm.js":"xterm.js", "/xterm.css":"xterm.css",
                        "/addon-fit.js":"addon-fit.js"}[url.path]
                content_type = {"/":"text/html; charset=utf-8", "/app.js":"text/javascript; charset=utf-8",
                                "/ansi.js":"text/javascript; charset=utf-8", "/timeline.js":"text/javascript; charset=utf-8",
                                "/visuals.js":"text/javascript; charset=utf-8",
                                "/style.css":"text/css; charset=utf-8", "/xterm.js":"text/javascript; charset=utf-8",
                                "/xterm.css":"text/css; charset=utf-8", "/addon-fit.js":"text/javascript; charset=utf-8"}[url.path]
                self.reply((ROOT / "host" / file).read_bytes(), content_type)
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


def main():
    ensure_final_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: all IPv4 interfaces)")
    parser.add_argument("--allow-host", action="append", default=[], help="Additional allowed browser hostname")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--ssh-host", help="SSH alias; reuse `odo` when it already points at Odo, "
                                           "otherwise the console manages `sc26-odo-<user>`")
    parser.add_argument("--odo-user", help="OLCF/UCAMS username (skips the prompt; also SC26_ODO_USER)")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh/config",
                        help="SSH config to read and update (default: ~/.ssh/config)")
    parser.add_argument("--account", default=os.environ.get("SC26_ACCOUNT", "trn047"))
    parser.add_argument("--remote-root", help="Dedicated portal directory, not an existing checkout")
    parser.add_argument("--control-path", type=Path, default=Path.home() / ".ssh/control/sc26-portal")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--reservation", default="")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip the startup SSH login; Connect then needs an existing connection")
    parser.add_argument("--no-auto-allocate", action="store_true",
                        help="Wait for Connect instead of allocating and warming Julia at startup")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.ssh_host and (not re.fullmatch(r"[\w@.-]+", args.ssh_host) or args.ssh_host.startswith("-")):
        parser.error("Invalid SSH host alias")
    args.control_path = args.control_path.expanduser().resolve()
    args.ssh_config = args.ssh_config.expanduser()
    try:
        args.odo_user, args.ssh_host = bootstrap_ssh(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    args.remote_root = args.remote_root or default_remote_root(args.account, args.odo_user)
    if not args.remote_root.startswith("/"):
        parser.error("--remote-root must be an absolute path")
    authenticated = False if args.no_auth else ensure_master(args)
    console = Console(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(console))
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
    if authenticated and not args.no_auto_allocate:
        def _auto_connect():
            time.sleep(0.5)
            try:
                console.connect()
            except Exception:
                pass
        threading.Thread(target=_auto_connect, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    if authenticated:
        connection_note = ""
    elif args.no_auth:
        connection_note = "\nSSH login skipped (--no-auth); Connect needs an existing connection"
    else:
        connection_note = "\nSSH login failed; use Connect & allocate after fixing SSH"
    print(f"Tutorial console: {url}\nOdo user: {args.odo_user} via {args.ssh_host}"
          f"\nOdo workspace: {args.remote_root}{connection_note}"
          + "\nCtrl-C releases the console's allocation.", flush=True)
    if authenticated and not args.no_auto_allocate:
        print("Allocating a node and precompiling Julia in the background…", flush=True)
    if args.host == "0.0.0.0":
        for ip in tailscale_ips:
            print(f"Tailscale: http://{ip}:{args.port}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        console.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()
