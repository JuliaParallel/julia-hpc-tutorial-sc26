#!/usr/bin/env python3
"""Local tutorial console. Python 3.9+, OpenSSH; no Python packages required."""
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
    ("00_julia", "Julia basics", "single"),
    ("01_tasks", "Tasks & dependencies", "single"),
    ("02_datadeps", "Mutable data & tiles", "single"),
    ("03_gpu", "Single GPU", "single"),
    ("05_multigpu", "Multiple GPUs", "single"),
    ("06_multinode", "Multi-node · CPU MPI", "mpi"),
    ("07_multinode_multigpu", "Multi-node + multi-GPU", "mpi"),
    ("08_stencil", "Stencils", "single"),
]
# Demos that share one page section. Everything else gets its own section.
SECTIONS = [
    ("julia", "Julia basics", ["00_julia"]),
    ("cpu", "CPU", ["01_tasks", "02_datadeps"]),
]
# Every run is its own Slurm job on andromeda's one node (two NVIDIA A30s).
# A demo's job gets these GPUs (MPI demos: in total, one per rank); the
# others get none. A REPL evaluation gets its section demo's GPUs.
CLUSTER_GPUS = 2
DEMO_GPUS = {"03_gpu": 1, "05_multigpu": 2, "07_multinode_multigpu": 2, "08_stencil": 1}
# One Exercise panel per section after Julia basics. Each one runs a Function
# editor and then its own Setup and Use code, which chooses the hardware and
# calls the function. The Cholesky sections share one Function (my_cholesky!);
# the Stencils section has its own (my_heat!). `target` and `mode` are the
# section's starting choices; the attendee can change both.
TARGETS = {"cpu": "CPU", "gpu": "GPU", "gpus": "All GPUs", "mixed": "Mixed"}
MODES = {"single": "One process", "mpi": "Two MPI ranks"}
# GPUs a run's Slurm job gets for each Target. Under MPI every GPU target
# gives each of the two ranks its own GPU.
TARGET_GPUS = {"cpu": 0, "gpu": 1, "gpus": CLUSTER_GPUS, "mixed": 1}


def run_gpus(target, mode):
    """GPUs for an exercise or final-exercise run with this Target and Processes."""
    if TARGET_GPUS[target] == 0:
        return 0
    return CLUSTER_GPUS if mode == "mpi" else TARGET_GPUS[target]


def eval_gpus(demo, mode):
    """GPUs for a REPL evaluation: those of the section's demo."""
    gpus = DEMO_GPUS.get(demo, 0)
    # Two ranks share the GPUs evenly, so one GPU becomes one per rank.
    return CLUSTER_GPUS if mode == "mpi" and gpus else gpus


EXERCISES = {
    "02_datadeps": dict(
        kind="cholesky", target="cpu", mode="single",
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
              cuBLAS/cuSOLVER on GPU tasks.
            - Do not use fetch or wait inside the Datadeps region.
            - Under MPI every rank must build the same graph; do not branch
              on rank.

            Run exercise runs the Function, then Setup and Use. Done when it
            prints PASS.
            Full handout: exercise/README.md""")),
    "03_gpu": dict(
        kind="cholesky", target="gpu", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: same code, on the GPU

            No changes to the Function. The my_cholesky! you wrote in
            exercise/my_cholesky.jl for the CPU must run unchanged on an
            NVIDIA GPU. Setup and Use now uses the GPU scope,
            Dagger.scope(cuda_gpu=1); Dagger maps the spawned BLAS/LAPACK
            calls to cuBLAS/cuSOLVER when a task lands on the GPU.

            If the GPU run fails inside your own helper, spawn the BLAS/LAPACK
            call itself so Dagger can substitute the GPU implementation.

            Full handout: exercise/README.md""")),
    "05_multigpu": dict(
        kind="cholesky", target="gpus", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: every GPU, or CPU and GPU together

            No changes to the Function. Setup and Use now uses both of
            andromeda's GPUs, Dagger.scope(cuda_gpus=:).
            - Run it as it is, then compare the timeline with one GPU.
            - Set Target to Mixed to let Dagger place tasks on both CPU and
              GPU. Is it faster? Why or why not?
            - Try other matrix and tile sizes (n and bs) in Setup and Use.
            - Look at solution/my_cholesky.jl only after you finish.

            Full handout: exercise/README.md""")),
    "06_multinode": dict(
        kind="cholesky", target="cpu", mode="mpi",
        prompt=textwrap.dedent("""\
            Exercise: same code, on MPI ranks (CPU)

            No changes to the Function. Processes is now Two MPI ranks:
            every rank runs the same Function and Setup and Use, and Dagger
            spreads the tiles and their tasks across the ranks. The test
            matrix is seeded, so every rank builds the same one, and only
            rank 0 prints.

            Your my_cholesky! from exercise/my_cholesky.jl is correct under
            MPI only if every In/InOut annotation is right: a tile written
            on another rank without InOut loses its update.

            Full handout: exercise/README.md""")),
    "07_multinode_multigpu": dict(
        kind="cholesky", target="gpu", mode="mpi",
        prompt=textwrap.dedent("""\
            Exercise: MPI ranks and GPUs together

            No changes to the Function. Two MPI ranks, each placing tasks on
            its GPU: the same my_cholesky! from exercise/my_cholesky.jl, the
            same Setup and Use, with Target GPU and Processes Two MPI ranks.
            Compare the timeline with the CPU run in the previous section.

            Full handout: exercise/README.md""")),
    "08_stencil": dict(
        kind="stencil", target="cpu", mode="single",
        prompt=textwrap.dedent("""\
            Exercise: heat diffusion with stencils

            The Function editor starts from exercise/my_stencil.jl: the
            temperature u of a periodic plate, a DArray split into tiles,
            and an empty time loop. One explicit step of the heat equation
            is
                u <- u + dt * D * laplacian(u)
            and exercise/serial_heat.jl does it with plain loops.

            1. Write laplacian(nb), the 5-point Laplacian of a 3 x 3
               neighbourhood: the four nearest neighbours minus 4 times the
               point itself, nb[2, 2].
            2. Make each time step one @stencil block of two statements:
               scratch[idx] gets the new temperature, read from
               @neighbors(u[idx], 1, Wrap()); then u[idx] gets scratch[idx].
            3. Return u.

            Rules:
            - A statement must not read the neighbours of the array it
              writes; that is what scratch is for.
            - Wrap() makes the plate periodic. Dagger reads the neighbours
              across tile edges for you; do not copy halos yourself.
            - Under MPI every rank runs the same code.

            Run exercise runs the Function, then Setup and Use, which checks
            u against the same steps in plain Julia. Done when it prints
            PASS. Then set Target to GPU, and Processes to Two MPI ranks:
            the same function, unchanged. The final exercise uses this
            stencil for its explicit terms.
            Full handout: exercise/README.md""")),
}


def _imports(target, mode, packages="using Dagger, LinearAlgebra, Random"):
    lines = [packages]
    if target != "cpu":
        lines.append("using CUDA")
    if mode == "mpi":
        lines += ["using MPI", "",
                  "rank = MPI.Comm_rank(MPI.COMM_WORLD)   # only rank 0 prints"]
    return lines


def _on_every_rank(mode):
    return ", on every MPI rank." if mode == "mpi" else "."


def _scope_lines(target, mode):
    """The scope line(s) for a target, as in harness/common.jl."""
    where = {"cpu": "the CPU threads", "gpu": "one GPU",
             "gpus": "both GPUs", "mixed": "the CPU threads and one GPU"}[target]
    if mode == "mpi":
        where = where.replace("the CPU threads", "every rank's CPU threads")
        where = where.replace("both GPUs", "every visible GPU")
        where += "" if target == "cpu" else " per rank"
    lines = [f"# (1) Where the tasks run: {where}."]
    if mode == "mpi" and target in ("cpu", "mixed"):
        lines.append("#     Constrain the MPI ranks to their CPUs; GPUs are opt-in.")
        cpu = "Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())"
    else:
        cpu = "Dagger.scope(worker=1)"
    gpu = "Dagger.scope(cuda_gpus=:)" if target == "gpus" else "Dagger.scope(cuda_gpu=1)"
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


def stencil_setup(target, mode):
    """Runnable Setup and Use for the stencil exercise."""
    lines = ["# Setup and Use: where the tasks run, the plate, the call to my_heat!,",
             "# and the check. Run exercise runs the Function first, then this code"
             + _on_every_rank(mode),
             ""] + _imports(target, mode, "using Dagger, Random")
    lines += [""] + _scope_lines(target, mode)
    lines += ["",
              "# (2) A periodic 512 x 512 plate: cold, with 24 seeded hot squares. The",
              "#     seed makes every MPI rank build the same plate.",
              "n, bs = 512, 128",
              "D, dt, nsteps = 0.2, 1.0, 100",
              "u0 = zeros(n, n)",
              "rng = Xoshiro(2026)",
              "for _ in 1:24",
              "    i, j = rand(rng, 1:n-7, 2)",
              "    u0[i:i+7, j:j+7] .= 1.0",
              "end",
              "",
              "# (3) Tile the plate, run my_heat! where the scope says, and gather u.",
              "u = Dagger.with_options(scope=scope) do",
              "    U = DArray(u0, Blocks(bs, bs))",
              "    my_heat!(U, similar(U), D, dt, nsteps)",
              "    collect(U)",
              "end",
              "",
              "# (4) Check u against the same steps in plain Julia, with circshift on the",
              "#     whole plate. A periodic plate also keeps its total heat.",
              "ref = copy(u0)",
              "for _ in 1:nsteps",
              "    ref .+= dt * D .* (circshift(ref, (1, 0)) .+ circshift(ref, (-1, 0)) .+",
              "                       circshift(ref, (0, 1)) .+ circshift(ref, (0, -1)) .- 4 .* ref)",
              "end",
              "err = maximum(abs, u - ref)",
              'status = err < 1e-10 ? "PASS" : "FAIL"',
              _say(mode, '"$status: $nsteps steps, tiles of $bs, max error ", err, '
                         '", total heat ", round(sum(u); digits=6), " of ", sum(u0)'),
              'err < 1e-10 || error("my_heat! does not match the serial steps")']
    return "\n".join(lines) + "\n"


def final_setup(target, mode):
    """Runnable Setup and Use for the final exercise."""
    lines = ["# Setup and Use: where the tasks run, the parameters, the initial fields,",
             "# the call to simulate!, and the check. Run final exercise runs the",
             "# Simulation first, then this code" + _on_every_rank(mode),
             "# The Visualization panel then draws u and v.",
             ""] + _imports(target, mode)
    lines += ["",
              "BLAS.set_num_threads(1)   # Dagger runs one solve per thread",
              ""]
    lines += _scope_lines(target, mode)
    lines += ["",
              "# (2) Parameters: grid size, time step, diffusion rates, feed and kill rates.",
              "n, dt = 256, 2.0",
              "Du, Dv = 0.16, 0.08",
              "F, k = 0.055, 0.062",
              "nsteps = 1800",
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
              "# (4) Split u and v into two panels of whole columns, run the simulation",
              "#     where the scope says, and gather the fields.",
              "u, v = Dagger.with_options(scope=scope) do",
              "    U = DArray(u, Blocks(n, n ÷ 2))",
              "    V = DArray(v, Blocks(n, n ÷ 2))",
              "    simulate!(U, V, (; n, dt, Du, Dv, F, k), nsteps)",
              "    collect(U), collect(V)",
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

        The feed and kill rates F = 0.055 and k = 0.062 grow a
        labyrinth of spots and stripes. The simulation combines the two
        earlier exercises: a stencil for the explicit terms, Cholesky
        for the implicit ones.

        ADI splitting makes each time step two half steps. In the
        first, y is explicit and x is implicit:
            ru = u + dt/2 * (Du * d2y(u) + reaction(u, v))   a stencil
            u  = A_u \\ ru, one column at a time            Cholesky
        with A_u = I - (dt/2) Du L, where L is the 1D periodic
        Laplacian; likewise for v with Dv. A_u and A_v are symmetric
        positive definite and constant, so factor each once with
        cholesky(Symmetric(A)) and reuse the factors every step.

        Setup and Use splits u and v into panels of whole columns, so
        each panel's solve is an independent task. The second half step
        is the first with x and y swapped: half_step! ends by
        transposing u and v, so calling it again treats y implicitly
        with the same code.

        Fill in d2y, half_step!, L, Cu, Cv, and the time loop of
        simulate!, updating u and v in place. Setup and Use sets the
        parameters and the initial fields, calls
        simulate!(u, v, p, nsteps) where the scope says, and checks the
        result; Target and Processes choose the hardware and whether it
        runs on MPI ranks. Every run is a fresh Julia in its own Slurm
        job; on the CPU it takes about a minute. The Visualization panel
        draws substrate u and activator v side by side and animates the
        sampled time steps; no plotting code belongs in your script.
    """),
    template=textwrap.dedent("""\
        # Final exercise: Gray-Scott reaction-diffusion on a periodic grid, with
        # a stencil for the explicit terms and Cholesky for the implicit ones.
        # Fill in the ??? and the TODOs. Setup and Use splits u and v into panels
        # of whole columns, calls simulate!, and checks the result.

        using Dagger, LinearAlgebra
        import Dagger: @stencil, Wrap
        import LinearAlgebra: LAPACK

        # Second difference along y, the second index, of a 3 x 3 neighbourhood
        # nb; nb[2, 2] is the point itself (as in the Stencils exercise).
        d2y(nb) = ???

        # solve_columns!('L', C, B) solves A X = B for every column of the panel
        # B, in place, from the lower Cholesky factor C of A. It is LAPACK.potrs!
        # under another name, so Dagger swaps in cuSOLVER when the task runs on
        # a GPU.
        const solve_columns! = LAPACK.potrs!

        # Rows `to` of the panel dst <- the transpose of rows `from` of the panel src.
        function transpose_block!(dst, src, from, to)
            dst[to, :] .= transpose(view(src, from, :))
            return nothing
        end

        # One ADI half step, from (u, v) back into (u, v), transposed.
        function half_step!(u, v, ru, rv, Cu, Cv, p)
            (; dt, Du, Dv, F, k) = p
            # (a) Explicit along y, plus the reaction, as one @stencil block:
            #       ru = u + dt/2 * (Du * d2y(u) - u v^2 + F (1 - u))
            #       rv = v + dt/2 * (Dv * d2y(v) + u v^2 - (F + k) v)
            #     where d2y(u) is d2y(@neighbors(u[idx], 1, Wrap())).
            # TODO

            # (b) Implicit along x: each panel of ru and rv holds whole columns,
            #     so solve_columns! on one panel is an independent task.
            # (c) Then u = ru' and v = rv': for every pair of panels (j, i),
            #     transpose_block! writes rows block(i) of panel j of u from rows
            #     block(j) of panel i of ru (the same for v and rv).
            #     Spawn (b) and (c) in one Datadeps region, with In and InOut.
            panels = size(u.chunks, 2)
            width = size(u, 2) ÷ panels
            block(i) = (i - 1) * width + 1:i * width
            # TODO
            return nothing
        end

        # Advance the substrate u and the activator v by nsteps ADI time steps, in
        # place. p holds the parameters from Setup and Use: n, dt, Du, Dv, F, k.
        function simulate!(u, v, p, nsteps)
            (; n, dt, Du, Dv) = p

            # 1. Build the 1D periodic Laplacian L: diagonal -2, neighbour
            #    entries +1, and the first/last rows wrap around.
            L = ???

            # 2. Factor A_u = I - (dt/2) Du L and A_v = I - (dt/2) Dv L with
            #    Cholesky. They are SPD and constant, so factor once and keep the
            #    lower factors, as plain matrices, for every half step.
            Cu = ???
            Cv = ???

            # 3. Time loop: two half steps per time step. The transpose at the end
            #    of a half step makes the next one treat y implicitly.
            ru, rv = similar(u), similar(v)
            for step in 1:nsteps
                # TODO
            end
            return u, v
        end
    """),
)


FINAL_WORKSPACE = ROOT / "results" / "final-exercise"
FINAL_FILE = FINAL_WORKSPACE / "final_exercise.jl"
FINAL_README = FINAL_WORKSPACE / "README.md"


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


def set_aside_pre_stencil_workspace():
    """The previous console's simulate! solved with Fu/Fv on plain matrices and
    had no stencil; this Setup and Use passes it column panels. Keep such a
    file aside once, at startup, and start from the stencil template."""
    if FINAL_FILE.exists():
        code = FINAL_FILE.read_text()
        if "function simulate!" in code and "Fu" in code and "@stencil" not in code:
            FINAL_FILE.replace(FINAL_FILE.with_name("final_exercise.pre-stencil.jl"))
            if FINAL_README.exists() and "stencil" not in FINAL_README.read_text():
                FINAL_README.replace(FINAL_README.with_name("README.pre-stencil.md"))
    ensure_final_workspace()


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
        if exercise["kind"] not in EXERCISE_KINDS:
            raise ValueError(f"Unknown exercise kind for {demo}")
    return {demo: dict(exercise) for demo, exercise in EXERCISES.items()}


# Each kind of exercise has its own Function file and Setup and Use builder.
# Sections of the same kind share one Function editor.
EXERCISE_KINDS = {
    "cholesky": dict(file="my_cholesky.jl", setup=exercise_setup),
    "stencil": dict(file="my_stencil.jl", setup=stencil_setup),
}


def exercise_template(kind="cholesky"):
    """Starting code shown in the exercise editor."""
    return (ROOT / "exercise" / EXERCISE_KINDS[kind]["file"]).read_text()


def exercise_templates():
    return {kind: exercise_template(kind) for kind in EXERCISE_KINDS}


def exercise_setups():
    return {kind: setup_templates(entry["setup"]) for kind, entry in EXERCISE_KINDS.items()}


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
        for name in ("Project.toml", "demos", "harness", "andromeda", "solution"):
            archive.add(ROOT / name, arcname=name,
                        filter=lambda item: None if "__pycache__" in item.name else item)
        for name in ("worker.jl", "remote_agent.py", "final_visuals.jl"):
            archive.add(ROOT / "host" / name, arcname="host/" + name)
    return stream.getvalue()


# The dry run uses andromeda (ICMC/USP): one Slurm node with two NVIDIA A30s,
# reached with SSH keys on a non-standard port.
CLUSTER = "andromeda"
CLUSTER_HOSTNAME = "andromeda.lasdpc.icmc.usp.br"
CLUSTER_HOSTNAMES = (CLUSTER_HOSTNAME, "andromeda.icmc.usp.br")
CLUSTER_PORT = 21200
DEFAULT_REMOTE_ROOT = "sc26-dagger/portal"   # relative to the remote home
# A run can print nothing for minutes. Probe every 30 s and give the network
# five minutes before the connection, and the run with it, is given up.
KEEPALIVE = ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
# Distinct markers, so the Odo console's own managed block stays untouched.
MANAGED_BEGIN = "# >>> sc26-dagger dry-run console: managed host block >>>"
MANAGED_END = "# <<< sc26-dagger dry-run console: managed host block <<<"


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


def is_cluster_hostname(hostname):
    return (hostname or "").lower().rstrip(".") in CLUSTER_HOSTNAMES


def valid_remote_root(root):
    """An absolute path, or one relative to the remote home without `..`."""
    if not root or root.startswith("-") or "\n" in root:
        return False
    return root.startswith("/") or ".." not in Path(root).parts


def remote_root_label(root):
    return root if root.startswith("/") else "~/" + root


def managed_alias(user):
    """A per-user alias that will not collide with a hand-written `Host andromeda`."""
    return "sc26-andromeda-" + re.sub(r"[^A-Za-z0-9_.-]", "-", user)


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


def alias_reaches_cluster(values):
    return is_cluster_hostname(values.get("hostname", "")) and values.get("port") == str(CLUSTER_PORT)


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


def write_managed_host(config_path, alias, user, control_path, server=CLUSTER_HOSTNAME,
                       port=CLUSTER_PORT, alternative=None):
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
             f"    Port {port}",
             f"    User {user}",
             "    ServerAliveInterval 30",
             "    ServerAliveCountMax 10",
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
    port = values.get("port", "")
    if not alias_reaches_cluster(values) or resolved != user:
        raise RuntimeError(
            f"SSH alias {alias!r} resolves to {resolved or '?'}@{hostname or '?'}:{port or '?'}, not "
            f"{user}@{CLUSTER_HOSTNAME}:{CLUSTER_PORT}. An earlier User/HostName/Port entry or an "
            "Include is overriding it")


def prompt_user(default):
    if not sys.stdin.isatty():
        raise ValueError(f"No {CLUSTER} username: pass --user, set SC26_USER, "
                         "or run the console from an interactive terminal")
    while True:
        try:
            answer = input(f"Username on {CLUSTER} ({CLUSTER_HOSTNAME}) [{default}]: ").strip()
        except EOFError:
            raise ValueError(f"No {CLUSTER} username provided")
        user = answer or default
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", user):
            return user
        print("Use letters, digits, dots, dashes or underscores.", flush=True)


def bootstrap_ssh(args):
    """Resolve the cluster identity, create a managed alias when needed, return (user, alias).

    A usable hand-written alias is never modified. Otherwise the console owns one
    explicitly marked block, so an attendee's `Host andromeda` keeps working unchanged.
    """
    local = getpass.getuser()
    config_path = Path(args.ssh_config).expanduser()
    explicit = (args.user or os.environ.get("SC26_USER", "")).strip()
    if explicit and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", explicit):
        raise ValueError(f"Invalid {CLUSTER} username: {explicit!r}")
    preferred = args.ssh_host or CLUSTER
    resolved = ssh_config_values(preferred, config_path)
    ready = alias_reaches_cluster(resolved)
    resolved_user = resolved.get("user", "")
    where = f"{CLUSTER_HOSTNAME}:{CLUSTER_PORT}"
    if explicit:
        user = explicit
    elif ready and resolved_user and resolved_user != local:
        # An existing, working alias wins; do not rewrite the file.
        print(f"Using existing SSH alias {preferred!r} as {resolved_user}@{where}", flush=True)
        return resolved_user, preferred
    else:
        user = prompt_user(resolved_user or local)
    if ready and resolved_user == user:
        print(f"Using existing SSH alias {preferred!r} as {user}@{where}", flush=True)
        return user, preferred
    alias = args.ssh_host or managed_alias(user)
    action = write_managed_host(config_path, alias, user, args.control_path,
                                alternative=None if args.ssh_host else managed_alias(user))
    verify_managed_host(alias, user, config_path)
    print(f"SSH alias {alias!r} {action} in {config_path}", flush=True)
    return user, alias


def ensure_master(args):
    """Open the multiplexed connection once. andromeda takes SSH keys, so no
    password is expected; a key passphrase can still be typed here."""
    socket_path = args.control_path
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    option = ssh_config_option(args.ssh_config)
    check = subprocess.run(["ssh"] + option + ["-S", str(socket_path), "-O", "check", args.ssh_host],
                           capture_output=True)
    if check.returncode == 0:
        print(f"Reusing the open SSH connection to {args.ssh_host}", flush=True)
        return True
    command = ["ssh"] + option + ["-S", str(socket_path),
               "-o", "ControlMaster=auto", "-o", "ControlPersist=8h"] + KEEPALIVE + [
               "-o", "ConnectTimeout=20", "-o", "RemoteCommand=none"]
    if sys.stdin.isatty():
        command += ["-o", "BatchMode=no"]
    else:
        command += ["-o", "BatchMode=yes"]
    command += ["-MNf", args.ssh_host]
    print(f"Opening SSH connection to {args.ssh_host} with your SSH key…", flush=True)
    if subprocess.run(command).returncode == 0:
        return True
    print("Could not open the SSH connection. Check that your key works, then reconnect the page:\n"
          + shlex.join(command), flush=True)
    return False


def remote_path(root):
    """A shell word for the workspace; relative paths start at the remote home."""
    return shlex.quote(root)


class Console:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.events = []
        self.seq = 0
        self.process = None
        self.connecting = False
        self.ready = False
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
            return dict(status=self.status, job=self.job, mode=self.mode, ready=self.ready,
                        connecting=self.connecting, busy=self.busy,
                        connected=self.process is not None and self.process.poll() is None,
                        events=[event for event in self.events if event["seq"] > after], seq=self.seq)

    def ssh_options(self):
        return ssh_config_option(getattr(self.args, "ssh_config", Path.home()/".ssh/config"))

    def ssh(self):
        return ["ssh"] + self.ssh_options() + [
                "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "RemoteCommand=none",
                "-S", str(self.args.control_path), self.args.ssh_host]

    def connect(self):
        with self.lock:
            if self.connecting or self.process is not None:
                raise ValueError("Already connected or connecting")
            self.connecting = True
            self.stop_event.clear()
            self.status = f"Connecting to {CLUSTER}…"
        threading.Thread(target=self._connect, daemon=True).start()

    def remote_environment(self):
        env = ["env", f"SC26_STEP_TIME={self.args.time}",
               f"SC26_PARTITION={self.args.partition}",
               f"SC26_LAUNCHER={getattr(self.args, 'launcher', 'slurm')}"]
        for name, value in (("SC26_ACCOUNT", self.args.account),
                            ("SC26_RESERVATION", self.args.reservation)):
            if value:
                env.append(f"{name}={value}")
        return env

    def _connect(self):
        try:
            socket = self.args.control_path
            socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            check = subprocess.run(["ssh"] + self.ssh_options()
                                   + ["-S", str(socket), "-O", "check", self.args.ssh_host],
                                   capture_output=True)
            if check.returncode:
                cmd = (["ssh"] + self.ssh_options() + KEEPALIVE
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
            root = remote_path(self.args.remote_root)
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
            self.status = "Preparing Julia and packages on the login node…"
            # prepare.sh downloads; the agent then compiles in a Slurm job and
            # submits every later run as its own job. Nothing is held between runs.
            script = "bash andromeda/prepare.sh </dev/null; exec python3 -u host/remote_agent.py"
            command = f"cd {root} && " + shlex.join(self.remote_environment() + ["bash", "-ec", script])
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
                        where = f" · job {self.job}" if self.job else ""
                        self.status = f"Running{where}…"
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
                raise ValueError(f"Connect to {CLUSTER} first and wait until it is ready")
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
                command["gpus"] = DEMO_GPUS.get(demo[0], 0)
                command["args"] = []
                command.setdefault("label", demo[1])
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
                command["label"] = f"Exercise · {TARGETS[target]} · {MODES[layout]}"
            elif command.get("op") == "eval":
                if command.get("mode") not in ("single", "mpi"):
                    raise ValueError("Unknown REPL layout")
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
                    command["label"] = f"Final Exercise · {TARGETS[target]} · {MODES[command['mode']]}"
                    visuals = command.get("visuals", [])
                    if not isinstance(visuals, list) or any(v not in FINAL_VISUAL_IDS for v in visuals):
                        raise ValueError("Unknown visualization")
                    command["visuals"] = visuals
                else:
                    command["gpus"] = eval_gpus(command.get("demo"), command["mode"])
            else:
                raise ValueError("Unknown operation")
            command["id"] = secrets.token_hex(8)
            self.send(command)
            self.busy = True
            self.status = "Submitting to Slurm…"
            self.run_id = command["id"]
            self.emit("run", run=command["id"], demo=command.get("demo"),
                      label=command.get("label") or command.get("demo", "REPL"),
                      mode=command["mode"], gpus=command["gpus"], args=command.get("args", []),
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
        # The remote agent's EOF/finally cancels the Slurm job of a run in
        # progress. Nothing else is allocated.


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
                                exercises=exercises(), exercise_templates=exercise_templates(),
                                exercise_setups=exercise_setups(),
                                targets=TARGETS, modes=MODES, demo_gpus=DEMO_GPUS,
                                final_exercise=dict(FINAL_EXERCISE, visuals=FINAL_VISUALS,
                                                    target="cpu", mode="single",
                                                    setups=setup_templates(final_setup)),
                                final_workspace=str(FINAL_WORKSPACE),
                                cluster=CLUSTER, host=console.args.ssh_host,
                                remote_root=remote_root_label(console.args.remote_root),
                                time=console.args.time))
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


def main():
    set_aside_pre_stencil_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: all IPv4 interfaces)")
    parser.add_argument("--allow-host", action="append", default=[], help="Additional allowed browser hostname")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--ssh-host", help=f"SSH alias; reuse `{CLUSTER}` when it already points at "
                                           f"{CLUSTER_HOSTNAME}:{CLUSTER_PORT}, otherwise the console "
                                           f"manages `sc26-{CLUSTER}-<user>`")
    parser.add_argument("--user", help=f"Your {CLUSTER} username (skips the prompt; also SC26_USER)")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh/config",
                        help="SSH config to read and update (default: ~/.ssh/config)")
    parser.add_argument("--partition", default=os.environ.get("SC26_PARTITION", "gpu"),
                        help="Slurm partition for every run (default: gpu)")
    parser.add_argument("--account", default=os.environ.get("SC26_ACCOUNT", ""),
                        help="Slurm account, if your association needs one")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT,
                        help="Dedicated workspace, absolute or relative to the remote home "
                             f"(default: ~/{DEFAULT_REMOTE_ROOT}); not an existing checkout")
    parser.add_argument("--control-path", type=Path,
                        default=Path.home() / f".ssh/control/sc26-{CLUSTER}")
    parser.add_argument("--time", default="00:20:00", help="Slurm time limit of each run")
    parser.add_argument("--reservation", default="")
    parser.add_argument("--launcher", choices=("slurm", "local"), default="slurm",
                        help="local runs each command on the login node without Slurm: for "
                             "testing only, while an account has no Slurm association")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip the startup SSH login; Connect then needs an existing connection")
    parser.add_argument("--no-auto-connect", action="store_true",
                        help="Wait for Connect instead of preparing Julia at startup")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.ssh_host and (not re.fullmatch(r"[\w@.-]+", args.ssh_host) or args.ssh_host.startswith("-")):
        parser.error("Invalid SSH host alias")
    if not valid_remote_root(args.remote_root):
        parser.error("--remote-root must be absolute or relative to your remote home, without ..")
    args.control_path = args.control_path.expanduser().resolve()
    args.ssh_config = args.ssh_config.expanduser()
    try:
        args.user, args.ssh_host = bootstrap_ssh(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
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
    if authenticated and not args.no_auto_connect:
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
        connection_note = "\nSSH login failed; use Connect after fixing SSH"
    print(f"Tutorial console: {url}\n{CLUSTER} user: {args.user} via {args.ssh_host}"
          f"\n{CLUSTER} workspace: {remote_root_label(args.remote_root)}{connection_note}"
          + "\nEvery run is its own Slurm job; Ctrl-C cancels a run in progress.", flush=True)
    if authenticated and not args.no_auto_connect:
        print("Preparing Julia and compiling packages in the background…", flush=True)
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
