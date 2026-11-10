"""Protocol, lifecycle and local HTTP boundary checks; no cluster required."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("console", BASE / "host/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def console(ready=False):
    c = app.Console(argparse.Namespace(ssh_host="andromeda", remote_root="sc26-dagger/portal with spaces",
        time="00:10:00", account="", reservation="", partition="gpu",
        control_path=Path("/tmp/portal-test-socket")))
    c.ready = ready
    return c


def load_agent():
    spec = importlib.util.spec_from_file_location("agent", BASE / "host/remote_agent.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    return agent


class ConsoleTests(unittest.TestCase):
    def test_selected_demo_source_matches_current_files(self):
        for demo, _, _ in app.DEMOS:
            sources = app.demo_sources(demo)["sources"]
            self.assertTrue(sources, demo)
            for source in sources:
                self.assertTrue(source["code"], demo)
                self.assertTrue(source["ansi"], demo)
                self.assertEqual(app.re.sub(r"\x1b\[[0-9;]*m", "", source["ansi"]), source["code"])
        gpu = app.demo_sources("05_multigpu")["sources"][0]["code"]
        cpu = app.demo_sources("06_multinode")["sources"][0]["code"]
        stencil = app.demo_sources("08_stencil")["sources"][0]["code"]
        # andromeda has NVIDIA GPUs: the dry run uses CUDA, not AMDGPU.
        self.assertIn("CUDA", gpu)
        self.assertIn("cuda_gpu=", gpu)
        for demo, _, _ in app.DEMOS:
            self.assertNotIn("AMDGPU", app.demo_sources(demo)["sources"][0]["code"], demo)
        self.assertIn("@stencil", stencil)
        self.assertIn("@neighbors(A[idx], 1, Wrap())", stencil)
        self.assertIn("Dagger.scope(cuda_gpu=1)", stencil)
        self.assertIn("using Dagger, MPI", cpu)
        self.assertNotIn("DEMO_", gpu)
        self.assertNotIn("DEMO_", cpu)
        self.assertNotIn("SC26_DEMO_LOAD_ONLY", gpu + cpu)
        with self.assertRaises(ValueError):
            app.demo_sources("../../Project")

    def test_every_panel_and_sidebar_group_is_collapsible(self):
        page = (BASE / "host/index.html").read_text()
        ids = set(app.re.findall(r'id="([^"]+)"', page))
        sections = app.re.findall(r'<section class="[^"]*(?:panel|side-section)[^"]*" id="([^"]+)"', page)
        controls = app.re.findall(r'aria-controls="([^"]+)"', page)
        self.assertTrue(sections)
        self.assertEqual(sorted(controls), sorted(section + "-body" for section in sections))
        self.assertEqual(set(controls) - ids, set())
        for target in controls:
            self.assertIn(f'aria-expanded="true" aria-controls="{target}"', page)
            self.assertIn(f'class="collapse"', page)
            self.assertIn(f'id="{target}"', page)

    def test_sections_cover_every_demo_once_in_sidebar_order(self):
        layout = app.section_layout()
        self.assertEqual([section["number"] for section in layout],
                         [f"{n:02d}" for n in range(1, len(layout) + 1)])
        demos = [demo for section in layout for demo in section["demos"]]
        self.assertEqual(demos, [demo for demo, _, _ in app.DEMOS])
        self.assertEqual(layout[0], dict(id="julia", title="Julia basics",
                                         number="01", demos=["00_julia"]))
        self.assertEqual(layout[1], dict(id="cpu", title="CPU", number="02",
                                         demos=["01_tasks", "02_datadeps"]))
        self.assertEqual(layout[2]["demos"], ["03_gpu"])
        self.assertEqual(layout[2]["title"], "Single GPU")
        # Stencils is the last section before the Final Exercise.
        self.assertEqual(layout[-1], dict(id="demo-08_stencil", title="Stencils",
                                          number=f"{len(layout):02d}", demos=["08_stencil"]))

    def test_every_section_after_julia_basics_has_an_exercise(self):
        exercises = app.exercises()
        sections = app.section_layout()
        for section in sections:
            # Julia basics has no exercise; every later section has one.
            expected = 0 if section["id"] == "julia" else 1
            self.assertEqual(sum(demo in exercises for demo in section["demos"]), expected,
                             section["id"])
        for demo, exercise in exercises.items():
            self.assertIn(demo, {entry[0] for entry in app.DEMOS})
            self.assertIn("Exercise:", exercise["prompt"])
            self.assertIn(app.EXERCISE_KINDS[exercise["kind"]]["file"], exercise["prompt"])
            self.assertIn(exercise["target"], app.TARGETS)
            self.assertIn(exercise["mode"], app.MODES)
            self.assertNotIn("rocm", exercise["prompt"])
        self.assertEqual(exercises["08_stencil"]["kind"], "stencil")
        self.assertEqual({e["kind"] for d, e in exercises.items() if d != "08_stencil"}, {"cholesky"})
        # The MPI sections start on two ranks, the others on one process.
        self.assertEqual(exercises["06_multinode"]["mode"], "mpi")
        self.assertEqual(exercises["07_multinode_multigpu"]["mode"], "mpi")
        self.assertEqual(exercises["02_datadeps"]["mode"], "single")

    def test_stencil_template_runs_but_fails_until_filled_in(self):
        template = app.exercise_template("stencil")
        self.assertIn("function my_heat!(u, scratch, D, dt, nsteps)", template)
        self.assertIn("import Dagger: @stencil, Wrap", template)
        self.assertIn('laplacian(nb) = error(', template)
        # The template parses: no ??? placeholders; Setup and Use reports FAIL.
        self.assertNotIn("???", template)
        solution = (BASE / "solution/my_stencil.jl").read_text()
        self.assertIn("@neighbors(u[idx], 1, Wrap())", solution)
        self.assertIn("u[idx] = scratch[idx]", solution)
        self.assertIn("function serial_heat!", (BASE / "exercise/serial_heat.jl").read_text())

    def test_exercise_template_defines_the_tile_operations(self):
        template = app.exercise_template("cholesky")
        self.assertIn("function my_cholesky!", template)
        for name, kernel in (("factor_diagonal_tile!", "LAPACK.potrf!"),
                             ("solve_panel_tile!", "BLAS.trsm!"),
                             ("update_diagonal_tile!", "BLAS.syrk!"),
                             ("update_offdiagonal_tile!", "BLAS.gemm!")):
            self.assertIn(f"const {name} = {kernel}", template)
        # They must stay aliases: Dagger selects the GPU kernel by function
        # identity, so a wrapper function would break the GPU run.
        self.assertNotIn("function factor_diagonal_tile!", template)

    def test_setup_and_use_is_runnable_for_every_target_and_layout(self):
        for build, call in ((app.exercise_setup, "my_cholesky!(DA.chunks)"),
                            (app.stencil_setup, "my_heat!(U, similar(U), D, dt, nsteps)"),
                            (app.final_setup, "simulate!(U, V,")):
            setups = app.setup_templates(build)
            self.assertEqual(set(setups), {f"{t}/{m}" for t in app.TARGETS for m in app.MODES})
            for key, setup in setups.items():
                target, mode = key.split("/")
                self.assertNotIn("???", setup, key)
                self.assertIn(call, setup, key)
                self.assertIn("Dagger.with_options(scope=scope)", setup, key)
                self.assertEqual("using CUDA" in setup, target != "cpu", key)
                self.assertNotIn("rocm", setup, key)
                self.assertEqual("using MPI" in setup, mode == "mpi", key)
                # Under MPI every rank runs the code; only rank 0 prints.
                if mode == "mpi":
                    self.assertIn("rank == 0 && println(", setup, key)
        cpu = app.exercise_setup("cpu", "single")
        self.assertIn("scope = Dagger.scope(worker=1)", cpu)
        self.assertIn("PASS", cpu)
        self.assertIn("scope = Dagger.scope(cuda_gpu=1)", app.exercise_setup("gpu", "single"))
        self.assertIn("scope = Dagger.scope(cuda_gpus=:)", app.exercise_setup("gpus", "single"))
        self.assertIn("PASS", app.stencil_setup("cpu", "single"))
        self.assertIn("circshift", app.stencil_setup("cpu", "single"))
        self.assertIn("Dagger.UnionScope(cpu, gpu)", app.exercise_setup("mixed", "mpi"))
        self.assertIn("mpi_ranks=:", app.exercise_setup("cpu", "mpi"))
        # The visualization reads these globals after the final exercise runs.
        final = app.final_setup("cpu", "single")
        for name in ("u = ", "v = ", "F, k = ", "nsteps = ", "u, v = Dagger.with_options"):
            self.assertIn(name, final)
        self.assertIn("Blocks(n, n ÷ 2)", final)
        # The solution file carries the console's CPU, one-process Setup and Use.
        solution = (BASE / "solution/final_exercise.jl").read_text()
        self.assertIn(final, solution)

    def test_console_generates_a_shared_exercise_editor(self):
        page = (BASE / "host/app.js").read_text()
        self.assertIn('data-role="exercise-code"', page)
        self.assertIn('data-role="exercise-setup"', page)
        self.assertIn('data-role="exercise-setup-reset"', page)
        self.assertIn('data-role="exercise-target"', page)
        self.assertIn('data-role="exercise-mode"', page)
        self.assertIn("Setup and Use", page)
        self.assertIn(">Function<", page)
        self.assertIn('op: "exercise"', page)
        self.assertIn("setup: panel.setup.value", page)
        self.assertIn("mode: panel.mode.value", page)
        self.assertIn("exercise_templates", page)
        self.assertIn("exercise_setups", page)
        self.assertIn("EXERCISE_SETUP_KEY", page)
        self.assertIn('data-kind="${exercise.kind}"', page)
        # Every editor is highlighted: Function, Setup and Use, Simulation, REPL.
        for editor in ("enhanceEditor(panel.code)", "enhanceEditor(panel.setup)",
                       "enhanceEditor(pane.code)"):
            self.assertIn(editor, page)
        self.assertIn('<script src="/highlight.js"></script>',
                      (BASE / "host/index.html").read_text())
        # The final exercise has the same Setup and Use and run controls.
        final = page[page.index("function finalExerciseMarkup"):]
        final = final[:final.index("\n}")]
        self.assertIn("setupMarkup(pid)", final)
        self.assertIn("runControlsMarkup()", final)
        self.assertIn("withoutComments", page)
        self.assertIn("finalExerciseMarkup", page)
        self.assertIn("showFinal", page)
        self.assertIn("Final Exercise", page)
        self.assertIn('dataset.final = "1"', page)
        self.assertIn("Use your agents", page)
        self.assertIn('data-agent="opencode"', page)
        self.assertIn('data-agent="claude"', page)
        self.assertIn('data-agent="codex"', page)
        self.assertIn("OpenCode", page)
        self.assertIn("Claude", page)
        self.assertIn("Codex", page)
        self.assertIn('data-role="agent-terminal"', page)
        self.assertIn('data-role="agent-terminal-body"', page)
        self.assertIn("/api/terminal/check", page)
        self.assertIn("/api/terminal/install", page)
        self.assertIn("window.confirm", page)
        self.assertIn("ensureAgent", page)
        self.assertIn("/api/final/file", page)
        self.assertIn("final_workspace", page)
        self.assertIn("syncFinalFile", page)
        self.assertGreaterEqual(page.count("Task timeline"), 2)
        self.assertGreaterEqual(page.count("Session output"), 2)
        self.assertIn('id="${pid}-visual"', page)
        self.assertIn('data-role="visual"', page)
        self.assertIn('class="visual-grid"', page)
        self.assertIn("renderVisualization", page)
        self.assertIn("visuals,", page)
        self.assertIn('item.kind === "visual"', page)
        # Session output comes first in every pane; the timeline follows it.
        for function in ("function analysisMarkup", "function paneMarkup"):
            body = page[page.index(function):]
            body = body[:body.index("\n}")]
            self.assertLess(body.index('id="${pid}-output"'),
                            body.index('id="${pid}-timeline"'), function)

    def test_final_exercise_is_configured(self):
        final = app.FINAL_EXERCISE
        self.assertEqual(final["title"], "Final Exercise")
        self.assertIn("Gray-Scott", final["prompt"])
        self.assertIn("A_u \\ ru", final["prompt"])
        for blank in ("d2y(nb) = ???", "L = ???", "Cu = ???", "Cv = ???"):
            self.assertIn(blank, final["template"])
        # Stencils for the explicit terms, Cholesky for the implicit ones.
        self.assertIn("import Dagger: @stencil, Wrap", final["template"])
        self.assertIn("const solve_columns! = LAPACK.potrs!", final["template"])
        self.assertIn("function half_step!(u, v, ru, rv, Cu, Cv, p)", final["template"])
        self.assertIn("function simulate!(u, v, p, nsteps)", final["template"])
        self.assertIn("for step in 1:nsteps", final["template"])
        solution = (BASE / "solution/final_exercise.jl").read_text()
        self.assertIn("@stencil begin", solution)
        self.assertIn("cholesky(Symmetric(", solution)
        # The visualization code lives in the harness, not in the template.
        self.assertNotIn("UnicodePlots", final["template"])
        self.assertNotIn("heatmap(", final["template"])
        self.assertEqual([item["id"] for item in app.FINAL_VISUALS],
                         ["substrate", "activator"])

    def test_visualization_has_time_controls(self):
        page = (BASE / "host/visuals.js").read_text(encoding="utf-8")
        self.assertIn('type = "range"', page)
        self.assertIn('play.textContent = "Play"', page)
        self.assertIn("data.frames", page)
        self.assertIn("setInterval", page)
        self.assertIn("_visualApply", page)
        self.assertIn("visualUnpack", page)
        self.assertIn("frame_range", page)
        script = (BASE / "host/app.js").read_text(encoding="utf-8")
        self.assertIn("_visualApply", script)
        fields = (BASE / "host/final_visuals.jl").read_text(encoding="utf-8")
        self.assertIn("max_frames::Integer = 32", fields)

    def test_final_exercise_eval_is_allowed(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="eval", mode="single", demo="final", code="x = 1", setup="y = 2"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["label"], "Final Exercise · CPU · One process")
        self.assertEqual(command["setup"], "y = 2")
        self.assertEqual(command["gpus"], 0)
        c.busy = False
        c.run(dict(op="eval", mode="mpi", target="gpu", demo="final", code="x = 1", setup="y = 2"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["label"], "Final Exercise · GPU · Two MPI ranks")
        self.assertEqual(command["gpus"], 2)
        c.busy = False
        with self.assertRaisesRegex(ValueError, "Setup and Use"):
            c.run(dict(op="eval", mode="single", demo="final", code="x = 1"))

    def test_final_exercise_visuals_are_whitelisted(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="eval", mode="single", demo="final", code="x = 1", setup="y = 2",
                   visuals=["substrate", "activator"]))
        command = c.send.call_args[0][0]
        self.assertEqual(command["visuals"], ["substrate", "activator"])
        c.busy = False
        with self.assertRaisesRegex(ValueError, "Unknown visualization"):
            c.run(dict(op="eval", mode="single", demo="final", code="x = 1", setup="y = 2",
                       visuals=["bogus"]))

    def test_agent_status_and_install_are_whitelisted(self):
        with patch.object(app.shutil, "which", return_value="/usr/bin/opencode"):
            self.assertTrue(app.agent_status("opencode")["installed"])
        with patch.object(app.shutil, "which", return_value=None):
            self.assertFalse(app.agent_status("opencode")["installed"])
        with self.assertRaisesRegex(ValueError, "Unknown agent"):
            app.install_agent("nope")
        def which(name, path=None):
            return {"npm": "/usr/bin/npm"}.get(name)
        with patch.object(app.shutil, "which", side_effect=which), \
                patch.object(app.subprocess, "run") as run:
            run.return_value = Mock(returncode=0, stdout="ok\n", stderr="")
            result = app.install_agent("codex")
            self.assertTrue(result["ok"])
            self.assertEqual(run.call_args[0][0][0], "/usr/bin/npm")

    def test_final_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(app, "FINAL_WORKSPACE", root), \
                    patch.object(app, "FINAL_FILE", root / "final_exercise.jl"), \
                    patch.object(app, "FINAL_README", root / "README.md"):
                app.ensure_final_workspace()
                self.assertTrue(app.FINAL_FILE.exists())
                app.write_final_file("x = 1\n")
                state = app.final_file_state()
                self.assertEqual(state["code"], "x = 1\n")
                self.assertGreater(state["mtime"], 0)

    def test_final_workspace_sets_aside_the_old_one_file_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(app, "FINAL_WORKSPACE", root), \
                    patch.object(app, "FINAL_FILE", root / "final_exercise.jl"), \
                    patch.object(app, "FINAL_README", root / "README.md"):
                old = "# Setup: parameters, fields, and the constant operators.\nn = 256\n"
                app.FINAL_FILE.write_text(old)
                app.ensure_final_workspace()
                self.assertEqual((root / "final_exercise.old.jl").read_text(), old)
                self.assertIn("function simulate!", app.FINAL_FILE.read_text())
                # The previous console's simulate! had no stencil: set aside once.
                adi = "function simulate!(u, v, p, nsteps)\n    Fu = cholesky(A)\nend\n"
                app.FINAL_FILE.write_text(adi)
                app.set_aside_pre_stencil_workspace()
                self.assertEqual((root / "final_exercise.pre-stencil.jl").read_text(), adi)
                self.assertIn("@stencil", app.FINAL_FILE.read_text())
                # A student's own file without a stencil is left alone.
                app.write_final_file("x = 1\n")
                app.set_aside_pre_stencil_workspace()
                self.assertEqual(app.FINAL_FILE.read_text(), "x = 1\n")

    def test_embedded_terminal_whitelists_and_streams(self):
        with self.assertRaisesRegex(ValueError, "Unknown agent"):
            app.open_agent_terminal("nope", 80, 24)
        with patch.object(app.shutil, "which", return_value="/bin/cat"):
            with self.assertRaisesRegex(ValueError, "Working directory"):
                app.open_agent_terminal("opencode", 80, 24, cwd="/no/such/path")
            session = app.open_agent_terminal("opencode", 80, 24)
        try:
            app.terminal_input(session, "hello\n")
            out = {"data": ""}
            for _ in range(40):
                out = app.terminal_output(session, 0)
                if "hello" in out["data"]:
                    break
                time.sleep(0.05)
            self.assertIn("hello", out["data"])
        finally:
            app.close_terminal_session(session)

    def test_exercise_run_sends_function_setup_target_and_layout(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="exercise", demo="02_datadeps", target="cpu",
                   code="x = 1", setup="y = 2"))
        started = [event for event in c.events if event["kind"] == "run"][-1]
        self.assertTrue(started["exercise"])
        command = c.send.call_args[0][0]
        self.assertEqual(command["mode"], "single")
        self.assertEqual(command["target"], "cpu")
        self.assertEqual(command["setup"], "y = 2")
        self.assertEqual(command["gpus"], 0)
        self.assertEqual(command["label"], "Exercise · CPU · One process")
        c.busy = False
        # Processes chooses the worker layout in any section, not only MPI ones.
        c.run(dict(op="exercise", demo="03_gpu", target="gpu", mode="mpi",
                   code="x = 1", setup="y = 2"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["mode"], "mpi")
        self.assertEqual(command["gpus"], 2)   # one GPU per rank
        self.assertEqual(command["label"], "Exercise · GPU · Two MPI ranks")
        c.busy = False
        c.run(dict(op="exercise", demo="08_stencil", target="gpus", code="x = 1", setup="y = 2"))
        command = c.send.call_args[0][0]
        self.assertEqual((command["mode"], command["gpus"]), ("single", 2))
        c.busy = False
        c.run(dict(op="exercise", demo="06_multinode", code="x = 1", setup="y = 2"))
        self.assertEqual(c.send.call_args[0][0]["mode"], "mpi")
        c.busy = False
        with self.assertRaisesRegex(ValueError, "no exercise"):
            c.run(dict(op="exercise", demo="01_tasks", code="x = 1", setup="y = 2"))
        with self.assertRaisesRegex(ValueError, "Write Julia code"):
            c.run(dict(op="exercise", demo="02_datadeps", code=" ", setup="y = 2"))
        with self.assertRaisesRegex(ValueError, "Setup and Use"):
            c.run(dict(op="exercise", demo="02_datadeps", code="x = 1", setup=""))
        with self.assertRaisesRegex(ValueError, "layout"):
            c.run(dict(op="exercise", demo="02_datadeps", code="x = 1", setup="y = 2",
                       mode="three"))

    def test_config_exposes_exercises(self):
        c = console()
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.handler_for(c))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/config") as response:
                config = json.load(response)
            self.assertEqual(set(config["exercises"]), set(app.EXERCISES))
            self.assertEqual(config["exercises"]["02_datadeps"]["target"], "cpu")
            self.assertIn("function my_cholesky!", config["exercise_templates"]["cholesky"])
            self.assertIn("function my_heat!", config["exercise_templates"]["stencil"])
            self.assertIn("my_cholesky!(DA.chunks)", config["exercise_setups"]["cholesky"]["cpu/single"])
            self.assertIn("my_heat!(U,", config["exercise_setups"]["stencil"]["gpu/mpi"])
            self.assertEqual(config["demo_gpus"]["05_multigpu"], 2)
            self.assertEqual((config["cluster"], config["remote_root"]),
                             ("andromeda", "~/sc26-dagger/portal with spaces"))
            self.assertEqual(list(config["targets"]), list(app.TARGETS))
            self.assertEqual(list(config["modes"]), ["single", "mpi"])
            self.assertEqual(config["final_exercise"]["title"], "Final Exercise")
            self.assertIn("L = ???", config["final_exercise"]["template"])
            self.assertIn("simulate!(U, V,", config["final_exercise"]["setups"]["cpu/mpi"])
            self.assertTrue(config["final_workspace"].endswith("final-exercise"))
        finally:
            server.shutdown()
            server.server_close()

    def test_run_tags_events_with_demo_and_clears_after_done(self):
        c = console(ready=True)
        c.process = Mock(stdin=io.StringIO())
        c.process.poll.return_value = None
        c.run(dict(op="demo", demo="01_tasks"))
        started = [event for event in c.events if event["kind"] == "run"][-1]
        self.assertEqual(started["demo"], "01_tasks")
        run_id = started["run"]
        p = Mock(stdout=io.StringIO("hello\n" + app.PREFIX
                                    + json.dumps(dict(kind="done", ok=True)) + "\n"))
        p.wait.return_value = 0
        c._read(p)
        output = [event for event in c.events if event["kind"] == "output"][-1]
        self.assertEqual(output["run"], run_id)
        self.assertIsNone(c.run_id)
        # The agent relays the worker's lines as output events; they belong to
        # the run that printed them, not to the selected pane.
        c.busy = False
        c.run(dict(op="exercise", demo="08_stencil", code="x = 1", setup="y = 2"))
        exercise_run = [event for event in c.events if event["kind"] == "run"][-1]["run"]
        p = Mock(stdout=io.StringIO(app.PREFIX + json.dumps(dict(kind="output", text="PASS\n")) + "\n"))
        p.wait.return_value = 0
        c._read(p)
        self.assertEqual([e for e in c.events if e["kind"] == "output"][-1]["run"], exercise_run)
        c.busy = False
        c.run(dict(op="eval", mode="single", code="1+1", demo="01_tasks"))
        evaluation = [event for event in c.events if event["kind"] == "run"][-1]
        self.assertEqual(evaluation["demo"], "01_tasks")
        self.assertEqual(evaluation["gpus"], 0)
        c.busy = False
        # A REPL evaluation gets its section demo's GPUs; two ranks get one each.
        c.run(dict(op="eval", mode="mpi", code="1+1", demo="03_gpu"))
        self.assertEqual([e for e in c.events if e["kind"] == "run"][-1]["gpus"], 2)
        c.busy = False
        with self.assertRaisesRegex(ValueError, "Unknown demo"):
            c.run(dict(op="eval", mode="single", code="1", demo="nope"))

    def test_remote_workspace_and_hostnames(self):
        self.assertEqual(app.remote_root_label("sc26-dagger/portal"), "~/sc26-dagger/portal")
        self.assertEqual(app.remote_root_label("/scratch/me/portal"), "/scratch/me/portal")
        self.assertTrue(app.valid_remote_root("sc26-dagger/portal"))
        self.assertFalse(app.valid_remote_root("../elsewhere"))
        self.assertFalse(app.valid_remote_root("-oProxyCommand=x"))
        self.assertTrue(app.is_cluster_hostname("andromeda.lasdpc.icmc.usp.br"))
        self.assertTrue(app.is_cluster_hostname("andromeda.icmc.usp.br."))
        self.assertFalse(app.is_cluster_hostname("andromeda"))

    def test_resources_follow_target_processes_and_demo(self):
        self.assertEqual([app.run_gpus(t, "single") for t in app.TARGETS], [0, 1, 2, 1])
        self.assertEqual([app.run_gpus(t, "mpi") for t in app.TARGETS], [0, 2, 2, 2])
        self.assertEqual(app.eval_gpus("05_multigpu", "single"), 2)
        self.assertEqual(app.eval_gpus("01_tasks", "mpi"), 0)
        self.assertEqual(app.eval_gpus(None, "single"), 0)
        agent = load_agent()
        self.assertEqual(agent.resources(dict(mode="mpi", gpus=2)), (2, 2))
        self.assertEqual(agent.describe(2, 2), "two MPI ranks · 4 CPUs each · 1 GPU each")
        self.assertEqual(agent.describe(1, 0), "one process · 8 CPUs · no GPU")
        for bad in (dict(mode="mpi", gpus=1), dict(gpus=3), dict(gpus="2")):
            with self.assertRaises(ValueError):
                agent.resources(bad)

    def test_existing_working_alias_is_reused_without_rewriting_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            original = ("Host andromeda\n    HostName andromeda.lasdpc.icmc.usp.br\n"
                        "    Port 21200\n    User alice\n")
            config.write_text(original)
            args = argparse.Namespace(ssh_host=None, user="alice",
                                      ssh_config=config, control_path=Path(directory)/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                user, alias = app.bootstrap_ssh(args)
            self.assertEqual((user, alias), ("alice", "andromeda"))
            self.assertEqual(config.read_text(), original)

    def test_managed_alias_is_added_first_and_leaves_other_hosts_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            # An alias without the port does not reach andromeda; the Odo
            # console's managed block is someone else's and must survive.
            original = ("# >>> sc26-dagger console: managed host block >>>\nHost sc26-odo-alice\n"
                        "    HostName login1.odo.olcf.ornl.gov\n"
                        "# <<< sc26-dagger console: managed host block <<<\n"
                        "# hand-written entry\nHost andromeda\n    HostName andromeda.lasdpc.icmc.usp.br\n"
                        "    User bob\n\nHost *\n    User ghost\n    Port 22\n")
            config.write_text(original)
            args = argparse.Namespace(ssh_host=None, user="alice",
                                      ssh_config=config, control_path=Path(directory)/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                user, alias = app.bootstrap_ssh(args)
            self.assertEqual((user, alias), ("alice", "sc26-andromeda-alice"))
            text = config.read_text()
            self.assertIn(original.strip(), text)
            # First value wins for User/HostName/Port, so the block must come first.
            self.assertLess(text.index(app.MANAGED_BEGIN), text.index("Host andromeda"))
            resolved = app.ssh_config_values(alias, config)
            self.assertEqual((resolved["user"], resolved["hostname"], resolved["port"]),
                             ("alice", app.CLUSTER_HOSTNAME, "21200"))
            self.assertEqual(app.write_managed_host(config, alias, "alice", args.control_path),
                             "unchanged")

    def test_foreign_alias_collision_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            config.write_text("Host sc26-andromeda-alice\n    HostName example.org\n")
            args = argparse.Namespace(ssh_host=None, user="alice",
                                      ssh_config=config, control_path=Path(directory)/"socket")
            with self.assertRaisesRegex(ValueError, "already defined"):
                with contextlib.redirect_stdout(io.StringIO()):
                    app.bootstrap_ssh(args)

    def test_managed_block_follows_a_symlinked_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "dotfiles-ssh-config"
            target.write_text("Host work\n    HostName example.org\n")
            link = root / "config"
            link.symlink_to(target)
            args = argparse.Namespace(ssh_host=None, user="alice", ssh_config=link,
                                      control_path=root/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                app.bootstrap_ssh(args)
            self.assertTrue(link.is_symlink())
            self.assertIn("Host sc26-andromeda-alice", target.read_text())
            self.assertIn("Host work", target.read_text())

    def test_command_layout_and_busy_guard(self):
        c = console()
        c.process = Mock(stdin=io.StringIO())
        c.process.poll.return_value = None
        with self.assertRaisesRegex(ValueError, "ready"):
            c.run(dict(op="demo", demo="07_multinode_multigpu"))
        c.ready = True
        c.run(dict(op="demo", demo="07_multinode_multigpu"))
        command = json.loads(c.process.stdin.getvalue())
        self.assertEqual(command["mode"], "mpi")
        self.assertEqual(command["gpus"], 2)
        self.assertEqual(command["args"], [])
        with self.assertRaisesRegex(ValueError, "in progress"):
            c.run(dict(op="eval", mode="single", code="1+1"))

    def test_failed_run_unlocks_and_expected_switch_stays_busy(self):
        c = console()
        c.busy = True
        events = [dict(kind="worker_exit", returncode=-15, expected=True),
                  dict(kind="worker_ready", mode="mpi")]
        p = Mock(stdout=io.StringIO("".join(app.PREFIX+json.dumps(e)+"\n" for e in events)))
        p.wait.return_value = 0
        c._read(p)
        self.assertTrue(c.busy)
        p.stdout = io.StringIO(app.PREFIX + json.dumps(dict(kind="done", ok=False)) + "\n")
        c._read(p)
        self.assertFalse(c.busy)

    def test_snapshot_excludes_results_and_credentials(self):
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(app.snapshot())) as archive:
            names = archive.getnames()
        self.assertIn("host/worker.jl", names)
        self.assertIn("host/final_visuals.jl", names)
        self.assertIn("demos/presentation.jl", names)
        self.assertIn("demos/08_stencil.jl", names)
        for script in ("env.sh", "prepare.sh", "precompile.sh", "launch.sh", "mpirun.jl", "rank.sh"):
            self.assertIn("andromeda/" + script, names)
        self.assertFalse(any(n.startswith("frontier") for n in names))
        self.assertFalse(any("results/" in n or "Manifest" in n or "Preferences" in n for n in names))

    def test_http_requires_token_and_has_no_path_traversal(self):
        c = console()
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.handler_for(c))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path, method, status in [("/api/run", "POST", 403), ("/../Project.toml", "GET", 404)]:
                request = urllib.request.Request(base+path, data=b'{}' if method=="POST" else None, method=method)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, status)
                caught.exception.close()
            request = urllib.request.Request(base+"/api/run", data=b'{"op":"eval","mode":"single","code":"1"}',
                                              headers={"X-Console-Token":c.token})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
            request = urllib.request.Request(base+"/api/config", headers={"Host":"untrusted.example"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            with urllib.request.urlopen(base+"/ansi.js") as response:
                self.assertEqual(response.status, 200)
            with urllib.request.urlopen(base+"/visuals.js") as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"renderVisualization", response.read())
            with urllib.request.urlopen(base+"/highlight.js") as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"function enhanceEditor", response.read())
        finally:
            server.shutdown()
            server.server_close()

    def test_wildcard_bind_accepts_tailscale_addresses(self):
        c = console()
        server = app.ThreadingHTTPServer(("0.0.0.0", 0), app.handler_for(c))
        server.allowed_hosts = {"127.0.0.1", "localhost", "myhost.example.ts.net"}
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}/api/config"
        try:
            for host in ("100.64.0.1", "myhost.example.ts.net"):
                request = urllib.request.Request(base, headers={"Host":f"{host}:{server.server_port}"})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 200)
            request = urllib.request.Request(base, headers={"Host":f"untrusted.example:{server.server_port}"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_agent_eof_stops_the_run_and_cancels_only_its_own_job(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "andromeda").mkdir()
            # A stand-in for launch.sh: records its arguments and input, then waits.
            launch = root / "andromeda/launch.sh"
            launch.write_text('echo "$@" > "$LOG/args"\ncat > "$LOG/stdin"\nsleep 30 & wait $!\n')
            (root / "bin").mkdir()
            scancel = root / "bin/scancel"
            scancel.write_text('#!/bin/sh\necho "$@" > "$LOG/cancelled"\n')
            scancel.chmod(0o755)
            script = ("import importlib.util, pathlib; s=importlib.util.spec_from_file_location('agent', "
                      + repr(str(BASE/"host/remote_agent.py")) + "); m=importlib.util.module_from_spec(s); "
                      "s.loader.exec_module(m); m.ROOT = pathlib.Path(" + repr(directory) + "); "
                      "m.configure_stdio(); m.main()")
            env = dict(os.environ, PATH=str(root/"bin")+os.pathsep+os.environ["PATH"], LOG=directory,
                       SC26_PRECOMPILE="0", USER="alice")
            env.pop("SC26_LAUNCHER", None)
            proc = subprocess.Popen(["python3", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env)
            try:
                proc.stdin.write(b'{"op":"demo","id":"abc123","mode":"mpi","gpus":2}\n')
                proc.stdin.flush()
                for _ in range(100):
                    if (root/"stdin").exists() and (root/"stdin").read_text():
                        break
                    time.sleep(0.05)
                proc.stdin.close()   # EOF: the console went away mid-run.
                out, err = proc.communicate(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()
            self.assertEqual(proc.returncode, 0, err)
            self.assertEqual((root/"args").read_text().split()[:3], ["2", "2", "--startup-file=no"])
            self.assertIn("--mpi --gpu", (root/"args").read_text())
            # The worker gets the one command and then quit.
            commands = [json.loads(line) for line in (root/"stdin").read_text().splitlines()]
            self.assertEqual([c["op"] for c in commands], ["demo", "quit"])
            self.assertEqual((root/"cancelled").read_text().split(), ["--name", "sc26-abc123", "--user", "alice"])
            events = [json.loads(line[len(app.PREFIX):]) for line in out.decode().splitlines()
                      if line.startswith(app.PREFIX)]
            self.assertEqual(events[0]["kind"], "ready")
            self.assertTrue(any(e["kind"] == "status" and "two MPI ranks" in e["message"] for e in events))

    def test_a_job_that_ends_without_done_reports_a_failed_run(self):
        agent = load_agent()
        events = []
        agent.emit = lambda kind, **fields: events.append(dict(kind=kind, **fields))
        child = ("print('srun: job 77 queued and waiting for resources', flush=True)\n"
                 "print('ERROR: something broke', flush=True)\nraise SystemExit(3)\n")
        proc = subprocess.Popen(["python3", "-c", child], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        run = dict(proc=proc, id="r1", name="sc26-r1", done=False, cancelled=False)
        agent.current = run
        agent.relay(run)
        self.assertIsNone(agent.current)
        self.assertIn(dict(kind="job", job="77"), events)
        self.assertTrue(all(e.get("run") == "r1" for e in events if e["kind"] == "output"))
        done = [e for e in events if e["kind"] == "done"]
        self.assertEqual(len(done), 1)
        self.assertFalse(done[0]["ok"])
        self.assertIn("status 3", done[0]["error"])
        exit_event = [e for e in events if e["kind"] == "worker_exit"][-1]
        self.assertEqual((exit_event["returncode"], exit_event["expected"]), (3, False))
        # A run that finished first is not reported twice.
        events.clear()
        finished = ("print('@@SC26@@' + '{\"kind\":\"done\",\"ok\":true}', flush=True)\n")
        proc = subprocess.Popen(["python3", "-c", finished], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        agent.relay(dict(proc=proc, id="r2", name="sc26-r2", done=False, cancelled=False))
        self.assertEqual([e["kind"] for e in events], ["done", "worker_exit"])
        self.assertTrue(events[-1]["expected"])

    def test_a_new_connection_stops_the_stale_coordinator(self):
        agent = load_agent()
        agent.emit = lambda kind, **fields: None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_script = root / "remote_agent.py"
            stale_script.write_text("import time\ntime.sleep(60)\n")
            stale = subprocess.Popen(["python3", str(stale_script)])
            try:
                pid = root / "agent.pid"
                pid.write_text(str(stale.pid))
                agent.replace_stale_agent(pid)
                self.assertEqual(stale.wait(timeout=10), -15)
                self.assertEqual(pid.read_text(), str(__import__("os").getpid()))
                agent.forget_pid(pid)
                self.assertFalse(pid.exists())
                # Another program's pid is left alone.
                other = subprocess.Popen(["sleep", "30"])
                pid.write_text(str(other.pid))
                agent.replace_stale_agent(pid)
                self.assertIsNone(other.poll())
                other.kill()
                other.wait()
            finally:
                if stale.poll() is None:
                    stale.kill()
                    stale.wait()

    def test_heartbeats_are_not_shown(self):
        c = console(ready=True)
        p = Mock(stdout=io.StringIO(app.PREFIX + json.dumps(dict(kind="heartbeat")) + "\n"))
        p.wait.return_value = 0
        c._read(p)
        self.assertEqual(c.events, [])

    def test_large_report_survives_nonblocking_inherited_stdout(self):
        import os
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        script = ("import importlib.util; s=importlib.util.spec_from_file_location('agent', "
                  + repr(str(BASE/"host/remote_agent.py")) + "); m=importlib.util.module_from_spec(s); "
                  "s.loader.exec_module(m); m.configure_stdio(); m.emit('timeline', payload='x'*250000)")
        proc = subprocess.Popen(["python3", "-c", script], stdout=write_fd, stderr=subprocess.PIPE)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb") as stream:
            report = stream.read()
        _, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(len(json.loads(report.decode()[len(app.PREFIX):])["payload"]), 250000)

    def test_batched_commands_do_not_wait_for_more_stdin(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sbatch", "scancel"):
                path = root/name
                path.write_text("#!/bin/sh\n" + ("echo 456\n" if name == "sbatch" else "exit 0\n"))
                path.chmod(0o755)
            script = ("import importlib.util; s=importlib.util.spec_from_file_location('agent', "
                      + repr(str(BASE/"host/remote_agent.py")) + "); m=importlib.util.module_from_spec(s); "
                      "s.loader.exec_module(m); m.execute=lambda c: None; m.precompile=lambda: None; "
                      "m.configure_stdio(); m.main()")
            proc = subprocess.Popen(["python3", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=dict(os.environ, PATH=directory+os.pathsep+os.environ["PATH"]))
            try:
                proc.stdin.write(b'{"op":"test"}\n{"op":"quit"}\n')
                proc.stdin.flush()  # Keep the pipe open; quit must already be consumed.
                proc.wait(timeout=5)
                self.assertEqual(proc.returncode, 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
