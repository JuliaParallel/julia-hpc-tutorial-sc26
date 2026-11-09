"""Protocol, lifecycle and local HTTP boundary checks; no Odo required."""
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


def console():
    return app.Console(argparse.Namespace(ssh_host="odo", remote_root="/tmp/portal with spaces",
        time="00:10:00", account="trn047", reservation="", control_path=Path("/tmp/portal-test-socket")))


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
        self.assertIn("DEMO_USE_GPU = true", gpu)
        self.assertIn("DEMO_USE_GPU = false", cpu)
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
        self.assertEqual(layout[0], dict(id="cpu", title="CPU", number="01",
                                         demos=["01_tasks", "02_datadeps"]))
        self.assertEqual(layout[1]["demos"], ["03_gpu"])
        self.assertEqual(layout[1]["title"], "Single GPU")

    def test_exercises_cover_the_hands_on_demos(self):
        exercises = app.exercises()
        self.assertEqual(set(exercises),
                         {"02_datadeps", "03_gpu", "04_mpi", "05_multigpu"})
        for demo, exercise in exercises.items():
            self.assertIn(demo, {entry[0] for entry in app.DEMOS})
            self.assertIn("Exercise:", exercise["prompt"])
            self.assertIn("my_cholesky.jl", exercise["prompt"])
            self.assertIn(exercise["target"], ("cpu", "gpu", "mixed"))
            self.assertIn(exercise["gpus"], ("1", "all"))

    def test_exercise_template_defines_the_tile_operations(self):
        template = app.exercise_template()
        self.assertIn("function my_cholesky!", template)
        for name, kernel in (("factor_diagonal_tile!", "LAPACK.potrf!"),
                             ("solve_panel_tile!", "BLAS.trsm!"),
                             ("update_diagonal_tile!", "BLAS.syrk!"),
                             ("update_offdiagonal_tile!", "BLAS.gemm!")):
            self.assertIn(f"const {name} = {kernel}", template)
        # They must stay aliases: Dagger selects the GPU kernel by function
        # identity, so a wrapper function would break the GPU run.
        self.assertNotIn("function factor_diagonal_tile!", template)

    def test_exercise_primer_is_an_incomplete_editable_template(self):
        primer = app.EXERCISE_PRIMER
        self.assertIn("DArray(A, ???)", primer)
        self.assertIn("my_cholesky!(???)", primer)
        self.assertIn("collect(???)", primer)
        self.assertIn("Dagger.with_options", primer)
        self.assertIn("Run exercise", primer)

    def test_console_generates_a_shared_exercise_editor(self):
        page = (BASE / "host/app.js").read_text()
        self.assertIn('data-role="exercise-code"', page)
        self.assertIn('data-role="exercise-primer"', page)
        self.assertIn('data-role="exercise-primer-reset"', page)
        self.assertIn("Setup and Use", page)
        self.assertIn(">Function<", page)
        self.assertIn('op: "exercise"', page)
        self.assertIn("exercise_template", page)
        self.assertIn("exercise_primer", page)
        self.assertIn("EXERCISE_PRIMER_KEY", page)
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
        self.assertIn("streamfunction-vorticity", final["prompt"])
        self.assertIn("A = ???", final["template"])
        self.assertIn("F = ???", final["template"])
        # The visualization code lives in the harness, not in the template.
        self.assertNotIn("UnicodePlots", final["template"])
        self.assertNotIn("heatmap(", final["template"])
        self.assertEqual([item["id"] for item in app.FINAL_VISUALS],
                         ["vorticity", "streamfunction"])

    def test_final_exercise_eval_is_allowed(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="eval", mode="single", demo="final", code="x = 1"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["label"], "Final Exercise")

    def test_final_exercise_visuals_are_whitelisted(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="eval", mode="single", demo="final", code="x = 1",
                   visuals=["vorticity", "streamfunction"]))
        command = c.send.call_args[0][0]
        self.assertEqual(command["visuals"], ["vorticity", "streamfunction"])
        c.busy = False
        with self.assertRaisesRegex(ValueError, "Unknown visualization"):
            c.run(dict(op="eval", mode="single", demo="final", code="x = 1",
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

    def test_exercise_run_uses_demo_layout_and_harness_defaults(self):
        c = console()
        c.send = Mock()
        c.run(dict(op="exercise", demo="02_datadeps", target="cpu",
                   strict=False, code="x = 1"))
        started = [event for event in c.events if event["kind"] == "run"][-1]
        self.assertTrue(started["exercise"])
        command = c.send.call_args[0][0]
        self.assertEqual(command["mode"], "single")
        self.assertEqual(command["target"], "cpu")
        self.assertEqual(command["sizes"], "1024")
        self.assertEqual(command["bs"], 256)
        self.assertEqual(command["gpus"], "1")
        self.assertEqual(command["label"], "Exercise · CPU")
        c.busy = False
        c.run(dict(op="exercise", demo="04_mpi", target="cpu",
                   strict=True, code="x = 1"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["mode"], "mpi")
        self.assertTrue(command["strict"])
        c.busy = False
        c.run(dict(op="exercise", demo="05_multigpu", target="gpu",
                   strict=False, code="x = 1"))
        command = c.send.call_args[0][0]
        self.assertEqual(command["gpus"], "all")
        self.assertEqual(command["bs"], 512)
        c.busy = False
        with self.assertRaisesRegex(ValueError, "no exercise"):
            c.run(dict(op="exercise", demo="01_tasks", code="x = 1"))
        with self.assertRaisesRegex(ValueError, "Write Julia code"):
            c.run(dict(op="exercise", demo="02_datadeps", code=" "))

    def test_config_exposes_exercises(self):
        c = console()
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.handler_for(c))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/config") as response:
                config = json.load(response)
            self.assertEqual(set(config["exercises"]),
                             {"02_datadeps", "03_gpu", "04_mpi", "05_multigpu"})
            self.assertEqual(config["exercises"]["02_datadeps"]["target"], "cpu")
            self.assertIn("function my_cholesky!", config["exercise_template"])
            self.assertIn("DArray(A, ???)", config["exercise_primer"])
            self.assertEqual(config["final_exercise"]["title"], "Final Exercise")
            self.assertIn("A = ???", config["final_exercise"]["template"])
            self.assertTrue(config["final_workspace"].endswith("final-exercise"))
        finally:
            server.shutdown()
            server.server_close()

    def test_run_tags_events_with_demo_and_clears_after_done(self):
        c = console()
        c.process = Mock(stdin=io.StringIO())
        c.process.poll.return_value = None
        c.job = "123"
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
        c.busy = False
        c.run(dict(op="eval", mode="single", code="1+1", demo="01_tasks"))
        evaluation = [event for event in c.events if event["kind"] == "run"][-1]
        self.assertEqual(evaluation["demo"], "01_tasks")
        c.busy = False
        with self.assertRaisesRegex(ValueError, "Unknown demo"):
            c.run(dict(op="eval", mode="single", code="1", demo="nope"))

    def test_ssh_config_username_replaces_the_local_user(self):
        self.assertEqual(app.default_remote_root("trn047", "godoywf"),
                         "/gpfs/wolf2/olcf/trn047/scratch/godoywf/sc26-dagger/portal")
        self.assertTrue(app.is_odo_hostname("login1.odo.olcf.ornl.gov"))
        self.assertTrue(app.is_odo_hostname("login2.odo.olcf.ornl.gov"))
        self.assertFalse(app.is_odo_hostname("odo"))

    def test_existing_working_alias_is_reused_without_rewriting_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            original = ("Host odo\n    HostName login1.odo.olcf.ornl.gov\n    User alice\n")
            config.write_text(original)
            args = argparse.Namespace(ssh_host=None, odo_user="alice",
                                      ssh_config=config, control_path=Path(directory)/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                user, alias = app.bootstrap_ssh(args)
            self.assertEqual((user, alias), ("alice", "odo"))
            self.assertEqual(config.read_text(), original)

    def test_managed_alias_is_added_first_and_leaves_other_hosts_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            original = ("# hand-written entry\nHost odo\n    HostName login1.odo.olcf.ornl.gov\n"
                        "    User bob\n\nHost *\n    User ghost\n")
            config.write_text(original)
            args = argparse.Namespace(ssh_host=None, odo_user="alice",
                                      ssh_config=config, control_path=Path(directory)/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                user, alias = app.bootstrap_ssh(args)
            self.assertEqual((user, alias), ("alice", "sc26-odo-alice"))
            text = config.read_text()
            self.assertIn(original.strip(), text)
            # First value wins for User/HostName, so the block must come first.
            self.assertLess(text.index(app.MANAGED_BEGIN), text.index("Host odo"))
            resolved = app.ssh_config_values(alias, config)
            self.assertEqual((resolved["user"], resolved["hostname"]), ("alice", app.ODO_HOSTNAME))
            self.assertEqual(app.write_managed_host(config, alias, "alice", args.control_path),
                             "unchanged")

    def test_foreign_alias_collision_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            config.write_text("Host sc26-odo-alice\n    HostName example.org\n")
            args = argparse.Namespace(ssh_host=None, odo_user="alice",
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
            args = argparse.Namespace(ssh_host=None, odo_user="alice", ssh_config=link,
                                      control_path=root/"socket")
            with contextlib.redirect_stdout(io.StringIO()):
                app.bootstrap_ssh(args)
            self.assertTrue(link.is_symlink())
            self.assertIn("Host sc26-odo-alice", target.read_text())
            self.assertIn("Host work", target.read_text())

    def test_command_layout_and_busy_guard(self):
        c = console()
        c.process = Mock(stdin=io.StringIO())
        c.process.poll.return_value = None
        c.job = "123"
        c.run(dict(op="demo", demo="07_multinode_multigpu"))
        command = json.loads(c.process.stdin.getvalue())
        self.assertEqual(command["mode"], "mpi")
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

    def test_agent_eof_cancels_only_its_own_job(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, code in {"sbatch":"echo 456", "scancel":'echo "$@" > "$CANCEL_LOG"', "squeue":r"echo RUNNING\|testnode"}.items():
                p = root / name
                p.write_text("#!/bin/sh\n"+code+"\n")
                p.chmod(0o755)
            env = dict(os.environ, PATH=directory+os.pathsep+os.environ["PATH"], CANCEL_LOG=str(root/"cancelled"))
            result = subprocess.run(["python3", str(BASE/"host/remote_agent.py")], input="", env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/"cancelled").read_text().strip(), "456")

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
                      "s.loader.exec_module(m); m.execute=lambda c: None; m.configure_stdio(); m.main()")
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
