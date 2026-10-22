"""Protocol, lifecycle and local HTTP boundary checks; no Odo required."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("console", BASE / "host/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def console():
    return app.Console(argparse.Namespace(ssh_host="odo", remote_root="/tmp/portal with spaces",
        time="00:10:00", account="trn047", reservation="", control_path=Path("/tmp/portal-test-socket")))


class ConsoleTests(unittest.TestCase):
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
