#!/usr/bin/env python3
"""Local tutorial console. Python 3.10+, OpenSSH; no Python packages required."""
import argparse
import atexit
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
import socket
import subprocess
import tarfile
import threading
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


def snapshot():
    """Upload course code only; results, credentials and preferences stay put."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ("Project.toml", "demos", "harness", "frontier", "solution"):
            archive.add(ROOT / name, arcname=name,
                        filter=lambda item: None if "__pycache__" in item.name else item)
        for name in ("worker.jl", "remote_agent.py"):
            archive.add(ROOT / "host" / name, arcname="host/" + name)
    return stream.getvalue()


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
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
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
            check = subprocess.run(["ssh", "-S", str(socket), "-O", "check", self.args.ssh_host],
                                   capture_output=True)
            if check.returncode:
                helper = shutil.which("odo-connect") if self.args.ssh_host == "odo" else None
                cmd = [helper, str(socket)] if helper else ["ssh", "-o", "BatchMode=yes",
                       "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto", "-o", "ControlPersist=8h",
                       "-S", str(socket), "-MNf", self.args.ssh_host]
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
                    self.emit("output", text=line)
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
            else:
                self.emit("output", text=line)
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
            elif command.get("op") == "eval":
                if command.get("mode") not in ("single", "mpi"):
                    raise ValueError("Unknown REPL layout")
                if not isinstance(command.get("code"), str) or not command["code"].strip():
                    raise ValueError("Enter Julia code")
            else:
                raise ValueError("Unknown operation")
            command["id"] = secrets.token_hex(8)
            self.send(command)
            self.busy = True
            self.status = "Running…"
            self.run_id = command["id"]
            self.emit("run", run=command["id"], label=command.get("demo", "REPL"),
                      mode=command["mode"], args=command.get("args", []))

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
                self.reply(dict(token=console.token, demos=DEMOS, host=console.args.ssh_host,
                                remote_root=console.args.remote_root, time=console.args.time))
            elif url.path in ("/", "/app.js", "/ansi.js", "/style.css"):
                file = {"/":"index.html", "/app.js":"app.js", "/ansi.js":"ansi.js", "/style.css":"style.css"}[url.path]
                content_type = {"/":"text/html; charset=utf-8", "/app.js":"text/javascript; charset=utf-8",
                                "/ansi.js":"text/javascript; charset=utf-8", "/style.css":"text/css; charset=utf-8"}[url.path]
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
                else:
                    self.reply(dict(error="Not found"), code=404)
                    return
                self.reply(dict(ok=True))
            except (ValueError, OSError) as exc:
                self.reply(dict(error=str(exc)), code=400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: all IPv4 interfaces)")
    parser.add_argument("--allow-host", action="append", default=[], help="Additional allowed browser hostname")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--ssh-host", default="odo")
    parser.add_argument("--account", default=os.environ.get("SC26_ACCOUNT", "trn047"))
    parser.add_argument("--remote-root", help="Dedicated portal directory, not an existing checkout")
    parser.add_argument("--control-path", type=Path, default=Path.home() / ".ssh/control/sc26-portal")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--reservation", default="")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w@.-]+", args.ssh_host) or args.ssh_host.startswith("-"):
        parser.error("Invalid SSH host alias")
    args.control_path = args.control_path.expanduser().resolve()
    args.remote_root = args.remote_root or f"/gpfs/wolf2/olcf/{args.account}/scratch/{getpass.getuser()}/sc26-dagger/portal"
    if not args.remote_root.startswith("/"):
        parser.error("--remote-root must be an absolute path")
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
    url = f"http://127.0.0.1:{args.port}"
    print(f"Tutorial console: {url}\nOdo workspace: {args.remote_root}\nCtrl-C releases the console's allocation.", flush=True)
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
