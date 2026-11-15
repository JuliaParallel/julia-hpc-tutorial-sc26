#!/usr/bin/env python3
"""Local, standard-library SSH transport for the shared-password gateway."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time


def descriptor(value, key=None, port=22):
    path = Path(value).expanduser()
    data = json.loads(path.read_text()) if path.is_file() else dict(host=value, port=port)
    host = data.get("host", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Invalid workshop gateway hostname")
    data.setdefault("port", port)
    if not isinstance(data["port"], int) or not 1 <= data["port"] <= 65535:
        raise ValueError("Invalid workshop SSH port")
    data["host_key"] = key or data.get("host_key")
    if not data["host_key"] or not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+", data["host_key"]):
        raise ValueError("A pinned workshop SSH host key is required; use the instructor's workshop.json")
    return data


class Transport:
    def __init__(self, gateway, backend, password, status):
        self.gateway, self.backend, self.status = gateway, backend, status
        self._password = password
        self.directory = tempfile.TemporaryDirectory(prefix="sc26-gateway-")
        self.root = Path(self.directory.name)
        self.socket = self.root / "commands"
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.metadata, self.error = None, None
        self.askpass_socket = socket.socket(socket.AF_UNIX)
        self.askpass_socket.bind(str(self.root / "password"))
        os.chmod(self.root / "password", 0o600)
        self.askpass_socket.listen(2)
        self.askpass_socket.settimeout(.5)
        self.thread = None
        self.close_lock = threading.Lock()
        self.closed = False

    def _credentials(self):
        while not self.stop_event.is_set() and not self.ready.is_set():
            try:
                conn, _ = self.askpass_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.sendall(self._password.encode("utf-8"))

    def start(self):
        # The password never enters argv, the environment, a file, or a log.
        helper = self.root / "askpass"
        helper.write_text("#!/bin/sh\nexec " + shlex.join([sys.executable, str(Path(__file__).resolve()),
                          "askpass", str(self.root / "password")]) + "\n")
        helper.chmod(0o700)
        host, port = self.gateway["host"], self.gateway["port"]
        known = self.root / "known_hosts"
        known.write_text("[{}]:{} {}\n{} {}\n".format(host, port, self.gateway["host_key"], host, self.gateway["host_key"]))
        known.chmod(0o600)
        env = dict(os.environ, SSH_ASKPASS=str(helper), SSH_ASKPASS_REQUIRE="force", DISPLAY="sc26:0")
        ssh = ["ssh", "-F", os.devnull, "-T", "-p", str(port),
               "-o", "BatchMode=no", "-o", "PreferredAuthentications=password",
               "-o", "PubkeyAuthentication=no", "-o", "NumberOfPasswordPrompts=1",
               "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(known),
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=3", "workshop@" + host, self.backend]
        relay_path = Path(__file__).with_name("relay.py")
        spec = importlib.util.spec_from_file_location("gateway_relay", relay_path)
        relay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relay)

        def on_ready(msg):
            self.metadata = msg
            self._password = ""
            self.ready.set()

        def on_status(msg):
            if msg["op"] == "error":
                self.error = msg.get("message", "Workshop connection failed")
            self.status(msg.get("message", "Connecting…"))

        def run():
            try:
                args = argparse.Namespace(socket=str(self.socket), local=False, upstream="", control_path=None)
                with open(self.root / "ssh-error", "wb") as error:
                    # relay.serve owns SSH's stdin/stdout; stderr is redirected
                    # through its own Popen option, without changing global fds.
                    relay.serve(args, upstream_command=ssh, env=env, on_ready=on_ready,
                                on_status=on_status, stop_event=self.stop_event, stderr=error)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._password = ""
                self.ready.set()

        threading.Thread(target=self._credentials, daemon=True).start()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        while not self.ready.wait(.25):
            if self.stop_event.is_set():
                raise RuntimeError("Workshop connection cancelled")
        self.askpass_socket.close()
        if not self.metadata:
            detail = (self.root / "ssh-error").read_text(errors="replace") if (self.root / "ssh-error").exists() else ""
            raise RuntimeError(self.error or detail.strip() or "Workshop SSH connection closed")
        return self.metadata

    def command(self, script):
        return [sys.executable, str(Path(__file__).with_name("relay.py")), "client", "--socket",
                str(self.socket), "--command", script]

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            self.closed = True
        self.stop_event.set()
        self._password = ""
        self.askpass_socket.close()
        if self.thread:
            self.thread.join(timeout=45)
        self.directory.cleanup()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "askpass":
        raise SystemExit(2)
    conn = socket.socket(socket.AF_UNIX)
    conn.connect(sys.argv[2])
    while True:
        data = conn.recv(4096)
        if not data:
            break
        sys.stdout.buffer.write(data)
    conn.close()
