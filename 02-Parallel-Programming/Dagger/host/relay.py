#!/usr/bin/env python3
"""Multiplex byte streams inside ONE SSH session, including with MaxSessions=1.

Run `serve` on the instructor's relay VM and `client` through SSH from each
console. The private Unix socket grants access as the upstream SSH account.
Only the instructor authenticates to Odo; participants authenticate to AWS.
Python 3.6+; standard library only, including on the Odo login node.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import uuid

CHUNK = 48 * 1024


def frame(stream, value):
    data = memoryview((json.dumps(value) + "\n").encode())
    while data:
        written = stream.write(data)
        if not written:
            raise OSError("Relay stream closed during write")
        data = data[written:]
    stream.flush()


def terminate(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def remote():
    """One login-node process owns independent command process groups."""
    for fd in (0, 1, 2):
        os.set_blocking(fd, True)
    processes, lock, output_lock = {}, threading.RLock(), threading.Lock()

    def emit(value):
        with output_lock:
            frame(sys.stdout.buffer, value)

    def output(ident, proc):
        try:
            while True:
                data = proc.stdout.read1(CHUNK)
                if not data:
                    break
                emit(dict(id=ident, op="output", data=base64.b64encode(data).decode()))
            emit(dict(id=ident, op="exit", returncode=proc.wait()))
        finally:
            with lock:
                processes.pop(ident, None)
            proc.stdout.close()
            if not proc.stdin.closed:
                proc.stdin.close()

    emit(dict(id=None, op="ready"))
    try:
        for line in sys.stdin.buffer:
            msg = json.loads(line)
            ident, op = msg["id"], msg["op"]
            with lock:
                proc = processes.get(ident)
                if op == "open":
                    proc = subprocess.Popen(["bash", "-lc", msg["command"]],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    processes[ident] = proc
                    threading.Thread(target=output, args=(ident, proc), daemon=True).start()
                elif proc is not None:
                    if op == "close":
                        # A participant disconnect only cancels that participant.
                        threading.Thread(target=terminate, args=(proc,), daemon=True).start()
                    else:
                        try:
                            if op == "stdin":
                                proc.stdin.write(base64.b64decode(msg["data"]))
                                proc.stdin.flush()
                            elif op == "eof":
                                proc.stdin.close()
                        except (OSError, ValueError):
                            pass
    finally:
        with lock:
            remaining = list(processes.values())
        workers = [threading.Thread(target=terminate, args=(p,)) for p in remaining]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()


def serve(args):
    path = Path(args.socket).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("Relay socket already exists: {} (stop the old relay first)".format(path))
    # Bootstrap in the one session: no scp, SFTP, second login, or remote port.
    import shlex
    source = Path(__file__).read_text()
    command = "python3 -u -c {} remote".format(shlex.quote(source))
    upstream = ([sys.executable, "-u", __file__, "remote"] if args.local else
                ["ssh", "-T", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"] +
                (["-S", args.control_path, "-o", "ProxyCommand=false", "-o", "BatchMode=yes"]
                 if args.control_path else []) + [args.upstream, command])
    # Keep the controlling terminal so OpenSSH can prompt on /dev/tty while
    # its stdin/stdout carry the protocol. This also works with SSH_ASKPASS.
    proc = subprocess.Popen(upstream, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    clients, lock, write_lock = {}, threading.RLock(), threading.Lock()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Private parent AND private socket, even with a permissive caller umask.
    listener.bind(str(path))
    path.chmod(0o600)
    listener.listen(64)
    listener.settimeout(1)
    stopped = threading.Event()

    def send(value):
        with write_lock:
            frame(proc.stdin, value)

    def dispatch():
        try:
            for line in proc.stdout:
                msg = json.loads(line)
                if msg.get("op") == "ready":
                    print("Odo dispatcher ready", file=sys.stderr, flush=True)
                    continue
                with lock:
                    target = clients.get(msg["id"])
                if target:
                    stream, stream_lock = target
                    try:
                        with stream_lock:
                            frame(stream, msg)
                    except OSError:
                        pass
        finally:
            stopped.set()
            with lock:
                for stream, stream_lock in list(clients.values()):
                    try:
                        with stream_lock:
                            frame(stream, dict(op="exit", returncode=255))
                    except OSError:
                        pass

    def handle(conn):
        ident = uuid.uuid4().hex
        stream = conn.makefile("rwb")
        try:
            msg = json.loads(stream.readline())
            with lock:
                clients[ident] = (stream, threading.Lock())
            send(dict(id=ident, op="open", command=msg["command"]))
            for line in stream:
                msg = json.loads(line)
                if msg.get("op") not in ("stdin", "eof"):
                    raise ValueError("Unknown relay operation")
                msg["id"] = ident
                send(msg)
        except (OSError, ValueError):
            pass
        finally:
            with lock:
                clients.pop(ident, None)
            try:
                send(dict(id=ident, op="close"))
            except (OSError, ValueError):
                pass
            stream.close()
            conn.close()

    threading.Thread(target=dispatch, daemon=True).start()
    print("Relay listening on {} (one upstream SSH session)".format(path), file=sys.stderr, flush=True)
    try:
        while not stopped.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    finally:
        listener.close()
        path.unlink()
        # EOF lets the remote dispatcher cancel all its children cleanly.
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def client(args):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(os.path.expanduser(args.socket))
    stream = conn.makefile("rwb")
    frame(stream, dict(command=args.command))

    def input_stream():
        try:
            while True:
                data = sys.stdin.buffer.read1(CHUNK)
                if not data:
                    frame(stream, dict(op="eof"))
                    return
                frame(stream, dict(op="stdin", data=base64.b64encode(data).decode()))
        except (OSError, ValueError):
            pass

    threading.Thread(target=input_stream, daemon=True).start()
    try:
        for line in stream:
            msg = json.loads(line)
            if msg["op"] == "exit":
                return msg["returncode"]
            sys.stdout.buffer.write(base64.b64decode(msg["data"]))
            sys.stdout.buffer.flush()
        return 255
    finally:
        conn.shutdown(socket.SHUT_RDWR)
        stream.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode")
    server = sub.add_parser("serve")
    server.add_argument("--upstream", default="odo")
    server.add_argument("--control-path")
    server.add_argument("--local", action="store_true", help="Test with a local dispatcher")
    server.add_argument("--socket", default="~/.sc26-relay/socket")
    participant = sub.add_parser("client")
    participant.add_argument("--socket", default="~/.sc26-relay/socket")
    participant.add_argument("--command", required=True)
    sub.add_parser("remote")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if args.mode == "remote":
        remote()
    elif args.mode == "serve":
        serve(args)
    elif args.mode == "client":
        sys.exit(client(args))
    else:
        parser.error("Choose serve, client, or remote")


if __name__ == "__main__":
    main()
