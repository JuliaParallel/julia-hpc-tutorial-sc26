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
import queue
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


def remote(command_factory=None):
    """One login-node process owns independent command process groups."""
    for fd in (0, 1, 2):
        os.set_blocking(fd, True)
    processes, inputs, flowing, lock, output_lock = {}, {}, {}, threading.RLock(), threading.Lock()
    closing = set()

    def emit(value):
        with output_lock:
            frame(sys.stdout.buffer, value)

    def output(ident, proc):
        try:
            while True:
                flowing[ident].wait()
                data = proc.stdout.read1(CHUNK)
                if not data:
                    break
                emit(dict(id=ident, op="output", data=base64.b64encode(data).decode()))
            emit(dict(id=ident, op="exit", returncode=proc.wait()))
        finally:
            terminate(proc)
            with lock:
                processes.pop(ident, None)
                inputs.pop(ident, None)
                flowing.pop(ident, None)
                closing.discard(ident)
            proc.stdout.close()
            if not proc.stdin.closed:
                proc.stdin.close()

    def input_stream(proc, messages):
        # A coordinator compiling packages (or a stalled command) must not
        # block the dispatcher from reading other attendees' commands.
        while proc.poll() is None:
            try:
                msg = messages.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if msg["op"] == "eof":
                    proc.stdin.close()
                    return
                proc.stdin.write(base64.b64decode(msg["data"]))
                proc.stdin.flush()
            except (OSError, ValueError):
                return

    emit(dict(id=None, op="ready"))
    try:
        for line in sys.stdin.buffer:
            msg = json.loads(line)
            ident, op = msg["id"], msg["op"]
            with lock:
                proc = processes.get(ident)
                if op == "open":
                    if proc is not None:
                        raise ValueError("Duplicate relay stream ID")
                    argv = command_factory(msg["command"]) if command_factory else ["bash", "-lc", msg["command"]]
                    proc = subprocess.Popen(argv,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    processes[ident] = proc
                    inputs[ident] = messages = queue.Queue(maxsize=64)
                    flowing[ident] = threading.Event()
                    flowing[ident].set()
                    threading.Thread(target=output, args=(ident, proc), daemon=True).start()
                    threading.Thread(target=input_stream, args=(proc, messages), daemon=True).start()
                elif proc is not None:
                    if ident in closing:
                        continue
                    if op in ("pause", "resume"):
                        (flowing[ident].clear if op == "pause" else flowing[ident].set)()
                    elif op == "close":
                        # A participant disconnect only cancels that participant.
                        flowing[ident].set()
                        closing.add(ident)
                        threading.Thread(target=terminate, args=(proc,), daemon=True).start()
                    else:
                        try:
                            inputs[ident].put_nowait(msg)
                        except queue.Full:
                            flowing[ident].set()
                            closing.add(ident)
                            threading.Thread(target=terminate, args=(proc,), daemon=True).start()
    finally:
        with lock:
            remaining = list(processes.values())
            for event in flowing.values():
                event.set()
        workers = [threading.Thread(target=terminate, args=(p,)) for p in remaining]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()


def serve(args, upstream_command=None, env=None, on_ready=None, on_status=None, stop_event=None, stderr=None):
    path = Path(args.socket).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit("Relay socket already exists: {} (stop the old relay first)".format(path))
    # Bootstrap in the one session: no scp, SFTP, second login, or remote port.
    import shlex
    source = Path(__file__).read_text()
    command = "python3 -u -c {} remote".format(shlex.quote(source))
    upstream = upstream_command or ([sys.executable, "-u", __file__, "remote"] if args.local else
                ["ssh", "-T", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"] +
                (["-S", args.control_path, "-o", "ProxyCommand=false", "-o", "BatchMode=yes"]
                 if args.control_path else []) + [args.upstream, command])
    # Keep the controlling terminal so OpenSSH can prompt on /dev/tty while
    # its stdin/stdout carry the protocol. This also works with SSH_ASKPASS.
    proc = subprocess.Popen(upstream, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, stderr=stderr)
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
                    if on_ready:
                        on_ready(msg)
                    else:
                        print("Odo dispatcher ready", file=sys.stderr, flush=True)
                    continue
                if msg.get("op") in ("status", "error"):
                    if on_status:
                        on_status(msg)
                    continue
                with lock:
                    target = clients.get(msg["id"])
                if target:
                    outgoing, conn, paused, flow_lock = target
                    try:
                        outgoing.put_nowait(msg)
                        if outgoing.qsize() >= 32:
                            with flow_lock:
                                if not paused.is_set():
                                    paused.set()
                                    send(dict(id=msg["id"], op="pause"))
                    except queue.Full:
                        # Bound buffering and drop only the slow participant.
                        try:
                            conn.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
        finally:
            stopped.set()
            with lock:
                for outgoing, conn, _, _ in list(clients.values()):
                    try:
                        outgoing.put_nowait(dict(op="exit", returncode=255))
                    except queue.Full:
                        try:
                            conn.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

    def handle(conn):
        ident = uuid.uuid4().hex
        stream = conn.makefile("rwb")
        outgoing, finished = queue.Queue(maxsize=256), threading.Event()
        paused, flow_lock = threading.Event(), threading.Lock()
        def write_client():
            try:
                while not finished.is_set():
                    try:
                        message = outgoing.get(timeout=1)
                    except queue.Empty:
                        continue
                    frame(stream, message)
                    if outgoing.qsize() <= 16:
                        with flow_lock:
                            if paused.is_set():
                                paused.clear()
                                send(dict(id=ident, op="resume"))
            except (OSError, ValueError):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        writer = threading.Thread(target=write_client, daemon=True)
        writer.start()
        try:
            msg = json.loads(stream.readline())
            with lock:
                clients[ident] = (outgoing, conn, paused, flow_lock)
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
            finished.set()
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            writer.join(timeout=2)
            stream.close()
            conn.close()

    dispatch_thread = threading.Thread(target=dispatch, daemon=True)
    dispatch_thread.start()
    print("Relay listening on {} (one upstream SSH session)".format(path), file=sys.stderr, flush=True)
    try:
        while not stopped.is_set() and not (stop_event and stop_event.is_set()):
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
        dispatch_thread.join(timeout=2)
        proc.stdout.close()


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
