#!/usr/bin/env python3
"""Shared-password workshop gateway. Only `serve` imports AWS/MCP packages.

sshd runs `attach` as the unprivileged workshop account. It can select only
cpu or odo; commands are executed on a worker or on Odo, never on this VM.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
import relay


def attach(path):
    backend = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if backend not in ("cpu", "odo"):
        raise SystemExit("Use the tutorial console to choose cpu or odo; shell access is disabled.")
    conn = socket.socket(socket.AF_UNIX)
    deadline = time.monotonic() + 10
    while True:
        try:
            conn.connect(path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                raise SystemExit("The workshop gateway is starting; reconnect shortly.")
            time.sleep(.2)
    conn.sendall((json.dumps(dict(backend=backend)) + "\n").encode())

    def incoming():
        try:
            while True:
                data = sys.stdin.buffer.read1(65536)
                if not data:
                    conn.shutdown(socket.SHUT_WR)
                    return
                conn.sendall(data)
        except OSError:
            pass
    threading.Thread(target=incoming, daemon=True).start()
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    finally:
        conn.close()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(tmp, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, indent=2)
    tmp.replace(path)


async def expire_session(config, disconnected, activity, status, clock=time.monotonic, interval=1):
    began = clock()
    while not disconnected.is_set():
        await asyncio.sleep(interval)
        now = clock()
        if now - began >= config["max_lifetime"] or now - activity[0] >= config["idle_timeout"]:
            try:
                await status("Workshop session timed out; reconnect to start a new session")
            finally:
                disconnected.set()


class Allocator:
    def __init__(self, config, client):
        import boto3
        self.config, self.client = config, client
        self.aws = boto3.Session(profile_name=config.get("profile") or None, region_name=config["region"])
        self.ec2 = self.aws.client("ec2")
        self.leases, self.queue = {}, []
        self.lock = asyncio.Lock()
        self.state = Path(config["state"])
        self.prefix = "sc26-gateway-worker-" + config["owner"] + "-"
        self.limit = min(config["vcpu_budget"], int(self.aws.client("service-quotas").get_service_quota(
            ServiceCode="ec2", QuotaCode="L-1216C47A")["Quota"]["Value"]))
        self.types = {}

    def persist(self):
        save(self.state, self.leases)

    def usage(self):
        instances = [i for page in self.ec2.get_paginator("describe_instances").paginate(
            Filters=[dict(Name="instance-state-name", Values=["pending", "running"])])
            for r in page["Reservations"] for i in r["Instances"] if i["InstanceType"][0] in "acdhimrtz"]
        missing = {i["InstanceType"] for i in instances} - self.types.keys()
        if missing:
            self.types.update({t["InstanceType"]: t["VCpuInfo"]["DefaultVCpus"] for t in
                self.ec2.describe_instance_types(InstanceTypes=sorted(missing))["InstanceTypes"]})
        used = sum(self.types[i["InstanceType"]] for i in instances)
        visible = {next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "vmbench-label"), "") for i in instances}
        # Count admitted launches that EC2 has not made visible yet as well.
        reserved = sum(self.config["threads"] for lease in self.leases.values()
                       if lease.get("reserved") and lease["label"] not in visible)
        return used + reserved

    async def rpc(self, name, **kwargs):
        from fleet import call
        return await call(self.client, name, **kwargs)

    async def acquire(self, ident, disconnected, status):
        lease = dict(label=self.prefix + ident, reserved=False, created=time.time(), phase="queued")
        self.leases[ident] = lease
        self.queue.append(ident)
        self.persist()
        last = 0
        while True:
            if disconnected.is_set():
                raise ConnectionError("Disconnected while waiting for a CPU VM")
            async with self.lock:
                if self.queue[0] == ident and await asyncio.to_thread(self.usage) + self.config["threads"] <= self.limit:
                    lease.update(reserved=True, phase="launching")
                    self.queue.remove(ident)
                    self.persist()
                    break
            if time.monotonic() - last > 10:
                await status("Waiting for an available CPU VM slot…")
                last = time.monotonic()
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=3)
            except asyncio.TimeoutError:
                pass
        await status("Starting your CPU VM…")
        # An instructor can promote a prepared image without interrupting
        # existing attendees. Only this root-owned config supplies the AMI.
        latest = json.loads(Path(self.config.get("config_path", "/etc/sc26-gateway.json")).read_text())
        data = await self.rpc("launch_vm", profile=self.config.get("profile", ""),
            cpus=self.config["threads"], memory=16, instance_type=self.config["instance_type"],
            ami=latest.get("ami", self.config.get("ami")), disk=40, timeout=self.config["max_lifetime"],
            label=lease["label"], network="none", no_login=True,
            ssh_cidr=self.config["public_ip"] + "/32")
        lease["handle"] = data["handle"]
        self.persist()
        # Do not cancel an in-flight MCP launch: first obtain its handle, then
        # release it if the attendee disappeared during provisioning.
        while not disconnected.is_set():
            data = await self.rpc("get_status", handle=lease["handle"])
            if data.get("instance_id"):
                lease["instance_id"] = data["instance_id"]
                self.persist()
            if data["provision_state"] == "failed":
                raise RuntimeError(data.get("provision_error", "CPU VM failed to start"))
            if data["provision_state"] == "ready":
                break
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        if disconnected.is_set():
            raise ConnectionError("Disconnected during CPU VM startup")
        lease.update(phase="preparing", instance_id=data["instance_id"])
        self.persist()
        await status("Checking the prepared Julia environment…" if latest.get("ami") else
                     "Preparing Julia on your CPU VM (first launch takes a few minutes)…")
        await self.rpc("upload", handle=lease["handle"], local=self.config["archive"], remote="/home/ubuntu/sc26.tar.gz")
        script = '''set -eu
mkdir -p /home/ubuntu/sc26
tar -xzf /home/ubuntu/sc26.tar.gz -C /home/ubuntu/sc26
rm /home/ubuntu/sc26.tar.gz
if [ ! -x /home/ubuntu/.juliaup/bin/julia ]; then
    curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12
fi
cd /home/ubuntu/sc26
bash cpu/prepare.sh
'''
        job = await self.rpc("exec_command", handle=lease["handle"], command=script, background=True)
        while not disconnected.is_set():
            result = await self.rpc("get_command_output", handle=lease["handle"], job_id=job["job_id"], tail=8)
            if not result.get("running"):
                if result.get("exit_code") != 0:
                    raise RuntimeError("CPU package setup failed; the instructor can inspect the coordinator log")
                break
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        if disconnected.is_set():
            raise ConnectionError("Disconnected during package preparation")
        from awsmcp.store import VMStore
        record = VMStore().get(lease["handle"])
        lease.update(phase="running", sg_id=record.sg_id)
        self.persist()
        import shlex
        command = "python3 -u -c {} remote".format(shlex.quote((ROOT / "host/relay.py").read_text()))
        return ["ssh", "-F", "/dev/null", "-T", "-i", record.key_path,
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "UserKnownHostsFile=" + str(Path(record.key_path).parent / "known_hosts"),
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                "ubuntu@" + record.ip, command]

    async def release(self, ident):
        lease = self.leases.get(ident)
        if not lease:
            return
        if ident in self.queue:
            self.queue.remove(ident)
        lease["phase"] = "releasing"
        self.persist()
        if lease.get("handle"):
            try:
                await self.rpc("terminate_vm", handle=lease["handle"])
            except Exception as exc:
                if "unknown handle" in str(exc).lower():
                    # A previous MCP process may have died before persisting
                    # its handle. Recover only this lease's tagged instances.
                    rows = await asyncio.to_thread(self.ec2.describe_instances, Filters=[
                        dict(Name="tag:vmbench-label", Values=[lease["label"]]),
                        dict(Name="instance-state-name", Values=["pending", "running"])])
                    ids = [i["InstanceId"] for r in rows["Reservations"] for i in r["Instances"]]
                    if ids:
                        await asyncio.to_thread(self.ec2.terminate_instances, InstanceIds=ids)
                    else:
                        self.leases.pop(ident, None)
                        self.persist()
                        return
                print("Cleanup retry needed for {}: {}".format(ident, exc), file=sys.stderr, flush=True)
                lease["phase"] = "cleanup_failed"
                self.persist()
                return
        self.leases.pop(ident, None)
        self.persist()

    async def recover(self):
        if self.state.exists():
            self.leases = json.loads(self.state.read_text())
            await asyncio.gather(*(self.release(i) for i in list(self.leases)))
        await self.reap()

    async def reap(self):
        # Recover launches whose handle could not be recorded before a crash.
        def orphaned():
            rows = self.ec2.describe_instances(Filters=[
                dict(Name="tag:vmbench-label", Values=[self.prefix + "*"]),
                dict(Name="instance-state-name", Values=["pending", "running"])])
            labels = {v["label"] for v in self.leases.values()}
            lost = [i["InstanceId"] for r in rows["Reservations"] for i in r["Instances"]
                    if next(t["Value"] for t in i["Tags"] if t["Key"] == "vmbench-label") not in labels]
            if lost:
                self.ec2.terminate_instances(InstanceIds=lost)
            groups = self.ec2.describe_security_groups(Filters=[dict(Name="tag:vmbench-label", Values=[self.prefix + "*"])])
            for group in groups["SecurityGroups"]:
                label = next((t["Value"] for t in group.get("Tags", []) if t["Key"] == "vmbench-label"), "")
                if label not in labels:
                    try:
                        self.ec2.delete_security_group(GroupId=group["GroupId"])
                    except self.ec2.exceptions.ClientError:
                        pass  # An instance is still terminating; retry next pass.
        await asyncio.to_thread(orphaned)


async def serve(config):
    from fleet import session
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)
    tasks = set()
    async with session() as client:
        allocator = Allocator(config, client)
        await allocator.recover()

        async def handle(reader, writer):
            task = asyncio.current_task()
            tasks.add(task)
            ident, proc = uuid.uuid4().hex, None
            disconnected = asyncio.Event()
            messages = asyncio.Queue(maxsize=128)
            activity = [time.monotonic()]
            read_task = output_task = forward_task = watch_task = None
            send_lock = asyncio.Lock()
            async def send(msg):
                async with send_lock:
                    writer.write((json.dumps(msg) + "\n").encode())
                    await asyncio.wait_for(writer.drain(), timeout=30)
            async def status(message):
                await send(dict(id=None, op="status", message=message))
            try:
                first = json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
                backend = first.get("backend")
                if backend not in ("cpu", "odo"):
                    raise ValueError("Choose CPU or Odo relay")
                async def read_client():
                    try:
                        while True:
                            line = await reader.readline()
                            if not line:
                                break
                            msg = json.loads(line)
                            if (msg.get("op") not in ("open", "stdin", "eof", "close", "pause", "resume") or
                                not isinstance(msg.get("id"), str) or len(msg["id"]) > 64):
                                raise ValueError("Invalid workshop stream operation")
                            activity[0] = time.monotonic()
                            await messages.put(line)
                    finally:
                        disconnected.set()
                read_task = asyncio.create_task(read_client())
                async def expired(message):
                    await send(dict(id=None, op="error", message=message))
                watch_task = asyncio.create_task(expire_session(config, disconnected, activity, expired))
                if backend == "cpu":
                    argv = await allocator.acquire(ident, disconnected, status)
                    meta = dict(backend="cpu", user="ubuntu", remote_root="/home/ubuntu/sc26-dagger/sessions",
                                threads=config["threads"])
                else:
                    if not Path(config["odo_socket"]).exists():
                        raise RuntimeError("The Odo relay is offline; choose AWS CPU or ask the instructor to connect Odo")
                    argv = [sys.executable, str(Path(__file__).resolve()), "mux-odo", "--socket", config["odo_socket"]]
                    meta = dict(backend="odo", user=config["odo_user"],
                        remote_root="/gpfs/wolf2/olcf/{}/scratch/{}/sc26-dagger/sessions".format(config["account"], config["odo_user"]))
                if disconnected.is_set():
                    raise ConnectionError("Workshop connection closed")
                proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=None, start_new_session=True, limit=131072)
                async def output():
                    try:
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            msg = json.loads(line)
                            if msg.get("op") == "ready":
                                msg.update(meta)
                            await send(msg)
                    finally:
                        disconnected.set()
                async def forward():
                    while True:
                        line = await messages.get()
                        proc.stdin.write(line)
                        await proc.stdin.drain()
                output_task = asyncio.create_task(output())
                forward_task = asyncio.create_task(forward())
                await disconnected.wait()
            except (Exception, asyncio.CancelledError) as exc:
                if not isinstance(exc, (ConnectionError, asyncio.CancelledError)):
                    print("Session {}: {}".format(ident, exc), file=sys.stderr, flush=True)
                    try:
                        await send(dict(id=None, op="error", message=str(exc)))
                    except (OSError, asyncio.TimeoutError):
                        pass
            finally:
                disconnected.set()
                for child in (read_task, output_task, forward_task, watch_task):
                    if child:
                        child.cancel()
                await asyncio.gather(*(t for t in (read_task, output_task, forward_task, watch_task) if t), return_exceptions=True)
                if proc and proc.returncode is None:
                    proc.stdin.close()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=25)
                    except asyncio.TimeoutError:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            os.killpg(proc.pid, signal.SIGKILL)
                            await proc.wait()
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, ConnectionError):
                    pass
                try:
                    await allocator.release(ident)
                finally:
                    tasks.discard(task)

        path = Path(config["socket"])
        if path.exists():
            path.unlink()  # systemd runs one coordinator; stale socket after a crash.
        server = await asyncio.start_unix_server(handle, path=str(path), limit=131072)
        os.chmod(path, 0o660)
        import grp
        os.chown(path, -1, grp.getgrnam("workshop").gr_gid)
        async def reaper():
            while not stopping.is_set():
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=30)
                except asyncio.TimeoutError:
                    try:
                        for ident, lease in list(allocator.leases.items()):
                            if lease["phase"] == "cleanup_failed":
                                await allocator.release(ident)
                        await allocator.reap()
                    except Exception as exc:
                        print("Cleanup pass: {}".format(exc), file=sys.stderr, flush=True)
        reaper_task = asyncio.create_task(reaper())
        print("Workshop gateway ready", flush=True)
        await stopping.wait()
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        await reaper_task
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "attach", "mux-odo"))
    parser.add_argument("--socket", default="/run/sc26-gateway/socket")
    parser.add_argument("--config", default="/etc/sc26-gateway.json")
    args = parser.parse_args()
    if args.mode == "attach":
        attach(args.socket)
    elif args.mode == "mux-odo":
        # Commands can reach only Odo's existing broker, never a local shell.
        relay.remote(lambda command: [sys.executable, str(ROOT / "host/relay.py"), "client",
                                       "--socket", args.socket, "--command", command])
    else:
        asyncio.run(serve(json.loads(Path(args.config).read_text())))
