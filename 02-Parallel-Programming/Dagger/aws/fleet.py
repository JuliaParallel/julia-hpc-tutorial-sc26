#!/usr/bin/env python3
"""Launch CPU or relay fallback VMs through ~/awsmcp; keep handles private.

`plan` is read-only. `launch` provisions only the requested count, records
handles immediately for recovery, and never exceeds the smaller of the
configured vCPU budget (256 by default) and the live AWS quota.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tarfile
import tempfile

import boto3
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def capacity(args):
    aws = boto3.Session(profile_name=args.profile)
    ec2 = aws.client("ec2")
    quota = aws.client("service-quotas").get_service_quota(
        ServiceCode="ec2", QuotaCode="L-1216C47A")["Quota"]["Value"]
    instances = [i for page in ec2.get_paginator("describe_instances").paginate(
        Filters=[dict(Name="instance-state-name", Values=["pending", "running"])])
        for r in page["Reservations"] for i in r["Instances"]
        if i["InstanceType"][0] in "acdhimrtz"]
    types = set(i["InstanceType"] for i in instances) | {args.instance_type}
    specs = {t["InstanceType"]: t for t in ec2.describe_instance_types(InstanceTypes=sorted(types))["InstanceTypes"]}
    used = sum(specs[i["InstanceType"]]["VCpuInfo"]["DefaultVCpus"] for i in instances)
    spec = specs[args.instance_type]
    cpus = spec["VCpuInfo"]["DefaultVCpus"]
    limit = min(args.vcpu_budget, int(quota))
    result = dict(region=aws.region_name, aws_quota=int(quota), budget=limit, used=used,
                  instance_type=args.instance_type, vcpus_each=cpus, count=args.count,
                  projected=used + cpus * args.count)
    print(json.dumps(result, indent=2), flush=True)
    if result["projected"] > limit:
        raise SystemExit("Requested fleet exceeds the vCPU budget; reduce --count")
    return cpus, spec["MemoryInfo"]["SizeInMiB"] // 1024


def read_state(path):
    return json.loads(path.read_text()) if path.exists() else {"vms": []}


def save_state(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    fd = os.open(temp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    os.chmod(temp, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    temp.replace(path)


@asynccontextmanager
async def session():
    directory = Path(os.environ.get("AWSMCP_DIR", str(Path.home() / "awsmcp")))
    async with stdio_client(StdioServerParameters(command=str(directory / "awsmcp"))) as (r, w):
        async with ClientSession(r, w) as client:
            await client.initialize()
            yield client


async def call(client, name, **kwargs):
    result = await client.call_tool(name, kwargs)
    blocks = [c.text for c in result.content if c.type == "text"]
    if result.is_error:
        raise RuntimeError("{}: {}".format(name, "\n".join(blocks)))
    data = json.loads(blocks[0])
    if data.get("ok") is False:
        raise RuntimeError("{}: {}".format(name, data.get("error", data)))
    return data


def archive(path):
    with tarfile.open(path, "w:gz") as tar:
        for name in ("Project.toml", "cpu", "host", "demos", "harness", "frontier", "solution"):
            tar.add(ROOT / name, arcname=name, filter=lambda i: None if (
                "__pycache__" in i.name or Path(i.name).name in
                ("Manifest.toml", "LocalPreferences.toml")) else i)


async def main(args):
    if args.action in ("plan", "launch"):
        cpus, memory = capacity(args)
        if args.action == "plan":
            return
    state = read_state(args.state)
    async with session() as client:
        if args.action == "launch":
            # Each participant VM is isolated; no shared cluster SSH key.
            new = []
            for _ in range(args.count):
                data = await call(client, "launch_vm", profile=args.profile, cpus=cpus,
                    memory=memory, instance_type=args.instance_type, disk=40,
                    timeout=args.timeout, label="sc26-fallback-" + args.role,
                    network="none", no_login=True, ssh_cidr=args.ssh_cidr)
                vm = dict(handle=data["handle"], role=args.role)
                state["vms"].append(vm)
                new.append(vm)
                save_state(args.state, state)
            # Keep this MCP session alive until all background provisioning finishes.
            pending = list(new)
            while pending:
                for vm in list(pending):
                    data = await call(client, "get_status", handle=vm["handle"])
                    if data["provision_state"] == "failed":
                        raise RuntimeError(data.get("provision_error", "provision failed"))
                    if data["provision_state"] == "ready":
                        vm.update(ip=data["public_ip"], instance_id=data["instance_id"], cpus=cpus)
                        pending.remove(vm)
                        save_state(args.state, state)
                        print("Ready: {} {}".format(vm["instance_id"], vm["ip"]), flush=True)
                if pending:
                    await asyncio.sleep(10)
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "sc26.tar.gz"
                archive(source)
                for vm in new:
                    await call(client, "upload", handle=vm["handle"], local=str(source), remote="/home/ubuntu/sc26.tar.gz")
                    setup = '''set -eu
mkdir -p /home/ubuntu/sc26
tar -xzf /home/ubuntu/sc26.tar.gz -C /home/ubuntu/sc26
rm /home/ubuntu/sc26.tar.gz
printf '%s\n' 'MaxStartups 100:30:200' 'MaxSessions 64' | sudo tee /etc/ssh/sshd_config.d/60-sc26.conf >/dev/null
sudo sshd -t
sudo systemctl reload ssh
'''
                    if args.role == "cpu":
                        setup += '''curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12
cd /home/ubuntu/sc26
bash cpu/prepare.sh
'''
                    data = await call(client, "exec_command", handle=vm["handle"], command=setup, background=True)
                    vm["setup_job"] = data["job_id"]
                    save_state(args.state, state)
                    print("Setup running: {} (use status to check it)".format(vm["ip"]), flush=True)
        elif args.action == "status":
            for vm in state["vms"]:
                if vm.get("terminated"):
                    continue
                data = await call(client, "get_status", handle=vm["handle"])
                print(json.dumps({k: data.get(k) for k in ("instance_id", "public_ip", "aws_state",
                    "provision_state", "remaining_seconds")}), flush=True)
                if vm.get("setup_job") and data.get("aws_state") == "running":
                    job = await call(client, "get_command_output", handle=vm["handle"], job_id=vm["setup_job"], tail=12)
                    print(json.dumps(job), flush=True)
        elif args.action == "terminate":
            for vm in state["vms"]:
                if vm.get("terminated"):
                    continue
                await call(client, "terminate_vm", handle=vm["handle"])
                vm["terminated"] = True
                save_state(args.state, state)
                print("Terminated: " + vm.get("instance_id", "provisioning VM"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "launch", "status", "terminate"))
    parser.add_argument("--profile", default="lab")
    parser.add_argument("--role", choices=("cpu", "relay"), default="cpu")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--instance-type", default="c7i.2xlarge")
    parser.add_argument("--vcpu-budget", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--ssh-cidr", help="Participant/instructor source CIDR; default: instructor IP/32")
    parser.add_argument("--state", type=Path, default=ROOT / "dev/aws-fleet.json")
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.timeout <= 86400 or args.vcpu_budget < 1:
        parser.error("count and budget must be positive; timeout must be 1..86400 seconds")
    asyncio.run(main(args))
