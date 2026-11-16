#!/usr/bin/env python3
"""Deploy and manage the instructor's shared-password workshop gateway."""
import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tarfile
import tempfile
import time

import boto3
from botocore.exceptions import ClientError
from fleet import ROOT, archive, call, read_state, save_state, session


def install_role(aws, owner, instance):
    iam, ec2 = aws.client("iam"), aws.client("ec2")
    name = "sc26-workshop-" + owner
    trust = dict(Version="2012-10-17", Statement=[dict(Effect="Allow",
        Principal=dict(Service="ec2.amazonaws.com"), Action="sts:AssumeRole")])
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    region = aws.region_name
    account = aws.client("sts").get_caller_identity()["Account"]
    arn = "arn:aws:ec2:{}:{}:".format(region, account)
    worker = "sc26-gateway-worker-" + owner + "-*"
    policy = dict(Version="2012-10-17", Statement=[
        dict(Effect="Allow", Action=["ec2:Describe*", "servicequotas:GetServiceQuota", "ssm:GetParameter"],
             Resource="*", Condition={"StringEquals": {"aws:RequestedRegion": region}}),
        dict(Effect="Allow", Action="sts:GetCallerIdentity", Resource="*"),
        dict(Effect="Allow", Action="ec2:RunInstances", Resource=arn + "instance/*",
             Condition={"StringLike": {"aws:RequestTag/vmbench-label": worker},
                        "StringEquals": {"ec2:InstanceType": "c7i.2xlarge"}}),
        dict(Effect="Allow", Action="ec2:RunInstances", Resource=[
            "arn:aws:ec2:{}::image/*".format(region)] + [arn + r + "/*" for r in
                ("volume", "network-interface", "subnet", "security-group", "key-pair")]),
        dict(Effect="Allow", Action="ec2:CreateSecurityGroup", Resource="*",
             Condition={"StringEquals": {"aws:RequestedRegion": region}}),
        dict(Effect="Allow", Action="ec2:CreateTags", Resource=[arn + r + "/*" for r in
             ("instance", "volume", "security-group")],
             Condition={"StringLike": {"aws:RequestTag/vmbench-label": worker}}),
        dict(Effect="Allow", Action=["ec2:AuthorizeSecurityGroupIngress", "ec2:DeleteSecurityGroup", "ec2:TerminateInstances"],
             Resource="*", Condition={"StringLike": {"ec2:ResourceTag/vmbench-label": worker},
                                      "StringEquals": {"aws:RequestedRegion": region}}),
    ])
    iam.put_role_policy(RoleName=name, PolicyName="WorkshopWorkers", PolicyDocument=json.dumps(policy))
    try:
        iam.create_instance_profile(InstanceProfileName=name)
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    profile = iam.get_instance_profile(InstanceProfileName=name)["InstanceProfile"]
    if not profile["Roles"]:
        iam.add_role_to_instance_profile(InstanceProfileName=name, RoleName=name)
    existing = ec2.describe_iam_instance_profile_associations(Filters=[dict(Name="instance-id", Values=[instance])])["IamInstanceProfileAssociations"]
    if not existing:
        for attempt in range(12):
            try:
                ec2.associate_iam_instance_profile(InstanceId=instance, IamInstanceProfile=dict(Name=name))
                break
            except ClientError as exc:
                if attempt == 11 or exc.response["Error"]["Code"] not in ("InvalidParameterValue", "NoSuchEntity"):
                    raise
                time.sleep(5)
    ec2.modify_instance_metadata_options(InstanceId=instance, HttpTokens="required", HttpPutResponseHopLimit=1)
    return name


async def deploy(args):
    aws = boto3.Session(profile_name=args.profile)
    state = read_state(args.state)
    async with session() as client:
        if not state["vms"]:
            # The coordinator stays available until explicit cleanup. Workers
            # have both connection ownership and a three-hour watchdog.
            from fleet import capacity
            capacity(argparse.Namespace(profile=args.profile, instance_type="c7i.xlarge", vcpu_budget=256, count=1))
            vm = await call(client, "launch_vm", profile=args.profile, instance_type="c7i.xlarge", cpus=4,
                memory=8, disk=40, timeout=0, label="sc26-workshop-gateway", network="none", no_login=True)
            state["vms"].append(dict(handle=vm["handle"], role="gateway"))
            save_state(args.state, state)
        vm = state["vms"][0]
        while True:
            info = await call(client, "get_status", handle=vm["handle"])
            if info["provision_state"] == "failed":
                raise RuntimeError(info.get("provision_error"))
            if info["provision_state"] == "ready":
                break
            await asyncio.sleep(5)
        vm.update(ip=info["public_ip"], instance_id=info["instance_id"], cpus=4)
        state.setdefault("owner", secrets.token_hex(4))
        state.setdefault("password", secrets.token_urlsafe(18))
        state["role"] = await asyncio.to_thread(install_role, aws, state["owner"], vm["instance_id"])
        save_state(args.state, state)
        # A deployment created through fleet's older helper may have a timer.
        await call(client, "reset_timeout", handle=vm["handle"], seconds=0)
        instance = aws.client("ec2").describe_instances(InstanceIds=[vm["instance_id"]])["Reservations"][0]["Instances"][0]
        for group in instance["SecurityGroups"]:
            try:
                aws.client("ec2").authorize_security_group_ingress(GroupId=group["GroupId"], IpPermissions=[
                    dict(IpProtocol="tcp", FromPort=22, ToPort=22, IpRanges=[dict(CidrIp="0.0.0.0/0", Description="Workshop SSH gateway")])])
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                    raise
        config = dict(owner=state["owner"], region=aws.region_name, public_ip=vm["ip"],
            state="/var/lib/sc26-gateway/leases.json", archive="/var/lib/sc26-gateway/source.tar.gz",
            socket="/run/sc26-gateway/socket", odo_socket="/run/sc26-odo/socket",
            odo_user=args.odo_user, account="trn047", threads=8, instance_type="c7i.2xlarge",
            vcpu_budget=256, max_lifetime=10800, idle_timeout=1800, profile="",
            ami=state.get("ami"))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "sc26.tar.gz"
            archive(source)
            # fleet's attendee snapshot excludes instructor tooling; deployment includes it.
            with tarfile.open(source, "r:gz") as old:
                entries = [(m, old.extractfile(m).read() if m.isfile() else None) for m in old.getmembers()]
            import io
            with tarfile.open(source, "w:gz") as tar:
                for member, data in entries:
                    tar.addfile(member, io.BytesIO(data) if data is not None else None)
                tar.add(ROOT / "aws", arcname="aws", filter=lambda m: None if "__pycache__" in m.name else m)
            mcp_dir = Path(os.environ.get("AWSMCP_DIR", str(Path.home() / "awsmcp")))
            mcp_source = temp / "awsmcp.tar.gz"
            with tarfile.open(mcp_source, "w:gz") as tar:
                for name in ("src", "pyproject.toml", "uv.lock", "awsmcp", "vmbench.py"):
                    tar.add(mcp_dir / name, arcname=name, filter=lambda m: None if "__pycache__" in m.name else m)
            for file in (source, mcp_source):
                await call(client, "upload", handle=vm["handle"], local=str(file), remote="/home/ubuntu/" + file.name)
        # Store only a salted password hash in the remote setup command.
        password_hash = subprocess.check_output(["openssl", "passwd", "-6", "-stdin"],
                                                input=state["password"].encode()).decode().strip()
        await call(client, "write_file", handle=vm["handle"], path="/home/ubuntu/gateway-config.json", content=json.dumps(config))
        script = '''set -eu
sudo mkdir -p /opt/sc26 /opt/awsmcp /var/lib/sc26-gateway /run/sc26-odo
sudo tar -xzf /home/ubuntu/sc26.tar.gz -C /opt/sc26
sudo tar -xzf /home/ubuntu/awsmcp.tar.gz -C /opt/awsmcp
sudo cp /home/ubuntu/sc26.tar.gz /var/lib/sc26-gateway/source.tar.gz
sudo install -m600 /home/ubuntu/gateway-config.json /etc/sc26-gateway.json
sudo chmod 700 /var/lib/sc26-gateway /run/sc26-odo
id workshop >/dev/null 2>&1 || sudo useradd --create-home --shell /bin/sh workshop
printf '%s\\n' PASSWORD_HASH | sudo chpasswd -e
sudo -H bash -c 'command -v /root/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -H /root/.local/bin/uv sync --directory /opt/awsmcp --frozen
sudo mkdir -p /etc/systemd/system
cat > /home/ubuntu/sc26-gateway.service <<'EOF'
[Unit]
Description=SC26 workshop gateway and CPU allocator
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=root
Group=workshop
Environment=PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=AWSMCP_DIR=/opt/awsmcp
Environment=AWSMCP_HOME=/var/lib/sc26-gateway/awsmcp
Environment=AWS_DEFAULT_REGION=@REGION@
RuntimeDirectory=sc26-gateway
RuntimeDirectoryMode=0750
ExecStart=/opt/awsmcp/.venv/bin/python /opt/sc26/aws/coordinator.py serve
Restart=on-failure
RestartSec=5
TimeoutStopSec=180
[Install]
WantedBy=multi-user.target
EOF
sudo cp /home/ubuntu/sc26-gateway.service /etc/systemd/system/sc26-gateway.service
cat > /home/ubuntu/sc26-workshop.conf <<'EOF'
Match User workshop
    PasswordAuthentication yes
    KbdInteractiveAuthentication no
    PubkeyAuthentication no
    AuthenticationMethods password
    MaxSessions 1
    ForceCommand /usr/bin/python3 /opt/sc26/aws/coordinator.py attach
    DisableForwarding yes
    PermitTTY no
    PermitUserRC no
    ClientAliveInterval 15
    ClientAliveCountMax 3
Match all
EOF
sudo cp /home/ubuntu/sc26-workshop.conf /etc/ssh/sshd_config.d/20-sc26-workshop.conf
printf '%s\\n' 'MaxStartups 100:30:200' 'MaxSessions 64' | sudo tee /etc/ssh/sshd_config.d/60-sc26.conf >/dev/null
sudo sshd -t
sudo systemctl reload ssh
sudo systemctl daemon-reload
sudo systemctl enable sc26-gateway
sudo systemctl restart sc26-gateway
rm -f /home/ubuntu/gateway-config.json /home/ubuntu/sc26.tar.gz /home/ubuntu/awsmcp.tar.gz
'''.replace("PASSWORD_HASH", shlex.quote("workshop:" + password_hash)).replace("@REGION@", aws.region_name)
        result = await call(client, "exec_command", handle=vm["handle"], command=script, timeout=600)
        key_result = await call(client, "exec_command", handle=vm["handle"], command="cat /etc/ssh/ssh_host_ed25519_key.pub")
        key = " ".join(key_result["stdout"].split()[:2])
        public = dict(host=vm["ip"], port=22, host_key=key)
        (ROOT / "host/workshop.json").write_text(json.dumps(public, indent=2) + "\n")
        access = args.state.parent / "gateway-access.txt"
        fd = os.open(access, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("Workshop gateway: {}\nPassword: {}\n\nAttendee command:\npython3 host/app.py --gateway\n".format(vm["ip"], state["password"]))
        print("Gateway deployed at {}. Password handout: {}".format(vm["ip"], access))


async def manage(args):
    state = read_state(args.state)
    if not state["vms"]:
        raise SystemExit("No gateway deployment in " + str(args.state))
    vm = state["vms"][0]
    async with session() as client:
        if args.action == "status":
            data = await call(client, "exec_command", handle=vm["handle"], command=
                "sudo systemctl is-active sc26-gateway sc26-odo; sudo cat /var/lib/sc26-gateway/leases.json; sudo journalctl -u sc26-gateway -n 8 --no-pager")
            print(data["stdout"])
        elif args.action == "cleanup":
            await call(client, "exec_command", handle=vm["handle"], command="sudo systemctl stop sc26-gateway", timeout=200)
            await call(client, "terminate_vm", handle=vm["handle"])
            aws = boto3.Session(profile_name=args.profile)
            if state.get("role"):
                iam, name = aws.client("iam"), state["role"]
                iam.remove_role_from_instance_profile(InstanceProfileName=name, RoleName=name)
                iam.delete_instance_profile(InstanceProfileName=name)
                iam.delete_role_policy(RoleName=name, PolicyName="WorkshopWorkers")
                iam.delete_role(RoleName=name)
            vm["terminated"] = True
            if state.get("ami"):
                ec2 = aws.client("ec2")
                ec2.deregister_image(ImageId=state["ami"])
                for snapshot in state.get("snapshots", []):
                    ec2.delete_snapshot(SnapshotId=snapshot)
            save_state(args.state, state)
            print("Gateway and its IAM role removed")


async def bake(args):
    """Build a clean CPU image once; worker connections then reuse its depot."""
    state = read_state(args.state)
    if state.get("ami"):
        raise SystemExit("A prepared image is already recorded: " + state["ami"])
    aws = boto3.Session(profile_name=args.profile)
    ec2 = aws.client("ec2")
    async with session() as client:
        vm = state["vms"][0]
        data = await call(client, "launch_vm", profile=args.profile, cpus=8, memory=16,
            instance_type="c7i.2xlarge", disk=40, timeout=3600, network="none", no_login=True,
            label="sc26-gateway-image-" + state["owner"])
        state["image_builder"] = data["handle"]
        save_state(args.state, state)
        try:
            while True:
                info = await call(client, "get_status", handle=data["handle"])
                if info["provision_state"] == "ready":
                    break
                if info["provision_state"] == "failed":
                    raise RuntimeError(info.get("provision_error"))
                await asyncio.sleep(5)
            print("Preparing the CPU image…", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "sc26.tar.gz"
                archive(source)
                await call(client, "upload", handle=data["handle"], local=str(source), remote="/home/ubuntu/sc26.tar.gz")
            job = await call(client, "exec_command", handle=data["handle"], background=True, command='''set -eu
mkdir -p /home/ubuntu/sc26
tar -xzf /home/ubuntu/sc26.tar.gz -C /home/ubuntu/sc26
rm /home/ubuntu/sc26.tar.gz
curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12
cd /home/ubuntu/sc26
bash cpu/prepare.sh
''')
            while True:
                result = await call(client, "get_command_output", handle=data["handle"], job_id=job["job_id"], tail=5)
                if not result["running"]:
                    if result.get("exit_code") != 0:
                        raise RuntimeError("Image preparation failed: " + result.get("stdout", ""))
                    break
                await asyncio.sleep(10)
            # Remove every launch key, log and machine identity BEFORE taking
            # a stopped-instance image. Cloud-init installs fresh keys on boot.
            await call(client, "exec_command", handle=data["handle"], command='''set -eu
rm -f /home/ubuntu/.ssh/authorized_keys /home/ubuntu/.ssh/known_hosts
sudo cloud-init clean --logs --machine-id
sudo rm -rf /home/ubuntu/work/*
''')
            iid = info["instance_id"]
            ec2.stop_instances(InstanceIds=[iid])
            await asyncio.to_thread(ec2.get_waiter("instance_stopped").wait, InstanceIds=[iid])
            image = ec2.create_image(InstanceId=iid, Name="sc26-cpu-{}-{}".format(state["owner"], int(time.time())),
                Description="SC26 prepared Julia CPU worker; no launch keys or AWS credentials",
                NoReboot=True, TagSpecifications=[dict(ResourceType="image", Tags=[dict(Key="sc26-gateway", Value=state["owner"])])])
            state["ami"] = image["ImageId"]
            save_state(args.state, state)

            print("Waiting for prepared image " + state["ami"], flush=True)
            await asyncio.to_thread(ec2.get_waiter("image_available").wait, ImageIds=[state["ami"]])
            spec = ec2.describe_images(ImageIds=[state["ami"]])["Images"][0]
            state["snapshots"] = [m["Ebs"]["SnapshotId"] for m in spec["BlockDeviceMappings"] if "Ebs" in m]
            save_state(args.state, state)
            current = await call(client, "exec_command", handle=vm["handle"], command="sudo cat /etc/sc26-gateway.json")
            config = json.loads(current["stdout"])
            config["ami"] = state["ami"]
            await call(client, "write_file", handle=vm["handle"], path="/home/ubuntu/gateway-image.json", content=json.dumps(config))
            await call(client, "exec_command", handle=vm["handle"], command="sudo install -m600 /home/ubuntu/gateway-image.json /etc/sc26-gateway.json.next && sudo mv /etc/sc26-gateway.json.next /etc/sc26-gateway.json && rm /home/ubuntu/gateway-image.json")
            print("Prepared image promoted: " + state["ami"], flush=True)
        finally:
            await call(client, "terminate_vm", handle=data["handle"])
            state.pop("image_builder", None)
            save_state(args.state, state)


async def connect_odo(args):
    state = read_state(args.state)
    vm = state["vms"][0]
    async with session() as client:
        probe = "sudo -H /usr/bin/python3 /opt/sc26/host/relay.py client --socket /run/sc26-odo/socket --command true"
        try:
            await call(client, "exec_command", handle=vm["handle"], command=probe, timeout=15)
            print("Odo relay is already connected")
            return
        except RuntimeError:
            pass
        config = await call(client, "exec_command", handle=vm["handle"], command="sudo cat /etc/sc26-gateway.json")
        user = json.loads(config["stdout"])["odo_user"]
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
            raise ValueError("Invalid Odo username")
        known = subprocess.check_output(["ssh-keygen", "-F", "login1.odo.olcf.ornl.gov"])
        await call(client, "write_file", handle=vm["handle"], path="/home/ubuntu/odo-known-hosts", content=known.decode())
        helper = '''#!/bin/sh
case "$1" in
 *[Pp]assword*) cat /run/sc26-odo/password;;
 *) exit 1;;
esac
'''
        await call(client, "write_file", handle=vm["handle"], path="/home/ubuntu/odo-askpass", content=helper)
        service = '''[Unit]
Description=SC26 single-connection Odo relay
After=network-online.target
[Service]
Type=simple
User=root
Environment=DISPLAY=sc26:0
Environment=SSH_ASKPASS=/run/sc26-odo/askpass
Environment=SSH_ASKPASS_REQUIRE=force
ExecStart=/usr/bin/python3 /opt/sc26/host/relay.py serve --socket /run/sc26-odo/socket --upstream USER@login1.odo.olcf.ornl.gov
TimeoutStopSec=40
'''.replace("USER@login", user + "@login")
        await call(client, "write_file", handle=vm["handle"], path="/home/ubuntu/sc26-odo.service", content=service)
        await call(client, "exec_command", handle=vm["handle"], command='''set -eu
sudo systemctl stop sc26-odo 2>/dev/null || true
sudo install -d -m700 /run/sc26-odo /root/.ssh
sudo rm -f /run/sc26-odo/socket
sudo sh -c 'cat /home/ubuntu/odo-known-hosts >> /root/.ssh/known_hosts'
sudo install -m700 /home/ubuntu/odo-askpass /run/sc26-odo/askpass
sudo install -m644 /home/ubuntu/sc26-odo.service /etc/systemd/system/sc26-odo.service
sudo systemctl daemon-reload
rm -f /home/ubuntu/odo-known-hosts /home/ubuntu/odo-askpass /home/ubuntu/sc26-odo.service
''')
        password = (subprocess.check_output([str(args.odo_askpass.expanduser()), "Password:"]).decode().rstrip("\r\n")
                    if args.odo_askpass else getpass.getpass("Odo password (used once; not saved): "))
        from awsmcp.store import VMStore
        record = VMStore().get(vm["handle"])
        public = json.loads((ROOT / "host/workshop.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            known_file = Path(tmp) / "known_hosts"
            known_file.write_text(vm["ip"] + " " + public["host_key"] + "\n")
            remote = "sudo python3 -c " + shlex.quote("import os,sys; fd=os.open('/run/sc26-odo/password',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); os.write(fd,sys.stdin.buffer.read()); os.close(fd)")
            subprocess.run(["ssh", "-F", "/dev/null", "-i", record.key_path,
                "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(known_file),
                "ubuntu@" + vm["ip"], remote], input=password.encode(), check=True)
        password = ""
        try:
            await call(client, "exec_command", handle=vm["handle"], command="sudo systemctl start sc26-odo")
            for attempt in range(20):
                try:
                    await call(client, "exec_command", handle=vm["handle"], command=probe, timeout=15)
                    print("Odo relay connected; temporary credential files removed")
                    break
                except RuntimeError:
                    if attempt == 19:
                        raise
                    await asyncio.sleep(1)
        finally:
            await call(client, "exec_command", handle=vm["handle"], command="sudo rm -f /run/sc26-odo/password /run/sc26-odo/askpass")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "bake", "connect-odo", "status", "cleanup"))
    parser.add_argument("--state", type=Path, default=ROOT / "dev/gateway-deployment.json")
    parser.add_argument("--profile", default="lab")
    parser.add_argument("--odo-user", default=os.environ.get("SC26_ODO_USER", getpass.getuser()))
    parser.add_argument("--odo-askpass", type=Path, help="Optional local Odo password helper; otherwise prompt in terminal")
    args = parser.parse_args()
    asyncio.run(deploy(args) if args.action == "deploy" else bake(args) if args.action == "bake" else
                connect_odo(args) if args.action == "connect-odo" else manage(args))
