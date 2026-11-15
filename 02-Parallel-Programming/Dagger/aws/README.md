# AWS fallbacks for the tutorial console

For attendee access without individual SSH keys or IP setup, use the
[shared-password workshop gateway](GATEWAY.md). Attendees run
`python3 host/app.py --gateway`, choose Odo relay or AWS CPU, and paste one
workshop password. CPU VMs are assigned and released automatically.
The instructions below also retain the original manual fleet/relay workflow.

Direct Odo remains the default (`python3 host/app.py`). These fallbacks are
explicit options and do not modify Odo's launchers, GPU setup, or SSH aliases.
Run the commands below from `02-Parallel-Programming/Dagger`.

## What was tested

Live validation on November 14, 2026 (Arizona):

| Path | Result |
| --- | --- |
| Ordinary Odo SSH multiplexing | A second simultaneous session is refused. A jump host or `ControlMaster` alone does not solve this. |
| Application relay on AWS | 30 concurrent independent commands through one Odo SSH session, including on a 2-vCPU `c7i.large`; two simultaneous console runs with distinct folders and outputs; the Single GPU demo passed on Odo. |
| CPU VM, `c7i.2xlarge` (8 vCPU, 16 GiB) | Tasks, mutable data, two-rank MPI, CPU Cholesky exercises in single-process and MPI modes, and a reduced-grid final simulation with visualization frames passed. Single-process and MPI cancellation left no worker processes, and a subsequent run passed. |
| Provisioning | AWS MCP launched the CPU pilot; `aws/fleet` launched a separate `c7i.large` relay VM and completed setup. Over-budget plans were rejected. |

The relay shares an SSH transport and account, while each console still owns
its session folder and Slurm job name. Odo continues to enforce its Slurm and
account limits. This test did not submit 30 simultaneous GPU jobs or use 30
physical attendee devices. The CPU MPI fallback runs ranks on **one VM**;
it demonstrates MPI scheduling and transfers, without demonstrating a physical
network between nodes.

## Capacity and provisioning

The live Standard On-Demand quota returned 300 vCPUs, but the launcher defaults
to a **256-vCPU budget**. It includes existing pending/running Standard instances
in its calculation. With the existing 8-vCPU instance, 30 attendee VMs at 8
vCPUs each plus a 2-vCPU relay use **250 vCPUs**. For 25 attendees this is 210.

AWS Pricing returned **$0.357/hour** for Linux `c7i.2xlarge` in `us-east-1`:
30 cost $10.71/hour for compute, plus the relay, disks, public IPv4 and transfer.
This is a capacity plan; the full 30-VM fleet was not launched during testing.

The wrapper uses `~/awsmcp/.venv/bin/python` and starts `~/awsmcp/awsmcp` as an
MCP stdio server. Set `AWSMCP_DIR` for a different installation. Authenticate
the `lab` AWS profile first. Defaults are On-Demand instances, a 40-GiB root
disk, and a **three-hour self-termination timer**. VM shutdown also terminates
the instance. Increase `--timeout` in seconds before launching when needed
(maximum 86400); setup time consumes part of that timer.

```bash
# Read-only capacity check.
aws/fleet plan --count 30

# Start with a single CPU VM, then inspect package preparation.
aws/fleet launch --count 1 --state dev/cpu-fleet.json
aws/fleet status --state dev/cpu-fleet.json

# Or launch 25/30 CPU VMs when the fallback is needed.
aws/fleet launch --count 30 --state dev/workshop-fleet.json

# A separate relay needs only two vCPUs; Julia stays on Odo.
aws/fleet launch --role relay --instance-type c7i.large --state dev/relay-fleet.json
aws/fleet status --state dev/relay-fleet.json
```

Do not run both CPU launch examples unless you want both sets of machines.
CPU package preparation is a background job; `status` shows its result. Wait
for exit status 0 before handing out a VM. The CPU environment pins the tested
Dagger commit and uses Julia 1.12 with MPICH's portable JLL binaries. No AMDGPU,
ROCm, Cray modules, or Slurm are required on AWS. Each VM has its own depot.

`dev/*fleet.json` contains secret MCP handles; the directory is Git-ignored and
the state file is written with mode 0600. Retain it until cleanup, including
after an interrupted launch. Rerunning `launch` adds more VMs; use `status`
to inspect an existing launch. Each participant CPU VM has a separate security
group and does not join an MCP shared cluster network.

By default, SSH is open only to the instructor's public IP, and the generated
key remains in the instructor's local MCP store. Before distributing access,
install each attendee's public SSH key on their assigned VM (on the shared
relay for relay users), and allow their source CIDR on port 22 in that VM's
security group. `--ssh-cidr CIDR` can set a common source range at launch; it
must also include the provisioning machine so AWS MCP can finish setup.
The browser server stays local to the attendee; no public web port is needed.

## CPU fallback

On the attendee's laptop, add a normal SSH alias with their key:

```sshconfig
Host sc26-cpu
    HostName ASSIGNED_VM_IP
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Then run:

```bash
python3 host/app.py --backend cpu --ssh-host sc26-cpu --threads 8
```

The console uploads its own workspace and reuses the VM's prepared package
depot. It offers Julia basics, the CPU demos, CPU MPI, Cholesky, and the final
exercise. GPU choices are removed and GPU commands are rejected by the server.
MPI divides eight Julia threads among the chosen ranks (two ranks get four
threads each); BLAS uses one thread. Cancel and Disconnect stop that run's
process group. `--time` remains an Odo/Slurm option; the AWS VM's termination
timer bounds the lifetime of the CPU fallback.

## Single-connection Odo relay

`host/relay.py` carries labelled binary streams **inside one SSH session**.
One Python 3.6 dispatcher on Odo starts the ordinary console coordinators;
the ordinary per-console Slurm lifecycle remains in effect. Uploads and live
output use this stream too, so no extra Odo SSH/SFTP channel is needed.

On the relay VM, in an instructor SSH terminal (`ssh -t ubuntu@RELAY_IP`), run:

```bash
# Confirm Odo's host key and connectivity using the normal SSH procedure.
ssh YOUR_ODO_USER@login1.odo.olcf.ornl.gov true

# Keep this running, e.g. inside tmux. Authenticate once in this terminal.
python3 /home/ubuntu/sc26/host/relay.py serve \
    --upstream YOUR_ODO_USER@login1.odo.olcf.ornl.gov
```

Wait for **Odo dispatcher ready**. The service listens on the private Unix
socket `/home/ubuntu/.sc26-relay/socket`; it does not open a TCP service.
Passwords stay in SSH's terminal prompt. For unattended operation, an
instructor-managed `SSH_ASKPASS` helper also works; remove temporary credential
files after authentication. The optional `--control-path` reuses an already
authenticated, idle master and fails closed if that socket is unavailable.

Each attendee adds an SSH alias for the AWS relay with their own SSH key, then:

```bash
python3 host/app.py --via-relay --ssh-host sc26-relay --odo-user SHARED_ODO_USER
```

The default relay script and socket paths match the provisioned VM. Override
them with `--relay-script` and `--relay-socket`. Attendees need no Odo password.
The shared AWS user/socket grants the shared Odo account's access; session
folders are organizational separation, with the same account permissions as
the branch's existing credential sharing. Use each attendee's CPU VM when
independent machines are desired.

Stopping the instructor relay ends its upstream stream and cancels its active
console runs. Restart the relay and reconnect the consoles after an upstream
failure. If a killed relay leaves its socket behind, verify no relay is
running before removing `~/.sc26-relay/socket`. The launcher raises AWS SSH's
burst-login and channel limits to admit the workshop's concurrent connections.

## Cleanup and checks

Disconnect the consoles and stop the relay before terminating VMs:

```bash
aws/fleet terminate --state dev/cpu-fleet.json
aws/fleet terminate --state dev/workshop-fleet.json
aws/fleet terminate --state dev/relay-fleet.json
```

Terminate only state files corresponding to fleets you launched. The MCP
cleanup removes each instance and its security group; the self-termination
timer is a backup and may leave a group for explicit cleanup.

Local regression checks need neither AWS nor Odo:

```bash
python3 -m unittest discover -s test -p 'test_*.py'
node test/ansi.cjs
node test/timeline.cjs
```

The fallback tests cover concurrent binary streams, command exit codes,
per-client disconnect cleanup, CPU resource rejection, core division, and
cancellation without Slurm. Existing Odo protocol/lifecycle tests still run.
