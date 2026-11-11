#!/bin/bash
# Download packages on the LOGIN node; compute nodes cannot reach package servers.
# Then run setup.sh in a compute step to precompile and check devices.
set -euo pipefail
if [[ "$(hostname -s)" != login* ]]; then
    echo "Run frontier/prepare.sh on an Odo login node before requesting an allocation." >&2
    exit 1
fi
source "$(dirname "$0")/env.sh"
mkdir -p "$SC26_DEPOT"
# Console sessions each have their own folder but share the package depot.
# Prepare one session at a time, and start every session from the manifest
# the last one resolved, so they all use the same package versions.
seed="$SC26_CACHE_ROOT/session-Manifest.toml"
if [[ -n "${SC26_SESSION:-}" ]]; then
    exec 9>"$SC26_CACHE_ROOT/prepare.lock"
    if ! flock -n 9; then
        echo "Another console session is preparing packages; waiting for it…"
        flock 9
    fi
    if [[ ! -f "$SC26_ROOT/Manifest.toml" && -f "$seed" ]]; then
        cp "$seed" "$SC26_ROOT/Manifest.toml"
    fi
fi
export JULIA_PKG_OFFLINE=false
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project="$SC26_ROOT" -e '
    using Pkg
    # A fresh depot has no registry; Pkg.resolve cannot resolve packages without
    # one. Install General on the login node, where the package server is reachable.
    if isempty(Pkg.Registry.reachable_registries())
        Pkg.Registry.add("General")
    end
    # Add newly introduced project dependencies and refresh copied stdlib entries
    # before instantiating an existing manifest.
    Pkg.resolve()
    Pkg.instantiate(; allow_autoprecomp=false)
    # Fetch lazy compiler helpers and highlighter themes on the login node.
    # Compute nodes cannot download artifacts when a demo first needs them.
    for (uuid, dep) in Pkg.dependencies()
        dep.name in ("LLVMDowngrader_jll", "Highlights") || continue
        artifacts = joinpath(dep.source, "Artifacts.toml")
        for name in keys(Pkg.Artifacts.select_downloadable_artifacts(
                artifacts; include_lazy=true, pkg_uuid=uuid))
            Pkg.Artifacts.ensure_artifact_installed(name, artifacts; pkg_uuid=uuid)
        end
    end
    using MPIPreferences
    MPIPreferences.use_system_binary(; vendor="cray", mpiexec="srun")
'
if [[ -n "${SC26_SESSION:-}" ]]; then
    cp "$SC26_ROOT/Manifest.toml" "$seed.$SC26_SESSION" && mv "$seed.$SC26_SESSION" "$seed"
fi
