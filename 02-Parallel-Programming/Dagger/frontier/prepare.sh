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
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project="$SC26_ROOT" -e '
    using Pkg
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
