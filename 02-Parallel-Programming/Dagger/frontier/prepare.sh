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
julia --project="$SC26_ROOT" -e '
    using Pkg
    Pkg.instantiate(; allow_autoprecomp=false)
    # Refresh stdlib entries if the copied manifest came from another Julia version.
    Pkg.resolve()
    # This compiler helper is lazy: instantiate alone does not fetch it, but
    # AMDGPU precompilation needs it on compute nodes without network access.
    for (uuid, dep) in Pkg.dependencies()
        dep.name == "LLVMDowngrader_jll" || continue
        artifacts = joinpath(dep.source, "Artifacts.toml")
        for name in keys(Pkg.Artifacts.select_downloadable_artifacts(
                artifacts; include_lazy=true, pkg_uuid=uuid))
            Pkg.Artifacts.ensure_artifact_installed(name, artifacts; pkg_uuid=uuid)
        end
    end
    using MPIPreferences
    MPIPreferences.use_system_binary(; vendor="cray", mpiexec="srun")
'
