#!/bin/bash
# Run on the andromeda login node: download Julia when it is missing, then
# resolve and download the packages. Compilation happens in a Slurm job
# afterwards (andromeda/precompile.sh), not here.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$SC26_CACHE_ROOT" "$SC26_DEPOT"
if [[ ! -x "$SC26_JULIA_HOME/bin/julia" ]]; then
    minor="${SC26_JULIA_VERSION%.*}"
    url="https://julialang-s3.julialang.org/bin/linux/x64/$minor/julia-$SC26_JULIA_VERSION-linux-x86_64.tar.gz"
    echo "Downloading Julia $SC26_JULIA_VERSION…"
    partial="$SC26_JULIA_HOME.partial"
    rm -rf "$partial"
    mkdir -p "$partial"
    curl -fsSL "$url" | tar -xz -C "$partial" --strip-components=1
    mv "$partial" "$SC26_JULIA_HOME"
fi
export JULIA_PKG_OFFLINE=false
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project="$SC26_ROOT" --startup-file=no -e '
    using Pkg
    if isempty(Pkg.Registry.reachable_registries())
        Pkg.Registry.add("General")
    end
    # The manifest is not in Git: resolve the project on first use, and pick
    # up dependencies added to Project.toml later.
    Pkg.resolve()
    Pkg.instantiate(; allow_autoprecomp=false)
    # Highlights keeps its themes in lazy artifacts; fetch them now.
    for (uuid, dep) in Pkg.dependencies()
        dep.name == "Highlights" || continue
        artifacts = joinpath(dep.source, "Artifacts.toml")
        for name in keys(Pkg.Artifacts.select_downloadable_artifacts(
                artifacts; include_lazy=true, pkg_uuid=uuid))
            Pkg.Artifacts.ensure_artifact_installed(name, artifacts; pkg_uuid=uuid)
        end
    end
'
