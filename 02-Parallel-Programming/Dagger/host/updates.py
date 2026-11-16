"""Updates from the public main branch, using only the Python standard library.

Git checkouts fast-forward normally. Without Git, compare pinned source ZIPs
before replacing Dagger files; keep edits when upstream did not change them,
and reject conflicts. Never extract archives over a working directory.
"""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile

UPSTREAM = "https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git"
ARCHIVE = "https://codeload.github.com/JuliaParallel/julia-hpc-tutorial-sc26/zip/"
SUBDIR = "02-Parallel-Programming/Dagger"
SHA = re.compile(r"[0-9a-f]{40}\Z")
MAX_DOWNLOAD = 120 * 1024 * 1024


def git_head(root):
    """Read HEAD even if Git isn't installed (including linked worktrees)."""
    for parent in (root, *root.parents):
        if parent != root and root.relative_to(parent).as_posix() != SUBDIR:
            continue
        marker = parent / ".git"
        if not marker.exists():
            continue
        directory = marker
        if marker.is_file():
            directory = (parent / marker.read_text().strip().removeprefix("gitdir: ")).resolve()
        if not (directory / "HEAD").is_file():
            continue
        value = (directory / "HEAD").read_text().strip()
        if value.startswith("ref: "):
            ref = value[5:]
            if ".." in PurePosixPath(ref).parts or not ref.startswith("refs/"):
                raise ValueError("Invalid Git HEAD")
            common = directory
            if (directory / "commondir").exists():
                common = (directory / (directory / "commondir").read_text().strip()).resolve()
            loose = common / ref
            if loose.exists():
                value = loose.read_text().strip()
            else:
                value = ""
                for line in (common / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        value = line.split()[0]
                        break
        return value if SHA.fullmatch(value) else None
    return None


def source_files(data):
    """Validate the ZIP and return Dagger's regular files and executable bits."""
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        revision = archive.comment.decode("ascii")
        if not SHA.fullmatch(revision):
            raise ValueError("The source archive has no valid commit identifier")
        entries = archive.infolist()
        if not entries:
            raise ValueError("Empty source archive")
        if len(entries) > 10000 or sum(e.file_size for e in entries) > 300 * 1024 * 1024:
            raise ValueError("Source archive is too large")
        prefix = entries[0].filename.split("/", 1)[0] + "/" + SUBDIR + "/"
        for entry in entries:
            name = entry.filename
            if "\\" in name or ".." in PurePosixPath(name).parts or name.startswith("/"):
                raise ValueError("Unsafe source archive path")
            if not name.startswith(prefix) or entry.is_dir():
                continue
            relative = name[len(prefix):]
            path = PurePosixPath(relative)
            if path.is_absolute() or path.as_posix() != relative or ":" in relative:
                raise ValueError("Unsafe source archive path")
            parts = path.parts
            if not parts or any(p in (".git", "dev", "results", "__pycache__") or
                                p.startswith(".env") or p == ".console-update.json" for p in parts):
                raise ValueError("Source archive contains a private or generated path")
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("Source archive contains a non-regular file")
            if relative in files:
                raise ValueError("Duplicate source archive path")
            files[relative] = (archive.read(entry), bool(mode & 0o111))
    if not {"host/app.py", "host/app.js", "console"} <= files.keys():
        raise ValueError("Source archive is missing console files")
    return revision, files


def fingerprint(data):
    # A Windows checkout may use CRLF. Binary files are compared byte for byte.
    try:
        data.decode("utf-8")
        if b"\0" not in data:
            data = data.replace(b"\r\n", b"\n")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(data).digest()


class Updater:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = self.root / "dev/console-updates"
        self.marker = self.root / ".console-update.json"
        self.lock = threading.RLock()
        self.last_check = 0
        self.details = dict(status="unchecked", available=False, message="")

    def status(self):
        with self.lock:
            return dict(self.details)

    def _git(self, *args):
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                                text=True, timeout=120, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if result.returncode:
            raise ValueError(result.stderr.strip() or "Git could not update this checkout")
        return result.stdout.strip()

    def installed(self):
        head = git_head(self.root)
        if self.marker.exists():
            state = json.loads(self.marker.read_text())
            if state.get("git_head") == head and SHA.fullmatch(state.get("revision", "")):
                return state["revision"], False
        if head:
            return head, bool(shutil.which("git"))
        version = self.root / "host/version.txt"
        value = version.read_text().strip() if version.exists() else ""
        return (value if SHA.fullmatch(value) else None), False

    def _download(self, ref):
        if ref != "refs/heads/main" and not SHA.fullmatch(ref):
            raise ValueError("Invalid source revision")
        with urllib.request.urlopen(ARCHIVE + ref, timeout=45) as response:
            data = response.read(MAX_DOWNLOAD + 1)
        if len(data) > MAX_DOWNLOAD:
            raise ValueError("Source download is too large")
        revision, files = source_files(data)
        if ref != "refs/heads/main" and revision != ref:
            raise ValueError("Source archive revision does not match the requested commit")
        return data, revision, files

    def _archive_latest(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        cached = self.cache / "main.zip"
        metadata = self.cache / "main.json"
        with urllib.request.urlopen(urllib.request.Request(ARCHIVE + "refs/heads/main", method="HEAD"),
                                    timeout=20) as response:
            etag = response.headers.get("ETag")
        old = json.loads(metadata.read_text()) if metadata.exists() else {}
        if etag and etag == old.get("etag") and cached.exists():
            try:
                return source_files(cached.read_bytes())
            except (ValueError, zipfile.BadZipFile):
                pass  # Recover from an interrupted or damaged local cache.
        data, revision, files = self._download("refs/heads/main")
        staging = self.cache / "main.zip.tmp"
        staging.write_bytes(data)
        os.replace(staging, cached)
        metadata.write_text(json.dumps(dict(etag=etag)))
        return revision, files

    def check(self, force=False):
        with self.lock:
            if self.details["status"] in ("checking", "updating", "restarting"):
                return self.status()
            if not force and time.monotonic() - self.last_check < 300:
                return self.status()
            self.last_check = time.monotonic()
            self.details.update(status="checking", message="Checking main for updates…")
        try:
            installed, use_git = self.installed()
            if use_git:
                lines = self._git("ls-remote", UPSTREAM, "refs/heads/main").split()
                latest = lines[0] if lines else ""
                if not SHA.fullmatch(latest):
                    raise ValueError("Could not determine the main revision")
                # A development branch ahead of main is not an old install.
                if latest != installed:
                    result = subprocess.run(["git", "-C", str(self.root), "merge-base",
                                             "--is-ancestor", latest, "HEAD"],
                                            capture_output=True, timeout=15)
                    if result.returncode == 0:
                        latest = installed
            else:
                latest, files = self._archive_latest()
                if installed is None:
                    # Bootstrap an unversioned source ZIP only if it matches;
                    # otherwise there is no safe baseline for attendee edits.
                    if all((self.root / name).is_file() and
                           fingerprint((self.root / name).read_bytes()) == fingerprint(value[0])
                           for name, value in files.items()):
                        self._save_revision(latest)
                        installed = latest
                    else:
                        raise ValueError("Cannot identify this install. Download main again, keeping your exercise edits.")
            available = latest != installed
            with self.lock:
                self.details = dict(status="available" if available else "current", available=available,
                                    installed=installed, latest=latest, method="git" if use_git else "archive",
                                    message="Updated code is available on main." if available else "Up to date with main.")
        except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            with self.lock:
                self.details.update(status="error", message=str(exc))
        return self.status()

    def _save_revision(self, revision):
        temporary = self.marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(revision=revision, git_head=git_head(self.root))))
        os.replace(temporary, self.marker)

    def _apply_archive(self, installed, latest):
        _, _, before = self._download(installed)
        _, _, after = self._download(latest)
        changes, conflicts = [], []
        for name in sorted(before.keys() | after.keys()):
            old, new = before.get(name), after.get(name)
            if old == new:
                continue
            target = self.root / name
            # Check all parents, including a symlink pointing back inside root.
            if any(p.is_symlink() for p in (target, *target.parents) if p != self.root.parent):
                raise ValueError("Refusing to update through a symlink: " + name)
            if target.exists() and not target.is_file():
                conflicts.append(name)
                continue
            actual = target.read_bytes() if target.exists() else None
            original = (actual, target.stat().st_mode) if actual is not None else None
            old_data, new_data = old[0] if old else None, new[0] if new else None
            if name == "host/version.txt" and actual == b"$Format:%H$\n":
                actual = old_data
            same = lambda a, b: a is b or (a is not None and b is not None and fingerprint(a) == fingerprint(b))
            if same(actual, new_data):
                continue
            if not same(actual, old_data):
                conflicts.append(name)
            else:
                changes.append((target, new, original))
        if conflicts:
            raise ValueError("Local edits overlap this update. Save them elsewhere or resolve them first: " +
                             ", ".join(conflicts[:12]))
        # Stage first; retain originals and roll back any failed replacement.
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.cache) as temporary:
            backups, applied = {}, []
            try:
                for i, (target, new, original) in enumerate(changes):
                    backups[target] = original
                    if new:
                        staged = Path(temporary) / str(i)
                        staged.write_bytes(new[0])
                        staged.chmod(0o755 if new[1] else 0o644)
                for i, (target, new, original) in enumerate(changes):
                    current = (target.read_bytes(), target.stat().st_mode) if target.exists() else None
                    if current != original or any(p.is_symlink() for p in (target, *target.parents)):
                        raise ValueError("A file changed during the update: " + str(target.relative_to(self.root)))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    applied.append(target)
                    if new:
                        os.replace(Path(temporary) / str(i), target)
                    else:
                        target.unlink()
                self._save_revision(latest)
            except Exception:
                for target in reversed(applied):
                    original = backups[target]
                    if original is None:
                        target.unlink(missing_ok=True)
                    else:
                        target.write_bytes(original[0])
                        target.chmod(original[1])
                raise

    def apply(self):
        with self.lock:
            info = self.status()
            if info["status"] != "available":
                raise ValueError("Check for an available update first")
            self.details.update(status="updating", message="Updating from main…")
        try:
            installed, use_git = self.installed()
            if installed != info["installed"]:
                raise ValueError("The checkout changed. Check for updates again.")
            if use_git:
                if self._git("status", "--porcelain", "--untracked-files=no"):
                    raise ValueError("Save or commit local tracked-file edits before updating with Git.")
                # Fetch only the canonical main, then use Git's fast-forward and
                # untracked-file protections. Never stash, reset or force.
                self._git("pull", "--ff-only", "--no-rebase", UPSTREAM, "main")
            else:
                self._apply_archive(installed, info["latest"])
            with self.lock:
                self.details.update(status="restarting", available=False,
                                    message="Updated. Restarting the console…")
        except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            with self.lock:
                self.details.update(status="error", message=str(exc))
            raise ValueError(str(exc)) from exc
