"""Exercise update safety with real local Git repos and synthetic source ZIPs."""
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
import updates

BASE = Path(__file__).resolve().parents[1]
OLD, NEW = "1" * 40, "2" * 40


def snapshot(revision, **extra):
    files = {"host/app.py": (b"old app\n", False), "host/app.js": (b"old js\n", False),
             "console": (b"#!/bin/sh\n", True), "exercise.jl": (b"old exercise\n", False),
             "host/version.txt": ((revision + "\n").encode(), False)}
    files.update({name: (value, False) for name, value in extra.items()})
    return files


def write_files(root, files):
    for name, (data, executable) in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if executable else 0o644)


def archive(files, revision=NEW):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.comment = revision.encode()
        for name, (data, executable) in files.items():
            info = zipfile.ZipInfo("repo-main/" + updates.SUBDIR + "/" + name)
            info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
            z.writestr(info, data)
    return buffer.getvalue()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.before = snapshot(OLD)
        self.after = snapshot(NEW, **{"host/app.py": b"new app\n"})
        write_files(self.root, self.before)
        self.updater = updates.Updater(self.root)
        self.updater._download = Mock(side_effect=lambda revision:
            (b"", revision, self.before if revision == OLD else self.after))
        self.updater._archive_latest = Mock(return_value=(NEW, self.after))
        self.updater._archive_commit_count = Mock(return_value=3)

    def test_no_git_check_apply_preserves_exercises_credentials_and_executable(self):
        (self.root / "exercise.jl").write_text("my solution\n")
        (self.root / ".env").write_text("private")
        (self.root / "results").mkdir()
        (self.root / "results/answer.jl").write_text("answer")
        with patch.object(updates.shutil, "which", return_value=None):
            self.assertTrue(self.updater.check()["available"])
            self.assertEqual(self.updater.status()["method"], "archive")
            self.updater.apply()
        self.assertEqual((self.root / "host/app.py").read_bytes(), b"new app\n")
        self.assertEqual((self.root / "exercise.jl").read_text(), "my solution\n")
        self.assertEqual((self.root / ".env").read_text(), "private")
        self.assertEqual((self.root / "results/answer.jl").read_text(), "answer")
        self.assertTrue((self.root / "console").stat().st_mode & stat.S_IXUSR)
        self.assertEqual(self.updater.installed(), (NEW, False))
        self.assertEqual(self.updater.status()["status"], "restarting")
        self.assertEqual(self.updater.status()["last_update"]["commits"], 3)

    def test_conflict_makes_no_changes(self):
        (self.root / "host/app.py").write_text("custom app\n")
        self.updater.check()
        with self.assertRaisesRegex(ValueError, "Local edits.*host/app.py"):
            self.updater.apply()
        self.assertEqual((self.root / "host/app.py").read_text(), "custom app\n")
        self.assertEqual((self.root / "host/version.txt").read_text().strip(), OLD)

    def test_untracked_collision_and_local_deletion_are_preserved(self):
        self.after["new.jl"] = (b"upstream", False)
        (self.root / "new.jl").write_text("student addition")
        self.updater.check()
        with self.assertRaisesRegex(ValueError, "new.jl"):
            self.updater.apply()
        self.after.pop("new.jl")
        (self.root / "host/app.py").unlink()
        self.updater.check(force=True)
        with self.assertRaisesRegex(ValueError, "host/app.py"):
            self.updater.apply()
        self.assertFalse((self.root / "host/app.py").exists())

    def test_deleted_upstream_file_and_added_executable(self):
        self.after.pop("exercise.jl")
        self.after["helper"] = (b"#!/bin/sh\nexit 0\n", True)
        self.updater.check()
        self.updater.apply()
        self.assertFalse((self.root / "exercise.jl").exists())
        self.assertTrue((self.root / "helper").stat().st_mode & stat.S_IXUSR)

    def test_failed_replace_rolls_back_prior_files(self):
        self.updater.check()
        replace = os.replace
        calls = []
        def fail_second(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("disk failure")
            return replace(source, target)
        with patch.object(updates.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "disk failure"):
                self.updater.apply()
        self.assertEqual((self.root / "host/app.py").read_bytes(), self.before["host/app.py"][0])
        self.assertEqual((self.root / "host/version.txt").read_text().strip(), OLD)

    def test_symlink_never_writes_outside_checkout(self):
        with tempfile.TemporaryDirectory() as outside:
            shutil.rmtree(self.root / "host")
            (self.root / "host").symlink_to(outside, target_is_directory=True)
            Path(outside, "version.txt").write_text(OLD)
            Path(outside, "app.py").write_text("private")
            self.updater.check()
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.updater.apply()
            self.assertEqual(Path(outside, "app.py").read_text(), "private")

    def test_edit_during_update_is_preserved_and_other_replacements_rolled_back(self):
        self.after["host/app.js"] = (b"new js", False)
        self.updater.check()
        replace = os.replace
        def edit_after_replace(source, target):
            replace(source, target)
            (self.root / "host/app.py").write_text("concurrent edit")
        with patch.object(updates.os, "replace", side_effect=edit_after_replace):
            with self.assertRaisesRegex(ValueError, "changed during the update"):
                self.updater.apply()
        self.assertEqual((self.root / "host/app.py").read_text(), "concurrent edit")
        self.assertEqual((self.root / "host/app.js").read_bytes(), self.before["host/app.js"][0])

    def test_zip_validation(self):
        revision, files = updates.source_files(archive(self.after))
        self.assertEqual(revision, NEW)
        self.assertTrue(files["console"][1])
        for name in ("../escape", "/escape", "host//bad", "host/./bad", "C:/escape",
                     "dev/private", "results/answer", ".env", "host\\bad"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                updates.source_files(archive(dict(self.after, **{name: (b"bad", False)})))
        with self.assertRaises(ValueError):
            updates.source_files(archive(self.after, "main"))
        data = io.BytesIO(archive(self.after))
        with zipfile.ZipFile(data, "a") as z:
            info = zipfile.ZipInfo("repo-main/" + updates.SUBDIR + "/link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(ValueError, "non-regular"):
            updates.source_files(data.getvalue())

    def test_offline_and_check_throttling(self):
        self.updater._archive_latest.side_effect = OSError("offline")
        self.assertEqual(self.updater.check()["status"], "error")
        self.updater.check()
        self.assertEqual(self.updater._archive_latest.call_count, 1)
        self.updater.check(force=True)
        self.assertEqual(self.updater._archive_latest.call_count, 2)

    def test_first_check_runs_immediately_on_a_freshly_booted_system(self):
        with patch.object(updates.time, "monotonic", return_value=0):
            self.assertTrue(self.updater.check()["available"])

    def test_archive_commit_count_and_unavailable_comparison(self):
        with patch.object(updates.urllib.request, "urlopen", return_value=io.StringIO('{"total_commits": 7}')):
            self.assertEqual(updates.Updater._archive_commit_count(self.updater, OLD, NEW), 7)
        with patch.object(updates.urllib.request, "urlopen", side_effect=OSError("rate limited")):
            self.assertIsNone(updates.Updater._archive_commit_count(self.updater, OLD, NEW))

    def test_archive_checks_reuse_download_when_etag_matches(self):
        updater = updates.Updater(self.root)
        payload = archive(self.after)
        updater._download = Mock(return_value=(payload, NEW, self.after))
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.headers = {"ETag": '"first"'}
        with patch.object(updates.urllib.request, "urlopen", return_value=response):
            self.assertEqual(updater._archive_latest()[0], NEW)
            self.assertEqual(updater._archive_latest()[0], NEW)
            updater._download.assert_called_once()
            response.headers = {"ETag": '"second"'}
            updater._archive_latest()
            self.assertEqual(updater._download.call_count, 2)


@unittest.skipUnless(shutil.which("git"), "Git integration")
class GitTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        parent = Path(self.temp.name)
        self.remote, self.repo = parent / "remote", parent / "checkout"
        self.remote.mkdir()
        self.git(self.remote, "init", "-b", "main")
        self.git(self.remote, "config", "user.name", "Test")
        self.git(self.remote, "config", "user.email", "test@example.invalid")
        (self.remote / "file").write_text("old")
        self.git(self.remote, "add", ".")
        self.git(self.remote, "commit", "-m", "initial")
        self.git(parent, "clone", str(self.remote), str(self.repo))
        self.old = self.git(self.repo, "rev-parse", "HEAD")
        (self.remote / "file").write_text("new")
        self.git(self.remote, "commit", "-am", "update")
        self.new = self.git(self.remote, "rev-parse", "HEAD")
        self.updater = updates.Updater(self.repo)
        self.context = patch.object(updates, "UPSTREAM", str(self.remote))
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_git_fast_forward(self):
        self.assertEqual(self.updater.check()["latest"], self.new)
        self.updater.apply()
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), self.new)
        self.assertEqual((self.repo / "file").read_text(), "new")
        receipt = self.updater.status()["last_update"]
        self.assertEqual(receipt["commits"], 1)
        with patch.dict(os.environ, SC26_CONSOLE_UPDATE_RESULT=json.dumps(receipt)):
            restarted = updates.Updater(self.repo)
            self.assertEqual(restarted.status()["last_update"], receipt)
            self.assertNotIn("SC26_CONSOLE_UPDATE_RESULT", os.environ)

    def test_git_dirty_checkout_rejected(self):
        (self.repo / "file").write_text("solution")
        self.updater.check()
        with self.assertRaisesRegex(ValueError, "local tracked-file"):
            self.updater.apply()
        self.assertEqual((self.repo / "file").read_text(), "solution")
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), self.old)

    def test_diverged_checkout_cannot_merge_or_reset(self):
        self.git(self.repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "--allow-empty", "-m", "local work")
        head = updates.git_head(self.repo)
        self.updater.check()
        with self.assertRaises(ValueError):
            self.updater.apply()
        self.assertEqual(updates.git_head(self.repo), head)

    def test_git_missing_still_reads_packed_head(self):
        self.git(self.repo, "pack-refs", "--all")
        with patch.object(updates.shutil, "which", return_value=None):
            self.assertEqual(self.updater.installed(), (self.old, False))


class LauncherTests(unittest.TestCase):
    def test_arguments_with_spaces_and_exit_code_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix="console launcher ") as directory:
            root = Path(directory)
            fake = root / "python3"
            fake.write_text("#!/bin/sh\nif [ \"$1\" = -c ]; then exit 0; fi\nprintf '%s\\n' \"$@\"\nexit 23\n")
            fake.chmod(0o755)
            args = ["--gateway", "a descriptor with spaces.json", "--port", "9123", "--reservation", "a; $(literal)"]
            result = subprocess.run([str(BASE / "console"), *args], cwd=root,
                                    env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"]),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 23)
            self.assertEqual(result.stdout.splitlines(), [str(BASE / "host/app.py"), *args])


if __name__ == "__main__":
    unittest.main()
