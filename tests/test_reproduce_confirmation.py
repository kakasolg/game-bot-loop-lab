"""Non-episode unit tests for reproduce_confirmation.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import reproduce_confirmation as rc  # noqa: E402


class TestReproduceConfirmation(unittest.TestCase):
    def test_refuses_repo_root_and_ancestors_and_protected_subdirs(self) -> None:
        with self.assertRaises(ValueError):
            rc.validate_destination(REPO_ROOT, REPO_ROOT)
        with self.assertRaises(ValueError):
            rc.validate_destination(REPO_ROOT.parent, REPO_ROOT)
        with self.assertRaises(ValueError):
            rc.validate_destination(REPO_ROOT / "docs", REPO_ROOT)
        with self.assertRaises(ValueError):
            rc.validate_destination(REPO_ROOT / "contributions", REPO_ROOT)
        with self.assertRaises(ValueError):
            rc.validate_destination(REPO_ROOT / ".venv", REPO_ROOT)

    def test_refuses_nonempty_dir_and_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nonempty = tmp_path / "nonempty"
            nonempty.mkdir()
            (nonempty / "existing.txt").write_text("hi", encoding="utf-8")
            with self.assertRaises(ValueError):
                rc.validate_destination(nonempty, REPO_ROOT)

            existing_file = tmp_path / "some_file.txt"
            existing_file.write_text("file", encoding="utf-8")
            with self.assertRaises(ValueError):
                rc.validate_destination(existing_file, REPO_ROOT)

    def test_real_source_inputs_and_prepare_only_isolation(self) -> None:
        plan, req = rc.load_required_inputs(REPO_ROOT)
        self.assertEqual(plan["seeds"]["count"], 200)
        expected_set = {
            "phase5_confirmation_plan.json",
            "phase1_weights.npz",
            "phase1_config.json",
            "run_phase1.py",
            "run_phase1_continuation.py",
            "external_wrapper.py",
            "phase5_stats.py",
            "run_phase5_confirmation.py",
        }
        self.assertEqual(set(req), expected_set)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "isolated_run"
            exit_code = rc.main(["--dest", str(dest), "--prepare-only"])
            self.assertEqual(exit_code, 0)
            actual_files = {p.name for p in dest.iterdir()}
            self.assertEqual(actual_files, expected_set)
            cmd, cwd = rc.build_runner_command(dest)
            self.assertEqual(cwd, dest.resolve())
            self.assertEqual(cmd, [sys.executable, str(dest.resolve() / "run_phase5_confirmation.py")])

            # Verify unchanged runner pre-flight checks & imports pass strictly inside isolated dest (0 episodes)
            probe = (
                "import json, pathlib, run_phase5_confirmation as r5; "
                "plan = json.loads(pathlib.Path('phase5_confirmation_plan.json').read_text(encoding='utf-8')); "
                "ok, rep = r5.verify_locked_plan(plan); "
                "assert ok, rep; "
                "assert r5.run_wrapper_unit_tests()['all_unit_tests_passed']; "
                "assert r5.run_statistical_unit_tests()['all_statistical_unit_tests_passed']; "
                "assert pathlib.Path(r5.__file__).resolve().parent == pathlib.Path('.').resolve(); "
                "print('ISOLATED_PREFLIGHT_OK')"
            )
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(dest),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("ISOLATED_PREFLIGHT_OK", proc.stdout)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "never_created"
            exit_code = rc.main(["--dest", str(dest), "--dry-run"])
            self.assertEqual(exit_code, 0)
            self.assertFalse(dest.exists())

    def test_input_validation_rejects_missing_or_corrupted_source(self) -> None:
        _, req = rc.load_required_inputs(REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            mock_src = Path(tmp) / "mock_src"
            mock_src.mkdir()
            for fname in req:
                (mock_src / fname).write_bytes((REPO_ROOT / fname).read_bytes())

            # Corrupt one locked file and ensure validation fails
            (mock_src / "phase5_stats.py").write_text("# modified\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rc.load_required_inputs(mock_src)

            # Remove a required file and ensure FileNotFoundError is raised
            (mock_src / "phase5_stats.py").unlink()
            with self.assertRaises(FileNotFoundError):
                rc.load_required_inputs(mock_src)

            # Remove plan completely
            (mock_src / "phase5_confirmation_plan.json").unlink()
            with self.assertRaises(FileNotFoundError):
                rc.load_required_inputs(mock_src)


if __name__ == "__main__":
    unittest.main()
