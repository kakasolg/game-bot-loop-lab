"""Isolated non-destructive reproduction launcher for Phase 5 confirmation evaluation.

NOTE: Seeds 30000..30199 are the public Phase 5 reproduction set (not an untouched
holdout). Running this launcher reproduces the frozen 200-seed evaluation inside a
new dedicated directory without modifying or overwriting published repository files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PLAN_FILENAME = "phase5_confirmation_plan.json"
PROTECTED_REPO_SUBDIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "docs",
    "contributions",
    "demo",
    "tests",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_required_inputs(source_dir: Path) -> tuple[dict, list[str]]:
    """Load the frozen Phase 5 plan and verify all required input files exist and match hashes."""
    plan_path = source_dir / PLAN_FILENAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing required confirmation plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    locked_hashes = plan.get("locked_artifacts_sha256")
    if not isinstance(locked_hashes, dict) or not locked_hashes:
        raise ValueError(f"Invalid or empty locked_artifacts_sha256 in {plan_path}")

    required_files = [PLAN_FILENAME, *locked_hashes.keys()]
    for fname in required_files:
        src = source_dir / fname
        if not src.is_file():
            raise FileNotFoundError(f"Missing required frozen input: {src}")

    mismatches: list[str] = []
    for fname, expected_sha in locked_hashes.items():
        observed_sha = sha256_file(source_dir / fname)
        if observed_sha != expected_sha:
            mismatches.append(f"{fname}: expected {expected_sha}, got {observed_sha}")

    if mismatches:
        raise ValueError(
            "Source artifact SHA-256 verification failed against frozen plan:\n  - "
            + "\n  - ".join(mismatches)
        )

    return plan, required_files


def validate_destination(dest_arg: Path | str, source_dir: Path) -> Path:
    """Validate that destination is a safe, dedicated empty/nonexistent directory."""
    raw = str(dest_arg).strip()
    if not raw:
        raise ValueError("Destination path must not be empty.")

    src = source_dir.resolve()
    dest_input = Path(dest_arg)
    if dest_input.is_symlink():
        raise ValueError(f"Refusing symlink destination: {dest_input}")

    dest = dest_input.resolve()

    if dest == src:
        raise ValueError(
            f"Refusing destination equal to repository root ({src}); "
            "published confirmation outputs must not be overwritten."
        )

    if src in dest.parents:
        raise ValueError(
            f"Refusing destination that is an ancestor of repository root ({dest})."
        )

    try:
        rel = dest.relative_to(src)
    except ValueError:
        rel = None

    if rel is not None and rel.parts:
        top_part = rel.parts[0]
        if top_part in PROTECTED_REPO_SUBDIRS:
            raise ValueError(
                f"Refusing protected repository subdirectory as destination: {dest}"
            )

    if dest.exists():
        if not dest.is_dir():
            raise ValueError(f"Refusing existing non-directory destination: {dest}")
        if any(dest.iterdir()):
            raise ValueError(
                f"Refusing non-empty destination directory: {dest}. "
                "Provide a new or empty dedicated directory."
            )

    return dest


def stage_frozen_inputs(
    source_dir: Path, dest: Path, required_files: list[str]
) -> list[Path]:
    """Copy only the explicitly required frozen inputs into the isolated destination."""
    dest.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for fname in required_files:
        src = source_dir / fname
        dst = dest / fname
        shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def build_runner_command(dest: Path, python_executable: str | None = None) -> tuple[list[str], Path]:
    """Build exact subprocess command and working directory for the unchanged runner."""
    py = python_executable or sys.executable
    runner = dest / "run_phase5_confirmation.py"
    return [py, str(runner)], dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the frozen Phase 5 evaluation (public seeds 30000..30199) inside "
            "a dedicated empty directory without modifying published repository files. "
            "This is a reproduction check on public seeds, not a new confirmation claim."
        )
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="New or empty dedicated directory for isolated inputs and outputs.",
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path(__file__).resolve().parent),
        help="Source directory containing frozen Phase 5 artifacts (defaults to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, destination safety, and command without copying or running.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and copy frozen inputs into --dest without running episodes.",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir).resolve()
    plan, required_files = load_required_inputs(source_dir)
    dest = validate_destination(args.dest, source_dir)
    cmd, cwd = build_runner_command(dest)

    print(
        "REPRODUCTION NOTICE: Seeds 30000..30199 are the public Phase 5 reproduction set "
        "(already evaluated). Running in an isolated destination reproduces the published "
        "result and does NOT establish a new holdout confirmation."
    )
    print(f"Source directory: {source_dir}")
    print(f"Isolated destination: {dest}")
    print(f"Verified required frozen inputs ({len(required_files)}): {', '.join(required_files)}")
    print(f"Runner command: {cmd} (cwd={cwd})")
    print(
        f"Planned public reproduction seed range: {plan['seeds']['start']}..{plan['seeds']['end_inclusive']} "
        f"({plan['seeds']['count']} paired episodes)"
    )

    if args.dry_run:
        print("Dry-run complete: no files copied and zero episodes executed.")
        return 0

    stage_frozen_inputs(source_dir, dest, required_files)
    print(f"Staged {len(required_files)} frozen files into {dest}.")

    if args.prepare_only:
        print("Prepare-only complete: isolated workspace ready; zero episodes executed.")
        return 0

    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        print(
            f"Isolated runner exited with code {proc.returncode}. Inspect outputs in {dest}.",
            file=sys.stderr,
        )
        return proc.returncode

    out_json = dest / "phase5_confirmation_results.json"
    out_jsonl = dest / "phase5_confirmation_episodes.jsonl"
    print(f"Completed isolated reproduction in {dest}.")
    print(f"Outputs written strictly outside published root:\n  - {out_json}\n  - {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
