#!/usr/bin/env python3
"""
Run one-song PianoMime generalist evaluation without video recording.

Expected layout:

workspace/
├── pianomime/
│   ├── eval_one_generalist_no_video.py   # put this file here
│   ├── multi_task/
│   ├── dataset_hl.zarr/
│   └── dataset_ll.zarr/
├── dataset/
│   ├── notes/
│   └── notes_test/
└── reproduced_ckpt/
    ├── checkpoint_ae.ckpt
    ├── checkpoint_high_level.ckpt
    └── checkpoint_low_level.ckpt

Usage from workspace:
    python pianomime/eval_one_generalist_no_video.py YourSongName

Usage from workspace/pianomime:
    python eval_one_generalist_no_video.py YourSongName

The original eval scripts hard-code checkpoint paths as:
    checkpoint_ae.ckpt
    checkpoint_high_level.ckpt
    checkpoint_low_level.ckpt
under workspace/. This script creates root-level symlinks pointing to
workspace/reproduced_ckpt/*.ckpt before running eval.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional



conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_PRELOAD"] = f"{conda_prefix}/lib/libstdc++.so.6"


# Edit this if you prefer running without a command-line song argument.
SONG_NAME = "Petrunko_3"

# Default checkpoint directory relative to workspace/.
DEFAULT_CKPT_DIRNAME = "reproduced_ckpt"


def repo_and_workspace() -> tuple[Path, Path]:
    """This file is expected to be placed directly under workspace/pianomime/."""
    repo_dir = Path(__file__).resolve().parent
    workspace_dir = repo_dir.parent
    return repo_dir, workspace_dir


def patch_low_level_no_video(repo_dir: Path) -> None:
    """Disable video/audio recording in multi_task/eval_low_level.py."""
    path = repo_dir / "multi_task" / "eval_low_level.py"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    text = path.read_text()
    original = text

    replacements = {
        'record_dir=".",': 'record_dir=None,',
        "record_dir='.',": 'record_dir=None,',
        'record_dir = ".",': 'record_dir=None,',
        "record_dir = '.',": 'record_dir=None,',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if text != original:
        path.write_text(text)
        print(f"[patch] Disabled video recording in {path}")
    elif "record_dir=None" in text:
        print(f"[patch] Video recording already disabled in {path}")
    else:
        print(
            f"[warn] Did not find record_dir='.' in {path}. "
            "Please check manually that get_env_ll(..., record_dir=None, ...) is used."
        )


def require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def resolve_ckpt_dir(workspace_dir: Path, ckpt_dir_arg: str) -> Path:
    ckpt_dir = Path(ckpt_dir_arg)
    if not ckpt_dir.is_absolute():
        ckpt_dir = workspace_dir / ckpt_dir
    return ckpt_dir.resolve()


def ensure_root_checkpoint_links(workspace_dir: Path, ckpt_dir: Path, force: bool = False) -> None:
    """
    Original eval scripts load checkpoints from workspace root. Create symlinks:
        workspace/checkpoint_*.ckpt -> workspace/reproduced_ckpt/checkpoint_*.ckpt
    """
    require_path(ckpt_dir, "Missing checkpoint directory")

    names = [
        "checkpoint_ae.ckpt",
        "dataset_hl_without_fingering.ckpt",
        "dataset_ll.ckpt",
    ]

    for name in names:
        target = ckpt_dir / name
        link = workspace_dir / name
        require_path(target, "Missing checkpoint")

        if link.exists() or link.is_symlink():
            try:
                if link.resolve() == target.resolve():
                    print(f"[ckpt] root checkpoint already points to {target}")
                    continue
            except FileNotFoundError:
                pass

            if link.is_symlink():
                link.unlink()
            elif force:
                backup = workspace_dir / f"{name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                link.rename(backup)
                print(f"[ckpt] Existing {link} backed up to {backup}")
            else:
                raise FileExistsError(
                    f"{link} already exists and does not point to {target}.\n"
                    "Either remove it manually, or rerun with --force-root-ckpt-links."
                )

        link.symlink_to(target)
        print(f"[ckpt] {link} -> {target}")


def check_required_paths(repo_dir: Path, workspace_dir: Path, ckpt_dir: Path, song: str) -> None:
    require_path(ckpt_dir / "checkpoint_ae.ckpt", "Missing AE checkpoint")
    require_path(ckpt_dir / "dataset_hl_without_fingering.ckpt", "Missing high-level checkpoint")
    require_path(ckpt_dir / "dataset_ll.ckpt", "Missing low-level checkpoint")

    require_path(workspace_dir / "dataset_hl.zarr", "Missing high-level zarr dataset")
    require_path(workspace_dir / "dataset_ll.zarr", "Missing low-level zarr dataset")

    notes_train = workspace_dir / "dataset" / "notes" / f"{song}.pkl"
    notes_test = workspace_dir / "dataset" / "notes_test" / f"{song}.pkl"
    if not notes_train.exists() and not notes_test.exists():
        raise FileNotFoundError(
            "Cannot find note trajectory for this song. Expected one of:\n"
            f"  {notes_train}\n"
            f"  {notes_test}"
        )

    if notes_train.exists() and notes_test.exists():
        print(
            f"[warn] {song}.pkl exists in both dataset/notes and dataset/notes_test. "
            "The original code usually loads dataset/notes first."
        )


def remove_old_trajectories(repo_dir: Path, song: str) -> None:
    traj_dir = repo_dir / "multi_task" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{song}_trajectory.npy",
        f"{song}_left_hand_action_list.npy",
        f"{song}_right_hand_action_list.npy",
    ]:
        path = traj_dir / name
        if path.exists():
            path.unlink()
            print(f"[clean] Removed old trajectory: {path}")


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WANDB_DIR", "/tmp/robopianist/")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    return env


def run_and_log(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    print(f"[log] {log_path}")

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def parse_metric(text: str, label: str) -> Optional[float]:
    # Format: "Precision: 0.123"
    pattern = rf"(?mi)^\s*{re.escape(label)}\s*:?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))

    # Fallback: label on one line, value on the next line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == label.lower() and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                pass
    return None


def append_result_csv(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = [
        "timestamp",
        "song",
        "precision",
        "recall",
        "f1",
        "high_level_status",
        "low_level_status",
        "ckpt_dir",
        "high_level_log",
        "low_level_log",
    ]
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PianoMime one-song two-stage generalist evaluation without video."
    )
    parser.add_argument(
        "song",
        nargs="?",
        default=SONG_NAME,
        help="Song name without .pkl. If omitted, uses SONG_NAME inside this script.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIRNAME,
        help="Checkpoint directory relative to workspace, or absolute path. Default: reproduced_ckpt",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Default: 0")
    parser.add_argument(
        "--egl-device",
        default=None,
        help="MUJOCO_EGL_DEVICE_ID value. Default: same as --gpu",
    )
    parser.add_argument(
        "--force-root-ckpt-links",
        action="store_true",
        help=(
            "If workspace/checkpoint_*.ckpt already exists and is not the desired symlink, "
            "back it up and create the symlink."
        ),
    )
    parser.add_argument(
        "--keep-traj",
        action="store_true",
        help="Do not delete existing generated high-level trajectories before evaluation.",
    )
    parser.add_argument(
        "--skip-high-level",
        action="store_true",
        help="Skip eval_high_level.py and reuse existing trajectories under pianomime/multi_task/trajectories/.",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Do not patch eval_low_level.py. Use only if you already set record_dir=None manually.",
    )
    args = parser.parse_args()

    if args.egl_device is None:
        args.egl_device = args.gpu

    song = args.song.strip()
    if not song:
        raise ValueError("Song name is empty.")
    if song.endswith(".pkl"):
        song = song[:-4]

    repo_dir, workspace_dir = repo_and_workspace()
    ckpt_dir = resolve_ckpt_dir(workspace_dir, args.ckpt_dir)
    logs_dir = workspace_dir / "logs" / "generalist_eval_single"

    print(f"[info] repo_dir      = {repo_dir}")
    print(f"[info] workspace_dir = {workspace_dir}")
    print(f"[info] ckpt_dir      = {ckpt_dir}")
    print(f"[info] song          = {song}")

    check_required_paths(repo_dir, workspace_dir, ckpt_dir, song)
    ensure_root_checkpoint_links(
        workspace_dir,
        ckpt_dir,
        force=args.force_root_ckpt_links,
    )

    if not args.no_patch:
        patch_low_level_no_video(repo_dir)

    if not args.keep_traj and not args.skip_high_level:
        remove_old_trajectories(repo_dir, song)

    env = build_env(args)

    high_level_log = logs_dir / f"{song}_high_level.log"
    low_level_log = logs_dir / f"{song}_low_level.log"

    high_status = 0
    if not args.skip_high_level:
        high_cmd = [
            sys.executable,
            "pianomime/multi_task/eval_high_level.py",
            song,
        ]
        high_status = run_and_log(high_cmd, workspace_dir, env, high_level_log)
        if high_status != 0:
            print(f"[error] High-level evaluation failed with exit code {high_status}")
            return high_status
    else:
        print("[skip] Skipping high-level evaluation and reusing existing trajectories.")

    left_traj = repo_dir / "multi_task" / "trajectories" / f"{song}_left_hand_action_list.npy"
    right_traj = repo_dir / "multi_task" / "trajectories" / f"{song}_right_hand_action_list.npy"
    require_path(left_traj, "Missing generated left-hand trajectory")
    require_path(right_traj, "Missing generated right-hand trajectory")

    low_cmd = [
        sys.executable,
        "pianomime/multi_task/eval_low_level.py",
        song,
    ]
    low_status = run_and_log(low_cmd, workspace_dir, env, low_level_log)

    text = low_level_log.read_text(encoding="utf-8", errors="replace") if low_level_log.exists() else ""
    precision = parse_metric(text, "Precision")
    recall = parse_metric(text, "Recall")
    f1 = parse_metric(text, "F1")

    print("\n========== Result ==========")
    print(f"song:      {song}")
    print(f"precision: {precision}")
    print(f"recall:    {recall}")
    print(f"f1:        {f1}")
    print(f"HL status: {high_status}")
    print(f"LL status: {low_status}")

    csv_path = logs_dir / "results.csv"
    append_result_csv(
        csv_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "song": song,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "high_level_status": high_status,
            "low_level_status": low_status,
            "ckpt_dir": str(ckpt_dir),
            "high_level_log": str(high_level_log),
            "low_level_log": str(low_level_log),
        },
    )
    print(f"[saved] {csv_path}")

    return low_status


if __name__ == "__main__":
    raise SystemExit(main())
