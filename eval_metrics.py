#!/usr/bin/env python3
"""Run one-song PianoMime generalist evaluation with explicit checkpoint paths.

This version supports both the original DDPM generalist and the new flow-matching
replacement. It intentionally does not rely on symlinks or a fixed reproduced_ckpt/
directory layout; pass the three checkpoints explicitly.

Examples from the workspace directory:

DDPM baseline:
    CUDA_VISIBLE_DEVICES=5 python pianomime/eval_metrics.py TwinkleTwinkleRousseau \
      --policy ddpm \
      --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt \
      --high-level-ckpt reproduced_ckpt/dataset_hl_without_fingering.ckpt \
      --low-level-ckpt reproduced_ckpt/dataset_ll.ckpt

Flow matching:
    CUDA_VISIBLE_DEVICES=5 python pianomime/eval_metrics.py TwinkleTwinkleRousseau \
      --policy flow \
      --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt \
      --high-level-ckpt flow/ckpts/checkpoint_FM-HL-dataset_hl_without_fingering.ckpt \
      --low-level-ckpt flow/ckpts/checkpoint_FM-LL-dataset_ll.ckpt \
      --flow-steps 20
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


DEFAULT_SONG_NAME = "Petrunko_3"


def repo_and_workspace() -> tuple[Path, Path]:
    """This file is expected to be placed directly under workspace/pianomime/."""
    repo_dir = Path(__file__).resolve().parent
    workspace_dir = repo_dir.parent
    return repo_dir, workspace_dir


def resolve_path(workspace_dir: Path, path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = workspace_dir / path
    return path.resolve()


def require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def check_required_paths(
    workspace_dir: Path,
    song: str,
    ae_ckpt: Path,
    high_level_ckpt: Path,
    low_level_ckpt: Path,
    dataset_hl: Path,
    dataset_ll: Path,
) -> None:
    require_path(ae_ckpt, "Missing AE checkpoint")
    require_path(high_level_ckpt, "Missing high-level checkpoint")
    require_path(low_level_ckpt, "Missing low-level checkpoint")
    require_path(dataset_hl, "Missing high-level zarr dataset")
    require_path(dataset_ll, "Missing low-level zarr dataset")

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
            "The environment loader usually tries dataset/notes first."
        )


def remove_old_trajectories(trajectory_dir: Path, song: str) -> None:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{song}_trajectory.npy",
        f"{song}_left_hand_action_list.npy",
        f"{song}_right_hand_action_list.npy",
    ]:
        path = trajectory_dir / name
        if path.exists():
            path.unlink()
            print(f"[clean] Removed old trajectory: {path}")


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WANDB_DIR", "/tmp/robopianist/")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Keep external CUDA_VISIBLE_DEVICES=... by default. Override only when --gpu is passed.
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.egl_device is not None:
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
        "label",
        "policy",
        "song",
        "precision",
        "recall",
        "f1",
        "high_level_status",
        "low_level_status",
        "ae_ckpt",
        "high_level_ckpt",
        "low_level_ckpt",
        "dataset_hl",
        "dataset_ll",
        "high_level_log",
        "low_level_log",
        "flow_steps",
        "flow_solver",
        "flow_clip_mode",
        "ddpm_hl_iters",
        "ddpm_ll_iters",
    ]
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_commands(
    args: argparse.Namespace,
    repo_dir: Path,
    song: str,
    ae_ckpt: Path,
    high_level_ckpt: Path,
    low_level_ckpt: Path,
    dataset_hl: Path,
    dataset_ll: Path,
    trajectory_dir: Path,
) -> tuple[list[str], list[str]]:
    if args.policy == "ddpm":
        high_script = repo_dir / "multi_task" / "eval_high_level.py"
        low_script = repo_dir / "multi_task" / "eval_low_level.py"
    else:
        high_script = repo_dir / "multi_task" / "flow_matching" / "eval_high_level_flow.py"
        low_script = repo_dir / "multi_task" / "flow_matching" / "eval_low_level_flow.py"

    require_path(high_script, "Missing high-level evaluation script")
    require_path(low_script, "Missing low-level evaluation script")

    high_cmd = [
        sys.executable,
        str(high_script),
        song,
        "--dataset-path",
        str(dataset_hl),
        "--ckpt-path",
        str(high_level_ckpt),
        "--ae-ckpt",
        str(ae_ckpt),
        "--trajectory-dir",
        str(trajectory_dir),
        "--lookahead",
        str(args.lookahead_hl),
    ]
    low_cmd = [
        sys.executable,
        str(low_script),
        song,
        "--dataset-path",
        str(dataset_ll),
        "--ckpt-path",
        str(low_level_ckpt),
        "--ae-ckpt",
        str(ae_ckpt),
        "--trajectory-dir",
        str(trajectory_dir),
        "--lookahead",
        str(args.lookahead_ll),
    ]

    if args.record_dir:
        high_cmd += ["--record-dir", args.record_dir]
        low_cmd += ["--record-dir", args.record_dir]
    if args.use_midi:
        high_cmd.append("--use-midi")
        low_cmd.append("--use-midi")
    if args.enable_ik:
        low_cmd.append("--enable-ik")

    if args.policy == "ddpm":
        high_cmd += ["--num-diffusion-iters", str(args.ddpm_hl_iters)]
        low_cmd += ["--num-diffusion-iters", str(args.ddpm_ll_iters)]
    else:
        flow_extra = [
            "--num-flow-steps",
            str(args.flow_steps),
            "--solver",
            args.flow_solver,
            "--clip-mode",
            args.flow_clip_mode,
            "--time-scale",
            str(args.flow_time_scale),
            "--noise-scale",
            str(args.flow_noise_scale),
        ]
        high_cmd += flow_extra
        low_cmd += flow_extra

    return high_cmd, low_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PianoMime one-song generalist evaluation with explicit ckpt paths.")
    parser.add_argument("song", nargs="?", default=DEFAULT_SONG_NAME, help="Song name without .pkl")
    parser.add_argument("--policy", choices=["ddpm", "flow"], required=True, help="Which sampler/eval scripts to use.")
    parser.add_argument("--ae-ckpt", required=True, help="Path to checkpoint_ae.ckpt")
    parser.add_argument("--high-level-ckpt", required=True, help="Path to high-level policy ckpt")
    parser.add_argument("--low-level-ckpt", required=True, help="Path to low-level policy ckpt")
    parser.add_argument("--dataset-hl", default="dataset_hl.zarr", help="Path to high-level zarr dataset")
    parser.add_argument("--dataset-ll", default="dataset_ll.zarr", help="Path to low-level zarr dataset")
    parser.add_argument("--trajectory-dir", default="pianomime/multi_task/trajectories")
    parser.add_argument("--label", default=None, help="Optional label written to results.csv, e.g. ddpm_100_50 or fm_20.")
    parser.add_argument("--log-dir", default="logs/generalist_eval_single")
    parser.add_argument("--gpu", default=None, help="Override CUDA_VISIBLE_DEVICES. By default, keep environment value.")
    parser.add_argument("--egl-device", default=None, help="Set MUJOCO_EGL_DEVICE_ID. By default, keep environment value.")
    parser.add_argument("--keep-traj", action="store_true", help="Do not delete existing generated high-level trajectories before evaluation.")
    parser.add_argument("--skip-high-level", action="store_true", help="Skip high-level eval and reuse existing trajectories.")
    parser.add_argument("--record-dir", default=None, help="Set to a directory to record video/audio. Default disables recording.")
    parser.add_argument("--lookahead-hl", type=int, default=10)
    parser.add_argument("--lookahead-ll", type=int, default=10)
    parser.add_argument("--enable-ik", action="store_true", help="Use IK residual mode in low-level eval.")
    parser.add_argument("--use-midi", action="store_true")

    # DDPM options.
    parser.add_argument("--ddpm-hl-iters", type=int, default=100)
    parser.add_argument("--ddpm-ll-iters", type=int, default=50)

    # Flow matching options.
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--flow-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--flow-clip-mode", choices=["none", "final", "step"], default="final")
    parser.add_argument("--flow-time-scale", type=float, default=100.0)
    parser.add_argument("--flow-noise-scale", type=float, default=1.0)

    args = parser.parse_args()

    song = args.song.strip()
    if not song:
        raise ValueError("Song name is empty.")
    if song.endswith(".pkl"):
        song = song[:-4]

    repo_dir, workspace_dir = repo_and_workspace()
    ae_ckpt = resolve_path(workspace_dir, args.ae_ckpt)
    high_level_ckpt = resolve_path(workspace_dir, args.high_level_ckpt)
    low_level_ckpt = resolve_path(workspace_dir, args.low_level_ckpt)
    dataset_hl = resolve_path(workspace_dir, args.dataset_hl)
    dataset_ll = resolve_path(workspace_dir, args.dataset_ll)
    trajectory_dir = resolve_path(workspace_dir, args.trajectory_dir)
    logs_dir = resolve_path(workspace_dir, args.log_dir)

    label = args.label or args.policy

    print(f"[info] repo_dir        = {repo_dir}")
    print(f"[info] workspace_dir   = {workspace_dir}")
    print(f"[info] policy          = {args.policy}")
    print(f"[info] label           = {label}")
    print(f"[info] song            = {song}")
    print(f"[info] ae_ckpt         = {ae_ckpt}")
    print(f"[info] high_level_ckpt = {high_level_ckpt}")
    print(f"[info] low_level_ckpt  = {low_level_ckpt}")

    check_required_paths(workspace_dir, song, ae_ckpt, high_level_ckpt, low_level_ckpt, dataset_hl, dataset_ll)

    if not args.keep_traj and not args.skip_high_level:
        remove_old_trajectories(trajectory_dir, song)

    env = build_env(args)
    high_level_log = logs_dir / f"{song}_{label}_high_level.log"
    low_level_log = logs_dir / f"{song}_{label}_low_level.log"

    high_cmd, low_cmd = build_commands(
        args,
        repo_dir,
        song,
        ae_ckpt,
        high_level_ckpt,
        low_level_ckpt,
        dataset_hl,
        dataset_ll,
        trajectory_dir,
    )

    high_status = 0
    if not args.skip_high_level:
        high_status = run_and_log(high_cmd, workspace_dir, env, high_level_log)
        if high_status != 0:
            print(f"[error] High-level evaluation failed with exit code {high_status}")
            return high_status
    else:
        print("[skip] Skipping high-level evaluation and reusing existing trajectories.")

    left_traj = trajectory_dir / f"{song}_left_hand_action_list.npy"
    right_traj = trajectory_dir / f"{song}_right_hand_action_list.npy"
    require_path(left_traj, "Missing generated left-hand trajectory")
    require_path(right_traj, "Missing generated right-hand trajectory")

    low_status = run_and_log(low_cmd, workspace_dir, env, low_level_log)

    text = low_level_log.read_text(encoding="utf-8", errors="replace") if low_level_log.exists() else ""
    precision = parse_metric(text, "Precision")
    recall = parse_metric(text, "Recall")
    f1 = parse_metric(text, "F1")

    print("\n========== Result ==========")
    print(f"label:     {label}")
    print(f"policy:    {args.policy}")
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
            "label": label,
            "policy": args.policy,
            "song": song,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "high_level_status": high_status,
            "low_level_status": low_status,
            "ae_ckpt": str(ae_ckpt),
            "high_level_ckpt": str(high_level_ckpt),
            "low_level_ckpt": str(low_level_ckpt),
            "dataset_hl": str(dataset_hl),
            "dataset_ll": str(dataset_ll),
            "high_level_log": str(high_level_log),
            "low_level_log": str(low_level_log),
            "flow_steps": args.flow_steps if args.policy == "flow" else "",
            "flow_solver": args.flow_solver if args.policy == "flow" else "",
            "flow_clip_mode": args.flow_clip_mode if args.policy == "flow" else "",
            "ddpm_hl_iters": args.ddpm_hl_iters if args.policy == "ddpm" else "",
            "ddpm_ll_iters": args.ddpm_ll_iters if args.policy == "ddpm" else "",
        },
    )
    print(f"[saved] {csv_path}")

    return low_status


if __name__ == "__main__":
    raise SystemExit(main())
