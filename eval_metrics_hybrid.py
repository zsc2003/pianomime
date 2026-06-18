#!/usr/bin/env python3
"""Run one-song PianoMime generalist evaluation with explicit checkpoint paths.

Supports three samplers/policies:
  - DDPM: original PianoMime diffusion sampler.
  - DDIM: deterministic/stochastic DDIM sampler using the same DDPM checkpoints.
  - Flow: conditional flow-matching replacement.
  - DDIM-HL + DDPM-LL: accelerated high-level DDIM with original DDPM low-level.
  - Flow-HL + DDPM-LL: accelerated high-level flow matching with original DDPM low-level.

This wrapper also records wall-clock runtime for high-level, low-level, and total evaluation.

Important default:
  The released low-level expert actions in this project are residual-style actions
  generated from specialist PPO + IK prior. Therefore this wrapper enables IK
  residual execution by default. Use ``--no-enable-ik`` only when evaluating a
  checkpoint/dataset that was explicitly trained to output full direct actions.

Examples:
  CUDA_VISIBLE_DEVICES=4 python pianomime/eval_metrics.py TwinkleTwinkleLittleStar \
    --policy ddpm \
    --ae-ckpt given_ckpt/checkpoint_ae.ckpt \
    --high-level-ckpt given_ckpt/checkpoint_high_level.ckpt \
    --low-level-ckpt given_ckpt/checkpoint_low_level.ckpt \
    --label ddpm_given --use-midi

  CUDA_VISIBLE_DEVICES=4 python pianomime/eval_metrics.py TwinkleTwinkleLittleStar \
    --policy ddim --ddim-steps 50 \
    --ae-ckpt given_ckpt/checkpoint_ae.ckpt \
    --high-level-ckpt given_ckpt/checkpoint_high_level.ckpt \
    --low-level-ckpt given_ckpt/checkpoint_low_level.ckpt \
    --label ddim_given50 --use-midi

  CUDA_VISIBLE_DEVICES=4 python pianomime/eval_metrics.py TwinkleTwinkleRousseau \
    --policy flow --flow-steps 20 \
    --ae-ckpt reproduced_ckpt/checkpoint_ae.ckpt \
    --high-level-ckpt flow/ckpts/checkpoint_FM-HL-dataset_hl_without_fingering.ckpt \
    --low-level-ckpt flow/ckpts/checkpoint_FM-LL-dataset_ll.ckpt \
    --label fm20 --use-midi
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_PRELOAD"] = f"{conda_prefix}/lib/libstdc++.so.6"


DEFAULT_SONG_NAME = "Petrunko_3"

MIDI_SONGS = {
    "TwinkleTwinkleLittleStar",
    "CMajorScaleOneHand",
    "CMajorScaleTwoHands",
    "DMajorScaleOneHand",
    "DMajorScaleTwoHands",
    "CMajorChordProgressionTwoHands",
    "TwinkleTwinkleRousseau",
    "NocturneRousseau",
}


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


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


@contextlib.contextmanager
def trajectory_lock(workspace_dir: Path, song: str) -> Iterator[None]:
    """Serialize evals for the same song.

    The low-level eval scripts still route trajectories through
    pianomime/multi_task/trajectories/<song>_*.npy. Running several policies for the
    same song concurrently can otherwise make one policy consume another policy's
    high-level trajectory. This lock keeps different songs parallel but serializes
    same-song runs.
    """
    lock_dir = workspace_dir / "pianomime" / "multi_task" / "trajectory_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{sanitize_name(song)}.lock"
    with lock_path.open("w") as lock_file:
        print(f"[lock] waiting for {lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[lock] acquired {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[lock] released {lock_path}")


def check_required_paths(
    workspace_dir: Path,
    song: str,
    ae_ckpt: Path,
    high_level_ckpt: Path,
    low_level_ckpt: Path,
    dataset_hl: Path,
    dataset_ll: Path,
    use_midi: bool = False,
) -> None:
    require_path(ae_ckpt, "Missing AE checkpoint")
    require_path(high_level_ckpt, "Missing high-level checkpoint")
    require_path(low_level_ckpt, "Missing low-level checkpoint")
    require_path(dataset_hl, "Missing high-level zarr dataset")
    require_path(dataset_ll, "Missing low-level zarr dataset")

    notes_train = workspace_dir / "dataset" / "notes" / f"{song}.pkl"
    notes_test = workspace_dir / "dataset" / "notes_test" / f"{song}.pkl"

    if use_midi:
        if song not in MIDI_SONGS:
            print(
                f"[warn] --use-midi was passed for '{song}', but it is not in the known MIDI_SONGS list. "
                "The underlying robopianist.music.load(song) may still work if this is a valid registered MIDI name."
            )
    else:
        if not notes_train.exists() and not notes_test.exists():
            raise FileNotFoundError(
                "Cannot find note trajectory for this song. Expected one of:\n"
                f"  {notes_train}\n"
                f"  {notes_test}\n"
                "If this is a RoboPianist built-in MIDI song, pass --use-midi."
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


def run_and_log(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    """Run a subprocess, stream stdout/stderr to both terminal and log, and return status + elapsed seconds."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    print(f"[log] {log_path}")

    start_time = time.perf_counter()
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
        status = process.wait()
    elapsed_sec = time.perf_counter() - start_time
    print(f"[time] {log_path.name}: {elapsed_sec:.3f} sec")
    return status, elapsed_sec


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
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
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
        "runtime_hl_sec",
        "runtime_ll_sec",
        "runtime_total_sec",
        "wall_runtime_sec",
        "ae_ckpt",
        "high_level_ckpt",
        "low_level_ckpt",
        "dataset_hl",
        "dataset_ll",
        "high_level_log",
        "low_level_log",
        "use_midi",
        "enable_ik",
        "flow_steps",
        "flow_solver",
        "flow_clip_mode",
        "flow_hl_steps",
        "flow_ll_steps",
        "flow_hl_solver",
        "flow_ll_solver",
        "flow_hl_clip_mode",
        "flow_ll_clip_mode",
        "ddpm_hl_iters",
        "ddpm_ll_iters",
        "ddim_steps",
        "ddim_hl_steps",
        "ddim_ll_steps",
        "ddim_eta",
        "ddim_train_timesteps",
    ]
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    """Build high-level and low-level eval commands.

    Hybrid policies are intentionally supported here:
      - ddim_hl_ddpm_ll: high-level DDIM sampler, low-level original DDPM sampler.
      - flow_hl_ddpm_ll: high-level Flow Matching sampler, low-level original DDPM sampler.

    In both hybrid modes, ``high_level_ckpt`` is the high-level accelerated-policy
    checkpoint and ``low_level_ckpt`` is the DDPM low-level checkpoint.
    """
    if args.policy == "ddpm":
        high_script = repo_dir / "multi_task" / "eval_high_level.py"
        low_script = repo_dir / "multi_task" / "eval_low_level.py"
    elif args.policy == "ddim":
        high_script = repo_dir / "multi_task" / "ddim" / "eval_high_level_ddim.py"
        low_script = repo_dir / "multi_task" / "ddim" / "eval_low_level_ddim.py"
    elif args.policy == "flow":
        high_script = repo_dir / "multi_task" / "flow_matching" / "eval_high_level_flow.py"
        low_script = repo_dir / "multi_task" / "flow_matching" / "eval_low_level_flow.py"
    elif args.policy == "ddim_hl_ddpm_ll":
        high_script = repo_dir / "multi_task" / "ddim" / "eval_high_level_ddim.py"
        low_script = repo_dir / "multi_task" / "eval_low_level.py"
    elif args.policy == "flow_hl_ddpm_ll":
        high_script = repo_dir / "multi_task" / "flow_matching" / "eval_high_level_flow.py"
        low_script = repo_dir / "multi_task" / "eval_low_level.py"
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"Unknown policy: {args.policy}")

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

    elif args.policy == "ddim":
        ddim_hl_steps = args.ddim_hl_steps if args.ddim_hl_steps is not None else args.ddim_steps
        ddim_ll_steps = args.ddim_ll_steps if args.ddim_ll_steps is not None else args.ddim_steps
        high_cmd += [
            "--train-timesteps",
            str(args.ddim_train_timesteps),
            "--ddim-steps",
            str(ddim_hl_steps),
            "--eta",
            str(args.ddim_eta),
        ]
        low_cmd += [
            "--train-timesteps",
            str(args.ddim_train_timesteps),
            "--ddim-steps",
            str(ddim_ll_steps),
            "--eta",
            str(args.ddim_eta),
        ]

    elif args.policy == "ddim_hl_ddpm_ll":
        ddim_hl_steps = args.ddim_hl_steps if args.ddim_hl_steps is not None else args.ddim_steps
        high_cmd += [
            "--train-timesteps",
            str(args.ddim_train_timesteps),
            "--ddim-steps",
            str(ddim_hl_steps),
            "--eta",
            str(args.ddim_eta),
        ]
        low_cmd += ["--num-diffusion-iters", str(args.ddpm_ll_iters)]

    elif args.policy == "flow":
        flow_hl_steps = args.flow_hl_steps if args.flow_hl_steps is not None else args.flow_steps
        flow_ll_steps = args.flow_ll_steps if args.flow_ll_steps is not None else args.flow_steps
        flow_hl_solver = args.flow_hl_solver if args.flow_hl_solver is not None else args.flow_solver
        flow_ll_solver = args.flow_ll_solver if args.flow_ll_solver is not None else args.flow_solver
        flow_hl_clip_mode = args.flow_hl_clip_mode if args.flow_hl_clip_mode is not None else args.flow_clip_mode
        flow_ll_clip_mode = args.flow_ll_clip_mode if args.flow_ll_clip_mode is not None else args.flow_clip_mode

        flow_common = [
            "--time-scale",
            str(args.flow_time_scale),
            "--noise-scale",
            str(args.flow_noise_scale),
        ]
        high_cmd += [
            "--num-flow-steps",
            str(flow_hl_steps),
            "--solver",
            flow_hl_solver,
            "--clip-mode",
            flow_hl_clip_mode,
        ] + flow_common
        low_cmd += [
            "--num-flow-steps",
            str(flow_ll_steps),
            "--solver",
            flow_ll_solver,
            "--clip-mode",
            flow_ll_clip_mode,
        ] + flow_common

    elif args.policy == "flow_hl_ddpm_ll":
        flow_hl_steps = args.flow_hl_steps if args.flow_hl_steps is not None else args.flow_steps
        flow_hl_solver = args.flow_hl_solver if args.flow_hl_solver is not None else args.flow_solver
        flow_hl_clip_mode = args.flow_hl_clip_mode if args.flow_hl_clip_mode is not None else args.flow_clip_mode
        high_cmd += [
            "--num-flow-steps",
            str(flow_hl_steps),
            "--solver",
            flow_hl_solver,
            "--clip-mode",
            flow_hl_clip_mode,
            "--time-scale",
            str(args.flow_time_scale),
            "--noise-scale",
            str(args.flow_noise_scale),
        ]
        low_cmd += ["--num-diffusion-iters", str(args.ddpm_ll_iters)]

    return high_cmd, low_cmd


def print_metric_diagnostics(
    precision: Optional[float],
    recall: Optional[float],
    f1: Optional[float],
    enable_ik: bool,
    policy: str,
    label: str,
) -> None:
    if precision is None or recall is None or f1 is None:
        print("[diag] Could not parse one or more metrics from low-level log.")
        return
    if precision > 0.75 and recall < 0.25:
        print(
            "[diag] Precision is high but recall is very low. This usually means the robot "
            "presses few keys, and sklearn zero_division=1 makes precision look deceptively high "
            "on frames with no predicted positives."
        )
        if not enable_ik:
            print(
                "[diag] IK residual mode is OFF. The released PianoMime low-level data/checkpoints "
                "are commonly residual-style because specialist PPO actions are saved before the IK prior "
                "is added. Re-run with --enable-ik. This is the most likely cause of TwinkleTwinkleLittleStar "
                "getting recall around 0.1."
            )
        else:
            print(
                "[diag] IK residual mode is already ON. If recall is still low, check that the AE checkpoint "
                "and zarr normalization stats match the policy checkpoint, and make sure this song should be "
                "evaluated with --use-midi only if it is a RoboPianist MIDI-library task."
            )
    print(f"[diag] policy={policy}, label={label}, enable_ik={enable_ik}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PianoMime one-song generalist evaluation with explicit ckpt paths.")
    parser.add_argument("song", nargs="?", default=DEFAULT_SONG_NAME, help="Song name without .pkl")
    parser.add_argument("--policy", choices=["ddpm", "ddim", "flow", "ddim_hl_ddpm_ll", "flow_hl_ddpm_ll"], required=True, help="Which sampler/eval scripts to use.")
    parser.add_argument("--ae-ckpt", required=True, help="Path to checkpoint_ae.ckpt")
    parser.add_argument("--high-level-ckpt", required=True, help="Path to high-level policy ckpt")
    parser.add_argument("--low-level-ckpt", required=True, help="Path to low-level policy ckpt")
    parser.add_argument("--dataset-hl", default="dataset_hl.zarr", help="Path to high-level zarr dataset")
    parser.add_argument("--dataset-ll", default="dataset_ll.zarr", help="Path to low-level zarr dataset")
    parser.add_argument("--trajectory-dir", default="pianomime/multi_task/trajectories")
    parser.add_argument("--label", default=None, help="Optional label written to results.csv, e.g. ddpm_given or ddim50.")
    parser.add_argument("--log-dir", default="logs/generalist_eval_single")
    parser.add_argument("--gpu", default=None, help="Override CUDA_VISIBLE_DEVICES. By default, keep environment value.")
    parser.add_argument("--egl-device", default=None, help="Set MUJOCO_EGL_DEVICE_ID. By default, keep environment value.")
    parser.add_argument("--keep-traj", action="store_true", help="Do not delete existing generated high-level trajectories before evaluation.")
    parser.add_argument("--skip-high-level", action="store_true", help="Skip high-level eval and reuse existing trajectories.")
    parser.add_argument("--record-dir", default=None, help="Set to a directory to record video/audio. Default disables recording.")
    parser.add_argument("--lookahead-hl", type=int, default=10)
    parser.add_argument("--lookahead-ll", type=int, default=10)
    parser.add_argument(
        "--enable-ik",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use IK residual mode in low-level eval. Default: True. Use --no-enable-ik for direct-action checkpoints.",
    )
    parser.add_argument("--use-midi", action="store_true")

    # DDPM options.
    parser.add_argument("--ddpm-hl-iters", type=int, default=100)
    parser.add_argument("--ddpm-ll-iters", type=int, default=50)

    # DDIM options.
    parser.add_argument("--ddim-steps", type=int, default=50, help="Use the same DDIM inference steps for high-level and low-level.")
    parser.add_argument("--ddim-hl-steps", type=int, default=None, help="Override DDIM high-level steps.")
    parser.add_argument("--ddim-ll-steps", type=int, default=None, help="Override DDIM low-level steps.")
    parser.add_argument("--ddim-train-timesteps", type=int, default=100)
    parser.add_argument("--ddim-eta", type=float, default=0.0)

    # Flow matching options.
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument("--flow-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--flow-clip-mode", choices=["none", "final", "step"], default="final")
    parser.add_argument("--flow-hl-steps", type=int, default=None, help="Override high-level Flow ODE steps.")
    parser.add_argument("--flow-ll-steps", type=int, default=None, help="Override low-level Flow ODE steps.")
    parser.add_argument("--flow-hl-solver", choices=["euler", "heun"], default=None, help="Override high-level Flow solver.")
    parser.add_argument("--flow-ll-solver", choices=["euler", "heun"], default=None, help="Override low-level Flow solver.")
    parser.add_argument("--flow-hl-clip-mode", choices=["none", "final", "step"], default=None, help="Override high-level Flow clip mode.")
    parser.add_argument("--flow-ll-clip-mode", choices=["none", "final", "step"], default=None, help="Override low-level Flow clip mode.")
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
    ddim_hl_steps = args.ddim_hl_steps if args.ddim_hl_steps is not None else args.ddim_steps
    ddim_ll_steps = args.ddim_ll_steps if args.ddim_ll_steps is not None else args.ddim_steps
    flow_hl_steps = args.flow_hl_steps if args.flow_hl_steps is not None else args.flow_steps
    flow_ll_steps = args.flow_ll_steps if args.flow_ll_steps is not None else args.flow_steps
    flow_hl_solver = args.flow_hl_solver if args.flow_hl_solver is not None else args.flow_solver
    flow_ll_solver = args.flow_ll_solver if args.flow_ll_solver is not None else args.flow_solver
    flow_hl_clip_mode = args.flow_hl_clip_mode if args.flow_hl_clip_mode is not None else args.flow_clip_mode
    flow_ll_clip_mode = args.flow_ll_clip_mode if args.flow_ll_clip_mode is not None else args.flow_clip_mode

    print(f"[info] repo_dir        = {repo_dir}")
    print(f"[info] workspace_dir   = {workspace_dir}")
    print(f"[info] policy          = {args.policy}")
    print(f"[info] label           = {label}")
    print(f"[info] song            = {song}")
    print(f"[info] use_midi        = {args.use_midi}")
    print(f"[info] enable_ik       = {args.enable_ik}")
    print(f"[info] ae_ckpt         = {ae_ckpt}")
    print(f"[info] high_level_ckpt = {high_level_ckpt}")
    print(f"[info] low_level_ckpt  = {low_level_ckpt}")
    if args.policy in {"flow", "flow_hl_ddpm_ll"}:
        print(f"[info] flow_hl_steps   = {flow_hl_steps}")
        if args.policy == "flow":
            print(f"[info] flow_ll_steps   = {flow_ll_steps}")
        print(f"[info] flow_hl_solver  = {flow_hl_solver}")
        if args.policy == "flow":
            print(f"[info] flow_ll_solver  = {flow_ll_solver}")
        print(f"[info] flow_hl_clip    = {flow_hl_clip_mode}")
        if args.policy == "flow":
            print(f"[info] flow_ll_clip    = {flow_ll_clip_mode}")

    check_required_paths(workspace_dir, song, ae_ckpt, high_level_ckpt, low_level_ckpt, dataset_hl, dataset_ll, args.use_midi)

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
    low_status = 1
    high_runtime_sec = 0.0
    low_runtime_sec = 0.0
    wall_eval_start = time.perf_counter()
    with trajectory_lock(workspace_dir, song):
        if not args.keep_traj and not args.skip_high_level:
            remove_old_trajectories(trajectory_dir, song)

        if not args.skip_high_level:
            high_status, high_runtime_sec = run_and_log(high_cmd, workspace_dir, env, high_level_log)
            if high_status != 0:
                print(f"[error] High-level evaluation failed with exit code {high_status}")
                return high_status
        else:
            print("[skip] Skipping high-level evaluation and reusing existing trajectories.")

        left_traj = trajectory_dir / f"{song}_left_hand_action_list.npy"
        right_traj = trajectory_dir / f"{song}_right_hand_action_list.npy"
        require_path(left_traj, "Missing generated left-hand trajectory")
        require_path(right_traj, "Missing generated right-hand trajectory")

        low_status, low_runtime_sec = run_and_log(low_cmd, workspace_dir, env, low_level_log)

    # Report policy evaluation runtime as high-level subprocess time + low-level subprocess time.
    # This intentionally excludes time waiting on the same-song trajectory lock.
    runtime_total_sec = high_runtime_sec + low_runtime_sec
    wall_runtime_sec = time.perf_counter() - wall_eval_start

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
    print(f"runtime_sec:      {runtime_total_sec:.3f}")
    print(f"HL runtime_sec:   {high_runtime_sec:.3f}")
    print(f"LL runtime_sec:   {low_runtime_sec:.3f}")
    print(f"wall_runtime_sec: {wall_runtime_sec:.3f}")
    print(f"HL status: {high_status}")
    print(f"LL status: {low_status}")
    print_metric_diagnostics(precision, recall, f1, args.enable_ik, args.policy, label)

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
            "runtime_hl_sec": f"{high_runtime_sec:.6f}",
            "runtime_ll_sec": f"{low_runtime_sec:.6f}",
            "runtime_total_sec": f"{runtime_total_sec:.6f}",
            "wall_runtime_sec": f"{wall_runtime_sec:.6f}",
            "ae_ckpt": str(ae_ckpt),
            "high_level_ckpt": str(high_level_ckpt),
            "low_level_ckpt": str(low_level_ckpt),
            "dataset_hl": str(dataset_hl),
            "dataset_ll": str(dataset_ll),
            "high_level_log": str(high_level_log),
            "low_level_log": str(low_level_log),
            "use_midi": args.use_midi,
            "enable_ik": args.enable_ik,
            "flow_steps": args.flow_steps if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_solver": args.flow_solver if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_clip_mode": args.flow_clip_mode if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_hl_steps": flow_hl_steps if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_ll_steps": flow_ll_steps if args.policy == "flow" else "",
            "flow_hl_solver": flow_hl_solver if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_ll_solver": flow_ll_solver if args.policy == "flow" else "",
            "flow_hl_clip_mode": flow_hl_clip_mode if args.policy in {"flow", "flow_hl_ddpm_ll"} else "",
            "flow_ll_clip_mode": flow_ll_clip_mode if args.policy == "flow" else "",
            "ddpm_hl_iters": args.ddpm_hl_iters if args.policy == "ddpm" else "",
            "ddpm_ll_iters": args.ddpm_ll_iters if args.policy in {"ddpm", "ddim_hl_ddpm_ll", "flow_hl_ddpm_ll"} else "",
            "ddim_steps": args.ddim_steps if args.policy in {"ddim", "ddim_hl_ddpm_ll"} else "",
            "ddim_hl_steps": ddim_hl_steps if args.policy in {"ddim", "ddim_hl_ddpm_ll"} else "",
            "ddim_ll_steps": ddim_ll_steps if args.policy == "ddim" else "",
            "ddim_eta": args.ddim_eta if args.policy in {"ddim", "ddim_hl_ddpm_ll"} else "",
            "ddim_train_timesteps": args.ddim_train_timesteps if args.policy in {"ddim", "ddim_hl_ddpm_ll"} else "",
        },
    )
    print(f"[saved] {csv_path}")

    return low_status


if __name__ == "__main__":
    raise SystemExit(main())
