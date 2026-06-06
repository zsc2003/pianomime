#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalar(log_dir: Path, tag: str):
    """Read a scalar tag from TensorBoard event files under log_dir."""
    event_dirs = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_dirs:
        return None, None
    # Use the parent directory of the event file
    event_dir = event_dirs[0].parent
    ea = EventAccumulator(str(event_dir))
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def discover_runs(log_root: Path):
    """Find all run directories that contain TensorBoard event files."""
    runs = []
    if not log_root.is_dir():
        return runs
    for d in sorted(log_root.iterdir()):
        if not d.is_dir():
            continue
        # A valid run dir must contain at least one event file
        if any(d.rglob("events.out.tfevents.*")):
            runs.append(d)
    return runs


def ema(values, smoothing: float):
    """Exponential moving average, matching TensorBoard's smoothing algorithm."""
    if smoothing == 0 or len(values) == 0:
        return list(values)
    smoothed = []
    last = values[0]
    for v in values:
        last = last * smoothing + v * (1 - smoothing)
        smoothed.append(last)
    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from TensorBoard logs. "
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="~/shared-nvme/pianomime_logs",
        help="Root log directory containing multiple run directories "
        "(default: ~/shared-nvme/pianomime_logs).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="Exponential moving average smoothing factor (0=no smoothing, "
        "closer to 1=more smooth). Matches TensorBoard's smoothing slider. "
        "(default: 0.6)",
    )
    args = parser.parse_args()

    log_root = Path(args.log_dir).expanduser().resolve()

    if not log_root.exists():
        print(f"Log directory not found: {log_root}", file=sys.stderr)
        sys.exit(1)

    runs = discover_runs(log_root)
    if not runs:
        print(f"No TensorBoard runs found under {log_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s) under {log_root}")
    for r in runs:
        print(f"  {r.name}")

    tags = [
        "eval/f1",
        "eval/reward_smoothness_reward",
        "eval/smoothness_action_rate",
        "eval/smoothness_accel_penalty",
    ]

    RUN_LABELS = {
        "PPO-NeverGonnaGiveYouUp_1-42-20260605-023428": "Baseline",
        "PPO-NeverGonnaGiveYouUp_1_smooth-42-20260605-210459": "Smooth1",
        "PPO-NeverGonnaGiveYouUp_1_smooth-42-20260606-150749": "Smooth2",
    }

    data = {tag: [] for tag in tags}
    for run_dir in runs:
        label = RUN_LABELS.get(run_dir.name, run_dir.name)
        for tag in tags:
            steps, values = read_scalar(run_dir, tag)
            if steps is not None:
                data[tag].append((label, steps, values))

    plots_dir = log_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        curves = data[tag]
        metric_name = tag.replace("/", "_")

        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not curves:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=18)
            ax.set_xlabel("Iteration", fontsize=16)
        else:
            for label, steps, values in curves:
                smoothed = ema(values, args.smoothing)
                ax.plot(steps, smoothed, linewidth=1.5, label=label)
            ax.set_xlabel("Iteration", fontsize=16)
            ax.set_ylabel(tag.split("/")[-1], fontsize=16)
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.3)
            if len(curves) > 1:
                ax.legend(fontsize=12)

        plt.tight_layout()
        out_path = plots_dir / f"{metric_name}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
