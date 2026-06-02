"""Recover zarr-episode to NoteTrajectory pickle mappings using an old AE.

The released PianoMime generalist zarr datasets store encoded goal latents, but
they do not store which ``notes/*.pkl`` clip each episode came from. This script
recreates the old 16-D goal latents from candidate NoteTrajectory pickles with a
provided ``checkpoint_ae.ckpt`` and matches them against the goal-latent slices
already stored in ``dataset_hl.zarr`` and ``dataset_ll.zarr``.

Outputs a JSON file containing the best candidate per zarr episode plus matching
errors. A near-zero error means the mapping is effectively verified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import pickle
import sys
from typing import Iterable

import numpy as np
import torch

try:
    import zarr
except ImportError as exc:  # pragma: no cover - runtime dependency hint.
    raise SystemExit(
        "zarr is required. Install project requirements first: "
        "pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "goal_auto_encoder") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "goal_auto_encoder"))

import network as ae_network  # noqa: E402


@dataclasses.dataclass
class _PianoNote:
    number: int
    velocity: int
    key: int
    name: str
    fingering: int = -1


@dataclasses.dataclass
class _NoteTrajectory:
    dt: float
    notes: list
    sustains: list


class _FallbackUnpickler(pickle.Unpickler):
    """Load NoteTrajectory pickles without importing optional note_seq deps."""

    def find_class(self, module: str, name: str):
        if module == "robopianist.music.midi_file" and name == "PianoNote":
            return _PianoNote
        if module == "robopianist.music.midi_file" and name == "NoteTrajectory":
            return _NoteTrajectory
        return super().find_class(module, name)


def load_note_trajectory(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with path.open("rb") as f:
            return _FallbackUnpickler(f).load()


def note_trajectory_to_goal88(note_traj) -> np.ndarray:
    goals = np.zeros((len(note_traj.notes), 88), dtype=np.float32)
    for t, timestep_notes in enumerate(note_traj.notes):
        for note in timestep_notes:
            key = int(note.key)
            if 0 <= key < 88:
                goals[t, key] = 1.0
    return goals


def load_encoder(checkpoint: Path, device: torch.device):
    ae = ae_network.Autoencoder(latent_dim=16, cond_dim=64, device=str(device)).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    ae.load_state_dict(state_dict)
    ae.eval()
    encoder = ae.encoder
    encoder.eval()
    return encoder


def encode_goals(
    encoder: torch.nn.Module,
    goals: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    latents = []
    with torch.no_grad():
        for start in range(0, len(goals), batch_size):
            batch = goals[start : start + batch_size]
            x = torch.from_numpy(batch[:, :, None]).to(device=device, dtype=torch.float32)
            z = encoder.forward_without_sampling(x)
            latents.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0) if latents else np.zeros((0, 16), np.float32)


def pad_latents(latents: np.ndarray, lookahead: int) -> np.ndarray:
    zero_goal = np.zeros((lookahead, 16), dtype=np.float32)
    return np.concatenate([latents, zero_goal], axis=0)


def hl_expected(latents: np.ndarray, indices: np.ndarray, lookahead: int = 10) -> np.ndarray:
    padded = pad_latents(latents, lookahead)
    windows = [padded[indices + offset] for offset in range(lookahead + 1)]
    return np.concatenate(windows, axis=1)


def ll_expected(latents: np.ndarray, indices: np.ndarray, lookahead: int = 3) -> np.ndarray:
    padded = pad_latents(latents, lookahead)
    windows = [padded[indices + offset] for offset in range(lookahead + 1)]
    return np.stack(windows, axis=1)


def sample_indices(length: int, samples_per_episode: int) -> np.ndarray:
    if length <= samples_per_episode:
        return np.arange(length, dtype=np.int64)
    # Include beginning/end and evenly spaced interior points.
    return np.unique(np.linspace(0, length - 1, samples_per_episode, dtype=np.int64))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-dimension target ~= source * scale + bias."""
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    denom = np.sum(src_centered * src_centered, axis=0)
    numer = np.sum(src_centered * tgt_centered, axis=0)
    scale = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    bias = tgt_mean - scale * src_mean
    return scale.astype(np.float32), bias.astype(np.float32)


def apply_affine(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x * scale.reshape((1,) * (x.ndim - 1) + (-1,)) + bias.reshape(
        (1,) * (x.ndim - 1) + (-1,)
    )


def fit_minmax_to_range(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map per-dimension source min/max to target min/max."""
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    src_min = src.min(axis=0)
    src_max = src.max(axis=0)
    tgt_min = tgt.min(axis=0)
    tgt_max = tgt.max(axis=0)
    denom = src_max - src_min
    scale = np.divide(tgt_max - tgt_min, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    bias = tgt_min - scale * src_min
    return scale.astype(np.float32), bias.astype(np.float32)


def episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1].astype(np.int64)])
    ends = episode_ends.astype(np.int64)
    return list(zip(starts.tolist(), ends.tolist()))


def iter_pickles(input_dirs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_dir in input_dirs:
        if input_dir.exists():
            paths.extend(sorted(input_dir.glob("*.pkl")))
        else:
            print(f"Skipping missing directory: {input_dir}")
    return sorted(paths)


def build_candidate_cache(
    pickle_paths: list[Path],
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    candidates: list[dict] = []
    for i, path in enumerate(pickle_paths, 1):
        note_traj = load_note_trajectory(path)
        goals = note_trajectory_to_goal88(note_traj)
        latents = encode_goals(encoder, goals, device=device, batch_size=batch_size)
        candidates.append(
            {
                "path": path,
                "name": path.stem,
                "length": len(goals),
                "latents": latents,
            }
        )
        print(f"[{i:04d}/{len(pickle_paths):04d}] {path.name}: {latents.shape}")
    return candidates


def match_high_level_episode(
    state: np.ndarray,
    start: int,
    end: int,
    candidates: list[dict],
    samples_per_episode: int,
    transform_mode: str,
) -> list[dict]:
    length = end - start
    idx = sample_indices(length, samples_per_episode)
    stored = state[start + idx, : 11 * 16]
    rows = []
    for cand in candidates:
        if cand["length"] != length:
            continue
        expected = hl_expected(cand["latents"], idx, lookahead=10)
        transform = None
        if transform_mode == "affine":
            scale, bias = fit_affine(expected.reshape(-1, 16), stored.reshape(-1, 16))
            expected = apply_affine(expected.reshape(-1, 16), scale, bias).reshape(expected.shape)
            transform = {
                "mode": "affine",
                "scale_mean": float(np.mean(scale)),
                "scale_std": float(np.std(scale)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            }
        elif transform_mode == "minmax":
            scale, bias = fit_minmax_to_range(
                expected.reshape(-1, 16), stored.reshape(-1, 16)
            )
            expected = apply_affine(expected.reshape(-1, 16), scale, bias).reshape(expected.shape)
            transform = {
                "mode": "minmax",
                "scale_mean": float(np.mean(scale)),
                "scale_std": float(np.std(scale)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            }
        rows.append(
            {
                "path": str(cand["path"]),
                "name": cand["name"],
                "length": cand["length"],
                "mse": mse(stored, expected),
                "transform": transform,
            }
        )
    return sorted(rows, key=lambda row: row["mse"])


def match_low_level_episode(
    state: np.ndarray,
    start: int,
    end: int,
    candidates: list[dict],
    samples_per_episode: int,
    transform_mode: str,
) -> list[dict]:
    length = end - start
    idx = sample_indices(length, samples_per_episode)
    cont = state[start + idx, : 4 * 52].reshape(len(idx), 4, 52)
    stored = cont[:, :, :16]
    rows = []
    for cand in candidates:
        if cand["length"] != length:
            continue
        expected = ll_expected(cand["latents"], idx, lookahead=3)
        transform = None
        if transform_mode == "affine":
            scale, bias = fit_affine(expected.reshape(-1, 16), stored.reshape(-1, 16))
            expected = apply_affine(expected.reshape(-1, 16), scale, bias).reshape(expected.shape)
            transform = {
                "mode": "affine",
                "scale_mean": float(np.mean(scale)),
                "scale_std": float(np.std(scale)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            }
        elif transform_mode == "minmax":
            scale, bias = fit_minmax_to_range(
                expected.reshape(-1, 16), stored.reshape(-1, 16)
            )
            expected = apply_affine(expected.reshape(-1, 16), scale, bias).reshape(expected.shape)
            transform = {
                "mode": "minmax",
                "scale_mean": float(np.mean(scale)),
                "scale_std": float(np.std(scale)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            }
        rows.append(
            {
                "path": str(cand["path"]),
                "name": cand["name"],
                "length": cand["length"],
                "mse": mse(stored, expected),
                "transform": transform,
            }
        )
    return sorted(rows, key=lambda row: row["mse"])


def recover_mapping(
    dataset_path: Path,
    dataset_kind: str,
    candidates: list[dict],
    samples_per_episode: int,
    top_k: int,
    transform_mode: str,
) -> dict:
    root = zarr.open(str(dataset_path), "r")
    state = root["data"]["state"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    slices = episode_slices(episode_ends)

    episodes = []
    for episode_idx, (start, end) in enumerate(slices):
        if dataset_kind == "hl":
            matches = match_high_level_episode(
                state,
                start,
                end,
                candidates,
                samples_per_episode=samples_per_episode,
                transform_mode=transform_mode,
            )
        elif dataset_kind == "ll":
            matches = match_low_level_episode(
                state,
                start,
                end,
                candidates,
                samples_per_episode=samples_per_episode,
                transform_mode=transform_mode,
            )
        else:
            raise ValueError(f"Unknown dataset kind: {dataset_kind}")

        best = matches[0] if matches else None
        second = matches[1] if len(matches) > 1 else None
        episodes.append(
            {
                "episode": episode_idx,
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "best": best,
                "second_best": second,
                "top": matches[:top_k],
            }
        )
        if best is None:
            print(f"{dataset_kind} episode {episode_idx}: no length-matched candidate")
        else:
            second_mse = second["mse"] if second is not None else None
            print(
                f"{dataset_kind} episode {episode_idx:04d}: "
                f"{best['name']} mse={best['mse']:.6g} second={second_mse}"
            )

    return {
        "dataset": str(dataset_path),
        "kind": dataset_kind,
        "num_episodes": len(episodes),
        "samples_per_episode": samples_per_episode,
        "transform_mode": transform_mode,
        "episodes": episodes,
    }


def default_note_dirs(project_root: Path) -> list[Path]:
    return [
        project_root / "dataset" / "dataset" / "notes",
        project_root / "dataset" / "dataset" / "notes_test",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "ckpts" / "checkpoint_ae.ckpt",
        help="Original auto-encoder checkpoint used to encode the released zarr goals.",
    )
    parser.add_argument(
        "--note-dirs",
        nargs="*",
        type=Path,
        default=default_note_dirs(PROJECT_ROOT),
        help="Candidate directories containing NoteTrajectory .pkl files.",
    )
    parser.add_argument(
        "--hl-zarr",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "dataset_hl.zarr",
    )
    parser.add_argument(
        "--ll-zarr",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "dataset_ll.zarr",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "generalist_episode_mapping.json",
    )
    parser.add_argument(
        "--dataset",
        choices=["hl", "ll", "both"],
        default="both",
        help="Which zarr mapping to recover.",
    )
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--transform-mode",
        choices=["none", "affine", "minmax"],
        default="none",
        help=(
            "Compare raw encoded latents, or fit a per-episode per-dimension "
            "linear transform before comparing. Use affine/minmax to diagnose "
            "whether zarr goal latents were normalized."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for encoding candidate goals.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = load_encoder(args.checkpoint, device=device)
    pickle_paths = iter_pickles(args.note_dirs)
    if not pickle_paths:
        raise ValueError("No candidate .pkl files found.")

    candidates = build_candidate_cache(
        pickle_paths=pickle_paths,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "note_dirs": [str(path) for path in args.note_dirs],
        "candidate_count": len(candidates),
        "datasets": {},
    }
    if args.dataset in ("hl", "both"):
        result["datasets"]["hl"] = recover_mapping(
            dataset_path=args.hl_zarr,
            dataset_kind="hl",
            candidates=candidates,
            samples_per_episode=args.samples_per_episode,
            top_k=args.top_k,
            transform_mode=args.transform_mode,
        )
    if args.dataset in ("ll", "both"):
        result["datasets"]["ll"] = recover_mapping(
            dataset_path=args.ll_zarr,
            dataset_kind="ll",
            candidates=candidates,
            samples_per_episode=args.samples_per_episode,
            top_k=args.top_k,
            transform_mode=args.transform_mode,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote mapping to {args.output}")


if __name__ == "__main__":
    main()
