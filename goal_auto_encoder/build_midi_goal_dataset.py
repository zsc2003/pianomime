"""Build an SDF auto-encoder dataset from PianoMime NoteTrajectory pickles.

The auto-encoder training code expects a zarr dataset with:

    data/state: (N, 88) float32

Each row is a multi-hot piano-key target for one control timestep. This script
converts ``dataset/notes/*.pkl`` and optionally ``dataset/notes_test/*.pkl`` into
that format.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np

try:
    import zarr
except ImportError as exc:  # pragma: no cover - helpful runtime error.
    raise SystemExit(
        "zarr is required to build this dataset. Install project requirements first: "
        "pip install -r requirements.txt"
    ) from exc


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


def _load_note_trajectory(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with path.open("rb") as f:
            return _FallbackUnpickler(f).load()


def _note_trajectory_to_goal88(note_traj) -> np.ndarray:
    goals = np.zeros((len(note_traj.notes), 88), dtype=np.float32)
    for t, timestep_notes in enumerate(note_traj.notes):
        for note in timestep_notes:
            key = int(note.key)
            if 0 <= key < 88:
                goals[t, key] = 1.0
    return goals


def _iter_pickles(input_dirs: Iterable[Path], clips_only: bool) -> list[Path]:
    paths: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Skipping missing directory: {input_dir}")
            continue
        for path in sorted(input_dir.glob("*.pkl")):
            if clips_only and "_" not in path.stem:
                continue
            paths.append(path)
    return paths


def build_dataset(
    input_dirs: list[Path],
    output: Path,
    clips_only: bool,
    chunk_size: int,
) -> None:
    pickle_paths = _iter_pickles(input_dirs, clips_only=clips_only)
    if not pickle_paths:
        raise ValueError("No .pkl files found. Check --input-dirs and --clips-only.")

    arrays: list[np.ndarray] = []
    episode_ends: list[int] = []
    total = 0

    for path in pickle_paths:
        note_traj = _load_note_trajectory(path)
        goals = _note_trajectory_to_goal88(note_traj)
        arrays.append(goals)
        total += len(goals)
        episode_ends.append(total)
        print(f"{path.name}: {goals.shape}")

    data = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)

    root = zarr.open(str(output), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    data_group.create_dataset(
        "state",
        data=data,
        chunks=(min(chunk_size, len(data)), 88),
        dtype="f4",
    )
    meta_group.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(min(len(episode_ends), chunk_size),),
        dtype="i8",
    )
    root.attrs["source_files"] = [str(path) for path in pickle_paths]
    root.attrs["format"] = "data/state is (N, 88) multi-hot piano key goal"

    print(f"\nWrote {output}")
    print(f"data/state: {data.shape}, dtype={data.dtype}")
    print(f"episodes: {len(episode_ends)}")


def _default_input_dirs(project_root: Path) -> list[Path]:
    nested_notes = project_root / "dataset" / "dataset" / "notes"
    nested_notes_test = project_root / "dataset" / "dataset" / "notes_test"
    if nested_notes.exists() or nested_notes_test.exists():
        return [nested_notes, nested_notes_test]
    return [
        project_root / "dataset" / "notes",
        project_root / "dataset" / "notes_test",
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="*",
        type=Path,
        default=_default_input_dirs(project_root),
        help="Directories containing NoteTrajectory .pkl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset" / "ae_midi_goal.zarr",
        help="Output zarr dataset path.",
    )
    parser.add_argument(
        "--clips-only",
        action="store_true",
        help="Use only files whose stem contains '_' to avoid full-song duplicates.",
    )
    parser.add_argument("--chunk-size", type=int, default=16384)
    args = parser.parse_args()

    build_dataset(
        input_dirs=args.input_dirs,
        output=args.output,
        clips_only=args.clips_only,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
