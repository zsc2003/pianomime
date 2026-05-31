from __future__ import annotations

import argparse
import collections
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

# Robust imports when running directly as:
#   python pianomime/multi_task/flow_matching/eval_low_level_flow.py SONG
_THIS_FILE = Path(__file__).resolve()
_FLOW_DIR = _THIS_FILE.parent
_MULTI_TASK_DIR = _FLOW_DIR.parent
_REPO_DIR = _MULTI_TASK_DIR.parent
_WORKSPACE_DIR = _REPO_DIR.parent
for _p in (str(_FLOW_DIR), str(_MULTI_TASK_DIR), str(_REPO_DIR), str(_WORKSPACE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import goal_auto_encoder.network  # noqa: E402
from dataset import normalize_data, read_dataset, unnormalize_data  # noqa: E402
from flow_matching_utils import resolve_existing_path, sample_flow, sanitize_name, set_seed  # noqa: E402
from network import ConditionalUnet1D, ConvEncoder  # noqa: E402
from utils import get_env_ll, get_flattend_obs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-level PianoMime policy with conditional flow matching.")
    parser.add_argument("task_name", type=str, help="Song/task name without .pkl")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default="./dataset_ll.zarr")
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", type=str, default=None)
    parser.add_argument("--ae-ckpt", "--ae_ckpt", dest="ae_ckpt", type=str, default=None)
    parser.add_argument("--trajectory-dir", "--trajectory_dir", dest="trajectory_dir", type=str, default="pianomime/multi_task/trajectories")
    parser.add_argument("--record-dir", "--record_dir", dest="record_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--num-flow-steps", "--flow-steps", "--flow_steps", dest="num_flow_steps", type=int, default=20)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--clip-mode", "--clip_mode", dest="clip_mode", choices=["none", "final", "step"], default="final")
    parser.add_argument("--time-scale", "--time_scale", dest="time_scale", type=float, default=100.0)
    parser.add_argument("--noise-scale", "--noise_scale", dest="noise_scale", type=float, default=1.0)
    parser.add_argument("--enable-ik", "--enable_ik", dest="enable_ik", action="store_true", help="Use IK residual mode. Default follows original eval_low_level.py: False.")
    parser.add_argument("--use-midi", "--use_midi", dest="use_midi", action="store_true")
    return parser.parse_args()


def build_model(device: torch.device) -> ConditionalUnet1D:
    obs_dim = 404
    action_dim = 46

    def create_midi_encoder(device=device):
        return ConvEncoder(
            in_channels=52,
            mid_channels=64,
            out_channels=128,
            horizon=4,
            noise_fingering=0,
            noise_ft=0,
        ).to(device)

    return ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        midi_dim=208,
        midi_cond_dim=0,
        midi_encoder=create_midi_encoder,
        freeze_encoder=False,
    ).to(device)


def default_ll_ckpt(dataset_path: str) -> str:
    return f"flow/ckpts/checkpoint_FM-LL-{sanitize_name(dataset_path)}.ckpt"


def load_goal_encoder(device: torch.device, ae_ckpt: Optional[str]):
    ae = goal_auto_encoder.network.Autoencoder(latent_dim=16, cond_dim=64).to(device)
    ckpt_path = resolve_existing_path(
        [ae_ckpt, "./reproduced_ckpt/checkpoint_ae.ckpt", "./checkpoint_ae.ckpt"],
        what="goal auto-encoder checkpoint",
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict(state_dict)
    ae.eval()
    print(f"[FM-LL eval] loaded goal AE: {ckpt_path}")
    return ae.encoder


def ensure_utils_can_find_trajectories(task_name: str, trajectory_dir: str) -> Path:
    """utils.get_env_ll loads hard-coded pianomime/multi_task/trajectories/*.npy.

    If a custom --trajectory-dir is used, copy the required files to the hard-coded directory
    before constructing the environment.
    """
    source_dir = Path(trajectory_dir)
    target_dir = Path("pianomime/multi_task/trajectories")
    names = [
        f"{task_name}_left_hand_action_list.npy",
        f"{task_name}_right_hand_action_list.npy",
    ]
    for name in names:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing high-level trajectory {src}. Run eval_high_level_flow.py first.")
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != target_dir.resolve():
        for name in names:
            shutil.copy2(source_dir / name, target_dir / name)
    return target_dir


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pred_horizon = 4
    action_horizon = 4
    obs_horizon = 1
    action_dim = 46
    batch_size = 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _dataloader, stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=args.dataset_path,
        normalization=True,
    )

    encoder = load_goal_encoder(device, args.ae_ckpt)

    velocity_net = build_model(device)
    ckpt_path = resolve_existing_path([args.ckpt_path, default_ll_ckpt(args.dataset_path)], what="low-level flow checkpoint")
    state_dict = torch.load(ckpt_path, map_location=device)
    velocity_net.load_state_dict(state_dict)
    velocity_net.eval()
    print(f"[FM-LL eval] loaded checkpoint: {ckpt_path}")
    print(
        f"[FM-LL eval] flow_steps={args.num_flow_steps}, solver={args.solver}, "
        f"clip_mode={args.clip_mode}, time_scale={args.time_scale}"
    )

    trajectory_dir = ensure_utils_can_find_trajectories(args.task_name, args.trajectory_dir)
    left_hand_action_list = np.load(trajectory_dir / f"{args.task_name}_left_hand_action_list.npy")
    max_steps = left_hand_action_list.shape[0]

    env = get_env_ll(
        task_name=args.task_name,
        enable_ik=args.enable_ik,
        lookahead=args.lookahead,
        record_dir=Path(args.record_dir) if args.record_dir else None,
        use_fingering_emb=False,
        use_midi=args.use_midi,
    )

    timestep = env.reset()
    obs = get_flattend_obs(
        timestep,
        lookahead=3,
        exclude_keys=["fingering", "prior_action"],
        encoder=encoder,
        sampling=False,
        concatenate_keys=["goal", "demo"],
    )
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    step_idx = 0

    with tqdm(total=max_steps, desc="FM-LL Eval Env") as pbar:
        while not timestep.last():
            nobs = np.stack(obs_deque)
            nobs = normalize_data(nobs, stats["obs"])
            nobs_t = torch.from_numpy(nobs).to(device=device, dtype=torch.float32)
            obs_cond = nobs_t.unsqueeze(0).flatten(start_dim=1)

            naction = sample_flow(
                velocity_net,
                sample_shape=(batch_size, pred_horizon, action_dim),
                global_cond=obs_cond,
                num_steps=args.num_flow_steps,
                solver=args.solver,
                clip_mode=args.clip_mode,
                time_scale=args.time_scale,
                noise_scale=args.noise_scale,
            )
            action_pred = naction.detach().cpu().numpy()[0]
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]

            for i in range(len(action)):
                action_i = unnormalize_data(action[i], stats=stats["action"])
                timestep = env.step(np.append(action_i, 0))
                if timestep.last():
                    break

                step_idx += 1
                if step_idx < left_hand_action_list.shape[0]:
                    obs = get_flattend_obs(
                        timestep,
                        lookahead=3,
                        exclude_keys=["fingering", "prior_action"],
                        encoder=encoder,
                        sampling=False,
                        concatenate_keys=["goal", "demo"],
                    )
                obs_deque.append(obs)
                pbar.update(1)

    metric = env.get_musical_metrics()
    precision = metric["precision"]
    recall = metric["recall"]
    f1 = metric["f1"]
    print(args.task_name)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


if __name__ == "__main__":
    main()
