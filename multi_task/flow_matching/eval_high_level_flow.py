from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

# Robust imports when running directly as:
#   python pianomime/multi_task/flow_matching/eval_high_level_flow.py SONG
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
from network import ConditionalUnet1D, VariationalConvMlpEncoder  # noqa: E402
from utils import adjust_ft_fingering, get_env_hl, get_flattend_obs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-level fingertip trajectories with a flow-matching policy.")
    parser.add_argument("task_name", type=str, help="Song/task name without .pkl")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default="./dataset_hl.zarr")
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
    parser.add_argument("--use-midi", "--use_midi", dest="use_midi", action="store_true")
    return parser.parse_args()


def build_model(device: torch.device) -> ConditionalUnet1D:
    obs_dim = 212
    action_dim = 36

    def create_midi_encoder(device=device):
        return VariationalConvMlpEncoder(
            in_channels=16,
            mid_channels=32,
            out_channels=64,
            latent_dim=32,
            noise=0.08,
        ).to(device)

    return ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        midi_dim=obs_dim,
        midi_cond_dim=36,
        midi_encoder=create_midi_encoder,
        freeze_encoder=False,
    ).to(device)


def default_hl_ckpt(dataset_path: str) -> str:
    return f"flow/ckpts/checkpoint_FM-HL-{sanitize_name(dataset_path)}_without_fingering.ckpt"


def load_goal_encoder(device: torch.device, ae_ckpt: Optional[str]):
    ae = goal_auto_encoder.network.Autoencoder(latent_dim=16, cond_dim=64).to(device)
    ckpt_path = resolve_existing_path(
        [ae_ckpt, "./reproduced_ckpt/checkpoint_ae.ckpt", "./checkpoint_ae.ckpt"],
        what="goal auto-encoder checkpoint",
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict(state_dict)
    ae.eval()
    print(f"[FM-HL eval] loaded goal AE: {ckpt_path}")
    return ae.encoder


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pred_horizon = 4
    action_horizon = 1
    obs_horizon = 1
    midi_channel = 16
    action_dim = 36
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
    ckpt_path = resolve_existing_path([args.ckpt_path, default_hl_ckpt(args.dataset_path)], what="high-level flow checkpoint")
    state_dict = torch.load(ckpt_path, map_location=device)
    velocity_net.load_state_dict(state_dict)
    velocity_net.eval()
    print(f"[FM-HL eval] loaded checkpoint: {ckpt_path}")
    print(
        f"[FM-HL eval] flow_steps={args.num_flow_steps}, solver={args.solver}, "
        f"clip_mode={args.clip_mode}, time_scale={args.time_scale}"
    )

    env, max_steps = get_env_hl(
        args.task_name,
        record_dir=Path(args.record_dir) if args.record_dir else None,
        lookahead=args.lookahead,
        use_midi=args.use_midi,
    )
    trajectory_lh = np.zeros((max_steps, 3, 6))
    trajectory_rh = np.zeros((max_steps, 3, 6))
    trajectory = []

    timestep = env.reset()
    lh_current, rh_current = env.task.get_fingertip_pos(env.physics)
    last_fingertip_pos = np.concatenate((lh_current, rh_current), axis=0).flatten()

    step = 0
    last_lh_ft = None
    last_rh_ft = None
    last_keys = None
    last_fingering = None

    with tqdm(total=max_steps, desc="FM-HL Eval Env") as pbar:
        while not timestep.last():
            goal = get_flattend_obs(
                timestep,
                lookahead=args.lookahead,
                exclude_keys=["fingering", "hand", "fingering", "demo", "prior_action", "q_piano"],
                encoder=encoder,
                sampling=False,
            )
            # Keep original shape assumption explicit: 4 lookahead goal tokens of 16 dims + current fingertip state.
            _ = goal[: 4 * midi_channel].reshape((4, -1))
            obs = np.concatenate((goal, last_fingertip_pos), axis=-1).astype(np.float32)
            obs = normalize_data(obs, stats["obs"])
            obs_cond = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)

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

            # High-level predicts 36 fingertip dims. Append dummy 10-D fingering so original 46-D action stats work.
            naction_np = naction.detach().cpu().numpy()
            naction_full = np.concatenate((naction_np, np.zeros((batch_size, pred_horizon, 10))), axis=2).flatten()
            naction_full = unnormalize_data(naction_full, stats["action"])
            naction_full = naction_full.reshape(batch_size, pred_horizon, -1)
            nft = naction_full[0, :, :36]

            goal_keys = timestep.observation["goal"][:88]
            keys = np.nonzero(goal_keys)
            lh_ft, rh_ft, fingering = adjust_ft_fingering(
                env,
                keys,
                nft[0][:18].reshape(6, 3).T,
                nft[0][18:].reshape(6, 3).T,
                last_keys,
                last_lh_ft,
                last_rh_ft,
                last_fingering,
            )
            last_lh_ft = lh_ft
            last_rh_ft = rh_ft
            last_keys = keys
            last_fingering = fingering

            ft = np.concatenate((lh_ft.T.flatten(), rh_ft.T.flatten()))
            trajectory_lh[step] = lh_ft
            trajectory_rh[step] = rh_ft
            trajectory.append(ft.copy())
            last_fingertip_pos = ft

            step += 1
            timestep = env.step(np.zeros(47))
            pbar.update(1)

    out_dir = Path(args.trajectory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.task_name}_trajectory.npy", np.array(trajectory, dtype=np.float32))
    np.save(out_dir / f"{args.task_name}_left_hand_action_list.npy", trajectory_lh)
    np.save(out_dir / f"{args.task_name}_right_hand_action_list.npy", trajectory_rh)
    print(f"[FM-HL eval] saved trajectories to {out_dir}")


if __name__ == "__main__":
    main()
