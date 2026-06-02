import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm.auto import tqdm

directory = "pianomime"
if directory not in sys.path:
    sys.path.append(directory)

from dataset import normalize_data, read_dataset, unnormalize_data
from network import ConditionalUnet1D, VariationalConvMlpEncoder
from utils import adjust_ft_fingering, get_env_hl, get_flattend_obs
import goal_auto_encoder.network


def _resolve_path(path: str) -> str:
    """Keep original script paths usable from either repo root or its parent."""
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if path.startswith("pianomime/") or path.startswith("pianomime\\"):
        fallback = Path(path.split("/", 1)[-1])
        if fallback.exists():
            return str(fallback)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-level PianoMime inference with DDIM sampling."
    )
    parser.add_argument("task_name", help="Song/clip name, same as eval_high_level.py.")

    # Core paths. Defaults mirror the original DDPM script.
    parser.add_argument("--dataset-path", default="pianomime/dataset_hl.zarr")
    parser.add_argument("--ae-ckpt", default="checkpoint_ae.ckpt")
    parser.add_argument("--high-level-ckpt", default="checkpoint_high_level.ckpt")
    parser.add_argument("--output-dir", default="pianomime/multi_task/trajectories")

    # DDIM hyperparameters. These are the main knobs to tune.
    parser.add_argument("--train-timesteps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0.0 gives deterministic DDIM; >0 adds stochasticity.")
    parser.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    parser.add_argument("--clip-sample", action=argparse.BooleanOptionalAction,
                        default=True)

    # Runtime controls.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=10)
    return parser.parse_args()


def create_midi_encoder(device="cuda"):
    return VariationalConvMlpEncoder(
        in_channels=16,
        mid_channels=32,
        out_channels=64,
        latent_dim=32,
        noise=0.08,
        device=device,
    ).to(device)


def main():
    args = parse_args()
    if args.ddim_steps > args.train_timesteps:
        raise ValueError("--ddim-steps must be <= --train-timesteps.")

    pred_horizon = 4
    action_horizon = 1
    obs_horizon = 1
    obs_dim = 212
    action_dim = 36
    midi_channel = 16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    _, stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=_resolve_path(args.dataset_path),
        normalization=True,
    )

    ae = goal_auto_encoder.network.Autoencoder(
        latent_dim=16,
        cond_dim=64,
    ).to(device)
    ae.load_state_dict(torch.load(_resolve_path(args.ae_ckpt), map_location=device))
    encoder = ae.encoder

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        midi_dim=obs_dim,
        midi_cond_dim=36,
        midi_encoder=lambda: create_midi_encoder(device=args.device),
    ).to(device)
    noise_pred_net.load_state_dict(
        torch.load(_resolve_path(args.high_level_ckpt), map_location=device)
    )
    noise_pred_net.eval()

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        clip_sample=args.clip_sample,
        prediction_type="epsilon",
    )

    env, max_steps = get_env_hl(args.task_name, lookahead=args.lookahead)
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

    with tqdm(total=max_steps, desc="Eval Env DDIM-HL") as pbar:
        while not timestep.last():
            goal = get_flattend_obs(
                timestep,
                lookahead=args.lookahead,
                exclude_keys=[
                    "fingering",
                    "hand",
                    "demo",
                    "prior_action",
                    "q_piano",
                ],
                encoder=encoder,
                sampling=False,
            )

            # Preserve the original high-level conditioning layout:
            # obs = [11 * 16-dim encoded goals, current 36-dim fingertip position].
            _ = np.zeros((pred_horizon, midi_channel + action_dim))
            _[:, :midi_channel] = goal[:pred_horizon * midi_channel].reshape(
                (pred_horizon, -1)
            )
            obs = torch.cat(
                (torch.from_numpy(goal), torch.from_numpy(last_fingertip_pos)),
                dim=-1,
            ).float()
            obs = normalize_data(obs, stats["obs"]).to(device)

            with torch.no_grad():
                obs = obs.unsqueeze(0)
                naction = torch.randn((1, pred_horizon, action_dim), device=device)
                noise_scheduler.set_timesteps(args.ddim_steps, device=device)

                for k in noise_scheduler.timesteps:
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs,
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction,
                        eta=args.eta,
                    ).prev_sample

            naction = naction.detach().cpu().numpy()
            naction = np.concatenate(
                (naction, np.zeros((1, pred_horizon, 10))), axis=2
            ).flatten()
            naction = unnormalize_data(naction, stats["action"])
            naction = naction.reshape(1, pred_horizon, -1)

            action = naction[0][0]
            nft = naction[0, :, :36]
            goal88 = timestep.observation["goal"][:88]
            keys = np.nonzero(goal88)

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
            last_fingertip_pos = np.concatenate((lh_ft.T.flatten(), rh_ft.T.flatten()))

            trajectory_lh[step] = lh_ft
            trajectory_rh[step] = rh_ft
            step += 1
            timestep = env.step(np.zeros(47))
            pbar.update(1)

    output_dir = Path(_resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{args.task_name}_trajectory.npy", trajectory)
    np.save(output_dir / f"{args.task_name}_left_hand_action_list.npy", trajectory_lh)
    np.save(output_dir / f"{args.task_name}_right_hand_action_list.npy", trajectory_rh)


if __name__ == "__main__":
    main()
