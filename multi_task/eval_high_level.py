import argparse
import os
import sys
from pathlib import Path

# Keep original import behavior, plus make direct execution from workspace robust.
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

import goal_auto_encoder.network
from dataset import normalize_data, read_dataset, unnormalize_data
from network import ConditionalUnet1D, VariationalConvMlpEncoder
from utils import adjust_ft_fingering, get_env_hl, get_flattend_obs

CTRL_TIMESTEP = 0.05


def resolve_existing_path(candidates, what="file"):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    tried = ", ".join(str(x) for x in candidates if x)
    raise FileNotFoundError(f"Could not find {what}. Tried: {tried}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PianoMime high-level trajectories with the original DDPM policy.")
    parser.add_argument("task_name", type=str, help="Song/task name without .pkl")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default="./dataset_hl.zarr")
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", type=str, default=None)
    parser.add_argument("--ae-ckpt", "--ae_ckpt", dest="ae_ckpt", type=str, default=None)
    parser.add_argument("--trajectory-dir", "--trajectory_dir", dest="trajectory_dir", type=str, default="pianomime/multi_task/trajectories")
    parser.add_argument("--record-dir", "--record_dir", dest="record_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--num-diffusion-iters", "--num_diffusion_iters", dest="num_diffusion_iters", type=int, default=100)
    parser.add_argument("--use-midi", "--use_midi", dest="use_midi", action="store_true")
    return parser.parse_args()


def build_model(device):
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
    ).to(device)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pred_horizon = 4
    action_horizon = 1
    obs_horizon = 1
    obs_dim = 212
    action_dim = 36
    midi_channel = 16
    batch_size = 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    _dataloader, stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=args.dataset_path,
        normalization=True,
    )

    ae = goal_auto_encoder.network.Autoencoder(latent_dim=16, cond_dim=64).to(device)
    ae_ckpt = resolve_existing_path(
        [args.ae_ckpt, "./reproduced_ckpt/checkpoint_ae.ckpt", "./checkpoint_ae.ckpt", "checkpoint_ae.ckpt"],
        what="goal auto-encoder checkpoint",
    )
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval()
    encoder = ae.encoder
    print(f"[DDPM-HL eval] loaded goal AE: {ae_ckpt}")

    noise_pred_net = build_model(device)
    ckpt_path = resolve_existing_path(
        [args.ckpt_path, "./reproduced_ckpt/dataset_hl_without_fingering.ckpt", "./checkpoint_high_level.ckpt", "checkpoint_high_level.ckpt"],
        what="high-level DDPM checkpoint",
    )
    noise_pred_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    noise_pred_net.eval()
    print(f"[DDPM-HL eval] loaded checkpoint: {ckpt_path}")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    task_name = args.task_name
    print(task_name)
    env, max_steps = get_env_hl(
        task_name,
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

    with tqdm(total=max_steps, desc="DDPM-HL Eval Env") as pbar:
        while not timestep.last():
            goal = get_flattend_obs(
                timestep,
                lookahead=args.lookahead,
                exclude_keys=["fingering", "hand", "fingering", "demo", "prior_action", "q_piano"],
                encoder=encoder,
                sampling=False,
            )
            _ = goal[: 4 * midi_channel].reshape((4, -1))
            current = last_fingertip_pos
            obs = np.concatenate((goal, current), axis=-1).astype(np.float32)
            obs = normalize_data(obs, stats["obs"])
            obs = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)

            noisy_action = torch.randn((batch_size, pred_horizon, action_dim), device=device)
            naction = noisy_action
            noise_scheduler.set_timesteps(args.num_diffusion_iters)
            for k in noise_scheduler.timesteps:
                noise_pred = noise_pred_net(sample=naction, timestep=k, global_cond=obs)
                naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            naction = naction.detach().cpu().numpy()
            naction = np.concatenate((naction, np.zeros((batch_size, pred_horizon, 10))), axis=2).flatten()
            naction = unnormalize_data(naction, stats["action"])
            naction = naction.reshape(batch_size, pred_horizon, -1)
            nft = naction[0, :, :36]

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
    np.save(out_dir / f"{task_name}_trajectory.npy", np.array(trajectory, dtype=np.float32))
    np.save(out_dir / f"{task_name}_left_hand_action_list.npy", trajectory_lh)
    np.save(out_dir / f"{task_name}_right_hand_action_list.npy", trajectory_rh)
    print(f"[DDPM-HL eval] saved trajectories to {out_dir}")


if __name__ == "__main__":
    main()
