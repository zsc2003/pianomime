import argparse
import collections
import os
import shutil
import sys
from pathlib import Path

# Keep original import behavior, plus make direct execution from workspace robust.
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm

import goal_auto_encoder.network
from dataset import normalize_data, read_dataset, unnormalize_data
from network import ConditionalUnet1D, ConvEncoder
from utils import get_env_ll, get_flattend_obs


def resolve_existing_path(candidates, what="file"):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    tried = ", ".join(str(x) for x in candidates if x)
    raise FileNotFoundError(f"Could not find {what}. Tried: {tried}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PianoMime low-level policy with the original DDPM sampler.")
    parser.add_argument("task_name", type=str, help="Song/task name without .pkl")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default="./dataset_ll.zarr")
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", type=str, default=None)
    parser.add_argument("--ae-ckpt", "--ae_ckpt", dest="ae_ckpt", type=str, default=None)
    parser.add_argument("--trajectory-dir", "--trajectory_dir", dest="trajectory_dir", type=str, default="pianomime/multi_task/trajectories")
    parser.add_argument("--record-dir", "--record_dir", dest="record_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--num-diffusion-iters", "--num_diffusion_iters", dest="num_diffusion_iters", type=int, default=50)
    parser.add_argument("--enable-ik", "--enable_ik", dest="enable_ik", action="store_true", help="Use IK residual mode. Default follows original script: False.")
    parser.add_argument("--use-midi", "--use_midi", dest="use_midi", action="store_true")
    return parser.parse_args()


def build_model(device):
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


def ensure_utils_can_find_trajectories(task_name, trajectory_dir):
    # utils.get_env_ll loads hard-coded pianomime/multi_task/trajectories/*.npy.
    source_dir = Path(trajectory_dir)
    target_dir = Path("pianomime/multi_task/trajectories")
    names = [
        f"{task_name}_left_hand_action_list.npy",
        f"{task_name}_right_hand_action_list.npy",
    ]
    for name in names:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing high-level trajectory {src}. Run eval_high_level.py first.")
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != target_dir.resolve():
        for name in names:
            shutil.copy2(source_dir / name, target_dir / name)
    return target_dir


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    ae = goal_auto_encoder.network.Autoencoder(latent_dim=16, cond_dim=64).to(device)
    ae_ckpt = resolve_existing_path(
        [args.ae_ckpt, "./reproduced_ckpt/checkpoint_ae.ckpt", "./checkpoint_ae.ckpt", "checkpoint_ae.ckpt"],
        what="goal auto-encoder checkpoint",
    )
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval()
    encoder = ae.encoder
    print(f"[DDPM-LL eval] loaded goal AE: {ae_ckpt}")

    noise_pred_net = build_model(device)
    ckpt_path = resolve_existing_path(
        [args.ckpt_path, "./reproduced_ckpt/dataset_ll.ckpt", "./checkpoint_low_level.ckpt", "checkpoint_low_level.ckpt"],
        what="low-level DDPM checkpoint",
    )
    noise_pred_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    noise_pred_net.eval()
    print(f"[DDPM-LL eval] loaded checkpoint: {ckpt_path}")

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

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    step_idx = 0
    with tqdm(total=max_steps, desc="DDPM-LL Eval Env") as pbar:
        while not timestep.last():
            nobs = np.stack(obs_deque)
            nobs = normalize_data(nobs, stats["obs"])
            nobs = torch.from_numpy(nobs).to(device=device, dtype=torch.float32)

            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
            noisy_action = torch.randn((batch_size, pred_horizon, action_dim), device=device)
            naction = noisy_action
            noise_scheduler.set_timesteps(args.num_diffusion_iters)
            for k in noise_scheduler.timesteps:
                noise_pred = noise_pred_net(sample=naction, timestep=k, global_cond=obs_cond)
                naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

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
