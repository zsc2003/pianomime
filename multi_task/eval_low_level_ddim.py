import argparse
import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

directory = "pianomime"
if directory not in sys.path:
    sys.path.append(directory)

from dataset import normalize_data, read_dataset, unnormalize_data
from network import ConditionalUnet1D, ConvEncoder
from utils import get_env_ll, get_flattend_obs
import goal_auto_encoder.network


def _resolve_path(path: str) -> str:
    """Keep original script paths usable from either repo root or its parent."""
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if path.startswith("pianomime/") or path.startswith("pianomime\\"):
        fallback = Path(path.replace("\\", "/").split("/", 1)[-1])
        if fallback.exists():
            return str(fallback)
    return path


def resolve_existing_path(candidates, what="file"):
    for candidate in candidates:
        if candidate and Path(_resolve_path(candidate)).exists():
            return _resolve_path(candidate)
    tried = ", ".join(str(x) for x in candidates if x)
    raise FileNotFoundError(f"Could not find {what}. Tried: {tried}")


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
            raise FileNotFoundError(
                f"Missing high-level trajectory {src}. Run eval_high_level_ddim.py first."
            )
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != target_dir.resolve():
        for name in names:
            shutil.copy2(source_dir / name, target_dir / name)
    return target_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-level PianoMime inference with DDIM sampling."
    )
    parser.add_argument("task_name", help="Song/clip name, same as eval_low_level.py.")

    # Core paths. Defaults mirror the original DDPM script.
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path",
                        default="pianomime/dataset_ll.zarr")
    parser.add_argument("--ae-ckpt", "--ae_ckpt", dest="ae_ckpt", default=None)
    parser.add_argument("--ckpt-path", "--ckpt_path", "--low-level-ckpt",
                        dest="ckpt_path", default=None)
    parser.add_argument(
        "--trajectory-dir",
        default="pianomime/multi_task/trajectories",
        help="High-level trajectory directory produced by eval_high_level*.py.",
    )
    parser.add_argument("--record-dir", "--record_dir", dest="record_dir",
                        default=None)

    # DDIM hyperparameters. These are the main knobs to tune.
    parser.add_argument("--train-timesteps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0.0 gives deterministic DDIM; >0 adds stochasticity.")
    parser.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    parser.add_argument("--clip-sample", action=argparse.BooleanOptionalAction,
                        default=True)

    # Runtime controls.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", "--env-lookahead", dest="env_lookahead",
                        type=int, default=10)
    parser.add_argument("--obs-lookahead", type=int, default=3)
    parser.add_argument("--enable-ik", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--use-midi", action=argparse.BooleanOptionalAction,
                        default=False)
    return parser.parse_args()


def create_midi_encoder(device="cuda"):
    return ConvEncoder(
        in_channels=52,
        mid_channels=64,
        out_channels=128,
        horizon=4,
        noise_fingering=0,
        noise_ft=0,
    ).to(device)


def main():
    args = parse_args()
    if args.ddim_steps > args.train_timesteps:
        raise ValueError("--ddim-steps must be <= --train-timesteps.")

    pred_horizon = 4
    action_horizon = 4
    obs_horizon = 1
    obs_dim = 404
    action_dim = 46

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    _, stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=resolve_existing_path(
            [
                args.dataset_path,
                "./dataset_ll.zarr",
                "dataset_ll.zarr",
                "./dataset/dataset_ll.zarr",
                "dataset/dataset_ll.zarr",
                "pianomime/dataset_ll.zarr",
            ],
            what="low-level zarr dataset",
        ),
        normalization=True,
    )

    ae = goal_auto_encoder.network.Autoencoder(
        latent_dim=16,
        cond_dim=64,
    ).to(device)
    ae_ckpt = resolve_existing_path(
        [
            args.ae_ckpt,
            "./reproduced_ckpt/checkpoint_ae.ckpt",
            "./ckpts/checkpoint_ae.ckpt",
            "./checkpoint_ae.ckpt",
            "checkpoint_ae.ckpt",
        ],
        what="goal auto-encoder checkpoint",
    )
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval()
    encoder = ae.encoder
    print(f"[DDIM-LL eval] loaded goal AE: {ae_ckpt}")

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        midi_dim=208,
        midi_cond_dim=0,
        midi_encoder=lambda: create_midi_encoder(device=args.device),
        freeze_encoder=False,
    ).to(device)
    ckpt_path = resolve_existing_path(
        [
            args.ckpt_path,
            "./reproduced_ckpt/dataset_ll.ckpt",
            "./ckpts/checkpoint_low_level.ckpt",
            "./checkpoint_low_level.ckpt",
            "checkpoint_low_level.ckpt",
        ],
        what="low-level diffusion checkpoint",
    )
    noise_pred_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    noise_pred_net.eval()
    print(f"[DDIM-LL eval] loaded checkpoint: {ckpt_path}")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        clip_sample=args.clip_sample,
        prediction_type="epsilon",
    )

    trajectory_dir = ensure_utils_can_find_trajectories(args.task_name, args.trajectory_dir)
    left_hand_action_list = np.load(
        trajectory_dir / f"{args.task_name}_left_hand_action_list.npy"
    )
    max_steps = left_hand_action_list.shape[0]

    env = get_env_ll(
        task_name=args.task_name,
        enable_ik=args.enable_ik,
        lookahead=args.env_lookahead,
        record_dir=Path(args.record_dir) if args.record_dir else None,
        use_fingering_emb=False,
        use_midi=args.use_midi,
    )

    timestep = env.reset()
    obs = get_flattend_obs(
        timestep,
        lookahead=args.obs_lookahead,
        exclude_keys=["fingering", "prior_action"],
        encoder=encoder,
        sampling=False,
        concatenate_keys=["goal", "demo"],
    )
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    precisions = []
    recalls = []
    f1s = []
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval Env DDIM-LL") as pbar:
        while not timestep.last():
            nobs = np.stack(obs_deque)
            nobs = normalize_data(nobs, stats["obs"])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            with torch.no_grad():
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
                naction = torch.randn((1, pred_horizon, action_dim), device=device)
                noise_scheduler.set_timesteps(args.ddim_steps, device=device)

                for k in noise_scheduler.timesteps:
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction,
                        eta=args.eta,
                    ).prev_sample

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
                        lookahead=args.obs_lookahead,
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
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(args.task_name)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))


if __name__ == "__main__":
    main()
