from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

# Robust imports when running directly as:
#   python pianomime/multi_task/flow_matching/train_high_level_flow.py dataset_hl.zarr
_THIS_FILE = Path(__file__).resolve()
_FLOW_DIR = _THIS_FILE.parent
_MULTI_TASK_DIR = _FLOW_DIR.parent
_REPO_DIR = _MULTI_TASK_DIR.parent
_WORKSPACE_DIR = _REPO_DIR.parent
for _p in (str(_FLOW_DIR), str(_MULTI_TASK_DIR), str(_REPO_DIR), str(_WORKSPACE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import read_dataset  # noqa: E402
from flow_matching_utils import FlowMatchingConfig, flow_matching_loss, sanitize_name, set_seed  # noqa: E402
from network import ConditionalUnet1D, VariationalConvMlpEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PianoMime high-level policy with conditional flow matching.")
    parser.add_argument("dataset_path", type=str, help="Path to high-level zarr dataset, e.g. dataset_hl.zarr")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=400)
    parser.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", type=str, default="flow/ckpts")
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", type=str, default=None)
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None)
    parser.add_argument(
        "--time-sampler",
        "--time_sampler",
        dest="time_sampler",
        choices=["uniform", "logit_normal", "logitnormal"],
        default="uniform",
    )
    parser.add_argument("--time-scale", "--time_scale", dest="time_scale", type=float, default=100.0)
    parser.add_argument("--t-eps", "--t_eps", dest="t_eps", type=float, default=1e-5)
    parser.add_argument("--logit-normal-mean", "--logit_normal_mean", dest="logit_normal_mean", type=float, default=0.0)
    parser.add_argument("--logit-normal-std", "--logit_normal_std", dest="logit_normal_std", type=float, default=1.0)
    parser.add_argument("--kl-weight", "--kl_weight", dest="kl_weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=1.0, help="0 disables gradient clipping.")
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


def default_ckpt_path(dataset_path: str, ckpt_dir: str, run_name: str | None) -> Path:
    run_name = run_name or f"FM-HL-{sanitize_name(dataset_path)}"
    return Path(ckpt_dir) / f"checkpoint_{run_name}_without_fingering.ckpt"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pred_horizon = 1
    action_horizon = 1
    obs_horizon = 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataloader, _stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=args.dataset_path,
        normalization=True,
    )

    velocity_net = build_model(device)
    velocity_net.train()

    ema = EMAModel(model=velocity_net, power=0.75)
    optimizer = torch.optim.AdamW(velocity_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.epochs,
    )

    fm_cfg = FlowMatchingConfig(
        time_scale=args.time_scale,
        time_sampler=args.time_sampler,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        t_eps=args.t_eps,
    )

    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else default_ckpt_path(args.dataset_path, args.ckpt_dir, args.run_name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[FM-HL train] dataset={args.dataset_path}")
    print(f"[FM-HL train] checkpoint will be saved to {ckpt_path}")
    print(f"[FM-HL train] time_sampler={fm_cfg.sampler_name}, time_scale={fm_cfg.time_scale}")

    with tqdm(range(args.epochs), desc="Epoch") as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            epoch_fm = []
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch["obs"].to(device)
                    naction = nbatch["action"].to(device)

                    # Original high-level DDPM stores a flattened 4-step action chunk:
                    # (B, 1, 4*46) -> (B, 4, 46), then uses first 36 fingertip coordinates.
                    naction = naction.reshape(naction.shape[0], 4, -1)[:, :, :36]
                    obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)

                    fm_loss, logs = flow_matching_loss(
                        velocity_net,
                        actions=naction,
                        global_cond=obs_cond,
                        cfg=fm_cfg,
                    )
                    kl = getattr(velocity_net, "kl", 0.0)
                    loss = fm_loss + args.kl_weight * kl

                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(velocity_net.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    ema.step(velocity_net)

                    loss_value = float(loss.detach().cpu())
                    fm_value = float(logs["fm_loss"].detach().cpu())
                    epoch_loss.append(loss_value)
                    epoch_fm.append(fm_value)
                    tepoch.set_postfix(loss=loss_value, fm=fm_value, t=float(logs["t_mean"].cpu()))

            tglobal.set_postfix(
                loss=float(np.mean(epoch_loss)),
                fm=float(np.mean(epoch_fm)),
                lr=lr_scheduler.get_last_lr()[0],
            )
            if args.save_every > 0 and epoch_idx % args.save_every == 0:
                torch.save(ema.averaged_model.state_dict(), ckpt_path)
                print(f"Saved high-level FM checkpoint at epoch {epoch_idx}: {ckpt_path}")

    torch.save(ema.averaged_model.state_dict(), ckpt_path)
    print(f"Done. Final high-level FM checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
