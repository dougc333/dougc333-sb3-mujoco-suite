#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

from stable_baselines3 import PPO

from rl_core.make_env import EnvConfig, make_train_env, make_eval_env
from rl_core.callbacks import JsonlMetricsCallback, FracHorizonAndEntropyCallback, EarlyStopOnFrac
from rl_core.checkpointing import PeriodicCheckpointCallback
from rl_core.utils import ensure_dir

# Headless MuJoCo defaults (works on Colab; harmless locally)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config, e.g. experiments/ant/ppo.yaml")
    ap.add_argument("--log-root", default="./logs", help="Root logs folder")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    env_id = cfg["env"]["id"]
    run_name = cfg.get("run_name", f"{env_id}_{time.strftime('%Y%m%d_%H%M%S')}")
    log_dir = os.path.join(args.log_root, run_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    # --- envs ---
    e = EnvConfig(
        env_id=env_id,
        n_train_envs=int(cfg["env"].get("n_train_envs", 8)),
        n_eval_envs=int(cfg["env"].get("n_eval_envs", 4)),
        seed=int(cfg["env"].get("seed", 0)),
        clip_obs=float(cfg["env"].get("clip_obs", 10.0)),
        norm_obs=bool(cfg["env"].get("norm_obs", True)),
        norm_reward_train=bool(cfg["env"].get("norm_reward_train", False)),
        norm_reward_eval=bool(cfg["env"].get("norm_reward_eval", False)),
    )

    train_env = make_train_env(e, log_dir=log_dir)
    eval_env = make_eval_env(e, log_dir=log_dir, obs_rms=train_env.obs_rms)

    # --- model ---
    device = cfg["algo"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=int(cfg["algo"]["n_steps"]),
        batch_size=int(cfg["algo"]["batch_size"]),
        n_epochs=int(cfg["algo"]["n_epochs"]),
        learning_rate=float(cfg["algo"]["learning_rate"]),
        gamma=float(cfg["algo"]["gamma"]),
        gae_lambda=float(cfg["algo"]["gae_lambda"]),
        clip_range=float(cfg["algo"]["clip_range"]),
        ent_coef=float(cfg["algo"]["ent_coef"]),
        vf_coef=float(cfg["algo"]["vf_coef"]),
        max_grad_norm=float(cfg["algo"]["max_grad_norm"]),
        target_kl=float(cfg["algo"]["target_kl"]),
        verbose=int(cfg["algo"].get("verbose", 1)),
        tensorboard_log=log_dir,  # ✅ keeps SB3 TB format
        device=device,
    )

    # --- callbacks ---
    metrics_jsonl = JsonlMetricsCallback(out_path=os.path.join(log_dir, "metrics.jsonl"))
    frac_cb = FracHorizonAndEntropyCallback(
        horizon=int(cfg["metrics"].get("horizon", 1000)),
        entropy_batch=int(cfg["metrics"].get("entropy_batch", 2048)),
        window=int(cfg["metrics"].get("window", 200)),
        jsonl_cb=metrics_jsonl,
    )
    earlystop = EarlyStopOnFrac(
        frac_cb,
        threshold=float(cfg["metrics"].get("earlystop_frac", 0.8)),
        patience=int(cfg["metrics"].get("earlystop_patience", 3)),
    )
    ckpt = PeriodicCheckpointCallback(
        save_dir=ckpt_dir,
        every_steps=int(cfg["checkpoint"].get("every_steps", 500_000)),
        frac_cb=frac_cb,
    )

    total_timesteps = int(cfg["train"]["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=[metrics_jsonl, frac_cb, earlystop, ckpt], log_interval=1)

    eval_env.close()
    train_env.close()
    print(f"done. logs: {log_dir}")


if __name__ == "__main__":
    print("Starting training...")
    main()