# experiments/benchmark.py
from __future__ import annotations
import argparse
import os
import time

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from rl_core.policies import SpectralActorCriticPolicy
from rl_core.callbacks_lip import EmpiricalLipschitzCallback
from rl_core.callbacks_ckpt import StepCheckpointCallback
# (optional) your Frac callback
from rl_core.callbacks_frac import Frac1000AndEntropyCallback  # you already have this


def make_envs(env_id: str, n_train: int, n_eval: int, log_dir: str):
    train = make_vec_env(env_id, n_envs=n_train, vec_env_cls=SubprocVecEnv)
    train = VecMonitor(train, filename=os.path.join(log_dir, "train_monitor.csv"))
    train = VecNormalize(train, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = make_vec_env(env_id, n_envs=n_eval, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "rgb_array"})
    eval_env = VecMonitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = train.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False
    return train, eval_env


def run_one(cfg_name: str, env_id: str, total_steps: int, out_root: str, spectral: bool, lip: bool):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(out_root, f"{env_id}_{cfg_name}_{stamp}")
    os.makedirs(log_dir, exist_ok=True)

    train_env, eval_env = make_envs(env_id, 8, 4, log_dir)

    device = "cuda" if th.cuda.is_available() else "cpu"

    policy = "MlpPolicy" if not spectral else SpectralActorCriticPolicy
    policy_kwargs = None if not spectral else dict(spectral=True, orthogonal=True, net_arch=[256, 256])

    model = PPO(
        policy=policy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.04,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )

    frac_cb = Frac1000AndEntropyCallback(horizon=1000, entropy_batch=2048, window=200)
    lip_cb = EmpiricalLipschitzCallback(jsonl_path=os
