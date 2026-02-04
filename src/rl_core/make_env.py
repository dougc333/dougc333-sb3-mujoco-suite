#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize


@dataclass
class EnvConfig:
    env_id: str
    n_train_envs: int = 8
    n_eval_envs: int = 4
    seed: int = 0
    clip_obs: float = 10.0
    norm_obs: bool = True
    norm_reward_train: bool = False
    # eval reward should generally be un-normalized
    norm_reward_eval: bool = False


def make_train_env(cfg: EnvConfig, log_dir: str, vec_env_cls=SubprocVecEnv):
    env = make_vec_env(
        cfg.env_id,
        n_envs=cfg.n_train_envs,
        seed=cfg.seed,
        vec_env_cls=vec_env_cls,
    )
    env = VecMonitor(env, filename=os.path.join(log_dir, "train_monitor.csv"))
    env = VecNormalize(
        env,
        norm_obs=cfg.norm_obs,
        norm_reward=cfg.norm_reward_train,
        clip_obs=cfg.clip_obs,
    )
    return env


def make_eval_env(cfg: EnvConfig, log_dir: str, obs_rms, vec_env_cls=SubprocVecEnv):
    env = make_vec_env(
        cfg.env_id,
        n_envs=cfg.n_eval_envs,
        seed=cfg.seed + 10_000,
        vec_env_cls=vec_env_cls,
        env_kwargs={"render_mode": "rgb_array"},
    )
    env = VecMonitor(env, filename=os.path.join(log_dir, "eval_monitor.csv"))
    env = VecNormalize(
        env,
        norm_obs=cfg.norm_obs,
        norm_reward=cfg.norm_reward_eval,
        clip_obs=cfg.clip_obs,
    )
    # ✅ share observation normalization stats from training
    env.obs_rms = obs_rms
    env.training = False
    env.norm_reward = cfg.norm_reward_eval
    return env