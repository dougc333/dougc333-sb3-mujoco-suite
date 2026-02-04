#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class JsonlMetricsCallback(BaseCallback):
    """
    Writes metrics to logs/metrics.jsonl so a dashboard can tail it.
    Each line is a JSON object: {"t": timesteps, "wall": seconds, "metrics": {...}}
    """
    def __init__(self, out_path: str, every_rollout: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.out_path = out_path
        self.every_rollout = every_rollout
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._t0 = None

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "training_start", "t": 0, "wall": 0.0}) + "\n")

    def write(self, metrics: Dict[str, float]):
        wall = time.time() - (self._t0 or time.time())
        rec = {"t": int(self.num_timesteps), "wall": float(wall), "metrics": metrics}
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _on_rollout_end(self) -> None:
        if not self.every_rollout:
            return
        # we write nothing unless someone calls write() from another callback
        return

    def _on_training_end(self) -> None:
        wall = time.time() - (self._t0 or time.time())
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "training_end", "t": int(self.num_timesteps), "wall": float(wall)}) + "\n")


class FracHorizonAndEntropyCallback(BaseCallback):
    """
    Logs:
      - custom/ep_len_frac_horizon   (your "frac_1000")
      - custom/frac_time_limit
      - custom/frac_terminated
      - custom/ep_len_mean_window
      - custom/ep_rew_mean_window
      - custom/policy_entropy_mean

    Also optionally emits one metric (frac_episode_len) to JSONL for dashboard.
    """
    def __init__(
        self,
        horizon: int = 1000,
        entropy_batch: int = 2048,
        window: int = 200,
        jsonl_cb: Optional[JsonlMetricsCallback] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.horizon = int(horizon)
        self.entropy_batch = int(entropy_batch)
        self.window = int(window)
        self.jsonl_cb = jsonl_cb

        self.ep_lens: Deque[float] = deque(maxlen=window)
        self.ep_rews: Deque[float] = deque(maxlen=window)
        self.was_time_limit: Deque[bool] = deque(maxlen=window)
        self.was_terminated: Deque[bool] = deque(maxlen=window)

        # exposed for other callbacks
        self.last_frac_horizon = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            l = float(ep.get("l", 0.0))
            r = float(ep.get("r", 0.0))
            self.ep_lens.append(l)
            self.ep_rews.append(r)

            truncated = bool(info.get("TimeLimit.truncated", False))
            self.was_time_limit.append(truncated)
            self.was_terminated.append(not truncated)
        return True

    def _on_rollout_end(self) -> None:
        # windowed stats
        if len(self.ep_lens) > 0:
            lens = np.asarray(self.ep_lens, dtype=np.float32)
            rews = np.asarray(self.ep_rews, dtype=np.float32)
            frac = float(np.mean(lens >= self.horizon))
            self.last_frac_horizon = frac

            tl = np.asarray(self.was_time_limit, dtype=np.float32)
            term = np.asarray(self.was_terminated, dtype=np.float32)
            frac_time_limit = float(tl.mean()) if tl.size else 0.0
            frac_terminated = float(term.mean()) if term.size else 0.0

            self.logger.record("custom/ep_len_frac_horizon", frac)
            self.logger.record("custom/ep_len_mean_window", float(lens.mean()))
            self.logger.record("custom/ep_rew_mean_window", float(rews.mean()))
            self.logger.record("custom/frac_time_limit", frac_time_limit)
            self.logger.record("custom/frac_terminated", frac_terminated)

            # ✅ dashboard metric name you asked: frac_episode_len
            if self.jsonl_cb is not None:
                self.jsonl_cb.write({"frac_episode_len": frac})

        # entropy estimate
        obs = self.model.rollout_buffer.observations.reshape(
            -1, self.model.rollout_buffer.observations.shape[-1]
        )
        if obs.shape[0] > self.entropy_batch:
            idx = np.random.choice(obs.shape[0], self.entropy_batch, replace=False)
            obs = obs[idx]

        obs_t = th.as_tensor(obs, device=self.model.device)
        with th.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            ent = dist.distribution.entropy()
            if ent.ndim == 2:
                ent = ent.sum(dim=-1)
            self.logger.record("custom/policy_entropy_mean", float(ent.mean().item()))


class EarlyStopOnFrac(BaseCallback):
    def __init__(self, frac_cb: FracHorizonAndEntropyCallback, threshold=0.8, patience=3, verbose=1):
        super().__init__(verbose)
        self.frac_cb = frac_cb
        self.threshold = float(threshold)
        self.patience = int(patience)
        self._streak = 0

    def _on_step(self) -> bool:
        if self._streak >= self.patience:
            if self.verbose:
                print(f"[early-stop] frac={self.frac_cb.last_frac_horizon:.2f} for {self.patience} rollouts")
            return False
        return True

    def _on_rollout_end(self) -> None:
        frac = float(self.frac_cb.last_frac_horizon)
        self._streak = self._streak + 1 if frac >= self.threshold else 0
        self.logger.record("custom/earlystop_streak", float(self._streak))