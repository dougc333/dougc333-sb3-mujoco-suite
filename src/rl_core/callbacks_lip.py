# rl_core/callbacks_lip.py
from __future__ import annotations

import json
import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class EmpiricalLipschitzCallback(BaseCallback):
    """
    Empirical local Lipschitz estimate of policy mean:
      lip = mean( ||mu(s+eps)-mu(s)|| / ||eps|| )
    Also logs approx_kl if SB3 provides it in logger (SB3 does).
    Optionally writes JSONL for a dashboard/plots.
    """

    def __init__(
        self,
        eps_scale: float = 1e-2,
        batch: int = 2048,
        jsonl_path: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eps_scale = float(eps_scale)
        self.batch = int(batch)
        self.jsonl_path = jsonl_path
        self._fh = None

    def _on_training_start(self) -> None:
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            self._fh = open(self.jsonl_path, "a", buffering=1)

    def _on_training_end(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    @th.no_grad()
    def _estimate_lip(self) -> float:
        rb = self.model.rollout_buffer
        obs = rb.observations
        obs = obs.reshape(-1, obs.shape[-1])
        if obs.shape[0] == 0:
            return 0.0
        if obs.shape[0] > self.batch:
            idx = np.random.choice(obs.shape[0], self.batch, replace=False)
            obs = obs[idx]

        obs_t = th.as_tensor(obs, device=self.model.device, dtype=th.float32)
        eps = th.randn_like(obs_t) * self.eps_scale

        # PPO policy is Gaussian; "mean actions" are stable for curvature measurement
        dist1 = self.model.policy.get_distribution(obs_t)
        mu1 = dist1.distribution.mean

        dist2 = self.model.policy.get_distribution(obs_t + eps)
        mu2 = dist2.distribution.mean

        num = th.norm(mu2 - mu1, dim=-1)
        den = th.norm(eps, dim=-1).clamp_min(1e-12)
        lip = (num / den).mean().item()
        return float(lip)

    def _on_rollout_end(self) -> None:
        lip = self._estimate_lip()
        self.logger.record("custom/empirical_lipschitz", lip)

        if self._fh:
            rec = {
                "t": int(self.num_timesteps),
                "emp_lip": lip,
            }
            self._fh.write(json.dumps(rec) + "\n")

    def _on_step(self) -> bool:
        return True
