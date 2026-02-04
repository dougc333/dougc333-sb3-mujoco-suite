import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class LipschitzProbeCallback(BaseCallback):
    """
    Empirical local Lipschitz estimates on rollout observations.
    Logs:
      - lipschitz/policy_mean, lipschitz/policy_p95
      - lipschitz/value_mean,  lipschitz/value_p95
    """
    def __init__(self, eps_std=1e-2, batch=512, every_rollout=True, verbose=0):
        super().__init__(verbose)
        self.eps_std = float(eps_std)
        self.batch = int(batch)
        self.every_rollout = every_rollout

    def _on_rollout_end(self) -> None:
        # sample obs from rollout buffer
        obs = self.model.rollout_buffer.observations
        obs = obs.reshape(-1, obs.shape[-1])
        n = obs.shape[0]
        if n == 0:
            return

        idx = np.random.choice(n, min(self.batch, n), replace=False)
        x = th.as_tensor(obs[idx], device=self.model.device).float()

        # random perturbation
        eps = th.randn_like(x) * self.eps_std
        x2 = x + eps
        denom = th.norm(eps, dim=-1).clamp_min(1e-12)

        with th.no_grad():
            # ---- policy: use mean action (deterministic output) as f(x)
            # Works for Box action spaces.
            a1 = self.model.policy.predict(x, deterministic=True)[0]
            a2 = self.model.policy.predict(x2, deterministic=True)[0]
            a1 = th.as_tensor(a1, device=self.model.device, dtype=th.float32)
            a2 = th.as_tensor(a2, device=self.model.device, dtype=th.float32)

            num_pi = th.norm(a2 - a1, dim=-1)
            L_pi = (num_pi / denom).detach().cpu().numpy()

            # ---- value: V(x)
            v1 = self.model.policy.predict_values(x).squeeze(-1)
            v2 = self.model.policy.predict_values(x2).squeeze(-1)
            num_v = (v2 - v1).abs()
            L_v = (num_v / denom).detach().cpu().numpy()

        self.logger.record("lipschitz/policy_mean", float(np.mean(L_pi)))
        self.logger.record("lipschitz/policy_p95", float(np.percentile(L_pi, 95)))
        self.logger.record("lipschitz/value_mean", float(np.mean(L_v)))
        self.logger.record("lipschitz/value_p95", float(np.percentile(L_v, 95)))
