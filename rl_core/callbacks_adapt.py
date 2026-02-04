# rl_core/callbacks_adapt.py
from __future__ import annotations

import json
import os
from stable_baselines3.common.callbacks import BaseCallback


class LipschitzAdaptiveLR(BaseCallback):
    """
    If empirical_lipschitz is too high -> reduce optimizer LR.
    If it's comfortably low -> slowly restore LR toward base_lr.

    Works best when combined with EmpiricalLipschitzCallback,
    because we read the logged value from that callback instance.
    """

    def __init__(
        self,
        lip_cb,  # EmpiricalLipschitzCallback instance
        lip_target: float = 5.0,
        lip_high: float = 7.0,
        decay: float = 0.7,
        grow: float = 1.02,
        min_lr: float = 1e-5,
        jsonl_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.lip_cb = lip_cb
        self.lip_target = float(lip_target)
        self.lip_high = float(lip_high)
        self.decay = float(decay)
        self.grow = float(grow)
        self.min_lr = float(min_lr)
        self.jsonl_path = jsonl_path
        self._fh = None
        self._base_lr = None

    def _on_training_start(self) -> None:
        # get base LR from optimizer
        opt = self.model.policy.optimizer
        self._base_lr = float(opt.param_groups[0]["lr"])
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            self._fh = open(self.jsonl_path, "a", buffering=1)

    def _on_training_end(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def _set_lr(self, new_lr: float) -> None:
        opt = self.model.policy.optimizer
        for g in opt.param_groups:
            g["lr"] = float(new_lr)

    def _get_lr(self) -> float:
        opt = self.model.policy.optimizer
        return float(opt.param_groups[0]["lr"])

    def _on_rollout_end(self) -> None:
        # Empirical Lipschitz estimate should have been computed at rollout end
        lip = getattr(self.lip_cb, "last_lip", None)
        if lip is None:
            # fallback: read from logger value isn't accessible here, so just skip
            return

        lr = self._get_lr()
        new_lr = lr

        if lip >= self.lip_high:
            new_lr = max(self.min_lr, lr * self.decay)
            if self.verbose:
                print(f"[lip-adapt] lip={lip:.2f} >= {self.lip_high:.2f} -> lr {lr:.2e} -> {new_lr:.2e}")
        elif lip <= self.lip_target:
            new_lr = min(self._base_lr, lr * self.grow)

        if new_lr != lr:
            self._set_lr(new_lr)
            self.logger.record("custom/adapt_lr", new_lr)

        if self._fh:
            self._fh.write(json.dumps({"t": int(self.num_timesteps), "lip": float(lip), "lr": float(new_lr)}) + "\n")

    def _on_step(self) -> bool:
        return True
