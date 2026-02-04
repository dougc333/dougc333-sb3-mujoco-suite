#!/usr/bin/env python3
from __future__ import annotations

import os
from stable_baselines3.common.callbacks import BaseCallback

from rl_core.utils import git_sha


class PeriodicCheckpointCallback(BaseCallback):
    """
    Save:
      - every `every_steps` timesteps
      - final checkpoint at end

    Filename:
      ckpt_t{steps}_frac{frac:.3f}_sha{sha}.zip
    """
    def __init__(self, save_dir: str, every_steps: int = 500_000, frac_cb=None, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_steps = int(every_steps)
        self.frac_cb = frac_cb
        self._next = self.every_steps
        os.makedirs(save_dir, exist_ok=True)
        self._sha = git_sha(short=True)

    def _save(self, tag: str):
        frac = None
        if self.frac_cb is not None:
            frac = float(getattr(self.frac_cb, "last_frac_horizon", 0.0))
        frac_s = f"_frac{frac:.3f}" if frac is not None else ""
        path = os.path.join(self.save_dir, f"ckpt_{tag}_t{self.num_timesteps}{frac_s}_sha{self._sha}.zip")
        self.model.save(path)
        if self.verbose:
            print(f"[ckpt] saved: {path}")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._save(tag="periodic")
            self._next += self.every_steps
        return True

    def _on_training_end(self) -> None:
        self._save(tag="final")