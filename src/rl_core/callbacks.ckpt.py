# rl_core/callbacks_ckpt.py
from __future__ import annotations

import os
import subprocess
from stable_baselines3.common.callbacks import BaseCallback


def git_short_sha(repo_dir: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir)
        return out.decode().strip()
    except Exception:
        return "nogit"


class StepCheckpointCallback(BaseCallback):
    """
    Save every `every_steps` timesteps, plus a final checkpoint at end.
    Filename includes timesteps + frac_1000 (from your frac callback) + git sha.
    """

    def __init__(
        self,
        save_dir: str,
        every_steps: int = 500_000,
        frac_cb=None,  # your Frac1000AndEntropyCallback
        repo_dir: str = ".",
        save_final: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_steps = int(every_steps)
        self.frac_cb = frac_cb
        self.repo_dir = repo_dir
        self.save_final = bool(save_final)
        self._next = self.every_steps
        os.makedirs(save_dir, exist_ok=True)

    def _save(self, tag: str) -> None:
        frac = float(getattr(self.frac_cb, "last_frac_1000", 0.0)) if self.frac_cb else 0.0
        sha = git_short_sha(self.repo_dir)
        path = os.path.join(self.save_dir, f"{tag}_t{self.num_timesteps}_frac{frac:.3f}_sha{sha}.zip")
        self.model.save(path)
        self.logger.record("custom/checkpoint_saved", 1.0)
        if self.verbose:
            print(f"[ckpt] saved: {path}")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._save("step")
            self._next += self.every_steps
        return True

    def _on_training_end(self) -> None:
        if self.save_final:
            self._save("final")
