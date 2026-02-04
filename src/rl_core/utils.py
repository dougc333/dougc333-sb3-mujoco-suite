#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def ensure_dir(p: str | Path) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)


def git_sha(short: bool = True) -> str:
    """
    Best-effort commit SHA. If not in git repo, returns 'nogit'.
    """
    try:
        args = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        return sha[:8] if short else sha
    except Exception:
        return "nogit"


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)