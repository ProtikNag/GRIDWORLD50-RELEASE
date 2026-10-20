"""One collected episode. Kept here so the env module imports unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    """One episode collected under a behavior policy."""

    obs: torch.Tensor  # [T, obs_dim]
    actions: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    behavior_logps: torch.Tensor  # [T] log pi_b(a_t | s_t) at collection time
    terminated: bool  # environment terminal state reached (not truncation)
