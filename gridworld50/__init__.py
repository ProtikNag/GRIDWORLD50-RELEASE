"""GridWorld-50: fifty verified maze-navigation tasks on a shared 50x50 grid.

    from gridworld50 import load_family
    family = load_family()          # all 50 tasks, read from the frozen .npz
    env = family.tasks[7].make_env()
    obs, info = env.reset(seed=0)

``load_family`` reads the frozen dataset by default, so every process builds
the same fifty tasks. See README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gridworld50.base import Task, TaskFamily, TaskSpec
from gridworld50.biggrid import BigGridFamily

_ROOT = Path(__file__).resolve().parent.parent
CONFIG = _ROOT / "configs" / "biggrid50v2.yaml"
DATASET = _ROOT / "data" / "biggrid50v2.npz"


def load_env_spec(config: str | Path = CONFIG) -> tuple[dict[str, Any],
                                                        list[dict[str, Any]]]:
    """Return the family-wide ``params`` and the ordered per-task dicts."""
    raw = yaml.safe_load(Path(config).read_text()) or {}
    env = raw.get("env", {})
    if env.get("family") != "biggrid":
        raise ValueError(f"{config} is not a biggrid config (family="
                         f"{env.get('family')!r})")
    return dict(env.get("params") or {}), list(env.get("tasks") or [])


def load_family(config: str | Path = CONFIG,
                dataset: str | Path | None = DATASET) -> BigGridFamily:
    """Build the task family.

    Args:
        config: the generated YAML holding the per-task axis values.
        dataset: the frozen ``.npz`` of layouts, spawn cells and goal
            distances. Pass ``None`` to rebuild those arrays from the config's
            ``obstacle_seed`` instead of reading them. Both paths produce the
            same tasks; ``verify.py`` checks that they still agree.

    Returns:
        A :class:`BigGridFamily` of fifty tasks in arrival order.
    """
    params, tasks = load_env_spec(config)
    params = dict(params)
    if dataset is None:
        params.pop("dataset_npz", None)
    else:
        params["dataset_npz"] = str(dataset)
    return BigGridFamily(params, tasks)


__all__ = ["load_family", "load_env_spec", "BigGridFamily", "Task",
           "TaskFamily", "TaskSpec", "CONFIG", "DATASET"]
