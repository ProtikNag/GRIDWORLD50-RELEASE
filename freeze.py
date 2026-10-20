"""Rebuild data/biggrid50v2.npz from the config.

    python freeze.py                      # rewrite the frozen dataset
    python freeze.py --check              # compare against the shipped one

The .npz is a cache, not a separate source of truth. Layouts, spawn cells and
goal distances all follow from the config's ``obstacle_seed``, so freezing is a
convenience: it skips a flood fill and a BFS per task at load time, and it lets
two machines prove they hold the same tasks by comparing one file.

Run this after ``generate.py`` changes the config, then run ``verify.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gridworld50 import CONFIG, DATASET, load_env_spec, load_family

NOTE = ("Load by setting env.params.dataset_npz to this file. The family then "
        "reads these arrays instead of regenerating, so every branch solves "
        "byte-identical tasks.")


def build(config: Path) -> dict[str, np.ndarray]:
    """Return the array payload for ``config``, keyed as the .npz stores it."""
    # dataset=None forces the flood fill and the BFS to run, so this is the
    # regeneration path rather than a copy of whatever .npz is already present.
    family = load_family(config=config, dataset=None)
    params, _ = load_env_spec(config)
    payload: dict[str, np.ndarray] = {}
    tasks_meta = []
    for task_id, task in enumerate(family.tasks):
        payload[f"blocked_{task_id}"] = task.blocked.numpy().astype(bool)
        payload[f"goal_dist_{task_id}"] = task.goal_dist.numpy().astype(np.int16)
        payload[f"starts_{task_id}"] = task.starts.numpy().astype(np.int16)
        tasks_meta.append({
            "task_id": task_id,
            "name": task.spec.name,
            "goal": list(task.goal),
            "slip": task.slip,
            "view_radius": task.view_radius,
            "obstacle_density": task.spec.params.get("obstacle_density"),
            "threshold": task.threshold,
            "n_blocked": int(task.blocked.sum()),
            "n_spawnable": int(task.starts.shape[0]),
        })
    meta = {
        "source_config": f"configs/{Path(config).name}",
        "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        "env_params": params,
        "n_tasks": len(family.tasks),
        "obs_dim": family.obs_dim,
        "num_actions": family.num_actions,
        "tasks": tasks_meta,
        "note": NOTE,
    }
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8)
    return payload


def compare(payload: dict[str, np.ndarray], path: Path) -> list[str]:
    """Return a list of human-readable differences against the .npz at ``path``."""
    shipped = np.load(path, allow_pickle=False)
    problems = []
    missing = set(payload) - set(shipped.files)
    extra = set(shipped.files) - set(payload)
    if missing:
        problems.append(f"absent from {path.name}: {sorted(missing)}")
    if extra:
        problems.append(f"only in {path.name}: {sorted(extra)}")
    for key in sorted(set(payload) & set(shipped.files)):
        if key == "meta_json":
            continue  # compared field-wise below; it carries a config hash
        if not np.array_equal(payload[key], shipped[key]):
            problems.append(f"{key} differs")
    a = json.loads(bytes(payload["meta_json"].tolist()).decode())
    b = json.loads(bytes(shipped["meta_json"].tolist()).decode())
    for field in ("config_sha256", "env_params", "n_tasks", "obs_dim", "tasks"):
        if a.get(field) != b.get(field):
            problems.append(f"meta_json.{field} differs")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=DATASET)
    ap.add_argument("--check", action="store_true",
                    help="compare against --out instead of overwriting it")
    args = ap.parse_args()

    print(f"building from {args.config}")
    payload = build(args.config)
    n = payload["meta_json"].size

    if args.check:
        problems = compare(payload, args.out)
        if problems:
            print(f"\nFAIL  {args.out} does not match the config:")
            for p in problems:
                print(f"  - {p}")
            raise SystemExit(1)
        print(f"OK    {args.out} matches the config "
              f"({len(payload) - 1} arrays, {n}-byte metadata)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}  ({len(payload) - 1} arrays, {n}-byte metadata)")


if __name__ == "__main__":
    main()
