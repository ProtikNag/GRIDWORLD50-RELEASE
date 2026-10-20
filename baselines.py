"""Measure the random FLOOR and the optimal CEILING of every BigGrid task.

    python baselines.py

A benchmark is only readable between two known reference points. The floor is
what a uniform-random policy scores: it must sit near 0, or a method is graded
mostly on the free reward it collects for existing. The ceiling is what a
shortest-path oracle scores: it must sit near 1 and succeed almost always, or
the task punishes an optimal policy and no learner can reach the top of the
scale.

The oracle descends the shipped BFS distance map, which IS the optimal path, so
the ceiling is exact up to slip. It is not a learned agent and does not look at
observations; it reads the env's true position. That is deliberate -- it bounds
what any policy could achieve under the dynamics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from gridworld50 import CONFIG, load_family

from gridworld50.baselines import MOVES, oracle_action, run_task  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--out", default="reference/floor_ceiling.json")
    args = ap.parse_args()

    family = load_family(args.config)
    print(f"{'task':>5}{'band':>10}{'slip':>6}{'dens':>6}"
          f"{'rand':>8}{'rand ok':>9}{'oracle':>8}{'orc ok':>8}{'orc len':>9}")
    rows = []
    for i, task in enumerate(family.tasks):
        rnd = run_task(task, args.episodes, args.seed, "random")
        orc = run_task(task, args.episodes, args.seed, "oracle")
        band = task.spec.params["start_band"]
        rows.append({"task": i + 1, "band": band, "slip": task.slip,
                     "density": task.spec.params["obstacle_density"],
                     "random": rnd, "oracle": orc})
        print(f"{i+1:>5}{str(band):>10}{task.slip:>6}"
              f"{task.spec.params['obstacle_density']:>6.2f}"
              f"{rnd['mean']:>8.3f}{rnd['success']:>9.0%}"
              f"{orc['mean']:>8.3f}{orc['success']:>8.0%}{orc['mean_len']:>9.1f}",
              flush=True)

    rf = [r["random"]["mean"] for r in rows]
    of = [r["oracle"]["mean"] for r in rows]
    os_ = [r["oracle"]["success"] for r in rows]
    print(f"\n{'':=<78}")
    print(f"random floor   mean {statistics.mean(rf):+.3f}  "
          f"range {min(rf):+.3f} .. {max(rf):+.3f}  "
          f"success {statistics.mean(r['random']['success'] for r in rows):.1%}")
    print(f"oracle ceiling mean {statistics.mean(of):+.3f}  "
          f"range {min(of):+.3f} .. {max(of):+.3f}  "
          f"success {statistics.mean(os_):.1%}")
    print(f"usable range   {statistics.mean(of) - statistics.mean(rf):.3f}")
    worst = min(rows, key=lambda r: r["oracle"]["success"])
    print(f"worst oracle   task {worst['task']} at {worst['oracle']['success']:.0%} "
          f"(slip {worst['slip']}, band {worst['band']})")
    # State what was measured rather than what we expected to measure: on this
    # dataset the oracle clears every task, and a stale claim to the contrary
    # would be the first thing a reader checks.
    print("\n  floor near 0 and ceiling near 1 -> the scale is readable.")
    orc_success = statistics.mean(os_)
    if orc_success >= 0.99:
        print(f"  oracle succeeds {orc_success:.1%} -> slip does not cap these tasks.")
    else:
        print(f"  oracle succeeds {orc_success:.1%} -> slip caps the task, "
              "not the policy.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
