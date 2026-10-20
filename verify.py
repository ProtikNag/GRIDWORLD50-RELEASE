"""Check that this copy of GridWorld-50 holds the tasks it claims to.

    python verify.py

Run this once after unpacking, and again after any change to the config or the
frozen dataset. If a check fails, two runs of the benchmark are not comparable,
because they were not solving the same tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from gridworld50 import CONFIG, DATASET, load_env_spec, load_family
from gridworld50.biggrid import _generate_blocked

CHECKS = []


def check(fn):
    """Register ``fn`` as a check; its docstring is the line that gets printed."""
    CHECKS.append(fn)
    return fn


@check
def frozen_dataset_matches_the_config() -> str:
    """The .npz holds the same grids the config's seed generates."""
    params, tasks = load_env_spec()
    frozen = load_family(dataset=DATASET)
    rebuilt = load_family(dataset=None)
    for i, (a, b) in enumerate(zip(frozen.tasks, rebuilt.tasks), start=1):
        assert torch.equal(a.blocked, b.blocked), f"task {i}: layout differs"
        assert torch.equal(a.goal_dist, b.goal_dist), f"task {i}: distances differ"
        assert torch.equal(a.starts, b.starts), f"task {i}: spawn cells differ"
    return f"{len(frozen.tasks)} tasks, frozen and regenerated agree"


@check
def layouts_are_a_property_of_the_task() -> str:
    """Building the family twice gives identical tasks."""
    a, b = load_family(), load_family()
    for i, (ta, tb) in enumerate(zip(a.tasks, b.tasks), start=1):
        assert ta.goal == tb.goal, f"task {i}: goal moved"
        assert ta.slip == tb.slip, f"task {i}: slip changed"
        assert ta.view_radius == tb.view_radius, f"task {i}: view radius changed"
        assert torch.equal(ta.blocked, tb.blocked), f"task {i}: layout changed"
    return "two constructions, no difference"


@check
def the_env_builds_the_grid_the_generator_validated() -> str:
    """Each layout seed is obstacle_seed + task_id, as the generator assumed."""
    params, tasks = load_env_spec()
    family = load_family()
    seed, size = int(params["obstacle_seed"]), int(params["size"])
    for i, spec in enumerate(tasks):
        want = _generate_blocked(size, spec["obstacle_density"],
                                 tuple(spec["goal"]), seed + i).numpy()
        assert np.array_equal(want, family.tasks[i].blocked.numpy()), \
            f"task {i + 1}: grid differs from the one validated at generation"
    return f"all {len(tasks)} layouts reproduce from obstacle_seed {seed}"


@check
def every_task_is_solvable_inside_the_horizon() -> str:
    """The 95th-percentile shortest path fits in half the episode budget."""
    params, _ = load_env_spec()
    horizon = int(params["max_steps"])
    worst = 0.0
    for i, task in enumerate(load_family().tasks, start=1):
        starts = task.starts.numpy()
        d = task.goal_dist.numpy()[starts[:, 0], starts[:, 1]]
        assert d.min() > 0, f"task {i}: a spawn sits on the goal"
        p95 = float(np.percentile(d, 95))
        assert p95 <= horizon * 0.5, \
            f"task {i}: p95 path {p95:.0f} exceeds half of {horizon}"
        worst = max(worst, p95)
    return f"worst p95 path {worst:.0f} steps, horizon {horizon}"


@check
def spawns_stay_inside_each_task_s_band() -> str:
    """Path length is the difficulty axis, so it must hold at run time."""
    for i, task in enumerate(load_family().tasks, start=1):
        lo, hi = task.spec.params["start_band"]
        starts = task.starts.numpy()
        d = task.goal_dist.numpy()[starts[:, 0], starts[:, 1]]
        assert d.min() >= lo and d.max() <= hi, \
            f"task {i}: spawn distances {d.min()}-{d.max()} outside {lo}-{hi}"
    return "all 50 bands respected"


@check
def the_same_seed_gives_the_same_episode() -> str:
    """Reset and step are reproducible, slip included."""
    family = load_family()
    task = next(t for t in family.tasks if t.slip > 0)
    runs = []
    for _ in range(2):
        env = task.make_env()
        env.reset(seed=99)
        rng = np.random.default_rng(4)  # identical action sequence both times
        trace = []
        for _ in range(160):
            _, reward, term, trunc, _ = env.step(int(rng.integers(4)))
            trace.append((round(float(reward), 10), bool(term), bool(trunc)))
            if term or trunc:
                break
        env.close()
        runs.append(trace)
    assert runs[0] == runs[1], "two identical rollouts diverged"
    return f"{len(runs[0])} steps replayed exactly (slip {task.slip})"


@check
def separate_envs_on_one_task_do_not_share_state() -> str:
    """Parallel workers on the same task stay independent."""
    task = load_family().tasks[0]
    a, b = task.make_env(), task.make_env()
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.array_equal(oa, ob), "same seed gave two different starts"
    oc, _ = b.reset(seed=12)
    assert not np.array_equal(oa, oc), "different seeds gave the same start"
    # Stepping one must not move the other.
    before = (a._row, a._col)
    for _ in range(5):
        b.step(0)
    assert (a._row, a._col) == before, "stepping one env moved another"
    a.close(); b.close()
    return "independent state, seed-addressable starts"


@check
def the_oracle_reaches_the_goal_and_random_does_not() -> str:
    """The scale is readable: a floor near 0 and a ceiling near 1."""
    from gridworld50.baselines import run_task
    task = load_family().tasks[0]
    rnd = run_task(task, episodes=10, seed=100000, policy="random")
    orc = run_task(task, episodes=10, seed=100000, policy="oracle")
    assert orc["success"] >= 0.8, f"oracle only reached the goal {orc['success']:.0%}"
    assert orc["mean"] > 0.5, f"oracle scored {orc['mean']:.3f}, expected near 1"
    assert rnd["mean"] < 0.2, f"random scored {rnd['mean']:.3f}, expected near 0"
    return f"task 1: random {rnd['mean']:+.3f}, oracle {orc['mean']:+.3f} " \
           f"at {orc['success']:.0%} success"


def main() -> int:
    print(f"config  {CONFIG}")
    print(f"dataset {DATASET}")
    print()
    failures = 0
    for fn in CHECKS:
        title = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            detail = fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {title}\n      {exc}")
        except Exception as exc:  # a broken copy, not a failed assertion
            failures += 1
            print(f"ERROR {title}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {title}\n      {detail}")
    print()
    if failures:
        print(f"{failures} of {len(CHECKS)} checks failed")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
