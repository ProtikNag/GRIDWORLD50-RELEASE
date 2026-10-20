"""Run several independent agents on one task, in parallel.

    python examples/run_agents.py --task 8 --agents 4 --episodes 20

Each agent gets its own environment and its own seed offset, so the runs are
independent but each one is reproducible on its own. Nothing is shared between
them, which is what makes this safe to fan out across processes: give every
worker a distinct ``--seed-base`` and the episodes never overlap.

The agents here are the two reference policies from ``baselines.py``. Swap
``policy_fn`` for your own and the harness is unchanged.
"""

from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gridworld50.baselines import oracle_action
from gridworld50 import load_family


def run_one_agent(task_id: int, agent_id: int, episodes: int, seed_base: int,
                  policy: str) -> dict[str, float]:
    """Roll ``episodes`` episodes of one agent on one task.

    Args:
        task_id: 0-based index into the fifty tasks.
        agent_id: which agent this is; it only shifts the seed stream.
        episodes: how many episodes to roll.
        seed_base: the first seed; agent ``i`` starts at ``seed_base + i*1000``.
        policy: ``"random"`` or ``"oracle"``.

    Returns:
        Mean score, success rate and mean episode length for this agent.
    """
    # Each worker builds its own family: the env holds torch tensors, so this is
    # cheaper and safer than pickling one across the process boundary.
    task = load_family().tasks[task_id]
    env = task.make_env()
    dist = task.goal_dist.numpy()
    seed = seed_base + agent_id * 1000
    rng = np.random.default_rng(seed)
    scores, lengths, successes = [], [], 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total, steps = 0.0, 0
        while True:
            action = (int(rng.integers(4)) if policy == "random"
                      else oracle_action(env, dist))
            _, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                if (env._row, env._col) == (env.g_row, env.g_col):
                    successes += 1
                break
        scores.append(total)
        lengths.append(steps)
    env.close()
    return {"agent": agent_id, "seed": seed, "mean": statistics.mean(scores),
            "success": successes / episodes, "mean_len": statistics.mean(lengths)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=0, help="0-based task index")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--policy", choices=("random", "oracle"), default="oracle")
    ap.add_argument("--serial", action="store_true",
                    help="run in this process instead of a process pool")
    args = ap.parse_args()

    family = load_family()
    task = family.tasks[args.task]
    print(f"task {args.task} ({task.spec.name})  slip {task.slip}  "
          f"band {task.spec.params['start_band']}  "
          f"spawns {task.starts.shape[0]}")
    print(f"{args.agents} agents x {args.episodes} episodes, policy {args.policy}\n")

    work = [(args.task, i, args.episodes, args.seed_base, args.policy)
            for i in range(args.agents)]
    if args.serial:
        rows = [run_one_agent(*w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=args.agents) as pool:
            rows = list(pool.map(run_one_agent, *zip(*work)))

    print(f"{'agent':>6}{'seed':>8}{'mean':>9}{'success':>9}{'len':>7}")
    for r in rows:
        print(f"{r['agent']:>6}{r['seed']:>8}{r['mean']:>9.3f}"
              f"{r['success']:>9.0%}{r['mean_len']:>7.1f}")
    means = [r["mean"] for r in rows]
    print(f"\nacross agents: mean {statistics.mean(means):.3f}"
          + (f", sd {statistics.stdev(means):.3f}" if len(means) > 1 else ""))


if __name__ == "__main__":
    main()
