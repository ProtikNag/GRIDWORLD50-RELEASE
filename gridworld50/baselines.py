"""The two reference policies that bracket the scale.

``random_action`` is the floor and ``oracle_action`` the ceiling. The
oracle descends the shipped BFS distance map, which IS the optimal path,
so it bounds what any policy could achieve under the dynamics. It reads
the env's true position rather than the observation, deliberately.
"""

from __future__ import annotations

import statistics

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def oracle_action(env, dist: np.ndarray) -> int:
    """Index of the move that most reduces the shortest-path distance.

    Never steps into a wall, so under ``wall_death`` the oracle only ever dies
    if SLIP pushes it into one -- which is exactly the irreducible part of the
    ceiling we want to measure.
    """
    r, c = env._row, env._col
    best, best_d = 0, None
    for a, (dr, dc) in enumerate(MOVES):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < env.size and 0 <= nc < env.size):
            continue
        d = dist[nr, nc]
        if d < 0:                      # wall or cut off from the goal
            continue
        if best_d is None or d < best_d:
            best, best_d = a, int(d)
    return best


def run_task(task, episodes: int, seed: int, policy: str) -> dict:
    """Roll ``episodes`` episodes of 'random' or 'oracle' on one task."""
    env = task.make_env()
    dist = task.goal_dist.numpy() if task.goal_dist is not None else None
    rng = np.random.default_rng(seed)
    scores, successes, lengths = [], 0, []
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
                # Reaching the goal is the only termination that lands ON it;
                # wall_death also terminates, so position is what distinguishes.
                if (env._row, env._col) == (env.g_row, env.g_col):
                    successes += 1
                break
        scores.append(total)
        lengths.append(steps)
    env.close()
    return {"mean": statistics.mean(scores), "std": statistics.pstdev(scores),
            "success": successes / episodes,
            "mean_len": statistics.mean(lengths)}
