# GridWorld-50

Fifty maze-navigation tasks on a shared 50x50 grid, built for continual
reinforcement learning. Every task is verified solvable before it ships, and
the whole set is deterministic, so two machines that pass `verify.py` are
solving the same fifty tasks.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
python verify.py          # 8 checks, about 20 seconds
```

`verify.py` must pass before you trust any number from this package. Run it
again after you change the config or the frozen dataset.

## Quick start

```python
from gridworld50 import load_family

family = load_family()            # 50 tasks, arrival order
task = family.tasks[7]            # 0-based
env = task.make_env()             # a standard Gymnasium env
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

Several agents on one task, each with its own env and seed stream:

```bash
python -m examples.run_agents --task 8 --agents 4 --episodes 20
```

Nothing is shared between agents. Give every worker a distinct `--seed-base`
and their episodes never overlap, which is what makes this safe to fan out
across processes or machines.

## What a task is

| | |
|---|---|
| Grid | 50 x 50, one goal cell per task |
| Actions | 4, up down left right |
| Observation | 321 floats, agent row and column one-hot, goal row and column one-hot, then an 11x11 egocentric wall patch |
| Episode horizon | 160 steps |
| Discount | 0.99 |
| Reward | potential-based shaping plus a time-decayed bonus for reaching the goal. A random policy nets about 0 and the shaping telescopes, so detours cost nothing |
| Termination | reaching the goal, or walking into a wall |

Walking into a wall ends the episode. A slip that pushes the agent into a wall
does not, so the penalty falls on the choice rather than on the noise.

The goal is in the observation, so direction is computable from the input
rather than something the network has to memorize per task.

## The four design axes

Each task draws one level per axis. The levels are balanced and checked
pairwise near-uncorrelated at generation, so a result cannot be read as a
confound between two axes.

| Axis | Levels | What it changes |
|---|---|---|
| Wall density | 5, 10, 15, 20, 25, 30 % | how much of the grid is blocked |
| Slip | 0.01, 0.02, 0.04, 0.06, 0.08, 0.10 | chance the executed action is replaced by a random one |
| Spawn distance band | 5-15, 15-30, 30-45, 45-60, 60-80 steps | the real difficulty axis |
| View radius | 3, 4, 5 cells | how far the agent sees around itself |

Spawns are restricted to a per-task shortest-path band, so distance to the goal
is a property of the task rather than a by-product of its layout. Density is
not a difficulty knob. Walls shrink the reachable region, so denser grids are
easier.

Tasks hold between 41 and 1101 legal spawn cells, 464 on average.

## Reference numbers

Measured on this exact dataset. Use them to check that a new setup is wired
correctly before reading anything into a learning curve.

| Policy | Mean score | Notes |
|---|---|---|
| Random | -0.00 | the floor, so the scale starts at zero |
| Specialist, one policy per task | 0.89 | 46 of 50 reach the 0.6 success bar |
| Joint, all 50 at once | 0.97 | 49 of 50 above 0.6, hidden [512, 512, 256] |
| Shortest-path oracle | 0.98 | 100% success, the ceiling under slip |

Raw per-task numbers are in `reference/`. Reproduce the floor and the ceiling
with:

```bash
python baselines.py --episodes 30        # rewrites reference/floor_ceiling.json
```

The oracle descends the shipped BFS distance map, which is the optimal path, so
the ceiling is exact up to slip. It reads the env's true position rather than
the observation, deliberately, because it bounds what any policy could reach
under these dynamics.

## Figures

In `figures/png` and `figures/svg`, rebuilt with
`python -m figures.make_figures`.

| File | Shows |
|---|---|
| `task_gallery` | eight sample layouts, one per difficulty band, with walls, goal and legal spawn cells |
| `axis_coverage` | the four axes over the fifty tasks |
| `path_length_profile` | per-task spawn distance spread against the solvability limit |
| `reference_scores` | random, specialist, joint and oracle per task |

## Changing or regenerating the dataset

Three files define the dataset and they are built in this order.

1. `configs/biggrid50v2.yaml` holds the per-task axis values. It is generated,
   so do not hand-edit it.
2. `configs/biggrid50v2.manifest.json` records the generator command, the seed,
   the config hash and the per-task validation report.
3. `data/biggrid50v2.npz` is a frozen cache of layouts, spawn cells and goal
   distances. It saves a flood fill and a BFS per task at load time, and it
   lets two machines prove they hold the same tasks by comparing one file.

To change the dataset, edit the axis constants near the top of `generate.py`,
then run the three steps:

```bash
python generate.py --out configs/biggrid50v2.yaml --seed 20260930
python freeze.py                 # rebuild the .npz from the new config
python verify.py                 # re-check solvability and determinism
```

Useful flags on `generate.py`:

| Flag | Effect |
|---|---|
| `--n-tasks` | how many tasks to emit |
| `--size` | grid side length |
| `--max-steps` | episode horizon, which also sets the solvability limit at half of it |
| `--seed` | everything follows from this, so the same seed rewrites a byte-identical config |
| `--easy-tasks` | tasks forced to the gentlest density and slip. Pass nothing to disable |

Generation rejects a task unless its band holds at least 40 spawn cells and its
95th-percentile shortest path fits in half the episode horizon. Nothing
unsolvable can ship.

Re-running `generate.py` with the shipped seed rewrites `configs/biggrid50v2.yaml`
byte for byte, and `freeze.py` then rewrites an identical `.npz`. That is how
you confirm the dataset has not drifted. The config header and the manifest
quote the generator's original module path rather than `generate.py`. That
string is inside the hashed config text, so correcting it would change the
config's sha256 and break the check. It is left as it was.

Generation prints a correlation table and flags `mean path vs spawn cells` at
-0.351. That pair is a design axis against a derived quantity, and it is
geometry rather than a confound, because a band further from the goal holds
fewer cells at that distance. Across the four design axes themselves the worst
correlation is 0.192. The manifest records the full table under
`axis_independence`.

To check the frozen cache against the config without rewriting it:

```bash
python freeze.py --check
```

The `.npz` is a cache and not a separate source of truth. You can delete it and
pass `load_family(dataset=None)` to rebuild those arrays from the config's
`obstacle_seed`. `verify.py` checks that both paths agree.

## Files

```
gridworld50/        the env and the two reference policies
  biggrid.py        the task family and the Gymnasium env
  base.py           Task, TaskFamily, TaskSpec
  baselines.py      random floor and shortest-path oracle
configs/            the generated config and its manifest
data/               the frozen dataset
reference/          per-task floor, ceiling, specialist and joint scores
figures/            dataset figures and the script that builds them
examples/           running several agents on one task
generate.py         rebuild the config
freeze.py           rebuild or check the .npz
baselines.py        measure the floor and the ceiling
verify.py           8 checks, run this first
```

Requires Python 3.10 or newer, with numpy, torch, gymnasium and pyyaml.
Plotly and kaleido are needed only to rebuild the figures.
