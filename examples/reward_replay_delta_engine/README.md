# Reward Replay Delta Engine

This example turns stored agent trajectories into maintained reward artifacts:

- a SQLite reward catalog (`output/reward_catalog.sqlite`)
- a training JSONL export (`output/training.jsonl`)
- one JSONL file per included training row (`output/training_rows/`)
- a Markdown pass/fail diff report (`output/reward_diff.md`)

It is intentionally small, but it shows the RL data/control-plane pattern: reward
logic and verifier prompts change often, while parsing and expensive verifier work
should only rerun when their real dependencies change.

## Run The Pipeline

```bash
cd examples/reward_replay_delta_engine
uv run cocoindex update main.py
uv run python main.py show
```

The first run backfills all seed trajectories. The simulated verifier sleeps for
each trajectory and appends a line to `output/verifier_calls.jsonl`, so repeated
runs make cache reuse visible:

```bash
uv run cocoindex update main.py
uv run python main.py show
```

The verifier call count should stay flat on the second run.

## Demo The Delta

Add a new trajectory. Only that new item needs the expensive verifier call.

```bash
uv run python main.py add-new-trajectory
uv run cocoindex update main.py
uv run python main.py show
```

Switch to the stricter reward policy. This changes reward scoring and the
verifier prompt, so the relevant memoized layers are invalidated and target rows,
training exports, and diff rows are reconciled.

```bash
uv run python main.py set-policy strict
uv run cocoindex update main.py
uv run python main.py show
```

Quarantine a trajectory. Its reward row changes and any owned training-row file is
cleaned up automatically.

```bash
uv run python main.py quarantine traj_weather_002
uv run cocoindex update main.py
uv run python main.py show
```

Reset everything:

```bash
uv run python main.py reset-demo
```

## What To Edit

- Edit `reward_policy.json` to change thresholds, penalties, or the verifier
  prompt.
- Edit `score_reward()` in `main.py` to change reward logic directly.
- Add, edit, delete, or quarantine files in `data/trajectories/`.

CocoIndex tracks stable per-trajectory components, memoized function inputs,
function logic, and declared target ownership. That is the key behavior this demo
is meant to make tangible: change the reward, replay the delta, trust the
artifacts.
