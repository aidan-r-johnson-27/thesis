# Flappy RL

A Q-learning agent learning to play Flappy Bird. The game runs on the left of the screen and live training charts run on the right.

## Quick start

```
./run.sh              # train with visuals (main mode)
./train.sh            # train without visuals, 5000 episodes
./run.sh --demo       # watch the learned policy, no learning
```

On first run the scripts create `.venv` (Python 3.14+) and install the dependencies in `requirements.txt` — `pygame-ce` for rendering and `numpy` for the Boids simulator. Progress is saved to `model.json` and picked up again by both modes, so you can train headless and then watch it. Add `--fresh` to start over.

If the model file is missing the agent starts from scratch (an untrained bird flaps randomly until it learns not to hit the ground).

## How it works

The Flappy project is four small modules:

| File          | Purpose                                                            |
| ------------- | ------------------------------------------------------------------ |
| `game.py`     | The environment: pipe physics, collision detection and rewards     |
| `agent.py`    | Table-based Q-learning agent plus rolling training statistics      |
| `main.py`     | Entry point: headless training loop and the interactive pygame mode |
| `render.py`   | pygame rendering — the game view and the charts panel               |

### The environment (`game.py`)

- The world is 400×600 px; the bird sits at a fixed `x = 90` and only moves vertically. Pipes scroll left at 3 px/frame.
- **State** (what the agent sees): horizontal distance to the next pipe, vertical offset from that pipe's gap centre, and the bird's velocity.
- **Actions**: `0` = glide, `1` = flap (an instant upward velocity impulse, then constant gravity).
- **Rewards**: `+1` each time a pipe is passed, `-1000` on a crash (ground, ceiling or pipe). Episodes run until a crash or the `--max-score` target is reached.

### The agent (`agent.py`)

- The continuous state is **discretised** into coarse integer buckets (`dx`/16 × `dy`/10 × `vel`/2) and stored in a Q-value table — one entry per bucket holding two values, one per action.
- Exploration uses **epsilon-greedy**: with probability `epsilon` a random action is taken (biased 70/30 toward gliding), otherwise the action with the higher estimated value wins.
- After each episode the recorded transitions are replayed **backwards** through the standard Q-learning update:

  `q[s][a] += alpha * (reward + gamma * max(q[s']) − q[s][a])`

  Backwards replay lets the terminal reward propagate through the whole episode in a single pass. `epsilon` decays by `0.995` after every episode down to a floor of `0.0`.

- Tuning knobs live in `QAgent.__init__`: `alpha` = 0.7 (learning rate), `gamma` = 0.95 (discount), `epsilon_decay` = 0.995.

### Training (`main.py`)

- **Headless** (`./train.sh` / `--headless`): pure Python loop, no pygame. Prints a progress line every 100 episodes (avg/best/success rate/table size/steps-per-second) and autosaves every 500.
- **Visual** (`./run.sh`): pygame loop that steps the simulation by `speed` frames per redraw, so you can watch up to 500× or uncapped "max" speed. With `V` visuals off, training runs at full speed with only the charts updating.

### Persistence

`save()` writes `model.json` atomically (temp file + rename) on exit, on `S`, and every 500 episodes headless. The file stores the epsilon value, the entire Q-table, and raw episode scores (averages/success rates are recomputed on load).

## Also in this repo: Boids

`boid.py` is a separate, self-contained project: a fullscreen simulation of Reynolds' boids tuned live through an on-screen panel (sliders for population, speed, perception radius and the separation/alignment/cohesion weights, plus three preset templates). Run it directly:

```
.venv/bin/python boid.py
```

It needs a display and `numpy`.

## Controls (visual mode)

| Key | Action |
| --- | --- |
| SPACE | pause |
| ↑ / → / + | speed up (1x … 500x, max) |
| ↓ / ← / − | speed down |
| V | turn the game view off/on (off trains at full speed, charts keep updating) |
| D | switch between demo and training |
| S | save |
| R | reset training (fresh model + stats) |
| ESC / Q | quit and save |

## Options

Run `./run.sh --help` for the full list.

| Option            | Default           | Description                                        |
| ----------------- | ----------------- | -------------------------------------------------- |
| `--headless`      | —                 | Train without a window                             |
| `--episodes N`    | 0 (5000 headless) | Stop after N episodes (0 = keep going)             |
| `--demo`          | —                 | Start in demo mode: greedy policy, no learning     |
| `--fresh`         | —                 | Ignore the saved model and start from scratch      |
| `--model PATH`    | `./model.json`    | Where to load/save the model                       |
| `--goal N`        | 50                | Score that counts as a "success" in the charts     |
| `--max-score N`   | 5000              | End an episode once this score is reached          |
| `--speed N`       | 1                 | Starting simulation speed                          |
| `--seed N`        | random            | Seed the environment RNG for reproducible runs     |

> [!NOTE]
> **This project's documentation has been generated or assisted by AI.** While it has been reviewed for accuracy, some edge cases, outdated dependencies, or minor errors may still exist. Please use with discretion.
