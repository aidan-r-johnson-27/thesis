import argparse
import copy
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch

from boids_rl.baselines import evaluate_policy, random_policy, reynolds
from boids_rl.config import (ABLATIONS, Config, add_config_args, add_dataclass_args, apply_config_args,
                             apply_dataclass_args, apply_overrides)
from boids_rl.env import to_world
from boids_rl.train import train

# random search over task settings, looking for the one whose learned flock looks most like
# hand-coded Reynolds boids
# score = RMS over the metrics of (x - reynolds) / (reynolds - random)
# so 0 means Reynolds-like and ~1 means no better than random steering

SCORE_METRICS = ("polarization", "nn_distance", "n_visible", "largest_group")

# the values to pick from for each "<section>.<field>"
SEARCH_SPACE = {
    "env.n_predators": [1, 2, 3],
    "env.predator_cooldown": [10, 20],
    "env.confusion": [1.0, 3.0],
    "env.confusion_radius": [2.0, 3.0, 4.0],
    "env.collision_radius": [0.1, 0.25],
    "reward.w_collision": [0.3, 1.0],
}


@dataclass
class SweepConfig:
    name: str = "sweep"
    results_dir: str = "results"
    n_configs: int = 23         # random configs; the reynolds-reward run is added on top
    seed: int = 0               # only for picking configs, the runs use train.seed
    workers: int = 8


def sample_configs(sweep):
    rng = np.random.default_rng(sweep.seed)
    seen = set()
    samples = []
    while len(samples) < sweep.n_configs:
        sample = {key: options[rng.integers(len(options))] for key, options in SEARCH_SPACE.items()}
        key = tuple(sample.values())
        if key not in seen:
            seen.add(key)
            samples.append(sample)
    return samples


def model_policy(model):
    # evaluate_policy wants world-frame accelerations but the model acts in each boid's own frame
    def policy(env):
        obs = torch.as_tensor(env._get_obs())
        action = model.act(obs, deterministic=True)[0].numpy()
        return to_world(action.astype(np.float64), env.heading)
    return policy


def run_one(cfg, overrides):
    torch.set_num_threads(1)
    eval_path = os.path.join(cfg.train.run_dir, "eval.json")
    if not os.path.exists(eval_path):
        resume = os.path.exists(os.path.join(cfg.train.run_dir, "checkpoints", "latest.pt"))
        model = train(cfg, resume=resume, verbose=False)
        results = {"overrides": overrides, **evaluate_policy(model_policy(model), cfg)}
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
    return cfg.train.run_dir


def reference(base):
    # scripted Reynolds and random boids in a plain world: the two ends of the score
    plain = apply_overrides(copy.deepcopy(base), {"env.n_predators": 0, "env.collision_radius": 0.0})
    return {"reynolds": evaluate_policy(reynolds, plain), "random": evaluate_policy(random_policy, plain)}


def score(metrics, ref):
    z = []
    for name in SCORE_METRICS:
        best, worst = ref["reynolds"][name], ref["random"][name]
        z.append((metrics[name] - best) / (best - worst))
    return float(np.sqrt(np.mean(np.square(z))))


def format_cell(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summarize(out_dir):
    with open(os.path.join(out_dir, "reference.json")) as f:
        ref = json.load(f)

    rows = []
    runs_dir = os.path.join(out_dir, "runs")
    for run_id in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, run_id, "eval.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            result = json.load(f)
        overrides = result.pop("overrides")
        row = {"run": run_id, "score": score(result, ref)}
        for key, value in overrides.items():
            row[key.split(".")[1]] = value
        row.update(result)
        rows.append(row)
    if not rows:
        print("no finished runs")
        return
    rows.sort(key=lambda row: row["score"])

    # rows can have different override keys, so collect every column name in order
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(os.path.join(out_dir, "sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)

    setting_names = [key.split(".")[1] for key in SEARCH_SPACE]
    metric_names = [*SCORE_METRICS, "caught_per_episode", "collisions_per_episode"]
    shown = ["run", "score", *setting_names, *metric_names]
    lines = ["| " + " | ".join(shown) + " |", "|---" * len(shown) + "|"]
    for name, result in (("REYNOLDS (scripted)", ref["reynolds"]), ("random (scripted)", ref["random"])):
        cells = [name, format_cell(score(result, ref))] + [""] * len(setting_names)
        cells += [format_cell(result.get(metric, "")) for metric in metric_names]
        lines.append("| " + " | ".join(cells) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col, "")) for col in shown) + " |")

    with open(os.path.join(out_dir, "sweep.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Random search for the most Reynolds-like survival setup.")
    parser.add_argument("--summarize-only", action="store_true")
    add_dataclass_args(parser, "sweep", SweepConfig())
    add_config_args(parser)
    args = parser.parse_args()
    sweep = apply_dataclass_args(SweepConfig(), args, "sweep")
    base = apply_config_args(Config(), args)

    out_dir = os.path.join(sweep.results_dir, sweep.name)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    if not args.summarize_only:
        with open(os.path.join(out_dir, "reference.json"), "w") as f:
            json.dump(reference(base), f, indent=2)
        with open(os.path.join(out_dir, "sweep.json"), "w") as f:
            json.dump({"sweep": asdict(sweep), "space": SEARCH_SPACE, "base_config": base.to_dict()}, f, indent=2)

        jobs = [(f"c{i:02d}", overrides) for i, overrides in enumerate(sample_configs(sweep))]
        # the hand-shaped reynolds reward run, as a reference point
        jobs.append(("reynolds_reward", dict(ABLATIONS["reynolds"])))

        configs, all_overrides = [], []
        for run_id, overrides in jobs:
            cfg = apply_overrides(copy.deepcopy(base), overrides)
            cfg.train.run_dir = os.path.join(out_dir, "runs", run_id)
            configs.append(cfg)
            all_overrides.append(overrides)
        with ProcessPoolExecutor(max_workers=sweep.workers) as pool:
            for i, run_dir in enumerate(pool.map(run_one, configs, all_overrides), start=1):
                print(f"[{i}/{len(configs)}] done: {run_dir}", flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
