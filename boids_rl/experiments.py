import argparse
import copy
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import torch

from boids_rl.config import (ABLATIONS, Config, ExperimentConfig, add_config_args, add_dataclass_args,
                             apply_config_args, apply_dataclass_args, apply_overrides)
from boids_rl.eval import evaluate
from boids_rl.rewards import REWARD_TERMS
from boids_rl.train import train

# every ablation x seed is just a normal training run; finished runs (eval.json exists) get
# skipped and half-done ones resumed, so re-running the same command picks up where it left off

METRICS = (["polarization", "mean_nn_distance", "caught_per_episode", "collisions_per_episode", "mean_return"]
           + [f"term_{name}" for name in REWARD_TERMS])

# one fixed colour per ablation
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def run_dir_for(exp_dir, ablation, seed):
    return os.path.join(exp_dir, "runs", ablation, f"seed_{seed}")


def run_config(base, exp_dir, ablation, seed):
    cfg = apply_overrides(copy.deepcopy(base), ABLATIONS[ablation])
    cfg.train.seed = seed
    cfg.train.run_dir = run_dir_for(exp_dir, ablation, seed)
    return cfg


def run_one(cfg, flock_sizes=()):
    # runs in a worker process
    # one torch thread each, since the parallelism comes from running lots of processes
    torch.set_num_threads(1)
    run_dir = cfg.train.run_dir
    eval_path = os.path.join(run_dir, "eval.json")
    if os.path.exists(eval_path):
        return run_dir

    resume = os.path.exists(os.path.join(run_dir, "checkpoints", "latest.pt"))
    model = train(cfg, resume=resume, verbose=False)
    for n_agents in flock_sizes:
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.env.n_agents = n_agents
        with open(os.path.join(run_dir, f"eval_n{n_agents}.json"), "w") as f:
            json.dump(evaluate(model, eval_cfg), f, indent=2)
    # eval.json is written last because its existence means "this run is done"
    with open(eval_path, "w") as f:
        json.dump(evaluate(model, cfg), f, indent=2)
    return run_dir


def load_evals(exp_dir, exp, filename):
    rows = []
    for ablation in exp.ablations:
        for seed in exp.seeds:
            path = os.path.join(run_dir_for(exp_dir, ablation, seed), filename)
            if os.path.exists(path):
                with open(path) as f:
                    rows.append({"ablation": ablation, "seed": seed, **json.load(f)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    # group rows that share the same values for `keys`, then mean and std over the seeds
    groups = {}
    for row in rows:
        group_key = tuple(row[k] for k in keys)
        groups.setdefault(group_key, []).append(row)

    table = []
    for group_key, group in groups.items():
        entry = dict(zip(keys, group_key))
        entry["n_seeds"] = len(group)
        for metric in METRICS:
            values = [row[metric] for row in group]
            entry[f"{metric}_mean"] = np.mean(values)
            entry[f"{metric}_std"] = np.std(values)
        table.append(entry)
    return table


def write_markdown(path, table, keys):
    with open(path, "w") as f:
        f.write("| " + " | ".join(keys) + " | seeds | " + " | ".join(METRICS) + " |\n")
        f.write("|---" * (len(keys) + len(METRICS) + 1) + "|\n")
        for entry in table:
            cells = [str(entry[k]) for k in keys] + [str(entry["n_seeds"])]
            for metric in METRICS:
                cells.append(f"{entry[metric + '_mean']:.3f} ± {entry[metric + '_std']:.3f}")
            f.write("| " + " | ".join(cells) + " |\n")


def summarize(exp_dir, exp):
    rows = load_evals(exp_dir, exp, "eval.json")
    if not rows:
        print("no finished runs to summarize")
        return
    write_csv(os.path.join(exp_dir, "summary.csv"), rows)
    table = aggregate(rows, ["ablation"])
    write_csv(os.path.join(exp_dir, "table.csv"), table)
    write_markdown(os.path.join(exp_dir, "table.md"), table, ["ablation"])
    with open(os.path.join(exp_dir, "table.md")) as f:
        print(f.read())

    # generalization: the same trained policies evaluated at other flock sizes
    if exp.eval_flock_sizes:
        gen_rows = list(rows)
        for n_agents in exp.eval_flock_sizes:
            gen_rows += load_evals(exp_dir, exp, f"eval_n{n_agents}.json")
        gen_rows.sort(key=lambda row: (exp.ablations.index(row["ablation"]), row["n_agents"]))
        gen_table = aggregate(gen_rows, ["ablation", "n_agents"])
        write_csv(os.path.join(exp_dir, "generalization.csv"), gen_table)
        write_markdown(os.path.join(exp_dir, "generalization.md"), gen_table, ["ablation", "n_agents"])

    plot_learning_curves(exp_dir, exp)


def read_progress(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def smooth(x, window=10):
    # moving average that skips NaNs (updates where no episode finished)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        chunk = x[max(0, i - window + 1):i + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
    return out


def plot_learning_curves(exp_dir, exp):
    # one panel per raw reward term (fine to compare across ablations) plus each ablation's
    # own return (only meaningful within that ablation)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [f"term_{name}" for name in REWARD_TERMS] + ["episode_return"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharex=True)
    for i, ablation in enumerate(exp.ablations):
        curves = []
        for seed in exp.seeds:
            path = os.path.join(run_dir_for(exp_dir, ablation, seed), "progress.csv")
            if os.path.exists(path):
                curves.append(read_progress(path))
        if not curves:
            continue

        # cut every seed to the shortest one so they line up
        length = min(len(curve["env_steps"]) for curve in curves)
        steps = curves[0]["env_steps"][:length]
        color = COLORS[i % len(COLORS)]
        for ax, key in zip(axes, panels):
            values = np.stack([smooth(curve[key][:length]) for curve in curves])
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            ax.plot(steps, mean, color=color, linewidth=2, label=ablation)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    for ax, key in zip(axes, panels):
        title = key.replace("term_", "raw ")
        if key == "episode_return":
            title += " (own weights)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Task and reward ablations across seeds.")
    parser.add_argument("--summarize-only", action="store_true", help="only rebuild tables and plots")
    add_dataclass_args(parser, "exp", ExperimentConfig())
    add_config_args(parser)
    args = parser.parse_args()

    exp = apply_dataclass_args(ExperimentConfig(), args, "exp")
    base = apply_config_args(Config(), args)
    unknown = set(exp.ablations) - set(ABLATIONS)
    if unknown:
        parser.error(f"unknown ablations {sorted(unknown)}; defined: {sorted(ABLATIONS)}")

    exp_dir = os.path.join(exp.results_dir, exp.name)
    os.makedirs(exp_dir, exist_ok=True)
    if not args.summarize_only:
        with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
            json.dump({"experiment": asdict(exp), "base_config": base.to_dict()}, f, indent=2)

        configs = []
        for ablation in exp.ablations:
            for seed in exp.seeds:
                configs.append(run_config(base, exp_dir, ablation, seed))
        flock_sizes = [exp.eval_flock_sizes] * len(configs)
        with ProcessPoolExecutor(max_workers=exp.workers) as pool:
            for i, run_dir in enumerate(pool.map(run_one, configs, flock_sizes), start=1):
                print(f"[{i}/{len(configs)}] done: {run_dir}")
    summarize(exp_dir, exp)


if __name__ == "__main__":
    main()
