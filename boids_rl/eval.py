import argparse
import json
import os

import numpy as np
import torch

from boids_rl.config import Config, add_config_args, apply_config_args
from boids_rl.env import BoidsEnv, mean_nn_distance, obs_dim, polarization
from boids_rl.rewards import REWARD_TERMS
from boids_rl.train import load_model, set_seed


def run_episode(model, env):
    # generator: after each step the caller can look at env.pos, env.data, ...
    obs = env.reset()
    for _ in range(env.cfg.max_steps):
        action, _, _ = model.act(torch.as_tensor(obs), deterministic=True)
        obs, reward, _, _, info = env.step(action.numpy())
        yield reward, info["terms"]


def evaluate(model, cfg):
    # mean_return depends on the run's own reward weights, so don't compare it across ablations
    # the raw term_* values and the flock metrics are fine to compare
    set_seed(cfg.eval.seed)
    env = BoidsEnv(cfg.env, cfg.reward, seed=cfg.eval.seed, auto_reset=False)
    returns, polarizations, nn_dists = [], [], []
    caught, collided = [], []
    term_values = {name: [] for name in REWARD_TERMS}

    for _ in range(cfg.eval.n_episodes):
        episode_return = np.zeros(env.n_agents)
        for reward, terms in run_episode(model, env):
            episode_return += reward
            for name, value in terms.items():
                term_values[name].append(value.mean())
            polarizations.append(polarization(env.vel))
            nn_dists.append(mean_nn_distance(env.data))
        returns.append(episode_return.mean())
        caught.append(env.deaths["caught"])
        collided.append(env.deaths["collided"])

    results = {
        "n_agents": cfg.env.n_agents,
        "n_episodes": cfg.eval.n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "polarization": float(np.mean(polarizations)),
        "mean_nn_distance": float(np.mean(nn_dists)),
        "caught_per_episode": float(np.mean(caught)),
        "collisions_per_episode": float(np.mean(collided)),
    }
    for name, values in term_values.items():
        results[f"term_{name}"] = float(np.mean(values))
    return results


def load_run(run_dir, checkpoint=None, args=None):
    cfg = Config.load(os.path.join(run_dir, "config.json"))
    if args is not None:
        apply_config_args(cfg, args)
    if checkpoint is None:
        checkpoint = os.path.join(run_dir, "checkpoints", "latest.pt")
    return cfg, load_model(checkpoint, obs_dim(cfg.env), 2, cfg.ppo)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy deterministically.")
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", help="default: <run_dir>/checkpoints/latest.pt")
    parser.add_argument("--out", help="also write the metrics to this JSON file")
    parser.add_argument("--render", metavar="PATH", help="save an animation of one episode (.gif or .mp4)")
    add_config_args(parser)
    args = parser.parse_args()

    cfg, model = load_run(args.run_dir, args.checkpoint, args)
    metrics = evaluate(model, cfg)
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    if args.render:
        # imported here so plain evaluation doesn't need matplotlib
        from boids_rl.render import render_episode
        render_episode(model, cfg, args.render)
        print(f"saved animation to {args.render}")


if __name__ == "__main__":
    main()
