import argparse
import json

import numpy as np

from boids_rl.config import Config, add_config_args, apply_config_args
from boids_rl.env import BoidsEnv, clip_norm, to_body

# hand-written policies to compare the trained one against
# a policy here takes the env and returns world-frame accelerations, one row per boid

# classic boids weights: separation wins up close, alignment and cohesion shape the flock
W_SEPARATION = 1.5
W_ALIGNMENT = 1.0
W_COHESION = 1.0
SEPARATION_DIST = 1.0


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def reynolds(env):
    # Reynolds steering, but only using what the learned policy can see (env.data),
    # so it's a fair "hand-written answer"
    cfg, data = env.cfg, env.data
    n_visible = np.maximum(data.mask.sum(axis=1), 1)[:, None]
    sees_anyone = data.mask.any(axis=1)[:, None]

    # push away from close neighbours, harder the closer they are (1 / dist)
    too_close = data.mask & (data.dist < SEPARATION_DIST)
    push = data.rel_pos / np.maximum(data.dist, 1e-8)[..., None] ** 2
    away = -(push * too_close[..., None]).sum(axis=1)
    separation = np.where(too_close.any(axis=1)[:, None], unit(away) * cfg.max_speed - data.vel, 0.0)

    # each rule is "desired velocity minus current velocity"
    mean_neighbour_vel = data.rel_vel.sum(axis=1) / n_visible + data.vel
    alignment = unit(mean_neighbour_vel) * cfg.max_speed - data.vel
    centre = data.rel_pos.sum(axis=1) / n_visible
    cohesion = unit(centre) * cfg.max_speed - data.vel

    steer = (W_SEPARATION * clip_norm(separation, cfg.max_accel)
             + W_ALIGNMENT * clip_norm(alignment, cfg.max_accel)
             + W_COHESION * clip_norm(cohesion, cfg.max_accel))
    # boids that see nobody just fly straight
    return np.where(sees_anyone, clip_norm(steer, cfg.max_accel), 0.0)


def random_policy(env):
    # same as the untrained network: N(0, 1) accelerations
    return env.rng.normal(size=(env.n_agents, 2))


POLICIES = {"reynolds": reynolds, "random": random_policy}


def evaluate_policy(policy, cfg, n_episodes=None, skip=50):
    # skip the first steps of each episode so the flock has time to form
    env = BoidsEnv(cfg.env, cfg.reward, seed=cfg.eval.seed, auto_reset=False)
    if n_episodes is None:
        n_episodes = cfg.eval.n_episodes
    metrics, caught, collided = [], [], []
    for _ in range(n_episodes):
        env.reset()
        for t in range(cfg.env.max_steps):
            env.step(to_body(policy(env), env.heading))
            if t >= skip:
                metrics.append(env.flock_metrics())
        caught.append(env.deaths["caught"])
        collided.append(env.deaths["collided"])

    results = {name: float(np.mean([m[name] for m in metrics])) for name in metrics[0]}
    results["caught_per_episode"] = float(np.mean(caught))
    results["collisions_per_episode"] = float(np.mean(collided))
    return results


def main():
    parser = argparse.ArgumentParser(description="Flock metrics of a scripted (non-learned) policy.")
    parser.add_argument("policy", choices=sorted(POLICIES))
    add_config_args(parser)
    args = parser.parse_args()
    cfg = apply_config_args(Config(), args)
    print(json.dumps(evaluate_policy(POLICIES[args.policy], cfg), indent=2))


if __name__ == "__main__":
    main()
