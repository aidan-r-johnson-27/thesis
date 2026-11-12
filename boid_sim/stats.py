import argparse
import csv
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import boid

# same metrics as boids_rl (env.flock_metrics, rewards.py terms, collisions), reported in boids_rl units
# 1 boids_rl unit = PERCEPTION_RADIUS / RL_NEIGHBOR_RADIUS px, so both sims see the same number of radii
RL_NEIGHBOR_RADIUS = 4.0
RL_SEPARATION_DIST = 1.0
K_NEIGHBORS = 4
CHUNK = 1024

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

PARAMS = ("perception", "separation", "max_speed", "max_force", "sep_weight", "ali_weight", "coh_weight")


def unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)


def wrap(delta, size):
    return delta - size * np.round(delta / size)


def neighbours(pos, vel, size, link):
    # k nearest per boid and every pair closer than link, in chunks so big flocks don't need an (N, N) array
    n = len(pos)
    n_found = min(K_NEIGHBORS, n - 1)
    nearest = np.zeros((n, n_found), dtype=np.intp)
    dist = np.full((n, K_NEIGHBORS), np.inf)
    rel_pos = np.zeros((n, K_NEIGHBORS, 2))
    edges = []
    for start in range(0, n, CHUNK):
        rows = np.arange(start, min(start + CHUNK, n))
        offsets = wrap(pos[None, :, :] - pos[rows, None, :], size)
        d = np.hypot(offsets[..., 0], offsets[..., 1])
        d[np.arange(len(rows)), rows] = np.inf
        closest = np.argpartition(d, n_found - 1, axis=1)[:, :n_found]
        r = np.arange(len(rows))[:, None]
        closest = closest[r, np.argsort(d[r, closest], axis=1)]
        nearest[rows] = closest
        dist[rows, :n_found] = d[r, closest]
        rel_pos[rows, :n_found] = offsets[r, closest]
        i, j = np.nonzero(d < link)
        edges.append((rows[i], j))
    rel_vel = np.zeros((n, K_NEIGHBORS, 2))
    rel_vel[:, :n_found] = vel[nearest] - vel[:, None, :]
    return dist, rel_pos, rel_vel, np.concatenate([e[0] for e in edges]), np.concatenate([e[1] for e in edges])


def largest_group_fraction(n, src, dst):
    labels = np.arange(n)
    while True:
        new_labels = labels.copy()
        np.minimum.at(new_labels, src, labels[dst])
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return np.bincount(labels).max() / n


def measure(pos, vel, size, collision_radius):
    neighbor_radius = RL_NEIGHBOR_RADIUS
    dist, rel_pos, rel_vel, src, dst = neighbours(pos, vel, size, neighbor_radius / 2)
    mask = dist < neighbor_radius
    rel_pos[~mask] = 0.0
    rel_vel[~mask] = 0.0
    n_visible = mask.sum(axis=1)
    collided = dist[:, 0] < collision_radius

    crowding = np.where(mask, np.clip(1 - dist / RL_SEPARATION_DIST, 0.0, None) ** 2, 0.0)
    cosines = (unit(rel_vel + vel[:, None, :]) * unit(vel)[:, None, :]).sum(axis=-1)
    alignment = np.where(mask, (cosines - 1) / 2, 0.0).sum(axis=1) / np.maximum(n_visible, 1)
    centre = rel_pos.sum(axis=1) / np.maximum(n_visible, 1)[:, None]
    slack = max(neighbor_radius - RL_SEPARATION_DIST, 1e-8)
    cohesion = -np.clip((np.linalg.norm(centre, axis=1) - RL_SEPARATION_DIST) / slack, 0.0, 1.0)

    metrics = {
        "polarization": float(np.linalg.norm(unit(vel).mean(axis=0))),
        "nn_distance": float(dist[:, 0].mean()),
        "n_visible": float(n_visible.mean()),
        "largest_group": float(largest_group_fraction(len(pos), src, dst)),
        "term_collision": float(-collided.mean()),
        "term_separation": float(-np.minimum(crowding.sum(axis=1), 1.0).mean()),
        "term_alignment": float(np.where(n_visible > 0, alignment, -0.5).mean()),
        "term_cohesion": float(np.where(n_visible > 0, cohesion, -1.0).mean()),
    }
    return metrics, collided


def configure(args):
    params = {"perception": boid.PERCEPTION_RADIUS, "separation": boid.SEPARATION_RADIUS,
              "max_speed": boid.MAX_SPEED, "max_force": boid.MAX_FORCE,
              "sep_weight": boid.WEIGHT_SEPARATION, "ali_weight": boid.WEIGHT_ALIGNMENT,
              "coh_weight": boid.WEIGHT_COHESION}
    if args.template:
        template = next(t for t in boid.TEMPLATES if t["name"] == args.template)
        params.update({k: v for k, v in template.items() if k in params})
    for name in PARAMS:
        if getattr(args, name) is not None:
            params[name] = getattr(args, name)

    boid.PERCEPTION_RADIUS = params["perception"]
    boid.SEPARATION_RADIUS = params["separation"]
    boid.MAX_SPEED = params["max_speed"]
    boid.MAX_FORCE = params["max_force"]
    boid.WEIGHT_SEPARATION = params["sep_weight"]
    boid.WEIGHT_ALIGNMENT = params["ali_weight"]
    boid.WEIGHT_COHESION = params["coh_weight"]

    px_per_unit = params["perception"] / RL_NEIGHBOR_RADIUS
    width = args.width if args.width is not None else args.world_size
    height = args.height if args.height is not None else args.world_size
    boid.WIDTH, boid.HEIGHT = width * px_per_unit, height * px_per_unit
    boid.HALF_WIDTH, boid.HALF_HEIGHT = boid.WIDTH / 2.0, boid.HEIGHT / 2.0
    boid.recompute_derived()
    return params, px_per_unit, np.array([width, height])


def evaluate(args):
    params, px_per_unit, size = configure(args)
    np.random.seed(args.seed)
    metrics, collided = [], []
    series, episodes = [], []
    for episode in range(args.n_episodes):
        boid.init_boids(args.n_boids)
        n_collided = 0
        episode_metrics = []
        for t in range(args.max_steps):
            boid.update_all()
            pos = boid.positions.astype(np.float64) / px_per_unit
            vel = boid.velocities.astype(np.float64) / px_per_unit
            step_metrics, dead = measure(pos, vel, size, args.collision_radius)
            n_collided += int(dead.sum())
            series.append({"episode": episode, "step": t, "burn_in": t < args.skip,
                           **step_metrics, "collisions": int(dead.sum())})
            if dead.any():
                # boids_rl respawns colliders somewhere random
                idx = np.flatnonzero(dead)
                boid.positions[idx], boid.velocities[idx] = boid._spawn(len(idx))
            if t >= args.skip:
                episode_metrics.append(step_metrics)
        metrics.extend(episode_metrics)
        collided.append(n_collided)
        episodes.append({"episode": episode,
                         **{name: float(np.mean([m[name] for m in episode_metrics])) for name in episode_metrics[0]},
                         "collisions": n_collided})

    results = {
        "n_agents": args.n_boids,
        "n_episodes": args.n_episodes,
        "world_size": size.tolist(),
        "px_per_unit": px_per_unit,
        "params": params,
    }
    results.update({name: float(np.mean([m[name] for m in metrics])) for name in metrics[0]})
    results["collisions_per_episode"] = float(np.mean(collided))
    return results, episodes, series


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_logs(log_dir, args, results, episodes, series):
    # episodes.csv averages leave out the burn-in, timeseries.csv keeps every step with burn_in flagged
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    write_csv(os.path.join(log_dir, "episodes.csv"), episodes)
    write_csv(os.path.join(log_dir, "timeseries.csv"), series)


def main():
    parser = argparse.ArgumentParser(description="Flock metrics of boid_sim, comparable to boids_rl.")
    parser.add_argument("--n-boids", type=int, default=64)
    parser.add_argument("--world-size", type=float, default=64.0, help="square world side, boids_rl units")
    parser.add_argument("--width", type=float, help="boids_rl units, overrides --world-size")
    parser.add_argument("--height", type=float, help="boids_rl units, overrides --world-size")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--skip", type=int, default=50, help="steps ignored at the start of each episode")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--collision-radius", type=float, default=0.0, help="boids_rl units, 0 turns it off")
    parser.add_argument("--template", choices=[t["name"] for t in boid.TEMPLATES])
    for name in PARAMS:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    parser.add_argument("--name", help="log folder name under logs/, defaults to template and time")
    args = parser.parse_args()
    if args.skip >= args.max_steps:
        parser.error("--skip has to be less than --max-steps")

    results, episodes, series = evaluate(args)
    print(json.dumps(results, indent=2))
    name = args.name or f"{args.template or 'default'}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(LOG_ROOT, name)
    write_logs(log_dir, args, results, episodes, series)
    print(f"wrote summary.json, args.json, episodes.csv, timeseries.csv to {log_dir}")


if __name__ == "__main__":
    main()
