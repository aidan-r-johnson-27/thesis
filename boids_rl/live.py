import argparse
import csv
import os
import time

import numpy as np
import torch

from boids_rl.config import Config, add_config_args, apply_config_args
from boids_rl.env import BoidsEnv
from boids_rl.train import CSV_FIELDS, Trainer

# env steps per frame; 0 means "as many as fit in the frame" and is only reachable with the arrow keys
SPEEDS = [1, 2, 4, 8, 16, 32, 128, 0]
DEFAULT_RUN_DIR = "runs/live"


def parse_args():
    parser = argparse.ArgumentParser(description="Train (or watch) the boid policy live.")
    parser.add_argument("--demo", action="store_true", help="start in demo mode (deterministic policy, no learning)")
    parser.add_argument("--fresh", action="store_true", help="ignore the run's checkpoint and start from scratch")
    parser.add_argument("--speed", type=int, default=1, choices=[s for s in SPEEDS if s > 0],
                        help="starting speed (env steps per frame)")
    parser.add_argument("--updates", type=int, default=0, help="stop after this many more PPO updates (0 = keep going)")
    add_config_args(parser)
    return parser.parse_args()


def make_trainer(args, fresh):
    # resume the run if it has a checkpoint (its saved config plus any flags on top), else start a new one
    run_dir = getattr(args, "train.run_dir") or DEFAULT_RUN_DIR
    resume = not fresh and os.path.exists(os.path.join(run_dir, "checkpoints", "latest.pt"))
    cfg = Config.load(os.path.join(run_dir, "config.json")) if resume else Config()
    cfg.train.run_dir = run_dir
    apply_config_args(cfg, args)
    return Trainer(cfg, resume=resume, verbose=False), resume


def read_history(trainer):
    # charts start from what's already in progress.csv (already trimmed back to the checkpoint)
    history = {name: [] for name in CSV_FIELDS}
    with open(trainer.logger.path) as f:
        for row in csv.DictReader(f):
            for name in CSV_FIELDS:
                history[name].append(float(row.get(name) or "nan"))
    return history


def main():
    args = parse_args()
    # imported here so the rest of the package doesn't need pygame
    import pygame

    from boids_rl.live_render import Renderer

    trainer, resumed = make_trainer(args, args.fresh)
    history = read_history(trainer)
    cfg = trainer.cfg
    ui = Renderer(cfg.env.world_size)
    clock = pygame.time.Clock()

    speed_index = SPEEDS.index(args.speed)
    show_world, paused, learning = True, False, not args.demo
    show_trails, show_focus, color_mode = True, True, "plain"
    view_index = 0   # which training env is on screen

    message = ""
    if resumed:
        message = f"resumed {cfg.train.run_dir} at update {trainer.update}"
        if trainer.new_worlds:
            message += f" (new worlds: {cfg.env.n_agents} boids, size {cfg.env.world_size:g})"
    message_until = time.time() + 2.5
    sps, sps_steps, sps_since = 0.0, 0, time.time()
    stop_at_update = trainer.update + args.updates if args.updates else None
    updating_until = 0.0  # the badge says "PPO UPDATE" until this time

    # demo mode runs its own env with the mean action on the eval seed,
    # so the training envs and the buffer are left alone
    demo_env = BoidsEnv(cfg.env, cfg.reward, seed=cfg.eval.seed)
    demo_obs = demo_env.reset()
    demo_reward = np.zeros(cfg.env.n_agents)
    demo_return = np.zeros(cfg.env.n_agents)

    def notify(text):
        nonlocal message, message_until
        message = text
        message_until = time.time() + 2.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif key == pygame.K_SPACE:
                    paused = not paused
                elif key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_EQUALS, pygame.K_PLUS):
                    speed_index = min(len(SPEEDS) - 1, speed_index + 1)
                elif key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                    speed_index = max(0, speed_index - 1)
                elif key == pygame.K_v:
                    show_world = not show_world
                elif key == pygame.K_d:
                    learning = not learning
                    if learning:
                        notify("training resumed")
                    else:
                        demo_obs = demo_env.reset()
                        demo_return[:] = 0.0
                        notify("demo: deterministic policy, no learning")
                elif key == pygame.K_e:
                    view_index = (view_index + 1) % trainer.env.n_envs
                    notify(f"viewing training env {view_index + 1}/{trainer.env.n_envs}")
                elif key == pygame.K_c:
                    color_mode = "reward" if color_mode == "plain" else "plain"
                elif key == pygame.K_n:
                    show_focus = not show_focus
                elif key == pygame.K_t:
                    show_trails = not show_trails
                elif key == pygame.K_s:
                    trainer.save()
                    notify(f"saved checkpoint (update {trainer.update})")
                elif key == pygame.K_r:
                    trainer, _ = make_trainer(args, fresh=True)
                    history = read_history(trainer)
                    learning, view_index = True, 0
                    stop_at_update = trainer.update + args.updates if args.updates else None
                    sps, sps_steps, sps_since = 0.0, 0, time.time()
                    notify("training reset")

        # with the world hidden there's nothing to watch, so always go max speed
        speed = SPEEDS[speed_index] if show_world else 0
        if not paused:
            deadline = time.perf_counter() + (0.014 if show_world else 0.03)
            steps_done = 0
            while True:
                if learning:
                    row = trainer.step()
                    if row is not None:
                        updating_until = time.time() + 0.4
                        for name in CSV_FIELDS:
                            history[name].append(float(row.get(name, float("nan"))))
                        if stop_at_update and trainer.update >= stop_at_update:
                            running = False
                            break
                else:
                    action, _, _ = trainer.model.act(torch.as_tensor(demo_obs), deterministic=True)
                    demo_obs, demo_reward, _, truncated, _ = demo_env.step(action.numpy())
                    demo_return = np.where(truncated, 0.0, demo_return + demo_reward)
                steps_done += 1
                if speed and steps_done >= speed:
                    break
                if time.perf_counter() > deadline:
                    break
            sps_steps += steps_done

        now = time.time()
        if now - sps_since >= 0.5:
            sps = sps_steps / (now - sps_since)
            sps_steps, sps_since = 0, now

        n_agents = cfg.env.n_agents
        if learning:
            world = trainer.env.envs[view_index]
            columns = slice(view_index * n_agents, (view_index + 1) * n_agents)
            rewards, all_rewards = trainer.last_reward[columns], trainer.last_reward
            episode_return = float(trainer.tracker.running[columns].mean())
            world_label = f"TRAINING  env {view_index + 1}/{trainer.env.n_envs}  (E)"
        else:
            world, rewards, all_rewards = demo_env, demo_reward, demo_reward
            episode_return = float(demo_return.mean())
            world_label = "DEMO  deterministic policy, eval seed"
        # colour scale for "shade by reward", ignoring the extremes
        low, high = np.percentile(all_rewards, [5, 95])

        if paused:
            mode = "PAUSED"
        elif not learning:
            mode = "DEMO"
        elif now < updating_until:
            mode = "PPO UPDATE"
        else:
            mode = "TRAINING"
        info = {
            "mode": mode,
            "speed": "max" if speed == 0 else f"{speed}x",
            "steps_per_frame": SPEEDS[speed_index] or 10 ** 9,
            "update": trainer.update,
            "env_steps": trainer.update * trainer.steps_per_update + trainer.collector.t * trainer.env.n_envs,
            "rollout_t": trainer.collector.t,
            "rollout_T": trainer.T,
            "sps": sps * (trainer.env.n_envs if learning else 1),
            "rewards": rewards,
            "reward_range": (low, high),
            "episode_return": episode_return,
            "world_label": world_label,
            "color_mode": color_mode,
            "show_trails": show_trails,
            "show_focus": show_focus,
            "toast": message if now < message_until else "",
        }
        ui.draw(world, history, info, show_world)
        clock.tick(60 if show_world else 0)

    trainer.save()
    pygame.quit()
    print(f"saved {trainer.latest}  update {trainer.update:,}  "
          f"env steps {trainer.update * trainer.steps_per_update:,}")


if __name__ == "__main__":
    main()
