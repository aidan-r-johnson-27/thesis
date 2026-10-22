import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch

from boids_rl.config import Config, add_config_args, apply_config_args
from boids_rl.env import VecBoidsEnv
from boids_rl.networks import ActorCritic
from boids_rl.ppo import RolloutBuffer, compute_gae, ppo_update
from boids_rl.rewards import REWARD_TERMS


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class EpisodeTracker:
    # adds up each column's reward over a whole world episode
    # a column holds every boid that lived in that slot, so under the survival reward
    # its return is just minus the number of deaths in it

    def __init__(self, n_columns):
        self.running = np.zeros(n_columns)
        self.finished = []

    def add(self, reward, world_end, n_agents):
        self.running += reward
        if world_end.any():
            # columns come in blocks of n_agents per env, and a whole env ends at once,
            # so log one mean return per env that just finished
            per_env = self.running.reshape(-1, n_agents)
            env_ended = world_end.reshape(-1, n_agents)[:, 0]
            for env_return in per_env[env_ended].mean(axis=1):
                self.finished.append(float(env_return))
            self.running[world_end] = 0.0

    def pop(self):
        finished = self.finished
        self.finished = []
        return finished


class RolloutCollector:
    # fills the buffer one env step at a time so the live viewer can draw in between
    # the env is never reset between rollouts, so episodes carry across them

    def __init__(self, buf):
        self.buf = buf
        self.T = buf.rewards.shape[0]
        self.t = 0
        self.final_values = np.zeros_like(buf.values)
        self.terms = {name: 0.0 for name in REWARD_TERMS}
        self.metrics = {}

    @property
    def full(self):
        return self.t == self.T

    def step(self, env, model, obs, tracker):
        buf, t = self.buf, self.t
        action, log_prob, value = model.act(torch.as_tensor(obs))
        next_obs, reward, terminated, truncated, info = env.step(action.numpy())

        buf.obs[t] = obs
        buf.actions[t] = action.numpy()
        buf.log_probs[t] = log_prob.numpy()
        buf.values[t] = value.numpy()
        buf.rewards[t] = reward
        buf.terminated[t] = terminated
        buf.episode_end[t] = terminated | truncated
        if "final_obs" in info:
            self.final_values[t] = model.value(torch.as_tensor(info["final_obs"])).detach().numpy()

        # running means over the rollout, for logging
        for name, term in info["terms"].items():
            self.terms[name] += term.mean() / self.T
        for name, metric in env.flock_metrics().items():
            self.metrics[name] = self.metrics.get(name, 0.0) + metric / self.T

        tracker.add(reward, truncated, env.n_agents)
        self.t += 1
        return next_obs, reward

    def finish(self, model, obs):
        # V(s_t+1) is normally just the next row's value, but where the world got cut off
        # the next row is already a new episode, so use the value of the final obs instead
        buf = self.buf
        buf.next_values[:-1] = buf.values[1:]
        buf.next_values[-1] = model.value(torch.as_tensor(obs)).detach().numpy()
        cut_off = buf.episode_end & ~buf.terminated
        buf.next_values[cut_off] = self.final_values[cut_off]


def collect_rollout(env, model, obs, buf, tracker):
    collector = RolloutCollector(buf)
    while not collector.full:
        obs, _ = collector.step(env, model, obs, tracker)
    collector.finish(model, obs)
    return obs, collector.terms, collector.metrics


CSV_FIELDS = (["update", "env_steps", "wall_time", "episode_return", "n_episodes"]
              + [f"term_{name}" for name in REWARD_TERMS]
              + ["polarization", "nn_distance", "n_visible", "largest_group"]
              + ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "sps"])


class CSVLogger:
    # one row per PPO update in progress.csv

    def __init__(self, path, resume_update=0):
        self.path = path
        old_rows = []
        if resume_update > 0 and os.path.exists(path):
            # throw away anything logged after the checkpoint I'm resuming from
            with open(path) as f:
                old_rows = [row for row in csv.DictReader(f) if int(row["update"]) <= resume_update]
        with open(path, "w", newline="") as f:
            # restval fills in nan for columns that didn't exist when an old row was written
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="nan", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old_rows)

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# saved in checkpoints so an old world-frame policy never gets run in the body frame
FRAME = "body"


def save_checkpoint(path, update, model, optimizer, env, obs, tracker, shuffle_rng):
    checkpoint = {
        "frame": FRAME,
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_state": env.get_state(),
        "obs": obs,
        "tracker_running": tracker.running,
        "shuffle_rng": shuffle_rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    # write to a temp file then rename, so a crash can't leave a half-written checkpoint
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def check_checkpoint(checkpoint, obs_dim, path):
    saved_obs_dim = checkpoint["model"]["actor.0.weight"].shape[1]
    if saved_obs_dim != obs_dim:
        raise SystemExit(f"{path} expects {saved_obs_dim} observation inputs but the env now gives {obs_dim} "
                         "(it predates the predator observations). Start a new run: --fresh, "
                         "or another --train.run_dir.")
    # same size inputs but they meant something else back when everything was in the world frame
    if checkpoint.get("frame") != FRAME:
        raise SystemExit(f"{path} is a world-frame policy from before observations and actions "
                         "moved to the body frame. Start a new run: --fresh, or another --train.run_dir.")


def load_model(path, obs_dim, act_dim, ppo_cfg):
    model = ActorCritic(obs_dim, act_dim, ppo_cfg.hidden_size, ppo_cfg.init_log_std)
    checkpoint = torch.load(path, weights_only=False)
    check_checkpoint(checkpoint, obs_dim, path)
    model.load_state_dict(checkpoint["model"])
    return model


class Trainer:
    # one training run, moved forward one env step at a time
    # step() does a PPO update whenever the rollout is full; train() just calls it in a loop,
    # and live.py calls it between frames

    def __init__(self, cfg, resume=False, verbose=True):
        self.cfg = cfg
        self.verbose = verbose
        run_dir = cfg.train.run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        set_seed(cfg.train.seed)
        self.env = VecBoidsEnv(cfg.env, cfg.reward, cfg.train.n_envs, seed=cfg.train.seed)
        self.model = ActorCritic(self.env.obs_dim, self.env.act_dim, cfg.ppo.hidden_size, cfg.ppo.init_log_std)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self.shuffle_rng = np.random.default_rng(cfg.train.seed)

        self.T = cfg.train.rollout_steps
        self.B = self.env.n_envs * self.env.n_agents
        self.tracker = EpisodeTracker(self.B)
        self.obs = self.env.reset()
        self.last_reward = np.zeros(self.B, np.float32)
        self.update = 0

        self.latest = os.path.join(self.ckpt_dir, "latest.pt")
        # set when resuming with a different env config: keep the policy but start new worlds
        self.new_worlds = False
        if resume:
            self._resume(run_dir)
        # saved after resuming so a failed resume doesn't overwrite the old config
        cfg.save(os.path.join(run_dir, "config.json"))
        self.logger = CSVLogger(os.path.join(run_dir, "progress.csv"), resume_update=self.update)

        self.steps_per_update = self.T * self.env.n_envs
        self.n_updates = cfg.train.total_env_steps // self.steps_per_update
        if self.n_updates == 0:
            raise ValueError(f"total_env_steps must be at least rollout_steps * n_envs = {self.steps_per_update}")
        self.first_update = self.update + 1
        self.start_time = time.time()
        self._new_rollout()

    def _resume(self, run_dir):
        checkpoint = torch.load(self.latest, weights_only=False)
        check_checkpoint(checkpoint, self.env.obs_dim, self.latest)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # the saved worlds only make sense with the env config they were made with
        # (e.g. same number of boids), otherwise just start fresh worlds
        cfg_path = os.path.join(run_dir, "config.json")
        saved_cfg = Config.load(cfg_path) if os.path.exists(cfg_path) else self.cfg
        same_env = saved_cfg.env == self.cfg.env and saved_cfg.train.n_envs == self.cfg.train.n_envs
        if same_env:
            self.env.set_state(checkpoint["env_state"])
            self.obs = checkpoint["obs"]
            self.tracker.running = checkpoint["tracker_running"]
        else:
            self.new_worlds = True

        # restore every random number generator so the resumed run matches an uninterrupted one
        self.shuffle_rng.bit_generator.state = checkpoint["shuffle_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        self.update = checkpoint["update"]
        if self.verbose:
            note = " (env config changed: new worlds)" if self.new_worlds else ""
            print(f"resumed from {self.latest} at update {self.update}{note}")

    def _new_rollout(self):
        buf = RolloutBuffer(self.T, self.B, self.env.obs_dim, self.env.act_dim)
        self.collector = RolloutCollector(buf)

    def step(self):
        # returns the progress.csv row if this step finished a rollout (and so ran an update)
        self.obs, self.last_reward = self.collector.step(self.env, self.model, self.obs, self.tracker)
        if self.collector.full:
            return self._ppo_update()
        return None

    def _ppo_update(self):
        cfg, buf = self.cfg, self.collector.buf
        self.collector.finish(self.model, self.obs)
        advantages, returns = compute_gae(buf.rewards, buf.values, buf.next_values, buf.terminated,
                                          buf.episode_end, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        stats = ppo_update(self.model, self.optimizer, buf, advantages, returns, cfg.ppo, self.shuffle_rng)

        self.update += 1
        update = self.update
        terms = self.collector.terms
        episode_returns = self.tracker.pop()
        env_steps = update * self.steps_per_update
        elapsed = time.time() - self.start_time
        steps_this_session = (update - self.first_update + 1) * self.steps_per_update

        row = {"update": update,
               "env_steps": env_steps,
               "wall_time": round(elapsed, 2),
               "episode_return": np.mean(episode_returns) if episode_returns else float("nan"),
               "n_episodes": len(episode_returns)}
        for name, value in terms.items():
            row[f"term_{name}"] = value
        row.update(self.collector.metrics)
        row.update(stats)
        row["sps"] = round(steps_this_session / elapsed)
        self.logger.log(row)

        if self.verbose:
            term_text = " ".join(f"{name} {value:+.3f}" for name, value in terms.items())
            print(f"update {update:4d} | steps {env_steps:8d} | return {row['episode_return']:8.2f} | "
                  f"{term_text} | kl {stats['approx_kl']:.4f} | {row['sps']} sps")

        if update % cfg.train.checkpoint_every == 0 or update == self.n_updates:
            self.save()
            shutil.copyfile(self.latest, os.path.join(self.ckpt_dir, f"update_{update:06d}.pt"))
        self._new_rollout()
        return row

    def save(self):
        # saving mid-rollout drops the half-filled rollout; resuming starts a new one
        save_checkpoint(self.latest, self.update, self.model, self.optimizer, self.env, self.obs,
                        self.tracker, self.shuffle_rng)


def train(cfg, resume=False, verbose=True):
    trainer = Trainer(cfg, resume=resume, verbose=verbose)
    while trainer.update < trainer.n_updates:
        trainer.step()
    return trainer.model


def main():
    parser = argparse.ArgumentParser(description="Train the shared boid policy with PPO.")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <run_dir>/checkpoints/latest.pt using <run_dir>/config.json; "
                             "flags given here still override it (e.g. a larger --train.total_env_steps)")
    add_config_args(parser)
    args = parser.parse_args()

    cfg = Config()
    if args.resume:
        run_dir = getattr(args, "train.run_dir") or cfg.train.run_dir
        cfg = Config.load(os.path.join(run_dir, "config.json"))
    train(apply_config_args(cfg, args), resume=args.resume)


if __name__ == "__main__":
    main()
