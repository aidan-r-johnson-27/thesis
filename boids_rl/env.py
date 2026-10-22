from dataclasses import dataclass

import numpy as np

from boids_rl.rewards import compute_rewards


def wrap(delta, size):
    # shortest way around the torus: take off however many whole world widths gets it closest to 0
    return delta - size * np.round(delta / size)


def clip_norm(vectors, max_norm):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))


def to_body(vectors, heading):
    # world frame -> each boid's own (forward, left) frame
    # vectors is (N, 2) or (N, k, 2) when there's one per neighbour slot
    if vectors.ndim == 3:
        heading = heading[:, None, :]
    hx, hy = heading[..., 0], heading[..., 1]
    forward = vectors[..., 0] * hx + vectors[..., 1] * hy
    left = vectors[..., 1] * hx - vectors[..., 0] * hy
    return np.stack([forward, left], axis=-1)


def to_world(vectors, heading):
    # undoes to_body for (N, 2) vectors
    hx, hy = heading[:, 0], heading[:, 1]
    x = vectors[:, 0] * hx - vectors[:, 1] * hy
    y = vectors[:, 0] * hy + vectors[:, 1] * hx
    return np.stack([x, y], axis=-1)


def obs_dim(env_cfg):
    # own velocity (2) + k neighbours * (rel pos, rel vel) (4 each) + predator pos, vel, visible (5)
    return 2 + 4 * env_cfg.k_neighbors + 5


@dataclass
class StepData:
    # everything the observations, rewards and metrics need, worked out once per step
    vel: np.ndarray                  # (N, 2) own velocity
    rel_pos: np.ndarray              # (N, k, 2) neighbour pos minus own pos, zero if not visible
    rel_vel: np.ndarray              # (N, k, 2) neighbour vel minus own vel, zero if not visible
    dist: np.ndarray                 # (N, k) distances to the k nearest boids, closest first
    mask: np.ndarray                 # (N, k) True where that neighbour is inside neighbor_radius
    neighbor_radius: float
    world_size: float
    pred_rel_pos: np.ndarray = None  # (N, 2) nearest predator, zero if not visible
    pred_rel_vel: np.ndarray = None  # (N, 2)
    pred_dist: np.ndarray = None     # (N,) inf if there are no predators
    pred_mask: np.ndarray = None     # (N,) True where the nearest predator is visible
    caught: np.ndarray = None        # (N,) caught by a predator this step
    collided: np.ndarray = None      # (N,) hit another boid this step
    all_dists: np.ndarray = None     # (N, N) every pairwise distance, inf on the diagonal


def compute_step_data(pos, vel, cfg, pred_pos=None, pred_vel=None):
    n_boids, k = len(pos), cfg.k_neighbors
    size = cfg.world_size

    # dx[i, j], dy[i, j] is where boid j is relative to boid i
    # (x and y kept as separate (N, N) arrays because that's a lot faster than one (N, N, 2))
    dx = wrap(pos[None, :, 0] - pos[:, None, 0], size)
    dy = wrap(pos[None, :, 1] - pos[:, None, 1], size)
    all_dists = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(all_dists, np.inf)

    # with fewer than k other boids the extra slots just stay empty
    n_found = min(k, n_boids - 1)
    rows = np.arange(n_boids)[:, None]
    # argpartition pulls out the k closest without sorting the whole row, then sort just those
    closest = np.argpartition(all_dists, n_found - 1, axis=1)[:, :n_found]
    order = np.argsort(all_dists[rows, closest], axis=1)
    nearest = closest[rows, order]

    dist = np.full((n_boids, k), np.inf)
    rel_pos = np.zeros((n_boids, k, 2))
    rel_vel = np.zeros((n_boids, k, 2))
    dist[:, :n_found] = all_dists[rows, nearest]
    rel_pos[:, :n_found, 0] = dx[rows, nearest]
    rel_pos[:, :n_found, 1] = dy[rows, nearest]
    rel_vel[:, :n_found] = vel[nearest] - vel[:, None, :]

    mask = dist < cfg.neighbor_radius
    rel_pos[~mask] = 0.0
    rel_vel[~mask] = 0.0

    pred_rel_pos = np.zeros((n_boids, 2))
    pred_rel_vel = np.zeros((n_boids, 2))
    pred_dist = np.full(n_boids, np.inf)
    if pred_pos is not None and len(pred_pos) > 0:
        pred_offsets = wrap(pred_pos[None, :, :] - pos[:, None, :], size)
        pred_dists = np.linalg.norm(pred_offsets, axis=-1)
        closest = pred_dists.argmin(axis=1)
        boids = np.arange(n_boids)
        pred_dist = pred_dists[boids, closest]
        pred_rel_pos = pred_offsets[boids, closest]
        pred_rel_vel = pred_vel[closest] - vel

    # predators are seen with the same radius as neighbours
    pred_mask = pred_dist < cfg.neighbor_radius
    pred_rel_pos[~pred_mask] = 0.0
    pred_rel_vel[~pred_mask] = 0.0

    return StepData(vel=vel.copy(), rel_pos=rel_pos, rel_vel=rel_vel, dist=dist, mask=mask,
                    neighbor_radius=cfg.neighbor_radius, world_size=size,
                    pred_rel_pos=pred_rel_pos, pred_rel_vel=pred_rel_vel,
                    pred_dist=pred_dist, pred_mask=pred_mask,
                    caught=np.zeros(n_boids, bool), collided=np.zeros(n_boids, bool),
                    all_dists=all_dists)


def polarization(vel):
    # length of the average heading: 1 = everyone flying the same way, ~0 = random
    heading = vel / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-8)
    return float(np.linalg.norm(heading.mean(axis=0)))


def mean_nn_distance(data):
    return float(data.dist[:, 0].mean())


def largest_group_fraction(linked):
    # every boid starts with its own label and keeps taking the smallest label of anyone
    # it's linked to; once nothing changes, each connected group shares a single label
    n_boids = len(linked)
    labels = np.arange(n_boids)
    while True:
        neighbour_min = np.where(linked, labels[None, :], n_boids).min(axis=1)
        new_labels = np.minimum(labels, neighbour_min)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return np.bincount(labels).max() / n_boids


def largest_group(pos, link, size):
    offsets = wrap(pos[None, :, :] - pos[:, None, :], size)
    linked = np.linalg.norm(offsets, axis=-1) < link
    return float(largest_group_fraction(linked))


def flock_metrics(vel, data):
    # groups link boids that are within half the perception radius of each other
    link = data.neighbor_radius / 2
    return {"polarization": polarization(vel),
            "nn_distance": mean_nn_distance(data),
            "n_visible": float(data.mask.sum(axis=1).mean()),
            "largest_group": float(largest_group_fraction(data.all_dists < link))}


class BoidsEnv:
    # one world; every boid is stepped at once and is its own row in every array
    # a boid that dies gets terminated=True and is respawned straight away, the world itself
    # just runs for max_steps and then gets truncated

    def __init__(self, env_cfg, reward_cfg, seed=None, auto_reset=True):
        self.cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.auto_reset = auto_reset
        self.rng = np.random.default_rng(seed)
        self.n_agents = env_cfg.n_agents
        self.obs_dim = obs_dim(env_cfg)
        self.act_dim = 2

    def _random_boids(self, count):
        cfg = self.cfg
        pos = self.rng.uniform(0, cfg.world_size, (count, 2))
        angle = self.rng.uniform(0, 2 * np.pi, count)
        speed = self.rng.uniform(cfg.min_speed, cfg.max_speed, count)
        vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]
        return pos, vel

    def _update_heading(self):
        # heading = direction of travel, but a boid that has stopped dead keeps its old one
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.heading = np.where(speed > 1e-8, self.vel / np.maximum(speed, 1e-8), self.heading)

    def _step_data(self):
        return compute_step_data(self.pos, self.vel, self.cfg, self.pred_pos, self.pred_vel)

    def reset(self):
        cfg = self.cfg
        self.pos, self.vel = self._random_boids(cfg.n_agents)
        self.pred_pos = self.rng.uniform(0, cfg.world_size, (cfg.n_predators, 2))
        self.pred_vel = np.zeros_like(self.pred_pos)
        # predators start on cooldown, which gives the boids a grace period
        self.pred_cooldown = np.full(cfg.n_predators, cfg.predator_cooldown)
        self.heading = np.tile([1.0, 0.0], (cfg.n_agents, 1))
        self._update_heading()
        self.t = 0
        self.deaths = {"caught": 0, "collided": 0}
        self.last_deaths = np.zeros((0, 2))
        self.data = self._step_data()
        return self._get_obs()

    def get_state(self):
        # enough to carry on exactly where we left off (goes into checkpoints)
        return {"pos": self.pos.copy(), "vel": self.vel.copy(), "heading": self.heading.copy(),
                "t": self.t, "pred_pos": self.pred_pos.copy(), "pred_vel": self.pred_vel.copy(),
                "pred_cooldown": self.pred_cooldown.copy(), "deaths": dict(self.deaths),
                "rng": self.rng.bit_generator.state}

    def set_state(self, state):
        self.pos = np.array(state["pos"])
        self.vel = np.array(state["vel"])
        self.heading = np.array(state["heading"])
        self.t = state["t"]
        self.pred_pos = np.array(state["pred_pos"])
        self.pred_vel = np.array(state["pred_vel"])
        self.pred_cooldown = np.array(state["pred_cooldown"])
        self.deaths = {kind: int(count) for kind, count in state["deaths"].items()}
        self.last_deaths = np.zeros((0, 2))
        self.rng.bit_generator.state = state["rng"]
        self.data = self._step_data()
        return self._get_obs()

    def _get_obs(self):
        # everything is in the boid's own frame, and divided by a scale that puts it in about [-1, 1]
        cfg, data, heading = self.cfg, self.data, self.heading
        own_vel = to_body(data.vel, heading) / cfg.max_speed
        nbr_pos = to_body(data.rel_pos, heading) / cfg.neighbor_radius
        nbr_vel = to_body(data.rel_vel, heading) / (2 * cfg.max_speed)
        # one block of 4 numbers per neighbour slot: (pos fwd, pos left, vel fwd, vel left)
        neighbours = np.concatenate([nbr_pos, nbr_vel], axis=-1).reshape(len(heading), -1)
        pred_pos = to_body(data.pred_rel_pos, heading) / cfg.neighbor_radius
        pred_vel = to_body(data.pred_rel_vel, heading) / (cfg.max_speed + cfg.predator_max_speed)
        pred_visible = data.pred_mask[:, None]
        obs = np.concatenate([own_vel, neighbours, pred_pos, pred_vel, pred_visible], axis=1)
        return obs.astype(np.float32)

    def _move_predators(self):
        cfg = self.cfg
        # offsets[p, i] is where boid i is relative to predator p
        offsets = wrap(self.pos[None, :, :] - self.pred_pos[:, None, :], cfg.world_size)
        dists = np.linalg.norm(offsets, axis=-1)
        targets = dists.argmin(axis=1)
        preds = np.arange(len(self.pred_pos))

        # steer straight at the nearest boid at full speed
        direction = offsets[preds, targets] / np.maximum(dists[preds, targets], 1e-8)[:, None]
        desired_vel = direction * cfg.predator_max_speed
        accel = clip_norm((desired_vel - self.pred_vel) / cfg.dt, cfg.predator_max_accel)
        self.pred_vel = clip_norm(self.pred_vel + accel * cfg.dt, cfg.predator_max_speed)
        self.pred_pos = (self.pred_pos + self.pred_vel * cfg.dt) % cfg.world_size

        # the predator has moved, so measure the distance to its target again
        target_offsets = wrap(self.pos[targets] - self.pred_pos, cfg.world_size)
        return targets, np.linalg.norm(target_offsets, axis=-1)

    def _attack(self, targets, target_dists):
        cfg = self.cfg
        caught = np.zeros(self.n_agents, bool)
        self.pred_cooldown = np.maximum(self.pred_cooldown - 1, 0)
        ready = (self.pred_cooldown == 0) & (target_dists < cfg.catch_radius)
        for p in np.flatnonzero(ready):
            target = targets[p]
            dists_to_target = np.linalg.norm(wrap(self.pos - self.pos[target], cfg.world_size), axis=1)
            # confusion effect: the more boids crowd the target, the more likely the attack misses
            n_crowd = np.sum(dists_to_target < cfg.confusion_radius) - 1
            if self.rng.random() < 1.0 / (1.0 + cfg.confusion * n_crowd):
                caught[target] = True
            # hit or miss, the predator has to wait before attacking again
            self.pred_cooldown[p] = cfg.predator_cooldown
        return caught

    def _respawn(self, dead):
        cfg = self.cfg
        idx = np.flatnonzero(dead)
        self.pos[idx], self.vel[idx] = self._random_boids(len(idx))
        # re-roll any spawn that a predator could see (give up after 20 tries)
        if cfg.n_predators > 0:
            for _ in range(20):
                offsets = wrap(self.pred_pos[None, :, :] - self.pos[idx, None, :], cfg.world_size)
                seen = (np.linalg.norm(offsets, axis=-1) < cfg.neighbor_radius).any(axis=1)
                bad = idx[seen]
                if len(bad) == 0:
                    break
                self.pos[bad] = self.rng.uniform(0, cfg.world_size, (len(bad), 2))
        self._update_heading()

    def step(self, actions):
        cfg = self.cfg

        # actions are (forward, left) accelerations in each boid's own frame
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n_agents, 2)
        accel = clip_norm(actions, cfg.max_accel)
        self.vel = clip_norm(self.vel + to_world(accel, self.heading) * cfg.dt, cfg.max_speed)

        # boids can't hover: too slow gets pushed back up to min_speed in the direction it's
        # now going (so turns still count), or along the old heading if it stopped dead
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        direction = np.where(speed > 1e-8, self.vel / np.maximum(speed, 1e-8), self.heading)
        self.vel = np.where(speed < cfg.min_speed, direction * cfg.min_speed, self.vel)
        self._update_heading()
        self.pos = (self.pos + self.vel * cfg.dt) % cfg.world_size

        if cfg.n_predators > 0:
            targets, target_dists = self._move_predators()
            caught = self._attack(targets, target_dists)
        else:
            caught = np.zeros(self.n_agents, bool)
        self.t += 1

        self.data = self._step_data()
        self.data.caught = caught
        self.data.collided = self.data.dist[:, 0] < cfg.collision_radius
        reward, terms = compute_rewards(self.data, self.reward_cfg)

        dead = caught | self.data.collided
        self.deaths["caught"] += int(caught.sum())
        self.deaths["collided"] += int(self.data.collided.sum())
        self.last_deaths = self.pos[dead]
        if dead.any():
            # dead boids come back somewhere else as a new episode in the same slot
            self._respawn(dead)
            self.data = self._step_data()
        obs = self._get_obs()

        info = {"terms": terms}
        truncated = np.full(self.n_agents, self.t >= cfg.max_steps)
        if truncated[0] and self.auto_reset:
            # hang on to the real last obs so GAE can still bootstrap from it
            info["final_obs"] = obs
            obs = self.reset()
        return obs, reward.astype(np.float32), dead, truncated, info

    def flock_metrics(self):
        return flock_metrics(self.vel, self.data)


class VecBoidsEnv:
    # several independent worlds; their boids get stacked into one batch (world 0's boids,
    # then world 1's, ...) so the policy only needs one forward pass for all of them

    def __init__(self, env_cfg, reward_cfg, n_envs, seed=None, auto_reset=True):
        self.cfg = env_cfg
        self.n_envs = n_envs
        self.n_agents = env_cfg.n_agents
        self.obs_dim = obs_dim(env_cfg)
        self.act_dim = 2
        # each world gets its own independent random stream
        seeds = np.random.SeedSequence(seed).spawn(n_envs)
        self.envs = [BoidsEnv(env_cfg, reward_cfg, s, auto_reset) for s in seeds]

    @property
    def t(self):
        return self.envs[0].t

    def reset(self):
        return np.concatenate([env.reset() for env in self.envs])

    def get_state(self):
        return [env.get_state() for env in self.envs]

    def set_state(self, states):
        return np.concatenate([env.set_state(s) for env, s in zip(self.envs, states)])

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n_envs, self.n_agents, 2)
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs = np.concatenate([r[0] for r in results])
        reward = np.concatenate([r[1] for r in results])
        terminated = np.concatenate([r[2] for r in results])
        truncated = np.concatenate([r[3] for r in results])

        infos = [r[4] for r in results]
        terms = {name: np.concatenate([i["terms"][name] for i in infos]) for name in infos[0]["terms"]}
        info = {"terms": terms}
        # all worlds start together, so they all get truncated on the same step
        if "final_obs" in infos[0]:
            info["final_obs"] = np.concatenate([i["final_obs"] for i in infos])
        return obs, reward, terminated, truncated, info

    def flock_metrics(self):
        per_env = [env.flock_metrics() for env in self.envs]
        return {name: float(np.mean([m[name] for m in per_env])) for name in per_env[0]}
