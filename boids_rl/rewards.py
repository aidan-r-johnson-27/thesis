import numpy as np


def unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)


# every term is in [-1, 0] per step, 0 is best

# predators (off)
# def caught(data, cfg):
#     return -data.caught.astype(float)


def collision(data, cfg):
    return -data.collided.astype(float)


def separation(data, cfg):
    # visible neighbours closer than separation_dist cost (1 - d / separation_dist)^2 each, capped at 1
    crowding = np.clip(1 - data.dist / cfg.separation_dist, 0.0, None) ** 2
    crowding = np.where(data.mask, crowding, 0.0)
    return -np.minimum(crowding.sum(axis=1), 1.0)


def alignment(data, cfg):
    # average of (cos - 1) / 2 between my heading and each visible neighbour's, -1 if I see nobody
    n_visible = data.mask.sum(axis=1)
    neighbour_vel = data.rel_vel + data.vel[:, None, :]
    cosines = (unit(neighbour_vel) * unit(data.vel)[:, None, :]).sum(axis=-1)
    scores = np.where(data.mask, (cosines - 1) / 2, 0.0)
    return np.where(n_visible > 0, scores.sum(axis=1) / np.maximum(n_visible, 1), -1.0)


def cohesion(data, cfg):
    # minus the distance to the middle of my visible neighbours, scaled into [-1, 0], -1 if I see nobody
    n_visible = data.mask.sum(axis=1)
    centre = data.rel_pos.sum(axis=1) / np.maximum(n_visible, 1)[:, None]
    reward = -np.linalg.norm(centre, axis=1) / data.neighbor_radius
    return np.where(n_visible > 0, reward, -1.0)


# to add a new term: write it above, add it here, and add a w_<name> weight to RewardConfig
REWARD_TERMS = {
    # "caught": caught,
    "collision": collision,
    "separation": separation,
    "alignment": alignment,
    "cohesion": cohesion,
}


def compute_rewards(data, reward_cfg):
    terms = {name: term(data, reward_cfg) for name, term in REWARD_TERMS.items()}
    total = sum(getattr(reward_cfg, f"w_{name}") * value for name, value in terms.items())
    return total, terms
