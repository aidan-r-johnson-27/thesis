import numpy as np


def unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)


def caught(data):
    return -data.caught.astype(float)


def collision(data):
    return -data.collided.astype(float)


# cohesion and alignment are the old hand-made Reynolds rewards, only switched on by the
# "reynolds" ablation (their weights are 0 by default)

def cohesion(data):
    # minus the distance to the middle of my visible neighbours, scaled into [-1, 0]
    # seeing nobody counts as the worst case, -1
    n_visible = data.mask.sum(axis=1)
    centre = data.rel_pos.sum(axis=1) / np.maximum(n_visible, 1)[:, None]
    reward = -np.linalg.norm(centre, axis=1) / data.neighbor_radius
    return np.where(n_visible > 0, reward, -1.0)


def alignment(data):
    # average cosine between my heading and each visible neighbour's heading, 0 if I see nobody
    n_visible = data.mask.sum(axis=1)
    neighbour_vel = data.rel_vel + data.vel[:, None, :]
    cosines = (unit(neighbour_vel) * unit(data.vel)[:, None, :]).sum(axis=-1)
    cosines = np.where(data.mask, cosines, 0.0)
    return cosines.sum(axis=1) / np.maximum(n_visible, 1)


# to add a new term: write it above, add it here, and add a w_<name> weight to RewardConfig
REWARD_TERMS = {
    "caught": caught,
    "collision": collision,
    "cohesion": cohesion,
    "alignment": alignment,
}


def compute_rewards(data, reward_cfg):
    terms = {name: term(data) for name, term in REWARD_TERMS.items()}
    total = sum(getattr(reward_cfg, f"w_{name}") * value for name, value in terms.items())
    return total, terms
