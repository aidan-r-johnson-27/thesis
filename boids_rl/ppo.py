import numpy as np
import torch


class RolloutBuffer:
    # a (T, B) grid: T time steps by B columns, one column per boid per env
    # with parameter sharing every cell is just another row of training data

    def __init__(self, T, B, obs_dim, act_dim):
        self.obs = np.zeros((T, B, obs_dim), np.float32)
        self.actions = np.zeros((T, B, act_dim), np.float32)
        self.log_probs = np.zeros((T, B), np.float32)
        self.values = np.zeros((T, B), np.float32)
        self.rewards = np.zeros((T, B), np.float32)
        # V of the real next state: after a time-limit cut that's the episode's final obs,
        # not the first obs of the next episode
        self.next_values = np.zeros((T, B), np.float32)
        self.terminated = np.zeros((T, B), bool)   # died: nothing to bootstrap from
        self.episode_end = np.zeros((T, B), bool)  # died or cut off: stop the advantage sum here


def compute_gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    # delta_t = r_t + gamma * V(s_t+1) * (1 - terminated_t) - V(s_t)
    # A_t     = delta_t + gamma * lam * (1 - episode_end_t) * A_t+1
    # a time-limit cut sets episode_end but not terminated, so it still bootstraps from V(s_t+1)
    # without letting the advantage leak into the next episode
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    next_advantage = np.zeros_like(rewards[0])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        next_advantage = delta + gamma * lam * (1 - episode_end[t]) * next_advantage
        advantages[t] = next_advantage
    returns = advantages + values
    return advantages, returns


def flatten(array):
    # (T, B, ...) -> (T * B, ...) as a tensor
    return torch.as_tensor(array.reshape(-1, *array.shape[2:]))


def ppo_update(model, optimizer, buf, advantages, returns, cfg, rng):
    obs = flatten(buf.obs)
    actions = flatten(buf.actions)
    old_log_probs = flatten(buf.log_probs)
    advantages = flatten(advantages)
    returns = flatten(returns)

    n_samples = obs.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for _ in range(cfg.n_epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, cfg.minibatch_size):
            batch = torch.as_tensor(order[start:start + cfg.minibatch_size])
            log_prob, entropy, value = model.evaluate(obs[batch], actions[batch])

            batch_adv = advantages[batch]
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
            log_ratio = log_prob - old_log_probs[batch]
            ratio = log_ratio.exp()

            # clipped surrogate: take the worse of the clipped and unclipped objective
            unclipped = ratio * batch_adv
            clipped = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns[batch]).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # low-variance KL estimate: mean of (r - 1) - log r
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["approx_kl"].append(approx_kl.item())
            stats["clip_frac"].append(clip_frac.item())

    return {name: float(np.mean(values)) for name, values in stats.items()}
