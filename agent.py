# Q-learning agent, episode stats and model persistence.
#
# The agent learns by TD (temporal-difference) update over an on-policy batch
# (a full episode of transitions), uses epsilon-greedy exploration and buckets
# the continuous game state into discrete cells so Q-values can be stored in a
# plain dict.

import json
import os
import random


class QAgent:
    # Table-based Q-learner.
    #
    # `q` maps a discretised state key -> (q[glide], q[flap]). After every
    # episode the collected transitions are replayed backwards (easy way to
    # propagate the terminal reward with the discount factor).

    def __init__(self, alpha=0.7, gamma=0.95, epsilon=0.02, epsilon_decay=0.995, epsilon_min=0.0):
        self.alpha = alpha          # learning rate: how much new info moves q
        self.gamma = gamma          # discount factor for future rewards
        self.epsilon = epsilon      # exploration probability (random actions)
        self.epsilon_decay = epsilon_decay  # epsilon multiplier per episode
        self.epsilon_min = epsilon_min      # floor epsilon never goes below
        self.q = {}                 # state key -> (q[0], q[1])
        self.trajectory = []        # transitions collected during this episode

    @staticmethod
    def key(state):
        # Discretise a raw state (dx, dy, vel) into a coarse integer bucket.
        #
        # The continuous values are quantised so similar states share one
        # table entry and learning generalises between them.
        dx, dy, vel = state
        dx = int(max(0, min(dx, 320)) // 16)
        dy = int(max(-360, min(dy, 360)) // 10)
        vel = int(vel // 2)
        return (dx, dy, vel)

    def values(self, state):
        # Current action values for a state, defaulting to all-zeros.
        return self.q.get(self.key(state), (0.0, 0.0))

    def act(self, state, greedy=False):
        # Pick an action: flap when its estimated value is higher.
        #
        # With probability `epsilon` (unless greedy) a random action is taken
        # instead to keep exploring. Random flaps happen 30% of the time, so
        # the bird is biased toward gliding when roaming.
        if not greedy and random.random() < self.epsilon:
            return 1 if random.random() < 0.3 else 0
        q = self.values(state)
        return 1 if q[1] > q[0] else 0

    def remember(self, state, action, reward, next_state, done):
        # Record one transition for the current episode (in bucket keys).
        self.trajectory.append((self.key(state), action, reward, self.key(next_state), done))

    def learn(self):
        # Run the Q-learning update over the episode just finished.
        #
        # target = reward + gamma * max_q(next_state) (or just the reward if
        # the episode ended), then q[state][action] += alpha * (target - q).
        for k, a, r, k2, done in reversed(self.trajectory):
            q = list(self.q.get(k, (0.0, 0.0)))
            target = r if done else r + self.gamma * max(self.q.get(k2, (0.0, 0.0)))
            q[a] += self.alpha * (target - q[a])
            self.q[k] = (q[0], q[1])
        self.trajectory.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def discard(self):
        # Throw away the current trajectory (used when switching modes).
        self.trajectory.clear()

    def to_dict(self):
        # Serialize to plain data; state keys become "dx,dy,vel" strings.
        return {
            "epsilon": self.epsilon,
            "q": {",".join(map(str, k)): v for k, v in self.q.items()},
        }

    def load_dict(self, data):
        # Restore from data produced by `to_dict`.
        self.epsilon = data.get("epsilon", self.epsilon)
        self.q = {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in data.get("q", {}).items()}


class Stats:
    # Rolling statistics of a training run, persisted alongside the model.

    def __init__(self, goal=50, window=50):
        self.goal = goal          # score that counts as a "success"
        self.window = window      # episodes used for averages / success rate
        self.scores = []          # score of every finished episode
        self.averages = []        # running window average per episode
        self.rates = []           # running success rate per episode
        self.best = 0

    @property
    def episodes(self):
        return len(self.scores)

    def add(self, score):
        # Record an episode's score and refresh derived statistics.
        self.scores.append(score)
        self.best = max(self.best, score)
        self.averages.append(self.average())
        self.rates.append(self.success_rate())

    def recent(self):
        # Scores of the last `window` episodes.
        return self.scores[-self.window:]

    def average(self):
        # Mean score over the last `window` episodes.
        r = self.recent()
        return sum(r) / len(r) if r else 0.0

    def success_rate(self):
        # Percentage of recent episodes reaching the goal score.
        r = self.recent()
        return 100.0 * sum(s >= self.goal for s in r) / len(r) if r else 0.0

    def to_dict(self):
        return {"scores": self.scores}

    def load_dict(self, data):
        # Rebuild from stored scores, recomputing the derived series.
        scores = list(data.get("scores", []))
        self.scores, self.averages, self.rates, self.best = [], [], [], 0
        for s in scores:
            self.add(s)


def save(path, agent, stats):
    # Atomically write model + stats to JSON (write tmp, then rename).
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"agent": agent.to_dict(), "stats": stats.to_dict()}, f)
    os.replace(tmp, path)


def load(path, agent, stats):
    # Restore an agent and stats from `path`; returns False if absent.
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    agent.load_dict(data.get("agent", {}))
    stats.load_dict(data.get("stats", {}))
    return True