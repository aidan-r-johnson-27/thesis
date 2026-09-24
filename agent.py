import json
import os
import random


class QAgent:
    # Table-based Q-learner. `q` maps a discretised state key to a
    # (q[glide], q[flap]) pair; after every episode the collected
    # transitions are replayed backwards, the cheap way to carry the
    # terminal reward across the run with the discount factor.

    def __init__(self, alpha=0.7, gamma=0.95, epsilon=0.02, epsilon_decay=0.995, epsilon_min=0.0):
        # alpha is how hard a new observation moves q; gamma discounts future
        # reward; epsilon is the random-action probability and gets multiplied
        # by epsilon_decay after every episode until it hits epsilon_min. The
        # trajectory is this episode's transitions, bucketed at remember time.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q = {}
        self.trajectory = []

    @staticmethod
    def key(state):
        # Bucket the raw continuous state (dx, dy, vel) into a coarse
        # integer key so similar states share one table entry and learning
        # generalises between them. dx/dy are clamped to a fixed range so a
        # single wild state can't spray the table with one-off buckets.
        dx, dy, vel = state
        dx = int(max(0, min(dx, 320)) // 16)
        dy = int(max(-360, min(dy, 360)) // 10)
        vel = int(vel // 2)
        return (dx, dy, vel)

    def values(self, state):
        # Current action values for a state, defaulting to all-zeros.
        return self.q.get(self.key(state), (0.0, 0.0))

    def act(self, state, greedy=False):
        # Pick an action, normally the higher-valued one. With probability
        # epsilon (skipped under greedy) a random action replaces it to
        # keep exploring; random flaps happen only 30% of the time so a
        # roaming bird drifts rather than flutters.
        if not greedy and random.random() < self.epsilon:
            return 1 if random.random() < 0.3 else 0
        q = self.values(state)
        return 1 if q[1] > q[0] else 0

    def remember(self, state, action, reward, next_state, done):
        # Stash one transition of the current episode, already bucketed.
        self.trajectory.append((self.key(state), action, reward, self.key(next_state), done))

    def learn(self):
        # Replay the finished episode backwards with the standard update,
        # target = reward + gamma * max_q(next) (just reward when done),
        # nudging q toward it by alpha. The backwards order is the trick:
        # each step reads the max_q of a state that has already been
        # updated in this same pass, so the terminal reward ripples through
        # the whole episode in a single replay.
        for k, a, r, k2, done in reversed(self.trajectory):
            q = list(self.q.get(k, (0.0, 0.0)))
            target = r if done else r + self.gamma * max(self.q.get(k2, (0.0, 0.0)))
            q[a] += self.alpha * (target - q[a])
            self.q[k] = (q[0], q[1])
        self.trajectory.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def discard(self):
        # Drop the current episode's transitions (used when mode switches).
        self.trajectory.clear()

    def to_dict(self):
        # Flatten to JSON-safe data; state keys become "dx,dy,vel" strings.
        return {
            "epsilon": self.epsilon,
            "q": {",".join(map(str, k)): v for k, v in self.q.items()},
        }

    def load_dict(self, data):
        # Undo to_dict: turn the string keys back into integer tuples.
        self.epsilon = data.get("epsilon", self.epsilon)
        self.q = {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in data.get("q", {}).items()}


class Stats:
    # Rolling statistics of a training run, persisted next to the model so
    # a restart picks up the same averages and success rate.

    def __init__(self, goal=50, window=50):
        # `goal` is the score that counts as a success, `window` how many of
        # the most recent episodes feed every running figure. Raw scores are
        # kept in full; averages and rates are derived from them on demand.
        self.goal = goal
        self.window = window
        self.scores = []
        self.averages = []
        self.rates = []
        self.best = 0

    @property
    def episodes(self):
        return len(self.scores)

    def add(self, score):
        # Record one finished episode and refresh everything derived from
        # the score history (best, rolling average, success rate).
        self.scores.append(score)
        self.best = max(self.best, score)
        self.averages.append(self.average())
        self.rates.append(self.success_rate())

    def recent(self):
        # The last `window` scores; the slice every running stat reads.
        return self.scores[-self.window:]

    def average(self):
        # Mean of the recent window (0 before any episode has finished).
        r = self.recent()
        return sum(r) / len(r) if r else 0.0

    def success_rate(self):
        # Percent of the recent window that reached the goal.
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
    # Write model + stats through a temp file and rename, so a crash mid-
    # write can never leave a truncated model behind.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"agent": agent.to_dict(), "stats": stats.to_dict()}, f)
    os.replace(tmp, path)


def load(path, agent, stats):
    # Restore an agent and stats from `path`, returning False when the file
    # isn't there yet (a first ever run starts from scratch anyway).
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    agent.load_dict(data.get("agent", {}))
    stats.load_dict(data.get("stats", {}))
    return True