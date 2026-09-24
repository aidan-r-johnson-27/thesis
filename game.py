import random

# ---- World geometry (pixels) ----
# The playspace is a fixed 400x600 column and the bird hangs at BIRD_X while
# only its height moves. Pipes slide left at PIPE_SPEED, sit PIPE_SPACING
# apart, and each one asks the bird to thread a PIPE_GAP-tall vertical hole.
WIDTH = 400
HEIGHT = 600
GROUND = 540
BIRD_X = 90
BIRD_R = 13
PIPE_W = 66
PIPE_GAP = 150
PIPE_SPACING = 230
PIPE_SPEED = 3.0

# ---- Physics ----
# Every frame gravity adds GRAVITY to the velocity (capped at MAX_FALL); a
# flap instead snaps the velocity straight to FLAP, overriding the fall.
GRAVITY = 0.45
FLAP = -7.5
MAX_FALL = 10.0

# ---- Pipe generation ----
# Consecutive gaps never jump more than MAX_GAP_SHIFT so the pipe stays
# fair, and GAP_MARGIN keeps a gap clear of the ground and the ceiling.
GAP_MARGIN = 70
MAX_GAP_SHIFT = 190

# ---- Episode control ----
# Once an episode hits this score it is force-ended as a win.
MAX_SCORE = 5000


class Pipe:
    # One pipe pair, defined entirely by its gap centre: where the pipes sit
    # across the screen and where the gap falls vertically. `passed` flips
    # the first time the bird clears the pair's far edge.

    def __init__(self, x, gap_y):
        self.x = x
        self.gap_y = gap_y
        self.passed = False

    @property
    def top(self):
        return self.gap_y - PIPE_GAP / 2

    @property
    def bottom(self):
        return self.gap_y + PIPE_GAP / 2


class FlappyEnv:
    # The environment the agent learns on. The state handed out is
    # (horizontal distance to the next pipe, vertical offset from its gap
    # centre, current velocity); the reward is +1 per pipe passed, -1000 on
    # a crash, and 0 otherwise.

    def __init__(self, seed=None, max_score=MAX_SCORE):
        self.rng = random.Random(seed)
        self.max_score = max_score
        self.reset()

    def reset(self):
        # Fresh episode: bird at rest in mid-air, no score, three pipes on
        # the way, the first one starting just off the right edge.
        self.y = HEIGHT / 2 - 40
        self.vel = 0.0
        self.score = 0
        self.frames = 0
        self.done = False
        self.crashed = False
        self.last_action = 0
        self.pipes = []
        x = WIDTH + 60
        gap = self._next_gap(HEIGHT / 2)
        for _ in range(3):
            self.pipes.append(Pipe(x, gap))
            x += PIPE_SPACING
            gap = self._next_gap(gap)
        return self.state()

    def _next_gap(self, prev):
        # Random gap height close to the previous one, kept off the edges.
        lo = max(PIPE_GAP / 2 + GAP_MARGIN, prev - MAX_GAP_SHIFT)
        hi = min(GROUND - PIPE_GAP / 2 - GAP_MARGIN, prev + MAX_GAP_SHIFT)
        return self.rng.uniform(lo, hi)

    def next_pipe(self):
        # The first pipe whose far edge the bird hasn't already reached.
        for p in self.pipes:
            if p.x + PIPE_W > BIRD_X - BIRD_R:
                return p
        return self.pipes[-1]

    def state(self):
        # Feature vector fed to the agent, measured against the next pipe.
        # Bucketing happens later in `QAgent.key`; here the raw continuous
        # values leave the door.
        p = self.next_pipe()
        return (p.x + PIPE_W - BIRD_X, self.y - p.gap_y, self.vel)

    def _collides(self):
        # True when the bird touches the ground, ceiling, or any pipe.
        if self.y + BIRD_R >= GROUND or self.y - BIRD_R <= 0:
            return True
        for p in self.pipes:
            if BIRD_X + BIRD_R > p.x and BIRD_X - BIRD_R < p.x + PIPE_W:
                if self.y - BIRD_R < p.top or self.y + BIRD_R > p.bottom:
                    return True
        return False

    def step(self, action):
        # Advance one frame with 0 = glide / 1 = flap, returning
        # (state, reward, done).
        self.last_action = action
        if action == 1:
            self.vel = FLAP
        self.vel = min(self.vel + GRAVITY, MAX_FALL)
        self.y += self.vel
        self.frames += 1
        for p in self.pipes:
            p.x -= PIPE_SPEED
        # A pipe that has scrolled fully off the left respawns at the back
        # of the queue with a fresh, fair gap height.
        if self.pipes[0].x + PIPE_W < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING, self._next_gap(self.pipes[-1].gap_y)))
        reward = 1.0
        for p in self.pipes:
            if not p.passed and p.x + PIPE_W < BIRD_X - BIRD_R:
                p.passed = True
                self.score += 1
        if self._collides():
            self.done = True
            self.crashed = True
            reward = -1000.0
        elif self.max_score and self.score >= self.max_score:
            self.done = True
        return self.state(), reward, self.done