# Flappy Bird environment: physics, pipes, collisions and rewards.
#
# This module is the "world" the agent interacts with. It exposes a small
# gym-style interface (reset/step) plus the state/reward logic, and stays
# independent of pygame so it can run headless during training.

import random

# --- World geometry (pixels) -----------------------------------------------
WIDTH = 400          # width of the play field
HEIGHT = 600         # height of the window / sky
GROUND = 540         # y position of the top of the ground
BIRD_X = 90          # fixed horizontal position of the bird (only y moves)
BIRD_R = 13          # bird hitbox radius
PIPE_W = 66          # pipe width
PIPE_GAP = 150       # vertical gap the bird must fly through
PIPE_SPACING = 230   # horizontal distance between consecutive pipes
PIPE_SPEED = 3.0     # how many pixels pipes slide left per frame

# --- Physics ----------------------------------------------------------------
GRAVITY = 0.45   # downward acceleration applied to the bird each frame
FLAP = -7.5      # upward velocity impulse applied when the bird flaps
MAX_FALL = 10.0  # terminal downward velocity (speed cap)

# --- Pipe generation --------------------------------------------------------
GAP_MARGIN = 70      # min distance from the gap edge to ceiling/floor
MAX_GAP_SHIFT = 190  # max vertical jump between two consecutive gaps

# --- Episode control --------------------------------------------------------
MAX_SCORE = 5000  # default score at which an episode is force-ended


class Pipe:
    # A single pipe pair, defined by its gap centre height.

    def __init__(self, x, gap_y):
        self.x = x              # x of the pipe's left edge
        self.gap_y = gap_y      # y of the centre of the gap between pipes
        self.passed = False     # True once the bird has flown past it

    @property
    def top(self):
        return self.gap_y - PIPE_GAP / 2

    @property
    def bottom(self):
        return self.gap_y + PIPE_GAP / 2


class FlappyEnv:
    # The Flappy Bird simulation the agent learns on.
    #
    # State: (horizontal distance to the next pipe, vertical offset from its
    # gap centre, current velocity). Reward: +1 per passed pipe, -1000 on a
    # crash, 0 otherwise.

    def __init__(self, seed=None, max_score=MAX_SCORE):
        self.rng = random.Random(seed)   # deterministic via --seed for debugging
        self.max_score = max_score
        self.reset()

    def reset(self):
        # Start a fresh episode: bird centred, no score, three pipes ahead.
        self.y = HEIGHT / 2 - 40
        self.vel = 0.0
        self.score = 0
        self.frames = 0
        self.done = False
        self.crashed = False
        self.last_action = 0
        self.pipes = []
        x = WIDTH + 60   # first pipe starts just off the right edge
        gap = self._next_gap(HEIGHT / 2)
        for _ in range(3):
            self.pipes.append(Pipe(x, gap))
            x += PIPE_SPACING
            gap = self._next_gap(gap)
        return self.state()

    def _next_gap(self, prev):
        # Random gap height close to the previous one, kept off the screen edges.
        lo = max(PIPE_GAP / 2 + GAP_MARGIN, prev - MAX_GAP_SHIFT)
        hi = min(GROUND - PIPE_GAP / 2 - GAP_MARGIN, prev + MAX_GAP_SHIFT)
        return self.rng.uniform(lo, hi)

    def next_pipe(self):
        # The first pipe the bird has not yet fully passed.
        for p in self.pipes:
            if p.x + PIPE_W > BIRD_X - BIRD_R:
                return p
        return self.pipes[-1]

    def state(self):
        # Feature vector fed to the agent.
        #
        # Discretisation happens in `QAgent.key`; here we export the raw
        # continuous values relative to the next pipe.
        p = self.next_pipe()
        return (p.x + PIPE_W - BIRD_X, self.y - p.gap_y, self.vel)

    def _collides(self):
        # True if the bird touches the ground, ceiling or any pipe.
        if self.y + BIRD_R >= GROUND or self.y - BIRD_R <= 0:
            return True
        for p in self.pipes:
            if BIRD_X + BIRD_R > p.x and BIRD_X - BIRD_R < p.x + PIPE_W:
                if self.y - BIRD_R < p.top or self.y + BIRD_R > p.bottom:
                    return True
        return False

    def step(self, action):
        # Advance one frame with the given action (0 = glide, 1 = flap).
        # Returns (state, reward, done).
        self.last_action = action
        if action == 1:
            self.vel = FLAP          # flap overrides current velocity
        self.vel = min(self.vel + GRAVITY, MAX_FALL)
        self.y += self.vel
        self.frames += 1
        for p in self.pipes:
            p.x -= PIPE_SPEED
        # Recycle a fully-scrolled-off pipe into a fresh one at the back.
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
            self.done = True   # reached the target score: stop the episode clean
        return self.state(), reward, self.done