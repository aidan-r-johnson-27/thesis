import math
import sys

import numpy as np
import pygame

# ---- Simulation config ----
NUM_BOIDS = 3000
PERCEPTION_RADIUS = 50.0
SEPARATION_RADIUS = 25.0
MAX_SPEED = 4.0
MAX_FORCE = 0.10

# Weight applied to each steering rule.
WEIGHT_SEPARATION = 1.4
WEIGHT_ALIGNMENT = 1.0
WEIGHT_COHESION = 1.0

# ---- Boid sprite geometry ----
BOID_SIZE = 7
WING_ANGLE = 2.35
WING_RATIO = 0.55

# Above this population, fall back to the cheaper dashed renderer.
TRIANGLE_RENDER_LIMIT = 4500

# ---- Window setup ----
pygame.init()
_display_info = pygame.display.Info()
WIDTH, HEIGHT = _display_info.current_w, _display_info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Boids Simulation")
clock = pygame.time.Clock()

# ---- UI colors and panel layout ----
C_BG = (2, 5, 2)
C_PANEL_BG = (5, 10, 5, 215)
C_PANEL_BORDER = (0, 100, 30)
C_TEXT = (0, 220, 60)
C_TEXT_DIM = (0, 100, 28)
C_TRACK_BG = (15, 25, 15)
C_TRACK_FILL = (0, 140, 40)
C_CONTROL = (0, 220, 60)
C_CONTROL_BORDER = (0, 80, 22)
C_ACTIVE_BG = (20, 40, 20)
C_IDLE_BG = (10, 18, 10)
C_HINT = (0, 80, 22)

PANEL_MARGIN = 18
PANEL_WIDTH = 340
PANEL_PAD = 14
PANEL_TOP = PANEL_MARGIN

TITLE_H = 24
LABEL_H = 18
TRACK_H = 10
HANDLE_R = 8
ROW_GAP = 10
ROW_H = LABEL_H + TRACK_H + ROW_GAP
HINT_H = 20

BUTTON_H = 30
BUTTON_GAP = 12
TEMPLATE_GAP = 6

NUM_SLIDERS = 8
PANEL_H = PANEL_PAD + TITLE_H + NUM_SLIDERS * ROW_H + BUTTON_GAP + BUTTON_H + BUTTON_GAP + BUTTON_H + HINT_H + PANEL_PAD

PANEL_X = WIDTH - PANEL_WIDTH - PANEL_MARGIN
SLIDER_W = PANEL_WIDTH - 2 * PANEL_PAD

PANEL_STRIP_H = PANEL_PAD + TITLE_H + PANEL_PAD

TOGGLE_W = 64
TOGGLE_H = 20
TOGGLE_X = PANEL_X + PANEL_WIDTH - PANEL_PAD - TOGGLE_W
TOGGLE_Y = PANEL_TOP + PANEL_PAD

# ---- 5x7 bitmap font ----
_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}

# ---- Cached text rendering ----
_glyph_cache = {}
_text_cache = {}
_TEXT_CACHE_MAX = 256


def _glyph_surface(ch, color, scale):
    key = (ch, color, scale)
    glyph = _glyph_cache.get(key)
    if glyph is None:
        glyph = pygame.Surface((5 * scale, 7 * scale), pygame.SRCALPHA)
        rows = _FONT.get(ch, _FONT[" "])
        for j in range(7):
            row = rows[j]
            for i in range(5):
                if row[i] == "1":
                    glyph.fill(color, (i * scale, j * scale, scale, scale))
        _glyph_cache[key] = glyph
    return glyph


def _text_surface(text, color, scale):
    key = (text, color, scale)
    cached = _text_cache.get(key)
    if cached is not None:
        return cached
    gw = 5 * scale + scale - 1
    gh = 7 * scale
    surf = pygame.Surface((len(text) * gw + 1, gh), pygame.SRCALPHA)
    surf.blits(
        [(_glyph_surface(ch, color, scale), (ci * gw, 0)) for ci, ch in enumerate(text)],
        doreturn=False,
    )
    if len(_text_cache) >= _TEXT_CACHE_MAX:
        _text_cache.clear()
    _text_cache[key] = surf
    return surf


def fmt_int(v):
    return f"{v:.0f}"


def fmt_1f(v):
    return f"{v:.1f}"


def fmt_2f(v):
    return f"{v:.2f}"


# ---- UI widgets ----
class Slider:
    def __init__(self, label, min_v, max_v, value, x, y, w, fmt):
        self.label = label
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.value = float(value)
        self.x = int(x)
        self.w = int(w)
        self.fmt = fmt
        self.track_y = int(y)
        self.dragging = False
        self._clamp()

    def _clamp(self):
        self.value = min(max(self.value, self.min_v), self.max_v)

    def frac(self):
        return (self.value - self.min_v) / (self.max_v - self.min_v)

    def handle_x(self):
        span = self.w - 2 * HANDLE_R
        return self.x + HANDLE_R + int(self.frac() * span)

    def value_from_x(self, mx):
        span = max(1.0, float(self.w - 2 * HANDLE_R))
        t = (mx - self.x - HANDLE_R) / span
        self.value = self.min_v + min(max(t, 0.0), 1.0) * (self.max_v - self.min_v)
        self._clamp()

    def hit(self, pos):
        rect = pygame.Rect(self.x, self.track_y - LABEL_H, self.w, LABEL_H + TRACK_H)
        return rect.collidepoint(pos)

    def draw(self, surface):
        hx = self.handle_x()
        cy = self.track_y + TRACK_H // 2
        surface.blit(_text_surface(
            f"{self.label}: {self.fmt(self.value)}", C_TEXT, 2
        ), (self.x, self.track_y - LABEL_H))
        pygame.draw.rect(surface, C_TRACK_BG, (self.x, self.track_y, self.w, TRACK_H))
        if hx > self.x:
            pygame.draw.rect(surface, C_TRACK_FILL, (self.x, self.track_y, hx - self.x, TRACK_H))
        pygame.draw.circle(surface, C_CONTROL, (hx, cy), HANDLE_R)
        pygame.draw.circle(surface, C_CONTROL_BORDER, (hx, cy), HANDLE_R, 1)


class Button:
    def __init__(self, label, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.hover = False

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, surface):
        fill = C_ACTIVE_BG if self.hover else C_IDLE_BG
        border = C_CONTROL if self.hover else C_CONTROL_BORDER
        pygame.draw.rect(surface, fill, self.rect)
        pygame.draw.rect(surface, border, self.rect, 1)
        label_surf = _text_surface(self.label, C_CONTROL, 2)
        surface.blit(label_surf, (
            self.rect.centerx - label_surf.get_width() // 2,
            self.rect.centery - label_surf.get_height() // 2,
        ))


# Instantiate one slider per setting, seeded from the config defaults.
_slider_y0 = PANEL_TOP + PANEL_PAD + TITLE_H
sliders = {
    "population":      Slider("POPULATION",        0.0, 10000.0, NUM_BOIDS,         PANEL_X + PANEL_PAD, _slider_y0 + 0 * ROW_H, SLIDER_W, fmt_int),
    "max_speed":       Slider("MAX SPEED",         0.5,    12.0, MAX_SPEED,         PANEL_X + PANEL_PAD, _slider_y0 + 1 * ROW_H, SLIDER_W, fmt_1f),
    "max_force":       Slider("MAX FORCE",        0.01,     1.0, MAX_FORCE,         PANEL_X + PANEL_PAD, _slider_y0 + 2 * ROW_H, SLIDER_W, fmt_2f),
    "perception":      Slider("PERCEPTION RADIUS", 10.0, 200.0, PERCEPTION_RADIUS, PANEL_X + PANEL_PAD, _slider_y0 + 3 * ROW_H, SLIDER_W, fmt_int),
    "separation":      Slider("SEPARATION RADIUS",  5.0, 100.0, SEPARATION_RADIUS, PANEL_X + PANEL_PAD, _slider_y0 + 4 * ROW_H, SLIDER_W, fmt_int),
    "sep_weight":      Slider("SEPARATION WEIGHT",  0.0,   4.0, WEIGHT_SEPARATION, PANEL_X + PANEL_PAD, _slider_y0 + 5 * ROW_H, SLIDER_W, fmt_2f),
    "ali_weight":      Slider("ALIGNMENT WEIGHT",   0.0,   4.0, WEIGHT_ALIGNMENT,  PANEL_X + PANEL_PAD, _slider_y0 + 6 * ROW_H, SLIDER_W, fmt_2f),
    "coh_weight":      Slider("COHESION WEIGHT",    0.0,   4.0, WEIGHT_COHESION,   PANEL_X + PANEL_PAD, _slider_y0 + 7 * ROW_H, SLIDER_W, fmt_2f),
}

TEMPLATES = [
    {"name": "T1", "perception": 50.0, "separation": 25.0, "max_speed": 4.0, "max_force": 0.10, "sep_weight": 1.4, "ali_weight": 1.0, "coh_weight": 0.3},
    {"name": "T2", "perception": 120.0, "separation": 15.0, "max_speed": 8.0, "max_force": 0.05, "sep_weight": 0.6, "ali_weight": 2.0, "coh_weight": 1.5},
    {"name": "T3", "perception": 30.0, "separation": 40.0, "max_speed": 2.0, "max_force": 0.30, "sep_weight": 2.5, "ali_weight": 0.3, "coh_weight": 0.1},
]

_template_btn_y = _slider_y0 + NUM_SLIDERS * ROW_H + BUTTON_GAP
TEMPLATE_BUTTON_W = (SLIDER_W - (len(TEMPLATES) - 1) * TEMPLATE_GAP) // len(TEMPLATES)
template_buttons = [
    Button(t["name"], PANEL_X + PANEL_PAD + i * (TEMPLATE_BUTTON_W + TEMPLATE_GAP),
           _template_btn_y, TEMPLATE_BUTTON_W, BUTTON_H)
    for i, t in enumerate(TEMPLATES)
]
_reset_btn_y = _template_btn_y + BUTTON_H + BUTTON_GAP
reset_button = Button("RESET FLOCK", PANEL_X + PANEL_PAD, _reset_btn_y, SLIDER_W, BUTTON_H)

panel_minimized = False
toggle_button = Button("HIDE", TOGGLE_X, TOGGLE_Y, TOGGLE_W, TOGGLE_H)


def _sync_population():
    # Keep the flock size in step with the population slider.
    target = int(sliders["population"].value)
    current = positions.shape[0]
    if current < target:
        add_boids(target - current)
    elif current > target:
        remove_boids(current - target)


def apply_sliders():
    # Push slider values into the live sim constants.
    global MAX_SPEED, MAX_FORCE, PERCEPTION_RADIUS, SEPARATION_RADIUS
    global WEIGHT_SEPARATION, WEIGHT_ALIGNMENT, WEIGHT_COHESION
    MAX_SPEED = sliders["max_speed"].value
    MAX_FORCE = sliders["max_force"].value
    PERCEPTION_RADIUS = sliders["perception"].value
    SEPARATION_RADIUS = sliders["separation"].value
    WEIGHT_SEPARATION = sliders["sep_weight"].value
    WEIGHT_ALIGNMENT = sliders["ali_weight"].value
    WEIGHT_COHESION = sliders["coh_weight"].value
    recompute_derived()
    _sync_population()


def apply_template(template):
    # Load a preset template into the sim and sync the sliders.
    global MAX_SPEED, MAX_FORCE, PERCEPTION_RADIUS, SEPARATION_RADIUS
    global WEIGHT_SEPARATION, WEIGHT_ALIGNMENT, WEIGHT_COHESION
    PERCEPTION_RADIUS = template["perception"]
    SEPARATION_RADIUS = template["separation"]
    MAX_SPEED = template["max_speed"]
    MAX_FORCE = template["max_force"]
    WEIGHT_SEPARATION = template["sep_weight"]
    WEIGHT_ALIGNMENT = template["ali_weight"]
    WEIGHT_COHESION = template["coh_weight"]
    for key, s in sliders.items():
        if key in template:
            s.value = template[key]
    recompute_derived()
    _sync_population()


def _panel_background(height):
    bg = pygame.Surface((PANEL_WIDTH, height), pygame.SRCALPHA)
    bg.fill(C_PANEL_BG)
    pygame.draw.rect(bg, C_PANEL_BORDER, bg.get_rect(), 1)
    return bg.convert_alpha()


_PANEL_BG = _panel_background(PANEL_H)
_PANEL_STRIP_BG = _panel_background(PANEL_STRIP_H)


def draw_panel(surface):
    if panel_minimized:
        surface.blit(_PANEL_STRIP_BG, (PANEL_X, PANEL_TOP))
        surface.blit(_text_surface("PARAMETERS", C_TEXT, 2),
                     (PANEL_X + PANEL_PAD, PANEL_TOP + PANEL_PAD))
        toggle_button.label = "SHOW"
        toggle_button.hover = toggle_button.hit(pygame.mouse.get_pos())
        toggle_button.draw(surface)
        return

    surface.blit(_PANEL_BG, (PANEL_X, PANEL_TOP))
    surface.blit(_text_surface("PARAMETERS", C_TEXT, 2),
                 (PANEL_X + PANEL_PAD, PANEL_TOP + PANEL_PAD))
    for s in sliders.values():
        s.draw(surface)
    for b in template_buttons:
        b.hover = b.hit(pygame.mouse.get_pos())
        b.draw(surface)
    reset_button.hover = reset_button.hit(pygame.mouse.get_pos())
    reset_button.draw(surface)
    toggle_button.label = "HIDE"
    toggle_button.hover = toggle_button.hit(pygame.mouse.get_pos())
    toggle_button.draw(surface)
    surface.blit(_text_surface("LMB DRAG  SPACE RESET  UP/DN POP  ESC QUIT",
                               C_HINT, 1),
                 (PANEL_X + PANEL_PAD, PANEL_TOP + PANEL_H - PANEL_PAD - HINT_H))


HALF_WIDTH = WIDTH / 2.0
HALF_HEIGHT = HEIGHT / 2.0

_FORWARD_OFFSETS = ((1, 0), (-1, 1), (0, 1), (1, 1))

_DTYPE = np.float32


def recompute_derived():
    global GRID_COLS, GRID_ROWS, NUM_CELLS, CELL_W, CELL_H, PERC_SQ, SEP_SQ
    GRID_COLS = max(3, int(WIDTH // PERCEPTION_RADIUS))
    GRID_ROWS = max(3, int(HEIGHT // PERCEPTION_RADIUS))
    NUM_CELLS = GRID_COLS * GRID_ROWS
    CELL_W = WIDTH / GRID_COLS
    CELL_H = HEIGHT / GRID_ROWS
    PERC_SQ = PERCEPTION_RADIUS * PERCEPTION_RADIUS
    SEP_SQ = SEPARATION_RADIUS * SEPARATION_RADIUS


recompute_derived()

positions = np.zeros((0, 2), dtype=_DTYPE)
velocities = np.zeros((0, 2), dtype=_DTYPE)


# ---- Flock creation ----
def init_boids(count):
    # Scatter the flock randomly across the screen with random headings.
    global positions, velocities
    positions = np.random.uniform(0.0, 1.0, (count, 2)).astype(_DTYPE)
    positions[:, 0] *= WIDTH
    positions[:, 1] *= HEIGHT
    angles = np.random.uniform(0.0, 2.0 * math.pi, count)
    velocities = np.column_stack((
        np.cos(angles) * MAX_SPEED,
        np.sin(angles) * MAX_SPEED,
    )).astype(_DTYPE)


def add_boids(count):
    # Append new animals; the existing flock stays right where it is.
    global positions, velocities
    n = positions.shape[0]
    new_pos = np.random.uniform(0.0, 1.0, (count, 2)).astype(_DTYPE)
    new_pos[:, 0] *= WIDTH
    new_pos[:, 1] *= HEIGHT
    angles = np.random.uniform(0.0, 2.0 * math.pi, count)
    new_vel = np.column_stack((
        np.cos(angles) * MAX_SPEED,
        np.sin(angles) * MAX_SPEED,
    )).astype(_DTYPE)
    if n == 0:
        positions, velocities = new_pos, new_vel
    else:
        positions = np.concatenate((positions, new_pos), axis=0)
        velocities = np.concatenate((velocities, new_vel), axis=0)


def remove_boids(count):
    global positions, velocities
    n = min(count, positions.shape[0])
    if n > 0:
        positions = positions[:-n]
        velocities = velocities[:-n]


def integrate():
    # Steering added up to MAX_FORCE per frame, so re-normalize speed back
    # down to MAX_SPEED, step the positions, then wrap around the torus so
    # a flock leaving one edge re-enters from the other.
    speed = np.hypot(velocities[:, 0], velocities[:, 1])
    speed[speed == 0] = 1.0
    scale = np.minimum(1.0, MAX_SPEED / speed)
    velocities[:, 0] *= scale
    velocities[:, 1] *= scale

    positions[:, 0] += velocities[:, 0]
    positions[:, 1] += velocities[:, 1]
    positions[:, 0] %= WIDTH
    positions[:, 1] %= HEIGHT


# ---- Spatial grid neighbor search ----
def update_all():
    n = positions.shape[0]
    if n == 0:
        return

    # Bucket each boid into a grid cell and sort by cell id so every animal
    # in a cell lands in one contiguous run. Position and velocity are
    # reordered identically, so a sorted index stays aligned across arrays.
    cols = (positions[:, 0] / CELL_W).astype(np.intp)
    rows = (positions[:, 1] / CELL_H).astype(np.intp)
    np.minimum(cols, GRID_COLS - 1, out=cols)
    np.minimum(rows, GRID_ROWS - 1, out=rows)
    cell_ids = rows * GRID_COLS + cols
    order = np.argsort(cell_ids)
    cell_ids = cell_ids[order]
    cols = cols[order]
    rows = rows[order]
    pos = positions[order]
    vel = velocities[order]

    # Prefix sum of the cell histogram; cell c occupies indices
    # [starts[c], starts[c+1]) with one spare slot to keep the +1 safe.
    starts = np.zeros(NUM_CELLS + 1, dtype=np.intp)
    np.cumsum(np.bincount(cell_ids, minlength=NUM_CELLS), out=starts[1:])

    # Encode every neighbor range as (block_start, block_len). Column 0 is
    # the boid's own cell minus its own sorted row (boid_range + 1 skips
    # itself); the other four come from _FORWARD_OFFSETS. Each pair is seen
    # from exactly one side, so the full 8-neighborhood never double-checks.
    boid_range = np.arange(n, dtype=np.intp)
    block_start = np.empty((n, 5), dtype=np.intp)
    block_len = np.empty((n, 5), dtype=np.intp)
    block_start[:, 0] = boid_range + 1
    block_len[:, 0] = starts[cell_ids + 1] - block_start[:, 0]
    for k, (dc, dr) in enumerate(_FORWARD_OFFSETS, 1):
        nb_cells = ((rows + dr) % GRID_ROWS) * GRID_COLS + (cols + dc) % GRID_COLS
        block_start[:, k] = starts[nb_cells]
        block_len[:, k] = starts[nb_cells + 1] - block_start[:, k]

    # Most compact trick in the file: unfold all (start, len) neighbor
    # blocks into one flat candidate list so the distance passes run
    # entirely in numpy. block_ends - lens is the flat offset where each
    # block begins, so I repeat start - begin_offset by its len, then add
    # arange(total); each block then walks start, start+1, ... start+len-1.
    lens = block_len.ravel()
    block_ends = np.cumsum(lens)
    total = int(block_ends[-1])
    if total == 0:
        integrate()
        return
    cand = np.repeat(block_start.ravel() - (block_ends - lens), lens)
    cand += np.arange(total, dtype=np.intp)
    row_ends = block_ends[4::5]
    query = np.repeat(boid_range, np.diff(row_ends, prepend=0))

    px = pos[:, 0]
    py = pos[:, 1]
    dx = px[cand]
    dx -= px[query]
    dy = py[cand]
    dy -= py[query]
    # The world wraps like a torus, so boid at x=1 and one at x=W-1 are
    # really 2px apart. Any diff beyond the half-width gets pulled back by a
    # full W or H, collapsing those huge floating deltas into the true
    # shortest distance straight across the seam.
    np.subtract(dx, WIDTH, out=dx, where=dx > HALF_WIDTH)
    np.add(dx, WIDTH, out=dx, where=dx < -HALF_WIDTH)
    np.subtract(dy, HEIGHT, out=dy, where=dy > HALF_HEIGHT)
    np.add(dy, HEIGHT, out=dy, where=dy < -HALF_HEIGHT)

    d2 = dx * dx
    d2 += dy * dy
    # Keep only neighbors inside the perception radius.
    keep = np.flatnonzero(d2 < PERC_SQ)
    if keep.size == 0:
        integrate()
        return
    q_counts = np.diff(np.searchsorted(keep, row_ends), prepend=0)
    cand = cand[keep]
    dx = dx[keep]
    dy = dy[keep]
    d2 = d2[keep]
    del keep
    query = np.repeat(boid_range, q_counts)

    has_q = np.flatnonzero(q_counts)
    seg_begins = (np.cumsum(q_counts) - q_counts)[has_q]

    # Each pair is stored in exactly one direction, so a symmetric sum over a
    # boid's neighbors needs both halves: edges pointing INTO it (neighbors
    # "behind", gathered by the bincount on cand) plus its own edges
    # (neighbors "ahead", summed by reduceat over its query segment). One
    # pass covers every neighbor exactly once, whichever side saw them.
    def sum_both(to_query, to_cand):
        acc = np.bincount(cand, weights=to_cand, minlength=n)
        acc[has_q] += np.add.reduceat(to_query, seg_begins)
        return acc

    neighbor_counts = q_counts + np.bincount(cand, minlength=n)

    vx = vel[:, 0].astype(np.float64)
    vy = vel[:, 1].astype(np.float64)
    ali_x_acc = sum_both(vel[cand, 0], vel[query, 0])
    ali_y_acc = sum_both(vel[cand, 1], vel[query, 1])

    coh_x_acc = sum_both(dx, -dx)
    coh_y_acc = sum_both(dy, -dy)

    # Separation breaks the symmetry: both ends of a close pair recoil, so
    # each pair is counted from cand and from query. That doubles the
    # impulse and sep_counts together, and dividing by the doubled count
    # re-normalizes it while the 1/distance factor makes nearer pairs push
    # harder, capped at MAX_SPEED.
    close = np.flatnonzero((d2 < SEP_SQ) & (d2 > 0.0))
    sep_q = query[close]
    sep_c = cand[close]
    inv_dist = 1.0 / np.sqrt(d2[close])
    ux = dx[close] * inv_dist
    uy = dy[close] * inv_dist
    sep_x_acc = np.bincount(sep_c, weights=ux, minlength=n) - np.bincount(sep_q, weights=ux, minlength=n)
    sep_y_acc = np.bincount(sep_c, weights=uy, minlength=n) - np.bincount(sep_q, weights=uy, minlength=n)
    sep_counts = np.bincount(sep_q, minlength=n) + np.bincount(sep_c, minlength=n)

    # Reynolds steering: build a "desired" velocity from the rule, then tug
    # the current velocity toward it by at most MAX_FORCE so turns are
    # gradual instead of snapping onto a new heading.
    def clamp_force(fx, fy):
        scale = MAX_FORCE / np.maximum(np.hypot(fx, fy), MAX_FORCE)
        return fx * scale, fy * scale

    has_sep = sep_counts > 0
    inv_sep = MAX_SPEED / np.maximum(sep_counts, 1)
    sep_x = np.where(has_sep, sep_x_acc * inv_sep - vx, 0.0)
    sep_y = np.where(has_sep, sep_y_acc * inv_sep - vy, 0.0)
    sep_x, sep_y = clamp_force(sep_x, sep_y)

    inv_count = 1.0 / np.maximum(neighbor_counts, 1)
    ali_x = ali_x_acc * (inv_count * MAX_SPEED) - vx
    ali_y = ali_y_acc * (inv_count * MAX_SPEED) - vy
    ali_x, ali_y = clamp_force(ali_x, ali_y)

    # Cohesion: steer toward the average position of local neighbors. The
    # average offset scaled to MAX_SPEED is the target velocity.
    coh_x = coh_x_acc * inv_count
    coh_y = coh_y_acc * inv_count
    coh_mag = np.hypot(coh_x, coh_y)
    non_zero = coh_mag > 1e-9
    coh_scale = MAX_SPEED / np.where(non_zero, coh_mag, 1.0)
    coh_x = np.where(non_zero, coh_x * coh_scale - vx, 0.0)
    coh_y = np.where(non_zero, coh_y * coh_scale - vy, 0.0)
    coh_x, coh_y = clamp_force(coh_x, coh_y)

    # A boid with no neighbors flies straight; nothing to steer toward.
    isolated = neighbor_counts == 0
    dvx = WEIGHT_SEPARATION * sep_x + WEIGHT_ALIGNMENT * ali_x + WEIGHT_COHESION * coh_x
    dvy = WEIGHT_SEPARATION * sep_y + WEIGHT_ALIGNMENT * ali_y + WEIGHT_COHESION * coh_y
    dvx[isolated] = 0.0
    dvy[isolated] = 0.0
    velocities[order, 0] += dvx.astype(_DTYPE)
    velocities[order, 1] += dvy.astype(_DTYPE)

    integrate()


# ---- Rendering ----
HEADING_BINS = 360
_SPRITE_HALF = BOID_SIZE + 1


def _heading_color(hue):
    h6 = hue * 6.0
    sector = int(h6) % 6
    frac = h6 - math.floor(h6)
    q = int(255.0 * (1.0 - frac))
    t = int(255.0 * frac)
    return (
        (255, t, 0), (q, 255, 0), (0, 255, t),
        (0, q, 255), (t, 0, 255), (255, 0, q),
    )[sector]


_BIN_COLORS = np.array(
    [_heading_color((b + 0.5) / HEADING_BINS) for b in range(HEADING_BINS)],
    dtype=np.int64,
)


# Pre-render one oriented triangle per heading bin so the per-frame draw
# loop only blits pre-baked surfaces, no trig per boid.
def _build_sprites():
    size = BOID_SIZE
    wing_len = size * WING_RATIO
    cw, sw = math.cos(WING_ANGLE), math.sin(WING_ANGLE)
    c = float(_SPRITE_HALF)
    sprites = []
    for b in range(HEADING_BINS):
        angle = (b + 0.5) / HEADING_BINS * 2.0 * math.pi - math.pi
        ca, sa = math.cos(angle), math.sin(angle)
        sprite = pygame.Surface((2 * _SPRITE_HALF + 1, 2 * _SPRITE_HALF + 1)).convert()
        sprite.fill((0, 0, 0))
        pygame.draw.polygon(sprite, tuple(int(v) for v in _BIN_COLORS[b]), (
            (c + ca * size, c + sa * size),
            (c + (ca * cw - sa * sw) * wing_len, c + (sa * cw + ca * sw) * wing_len),
            (c + (ca * cw + sa * sw) * wing_len, c + (sa * cw - ca * sw) * wing_len),
        ))
        sprite.set_colorkey((0, 0, 0))
        sprites.append(sprite)
    return sprites


_SPRITES = _build_sprites()

_packed_colors = {}


def heading_bins():
    # Quantize each velocity direction into a color-bin index.
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    bins = ((angles + math.pi) * (HEADING_BINS / (2.0 * math.pi))).astype(np.intp)
    bins %= HEADING_BINS
    return bins


def draw_boids_tri(surface, bins):
    xs = (positions[:, 0].astype(np.int32) - _SPRITE_HALF).tolist()
    ys = (positions[:, 1].astype(np.int32) - _SPRITE_HALF).tolist()
    sprites = _SPRITES
    surface.blits(
        [(sprites[b], (x, y)) for b, x, y in zip(bins.tolist(), xs, ys)],
        doreturn=False,
    )


def draw_boids_dash(surface, bins):
    # surfarray's 2D view is a raw packed-int buffer, not (r, g, b). The
    # surface's color masks reveal how each channel's bits sit inside that
    # int, so I fold every heading color into its packed form once per
    # display format, then stamp pixels by bin. Re-stamping each boid at
    # 2px and 4px ahead of its nose fakes a motion trail with zero extra
    # surface allocation.
    view = pygame.surfarray.pixels2d(surface)

    masks = surface.get_masks()
    packed = _packed_colors.get(masks)
    if packed is None:
        def _shift(mask):
            # Bit index of the mask's lowest set bit.
            return (mask & -mask).bit_length() - 1

        mask_r, mask_g, mask_b, _ = masks
        packed = (
            (_BIN_COLORS[:, 0] << _shift(mask_r))
            | (_BIN_COLORS[:, 1] << _shift(mask_g))
            | (_BIN_COLORS[:, 2] << _shift(mask_b))
        ).astype(view.dtype)
        _packed_colors[masks] = packed
    rgb = packed[bins]

    px = positions[:, 0]
    py = positions[:, 1]
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    for k in (0.0, 2.0, 4.0):
        ix = np.clip((px + vx * k).astype(np.int32), 0, WIDTH - 1)
        iy = np.clip((py + vy * k).astype(np.int32), 0, HEIGHT - 1)
        view[ix, iy] = rgb
    del view


def draw_boids(surface):
    # Triangles win under the limit; past it the packed dashed path is faster.
    n = positions.shape[0]
    if n == 0:
        return
    bins = heading_bins()
    if n <= TRIANGLE_RENDER_LIMIT:
        draw_boids_tri(surface, bins)
    else:
        draw_boids_dash(surface, bins)


# ---- Main loop ----
def main():
    global panel_minimized
    init_boids(NUM_BOIDS)
    background_color = C_BG

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    init_boids(positions.shape[0])
                elif event.key == pygame.K_UP:
                    sliders["population"].value = min(
                        sliders["population"].max_v,
                        sliders["population"].value + 100,
                    )
                elif event.key == pygame.K_DOWN:
                    sliders["population"].value = max(
                        sliders["population"].min_v,
                        sliders["population"].value - 100,
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if toggle_button.hit(event.pos):
                    panel_minimized = not panel_minimized
                elif not panel_minimized:
                    if reset_button.hit(event.pos):
                        init_boids(positions.shape[0])
                    elif any(b.hit(event.pos) for b in template_buttons):
                        for b, t in zip(template_buttons, TEMPLATES):
                            if b.hit(event.pos):
                                apply_template(t)
                                break
                    else:
                        for s in sliders.values():
                            if s.hit(event.pos):
                                s.dragging = True
                                s.value_from_x(event.pos[0])
                                break
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for s in sliders.values():
                    s.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                for s in sliders.values():
                    if s.dragging:
                        s.value_from_x(event.pos[0])

        # First sync controls, then step physics, then draw.
        apply_sliders()   # Push slider values into the sim constants.
        update_all()      # Advance one physics step.

        screen.fill(background_color)
        draw_boids(screen)
        draw_panel(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()