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

WEIGHT_SEPARATION = 1.4
WEIGHT_ALIGNMENT = 1.0
WEIGHT_COHESION = 1.0

# ---- Boid sprite geometry ----
BOID_SIZE = 4
WING_ANGLE = 2.35
WING_RATIO = 0.55

TRIANGLE_RENDER_LIMIT = 4500

# ---- Window setup ----
pygame.init()
_display_info = pygame.display.Info()
WIDTH, HEIGHT = _display_info.current_w, _display_info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Boids Simulation")
clock = pygame.time.Clock()

# ---- UI colors and panel layout ----
C_BG = (10, 10, 10)
C_PANEL_BORDER = (72, 72, 72)
C_TITLE = (164, 164, 164)
C_TEXT = (214, 214, 214)
C_TEXT_DIM = (112, 112, 112)
C_TRACK_BG = (40, 40, 40)
C_TRACK_FILL = (164, 164, 164)
C_CONTROL = (214, 214, 214)
C_CONTROL_BORDER = (72, 72, 72)
C_ACTIVE_BG = (26, 26, 26)
C_IDLE_BG = (10, 10, 10)
C_HINT = (112, 112, 112)

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

# ---- Monospace font ----
_MONO_FONTS = ("jetbrainsmono", "jetbrainsmononerdfontmono", "dejavusansmono",
               "liberationmono", "menlo", "consolas", "couriernew")
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    f = _font_cache.get(key)
    if f is None:
        for name in _MONO_FONTS:
            path = pygame.font.match_font(name, bold=bold)
            if path:
                f = pygame.font.Font(path, size)
                break
        else:
            # default font renders small, bump it up a bit
            f = pygame.font.Font(None, size + 6)
        _font_cache[key] = f
    return f


# ---- Cached text rendering ----
_text_cache = {}
_TEXT_CACHE_MAX = 256


def _text_surface(text, color, size=14, bold=True):
    key = (text, color, size, bold)
    cached = _text_cache.get(key)
    if cached is not None:
        return cached
    surf = font(size, bold).render(text, True, color)
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
        surface.blit(_text_surface(f"{self.label}: {self.fmt(self.value)}", C_TEXT),
                     (self.x, self.track_y - LABEL_H))
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
        label_surf = _text_surface(self.label, C_CONTROL)
        surface.blit(label_surf, label_surf.get_rect(center=self.rect.center))


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
    target = int(sliders["population"].value)
    current = positions.shape[0]
    if current < target:
        add_boids(target - current)
    elif current > target:
        remove_boids(current - target)


def apply_sliders():
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
    # just push into the sliders, apply_sliders picks it up next frame anyway
    for key, s in sliders.items():
        if key in template:
            s.value = template[key]
    apply_sliders()


# title sits on top of the border, so the surface needs half the title's height of extra room above the box
_TITLE_SURF = _text_surface(" PARAMETERS ", C_TITLE)
_TITLE_OVERHANG = _TITLE_SURF.get_height() // 2


def _panel_background(height):
    bg = pygame.Surface((PANEL_WIDTH, height + _TITLE_OVERHANG)).convert()
    bg.fill(C_BG)
    pygame.draw.rect(bg, C_PANEL_BORDER, (0, _TITLE_OVERHANG, PANEL_WIDTH, height), 1)
    bg.blit(_TITLE_SURF, _TITLE_SURF.get_rect(midleft=(PANEL_PAD - 4, _TITLE_OVERHANG)))
    return bg


_PANEL_BG = _panel_background(PANEL_H)
_PANEL_STRIP_BG = _panel_background(PANEL_STRIP_H)


def draw_panel(surface):
    mouse = pygame.mouse.get_pos()

    if panel_minimized:
        surface.blit(_PANEL_STRIP_BG, (PANEL_X, PANEL_TOP - _TITLE_OVERHANG))
        toggle_button.label = "SHOW"
        toggle_button.hover = toggle_button.hit(mouse)
        toggle_button.draw(surface)
        return

    surface.blit(_PANEL_BG, (PANEL_X, PANEL_TOP - _TITLE_OVERHANG))
    for s in sliders.values():
        s.draw(surface)
    for b in template_buttons + [reset_button]:
        b.hover = b.hit(mouse)
        b.draw(surface)
    toggle_button.label = "HIDE"
    toggle_button.hover = toggle_button.hit(mouse)
    toggle_button.draw(surface)
    surface.blit(_text_surface("SPACE RESET  UP/DN POP  ESC QUIT", C_HINT, 7),
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
def _spawn(count):
    pos = np.random.uniform(0.0, 1.0, (count, 2)).astype(_DTYPE)
    pos[:, 0] *= WIDTH
    pos[:, 1] *= HEIGHT
    angles = np.random.uniform(0.0, 2.0 * math.pi, count)
    vel = np.column_stack((np.cos(angles), np.sin(angles))) * MAX_SPEED
    return pos, vel.astype(_DTYPE)


def init_boids(count):
    global positions, velocities
    positions, velocities = _spawn(count)


def add_boids(count):
    global positions, velocities
    new_pos, new_vel = _spawn(count)
    positions = np.concatenate((positions, new_pos))
    velocities = np.concatenate((velocities, new_vel))


def remove_boids(count):
    global positions, velocities
    n = min(count, positions.shape[0])
    if n > 0:
        positions = positions[:-n]
        velocities = velocities[:-n]


def integrate():
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

    # sort by cell id so every cell is one contiguous run, pos/vel get the same order so indices line up
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

    # cell c lives in starts[c]:starts[c+1]
    starts = np.zeros(NUM_CELLS + 1, dtype=np.intp)
    np.cumsum(np.bincount(cell_ids, minlength=NUM_CELLS), out=starts[1:])

    # own cell (only boids after me) + 4 forward cells, so each pair only gets looked at once
    boid_range = np.arange(n, dtype=np.intp)
    block_start = np.empty((n, 5), dtype=np.intp)
    block_len = np.empty((n, 5), dtype=np.intp)
    block_start[:, 0] = boid_range + 1
    block_len[:, 0] = starts[cell_ids + 1] - block_start[:, 0]
    for k, (dc, dr) in enumerate(_FORWARD_OFFSETS, 1):
        nb_cells = ((rows + dr) % GRID_ROWS) * GRID_COLS + (cols + dc) % GRID_COLS
        block_start[:, k] = starts[nb_cells]
        block_len[:, k] = starts[nb_cells + 1] - block_start[:, k]

    lens = block_len.ravel()
    block_ends = np.cumsum(lens)
    total = int(block_ends[-1])
    if total == 0:
        integrate()
        return

    # flatten every (start, len) block into one index list: repeat (start - where the block lands) then add arange
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

    # torus, so anything over half the screen away is actually closer going across the seam
    np.subtract(dx, WIDTH, out=dx, where=dx > HALF_WIDTH)
    np.add(dx, WIDTH, out=dx, where=dx < -HALF_WIDTH)
    np.subtract(dy, HEIGHT, out=dy, where=dy > HALF_HEIGHT)
    np.add(dy, HEIGHT, out=dy, where=dy < -HALF_HEIGHT)

    d2 = dx * dx
    d2 += dy * dy
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

    # pairs only stored one way, so bincount on cand gets the ones pointing at me and reduceat gets my own
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

    # both boids in a close pair push away, count doubles too so dividing by it cancels out
    close = np.flatnonzero((d2 < SEP_SQ) & (d2 > 0.0))
    sep_q = query[close]
    sep_c = cand[close]
    inv_dist = 1.0 / np.sqrt(d2[close])
    ux = dx[close] * inv_dist
    uy = dy[close] * inv_dist
    sep_x_acc = np.bincount(sep_c, weights=ux, minlength=n) - np.bincount(sep_q, weights=ux, minlength=n)
    sep_y_acc = np.bincount(sep_c, weights=uy, minlength=n) - np.bincount(sep_q, weights=uy, minlength=n)
    sep_counts = np.bincount(sep_q, minlength=n) + np.bincount(sep_c, minlength=n)

    # steer = desired - current, capped at MAX_FORCE so turns stay smooth
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

    coh_x = coh_x_acc * inv_count
    coh_y = coh_y_acc * inv_count
    coh_mag = np.hypot(coh_x, coh_y)
    non_zero = coh_mag > 1e-9
    coh_scale = MAX_SPEED / np.where(non_zero, coh_mag, 1.0)
    coh_x = np.where(non_zero, coh_x * coh_scale - vx, 0.0)
    coh_y = np.where(non_zero, coh_y * coh_scale - vy, 0.0)
    coh_x, coh_y = clamp_force(coh_x, coh_y)

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
C_BOID = C_TEXT


def _make_backdrop():
    s = pygame.Surface((WIDTH, HEIGHT)).convert()
    s.fill(C_BG)
    step = 80
    for x in range(step, WIDTH, step):
        for y in range(step, HEIGHT, step):
            pygame.draw.line(s, C_TRACK_BG, (x - 3, y), (x + 3, y))
            pygame.draw.line(s, C_TRACK_BG, (x, y - 3), (x, y + 3))
    return s


_BACKDROP = _make_backdrop()


def _build_sprites():
    # one pre-rotated triangle per heading bin so drawing is just blits
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
        pygame.draw.polygon(sprite, C_BOID, (
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
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    bins = ((angles + math.pi) * (HEADING_BINS / (2.0 * math.pi))).astype(np.intp)
    bins %= HEADING_BINS
    return bins


def draw_boids_tri(surface):
    xs = (positions[:, 0].astype(np.int32) - _SPRITE_HALF).tolist()
    ys = (positions[:, 1].astype(np.int32) - _SPRITE_HALF).tolist()
    sprites = _SPRITES
    surface.blits(
        [(sprites[b], (x, y)) for b, x, y in zip(heading_bins().tolist(), xs, ys)],
        doreturn=False,
    )


def _packed_boid_color(surface):
    masks = surface.get_masks()
    packed = _packed_colors.get(masks)
    if packed is None:
        # lowest set bit of each mask = how far to shift that channel
        shift = [(m & -m).bit_length() - 1 for m in masks[:3]]
        packed = sum(ch << sh for ch, sh in zip(C_BOID, shift))
        _packed_colors[masks] = packed
    return packed


def draw_boids_dash(surface):
    # pixels2d is raw packed ints not rgb, so the color has to be packed to match the surface format
    packed = _packed_boid_color(surface)
    view = pygame.surfarray.pixels2d(surface)

    px = positions[:, 0]
    py = positions[:, 1]
    vx = velocities[:, 0]
    vy = velocities[:, 1]

    # stamp at 0, 2 and 4 px ahead for a cheap little trail
    for k in (0.0, 2.0, 4.0):
        ix = np.clip((px + vx * k).astype(np.int32), 0, WIDTH - 1)
        iy = np.clip((py + vy * k).astype(np.int32), 0, HEIGHT - 1)
        view[ix, iy] = packed
    del view


def draw_boids(surface):
    n = positions.shape[0]
    if n == 0:
        return
    if n <= TRIANGLE_RENDER_LIMIT:
        draw_boids_tri(surface)
    else:
        draw_boids_dash(surface)


# ---- Main loop ----
def main():
    global panel_minimized
    init_boids(NUM_BOIDS)
    pop = sliders["population"]

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
                    pop.value = min(pop.max_v, pop.value + 100)
                elif event.key == pygame.K_DOWN:
                    pop.value = max(pop.min_v, pop.value - 100)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if toggle_button.hit(event.pos):
                    panel_minimized = not panel_minimized
                elif not panel_minimized:
                    if reset_button.hit(event.pos):
                        init_boids(positions.shape[0])
                        continue
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

        apply_sliders()
        update_all()

        screen.blit(_BACKDROP, (0, 0))
        draw_boids(screen)
        draw_panel(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
