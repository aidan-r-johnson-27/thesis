import math

import numpy as np
import pygame

from boids_rl.rewards import REWARD_TERMS

# terminal look: one monospace font, greys only, 1px boxes with the title cut into the top border,
# and chart lines told apart by dash pattern instead of colour

VIEW = 900              # the world view is a VIEW x VIEW square on the left
PANEL_W = 640           # stats panel on the right
SCREEN_W = VIEW + PANEL_W
HEIGHT = VIEW
TRAIL_FADE = 216        # trail alpha gets multiplied by this / 256 every frame (~16 frame trails)

BG = (10, 10, 10)
FAINT = (40, 40, 40)    # world grid, chart gridlines
LINE = (72, 72, 72)     # box borders
DIM = (112, 112, 112)   # labels, axis text
MID = (164, 164, 164)
TEXT = (214, 214, 214)
BRIGHT = (250, 250, 250)

# (grey, dash) per reward term, dash is (pixels on, pixels off) or None for a solid line
TERM_STYLES = {"caught": (TEXT, (6, 4)), "collision": (DIM, (1, 3)),
               "cohesion": (MID, None), "alignment": (BRIGHT, None)}

MONO_FONTS = ("jetbrainsmono", "jetbrainsmononerdfontmono", "dejavusansmono", "liberationmono",
              "menlo", "consolas", "couriernew")


def load_font(size, bold=False):
    for name in MONO_FONTS:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    # pygame's built-in font runs small, hence the +6
    return pygame.font.Font(None, size + 6)


def nice_step(raw_step):
    # round a tick spacing up to 1, 2, 2.5 or 5 times a power of ten
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiple in (1, 2, 2.5, 5, 10):
        if raw_step <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def nice_range(low, high, ticks=4):
    # axis limits on round tick values that cover [low, high]
    # (rewards are usually negative so the axis doesn't have to start at 0)
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0, 1.0
    if high - low < 1e-9:
        pad = max(abs(high) * 0.1, 0.05)
        low, high = low - pad, high + pad
    step = nice_step((high - low) / ticks)
    while True:
        start = math.floor(low / step) * step
        if start + ticks * step >= high - 1e-12:
            return start, start + ticks * step
        # doesn't reach high yet, so try the next bigger nice step
        step = nice_step(step * 1.001)


def rolling_mean(values, window):
    # trailing mean that skips NaNs (updates where no episode finished)
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) > 0:
            out[i] = chunk.mean()
    return out


def best_rolling_return(history):
    returns = np.array(history.get("episode_return", []), float)
    if len(returns) == 0:
        return None
    averages = rolling_mean(returns, 20)
    if not np.isfinite(averages).any():
        return None
    return np.nanmax(averages)


def mix_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def text_bar(fraction, width):
    # "[#####.....]" with `width` cells between the brackets
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def draw_dashed_line(surface, color, points, dash=None, width=1):
    if dash is None:
        pygame.draw.lines(surface, color, False, points, width)
        return
    on, off = dash
    period = on + off
    # distance along the whole line so far, so the pattern carries on from one segment
    # to the next (otherwise lots of tiny segments would all just look solid)
    travelled = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < length:
            phase = (travelled + pos) % period
            if phase < on:
                piece = min(on - phase, length - pos)
                a, b = pos / length, (pos + piece) / length
                start = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
                end = (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)
                pygame.draw.line(surface, color, start, end, width)
            else:
                piece = min(period - phase, length - pos)
            pos += piece
        travelled += length


def draw_dotted_circle(surface, color, center, radius, spacing=5):
    n_dots = max(8, int(2 * math.pi * radius / spacing))
    for i in range(n_dots):
        angle = 2 * math.pi * i / n_dots
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        surface.fill(color, (x, y, 1, 1))


def arrow_points(x, y, angle, length, half_width):
    # a triangle pointing along `angle` (screen y points down, hence the minus)
    ux, uy = math.cos(angle), -math.sin(angle)
    back = length * 0.6
    return [(x + ux * length, y + uy * length),
            (x - ux * back - uy * half_width, y - uy * back + ux * half_width),
            (x - ux * back + uy * half_width, y - uy * back - ux * half_width)]


def pixel_columns(values, n_columns):
    # when there are more points than pixels, split them into one chunk per pixel column
    n = len(values)
    for column in range(n_columns):
        start = column * n // n_columns
        end = max(start + 1, (column + 1) * n // n_columns)
        chunk = values[start:end]
        yield column, chunk[np.isfinite(chunk)]


class Renderer:
    # owns the window and draws the world view and the stats panel every frame

    def __init__(self, world_size):
        pygame.init()
        pygame.display.set_caption("boids_rl")
        self.screen = pygame.display.set_mode((SCREEN_W, HEIGHT))
        self.font = load_font(14)
        self.font_bold = load_font(14, True)
        self.font_small = load_font(12)
        self.line_h = self.font.get_linesize()
        self.char_w = self.font.size("0")[0]
        self.world_size = world_size
        self.scale = VIEW / world_size
        self.backdrop = self._make_backdrop()

        # trails live on their own layer that fades a bit every frame
        self.trail_layer = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
        self.trail_prev = None   # boid screen positions last frame, None = start the trails over
        self.trail_key = None    # (which env, n boids): switching worlds clears the trails
        self.last_t = 0

        self.death_marks = []    # [x, y, frames left] for the fading crosses where boids died
        self.last_drawn_t = None

        # the charts only get redrawn when a new update comes in
        self.chart = None
        self.chart_key = None
        self.best_return = None

    def _text(self, surface, text, pos, color=TEXT, font=None, anchor="topleft", bg=None):
        # giving bg makes it opaque; color=BG with bg=TEXT is reverse video
        rendered = (font or self.font).render(text, True, color, bg)
        rect = rendered.get_rect(**{anchor: pos})
        surface.blit(rendered, rect)
        return rect

    def _box(self, surface, rect, title=None):
        surface.fill(BG, rect)
        pygame.draw.rect(surface, LINE, rect, 1)
        if title:
            self._text(surface, f" {title} ", (rect.x + 8, rect.y), MID, anchor="midleft", bg=BG)

    def _make_backdrop(self):
        # flat background with a small + at every grid point
        surface = pygame.Surface((VIEW, VIEW))
        surface.fill(BG)
        step = nice_step(self.world_size / 8) * self.scale
        for x in np.arange(step, VIEW, step):
            for y in np.arange(step, VIEW, step):
                pygame.draw.line(surface, FAINT, (x - 3, y), (x + 3, y))
                pygame.draw.line(surface, FAINT, (x, y - 3), (x, y + 3))
        return surface

    def draw(self, world, history, info, show_world=True):
        if show_world:
            self._draw_world(world, info)
        else:
            self._draw_idle(info)
        self._draw_panel(history, info)
        pygame.display.flip()

    def to_screen(self, points):
        # world y points up, screen y points down
        return points[..., 0] * self.scale, VIEW - points[..., 1] * self.scale

    def _draw_world(self, env, info):
        screen = self.screen
        screen.set_clip(pygame.Rect(0, 0, VIEW, VIEW))
        screen.blit(self.backdrop, (0, 0))
        n_boids = len(env.pos)
        xs, ys = self.to_screen(env.pos)
        headings = np.arctan2(env.vel[:, 1], env.vel[:, 0])

        if info["color_mode"] == "reward":
            # dim = low reward this step, white = high
            low, high = info["reward_range"]
            shade = np.clip((info["rewards"] - low) / max(high - low, 1e-6), 0, 1)
            colors = [mix_color(DIM, BRIGHT, s) for s in shade]
        else:
            colors = [TEXT] * n_boids

        self._draw_trails(env, info, xs, ys, colors)

        if info["show_focus"]:
            # boid 0's perception radius (plus its copies across the wrap) and lines to who it sees
            cx, cy = xs[0], ys[0]
            for ox in (-VIEW, 0, VIEW):
                for oy in (-VIEW, 0, VIEW):
                    draw_dotted_circle(screen, DIM, (cx + ox, cy + oy), env.cfg.neighbor_radius * self.scale)
            data = env.data
            for rel in data.rel_pos[0][data.mask[0]]:
                end = (cx + rel[0] * self.scale, cy - rel[1] * self.scale)
                pygame.draw.line(screen, DIM, (cx, cy), end)

        for i in range(n_boids):
            pygame.draw.polygon(screen, colors[i], arrow_points(xs[i], ys[i], headings[i], 9, 4))
        if info["show_focus"]:
            self._brackets(xs[0], ys[0], 11)

        self._draw_predators(env)
        self._draw_hud(env, info, headings)
        screen.set_clip(None)

    def _draw_trails(self, env, info, xs, ys, colors):
        # instead of redrawing each boid's whole history, fade the trail layer a bit and add
        # one new segment per boid every frame
        key = (id(env), len(env.pos))
        if key != self.trail_key or env.t < self.last_t:
            # different world, or the episode restarted
            self.trail_prev = None
            self.trail_key = key
        self.last_t = env.t

        # at high speeds trails would just be streaks
        if not info["show_trails"] or info["steps_per_frame"] > 8:
            self.trail_prev = None
            return

        layer = self.trail_layer
        if self.trail_prev is None:
            layer.fill((0, 0, 0, 0))
        else:
            layer.fill((255, 255, 255, TRAIL_FADE), special_flags=pygame.BLEND_RGBA_MULT)
            prev_xs, prev_ys = self.trail_prev
            # anything that moved further than a boid can fly in one frame wrapped around or respawned
            reach = env.cfg.max_speed * env.cfg.dt * info["steps_per_frame"] * self.scale + 1
            for i in range(len(xs)):
                if abs(xs[i] - prev_xs[i]) > reach or abs(ys[i] - prev_ys[i]) > reach:
                    continue
                pygame.draw.line(layer, (*colors[i], 110), (prev_xs[i], prev_ys[i]), (xs[i], ys[i]), 1)
        self.trail_prev = (xs, ys)
        self.screen.blit(layer, (0, 0))

    def _brackets(self, x, y, r):
        # selection-cursor corners around a point
        for dx in (-1, 1):
            for dy in (-1, 1):
                cx, cy = x + dx * r, y + dy * r
                pygame.draw.line(self.screen, BRIGHT, (cx, cy), (cx - dx * 4, cy))
                pygame.draw.line(self.screen, BRIGHT, (cx, cy), (cx, cy - dy * 4))

    def _draw_predators(self, env):
        screen = self.screen

        # deaths leave a fading cross (only deaths on steps that actually got drawn)
        if env.t != self.last_drawn_t:
            death_xs, death_ys = self.to_screen(env.last_deaths)
            for x, y in zip(death_xs, death_ys):
                self.death_marks.append([x, y, 30])
            self.last_drawn_t = env.t
        for mark in self.death_marks:
            x, y, frames_left = mark
            color = mix_color(BG, BRIGHT, frames_left / 30)
            pygame.draw.line(screen, color, (x - 5, y - 5), (x + 5, y + 5))
            pygame.draw.line(screen, color, (x - 5, y + 5), (x + 5, y - 5))
            mark[2] -= 1
        self.death_marks = [mark for mark in self.death_marks if mark[2] > 0]

        # a bigger hollow triangle with its catch radius, dimmed while on cooldown
        xs, ys = self.to_screen(env.pred_pos)
        for i in range(len(env.pred_pos)):
            ready = env.pred_cooldown[i] == 0
            catch_r = env.cfg.catch_radius * self.scale
            pygame.draw.circle(screen, LINE if ready else FAINT, (xs[i], ys[i]), catch_r, 1)
            angle = math.atan2(env.pred_vel[i, 1], env.pred_vel[i, 0])
            points = arrow_points(xs[i], ys[i], angle, 16, 9)
            pygame.draw.polygon(screen, BG, points)
            pygame.draw.polygon(screen, BRIGHT if ready else DIM, points, 2)

    def _draw_hud(self, env, info, headings):
        screen = self.screen
        self._text(screen, f" {info['world_label']} ", (10, 10), TEXT, bg=BG)

        # bottom left: key/value readout for the world on screen
        metrics = env.flock_metrics()
        rows = [("step", f"{env.t}/{env.cfg.max_steps}"),
                ("", text_bar(env.t / env.cfg.max_steps, 22)),
                ("polarization", f"{metrics['polarization']:.2f}"),
                ("nearest nbr", f"{metrics['nn_distance']:.2f}"),
                ("caught/collided", f"{env.deaths['caught']}/{env.deaths['collided']}"),
                ("return so far", f"{info['episode_return']:.1f}"),
                ("shade [c]", info["color_mode"])]
        box = pygame.Rect(10, 0, 24 * self.char_w + 20, self.line_h * len(rows) + 18)
        box.bottom = VIEW - 10
        self._box(screen, box, "world")
        for i, (name, value) in enumerate(rows):
            y = box.y + 11 + self.line_h * i
            if name:
                self._text(screen, name, (box.x + 10, y), DIM)
                self._text(screen, value, (box.right - 10, y), TEXT, anchor="topright")
            else:
                self._text(screen, value, (box.x + 10, y), MID)

        # bottom right: compass of the mean heading, the needle's length is the polarization
        box = pygame.Rect(0, 0, 96, 104)
        box.bottomright = (VIEW - 10, VIEW - 10)
        self._box(screen, box, "heading")
        center = (box.centerx, box.y + 46)
        draw_dotted_circle(screen, LINE, center, 30, 4)
        mean_x, mean_y = np.cos(headings).mean(), np.sin(headings).mean()
        tip = (center[0] + mean_x * 30, center[1] - mean_y * 30)
        pygame.draw.line(screen, TEXT, center, tip)
        screen.fill(BRIGHT, (tip[0] - 1, tip[1] - 1, 3, 3))
        degrees = math.degrees(math.atan2(mean_y, mean_x)) % 360
        self._text(screen, f"{degrees:03.0f}°", (center[0], box.bottom - 6), DIM, font=self.font_small,
                   anchor="midbottom")

    def _draw_idle(self, info):
        # shown instead of the world when it's hidden with V, with a blinking cursor
        screen = self.screen
        screen.fill(BG, (0, 0, VIEW, HEIGHT))
        y = HEIGHT // 2 - 2 * self.line_h
        rows = [("-- visuals off --", TEXT), ("training at full speed", DIM),
                (f"{info['sps']:,.0f} env steps/s", TEXT), ("", DIM), ("press v to watch again", DIM)]
        x = VIEW // 2 - max(self.font.size(text)[0] for text, _ in rows) // 2
        for i, (text, color) in enumerate(rows):
            rect = self._text(screen, text, (x, y + i * self.line_h), color)
        if (pygame.time.get_ticks() // 530) % 2 == 0:
            screen.fill(TEXT, (rect.right + 2, rect.y + 2, self.char_w, self.line_h - 4))

    def _draw_panel(self, history, info):
        screen = self.screen
        x0 = VIEW
        pad = 16
        screen.fill(BG, (x0, 0, PANEL_W, HEIGHT))
        pygame.draw.line(screen, LINE, (x0, 0), (x0, HEIGHT))

        # header: name, mode badge in reverse video, speed
        rect = self._text(screen, "boids_rl", (x0 + pad, 12), BRIGHT, font=self.font_bold)
        rect = self._text(screen, "/live", (rect.right, 12), DIM)
        self._text(screen, f" {info['mode']} ", (rect.right + 14, 12), BG, font=self.font_bold, bg=TEXT)
        self._text(screen, f"speed {info['speed']}", (SCREEN_W - pad, 12), DIM, anchor="topright")

        n_updates = len(history.get("update", []))
        if self.chart is None or n_updates != self.chart_key:
            self.best_return = best_rolling_return(history)

        latest = {name: values[-1] for name, values in history.items() if values}
        finite_returns = [r for r in history.get("episode_return", []) if np.isfinite(r)]

        def latest_value(name, fmt):
            if name in latest and np.isfinite(latest[name]):
                return fmt.format(latest[name])
            return "-"

        cells = [
            ("update", f"{info['update']:,}"),
            ("env steps", f"{info['env_steps']:,}"),
            ("episodes", f"{int(np.nansum(history.get('n_episodes', [0]))):,}"),
            ("steps/s", f"{info['sps']:,.0f}"),
            ("return", f"{finite_returns[-1]:.2f}" if finite_returns else "-"),
            ("best avg20", f"{self.best_return:.2f}" if self.best_return is not None else "-"),
            ("polarization", latest_value("polarization", "{:.2f}")),
            ("entropy", latest_value("entropy", "{:.3f}")),
        ]
        box = pygame.Rect(x0 + pad, 44, PANEL_W - 2 * pad, self.line_h * 5 + 22)
        self._box(screen, box, "status")
        half = (box.w - 20) // 2
        for i, (label, value) in enumerate(cells):
            # two columns of label ... value
            cx = box.x + 10 + (i % 2) * (half + 10)
            cy = box.y + 12 + (i // 2) * self.line_h
            self._text(screen, label, (cx, cy), DIM)
            self._text(screen, value, (cx + half - 10, cy), TEXT, anchor="topright")

        cy = box.y + 12 + 4 * self.line_h + 4
        label = self._text(screen, "rollout", (box.x + 10, cy), DIM)
        count = self._text(screen, f"{info['rollout_t']}/{info['rollout_T']}", (box.right - 10, cy), TEXT,
                           anchor="topright")
        bar_width = (count.x - label.right - 4 * self.char_w) // self.char_w
        bar = text_bar(info["rollout_t"] / info["rollout_T"], bar_width)
        self._text(screen, bar, (label.right + 2 * self.char_w, cy), MID)

        top = box.bottom + 14
        if self.chart is None or n_updates != self.chart_key:
            chart_h = HEIGHT - top - 2 * self.line_h - 18
            self.chart = self._make_charts(history, PANEL_W - pad * 2, chart_h)
            self.chart_key = n_updates
        screen.blit(self.chart, (x0 + pad, top))

        # key help at the bottom, nano style
        key_rows = [[("spc", "pause"), ("↑↓", "speed"), ("v", "visuals"), ("d", "demo"), ("s", "save"),
                     ("r", "reset"), ("esc", "quit")],
                    [("e", "next env"), ("c", "shade by reward"), ("n", "neighbours"), ("t", "trails")]]
        for i, row in enumerate(key_rows):
            y = HEIGHT - 10 - (len(key_rows) - i) * self.line_h
            x = x0 + pad
            for key, action in row:
                rect = self._text(screen, f" {key} ", (x, y), BG, font=self.font_small, bg=MID)
                rect = self._text(screen, " " + action, (rect.right, y), DIM, font=self.font_small)
                x = rect.right + 10

        if info.get("toast"):
            self._text(screen, f" {info['toast']} ", (VIEW // 2, 10), BG, anchor="midtop", bg=TEXT)

    def _make_charts(self, history, w, h):
        # return per update on top, raw reward terms in the middle, polarization and KL side by side
        surface = pygame.Surface((w, h))
        surface.fill(BG)
        gap = 12
        h_return = int((h - 2 * gap) * 0.37)
        h_terms = int((h - 2 * gap) * 0.35)
        h_bottom = h - 2 * gap - h_return - h_terms

        def series(name):
            return np.array(history.get(name, []), float)

        returns = series("episode_return")
        averages = rolling_mean(returns, 20) if len(returns) else returns
        self._chart(surface, pygame.Rect(0, 0, w, h_return), "episode return / update",
                    [(averages, BRIGHT, None)],
                    [("return", DIM, "dot"), ("avg20", BRIGHT, None), ("best", MID, (4, 4))],
                    points=returns, best=self.best_return)

        term_lines, term_legend = [], []
        for name in REWARD_TERMS:
            color, dash = TERM_STYLES.get(name, (TEXT, None))
            term_lines.append((series(f"term_{name}"), color, dash))
            term_legend.append((name, color, dash))
        self._chart(surface, pygame.Rect(0, h_return + gap, w, h_terms), "reward terms (mean/step)",
                    term_lines, term_legend)

        half = (w - gap) // 2
        y = h_return + h_terms + 2 * gap
        self._chart(surface, pygame.Rect(0, y, half, h_bottom), "polarization",
                    [(series("polarization"), BRIGHT, None)], [], low=0.0, high=1.0, fill=True, show_last=True)
        self._chart(surface, pygame.Rect(half + gap, y, w - half - gap, h_bottom), "approx kl",
                    [(series("approx_kl"), BRIGHT, None)], [], low=0.0, show_last=True)
        return surface

    def _chart(self, surface, rect, title, lines, legend, points=None, best=None, low=None, high=None,
               fill=False, show_last=False):
        # lines: (values, grey, dash), one value per PPO update
        # points: drawn as dots, or as one min..max bar per pixel column once there are too many
        # best: a dashed horizontal guide
        # NaNs (updates where no episode finished) just get skipped
        if show_last:
            first = lines[0][0]
            first = first[np.isfinite(first)]
            if len(first) > 0:
                title += f"  {first[-1]:.3g}"
        # box starts half a text row down so the title can sit on its top edge
        box = pygame.Rect(rect.x, rect.y + self.line_h // 2, rect.w, rect.h - self.line_h // 2)
        self._box(surface, box, title)
        self._draw_legend(surface, box, legend)

        plot = pygame.Rect(box.x + 54, box.y + 16, box.w - 68, box.h - 16 - 12 - self.line_h)
        n = len(lines[0][0])
        low, high = self._y_limits(lines, points, n, low, high)

        # dotted gridlines with tick labels on the left
        for i in range(5):
            value = low + (high - low) * i / 4
            y = int(plot.bottom - plot.h * i / 4)
            for x in range(plot.x, plot.right, 3):
                surface.fill(FAINT, (x, y, 1, 1))
            label = f"{value:.3g}" if abs(value) < 1000 else f"{value / 1000:.3g}k"
            self._text(surface, label, (plot.x - 8, y), DIM, font=self.font_small, anchor="midright")
        if n == 0:
            self._text(surface, "waiting for first update_", plot.center, DIM, font=self.font_small,
                       anchor="center")
            return

        # first and last update number under the x axis
        last_update = n if n > 1 else 1
        self._text(surface, "1", (plot.x, plot.bottom + 6), DIM, font=self.font_small, anchor="topleft")
        self._text(surface, f"{last_update:,}", (plot.right, plot.bottom + 6), DIM, font=self.font_small,
                   anchor="topright")
        self._text(surface, "update", (plot.centerx, plot.bottom + 6), DIM, font=self.font_small, anchor="midtop")

        def screen_x(i):
            if n == 1:
                return plot.x + plot.w / 2
            return plot.x + plot.w * i / (n - 1)

        def screen_y(value):
            value = min(max(value, low), high)
            return plot.bottom - plot.h * (value - low) / (high - low)

        n_columns = max(1, plot.w)
        if points is not None:
            if n <= n_columns:
                for i, value in enumerate(points):
                    if np.isfinite(value):
                        surface.fill(DIM, (int(screen_x(i)) - 1, int(screen_y(value)) - 1, 2, 2))
            else:
                for column, chunk in pixel_columns(points, n_columns):
                    if len(chunk) > 0:
                        x = plot.x + column
                        pygame.draw.line(surface, DIM, (x, screen_y(chunk.min())), (x, screen_y(chunk.max())))
        if best is not None:
            y = screen_y(best)
            draw_dashed_line(surface, MID, [(plot.x, y), (plot.right, y)], (4, 4))

        for values, color, dash in lines:
            if n <= n_columns:
                line = [(screen_x(i), screen_y(v)) for i, v in enumerate(values) if np.isfinite(v)]
            else:
                line = [(plot.x + column, screen_y(chunk.mean()))
                        for column, chunk in pixel_columns(values, n_columns) if len(chunk) > 0]
            if len(line) > 1:
                if fill:
                    # hatching under the line: a 1px column every 3px
                    xs, ys = zip(*line)
                    for x in range(int(xs[0]), int(xs[-1]) + 1, 3):
                        pygame.draw.line(surface, FAINT, (x, np.interp(x, xs, ys) + 2), (x, plot.bottom))
                draw_dashed_line(surface, color, line, dash)
            if line:
                surface.fill(color, (int(line[-1][0]) - 1, int(line[-1][1]) - 1, 3, 3))

    def _draw_legend(self, surface, box, legend):
        # right to left along the top border: a sample of the line style, then the name
        x = box.right - 10
        for name, color, dash in reversed(legend):
            label = self.font_small.render(" " + name + " ", True, DIM, BG)
            x -= label.get_width()
            surface.blit(label, label.get_rect(midleft=(x, box.y)))
            x -= 16
            surface.fill(BG, (x - 2, box.y - 1, 20, 3))
            if dash == "dot":
                for dot_x in (x + 2, x + 8, x + 14):
                    surface.fill(color, (dot_x - 1, box.y - 1, 2, 2))
            else:
                draw_dashed_line(surface, color, [(x, box.y), (x + 16, box.y)], dash)
            x -= 6

    def _y_limits(self, lines, points, n, low, high):
        # fixed limits win; anything not given gets a nice range around the data
        if low is not None and high is not None:
            return low, high
        all_values = [values for values, _, _ in lines]
        if points is not None:
            all_values.append(points)
        data = np.concatenate(all_values) if n > 0 else np.array([])
        data = data[np.isfinite(data)]
        data_low = data.min() if len(data) else 0.0
        data_high = data.max() if len(data) else 1.0
        nice_low, nice_high = nice_range(low if low is not None else data_low,
                                         high if high is not None else data_high)
        if low is None:
            low = nice_low
        if high is None:
            high = max(nice_high, low + 1e-6)
        return low, high
