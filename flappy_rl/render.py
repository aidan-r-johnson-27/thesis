import math
import pygame
from game import BIRD_X, GROUND, HEIGHT, PIPE_W, WIDTH

# ---- Screen layout ----
PANEL_W = 560
SCREEN_W = WIDTH + PANEL_W

# ---- Palette ----
BG = (10, 10, 10)
FAINT = (40, 40, 40)
LINE = (72, 72, 72)
DIM = (112, 112, 112)
MID = (164, 164, 164)
TEXT = (214, 214, 214)
BRIGHT = (250, 250, 250)

_MONO_FONTS = ("jetbrainsmono", "jetbrainsmononerdfontmono", "dejavusansmono",
               "liberationmono", "menlo", "consolas", "couriernew")


def font(size, bold=False):
    for name in _MONO_FONTS:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    # default font renders small, bump it up a bit
    return pygame.font.Font(None, size + 6)


def nice_max(v):
    # round up to 1/2/2.5/5 x 10^k so the gridlines get clean labels
    if v <= 4:
        return 4
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


# ---- Line helpers ----
def dashed_lines(surf, color, pts, dash=None, width=1):
    if dash is None:
        pygame.draw.lines(surf, color, False, pts, width)
        return

    on, off = dash
    period = on + off
    # s is distance along the whole polyline so the pattern doesn't restart at every vertex
    s = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        d = 0.0
        while d < seg:
            phase = (s + d) % period
            step = min(on - phase if phase < on else period - phase, seg - d)
            if phase < on:
                a, b = d / seg, (d + step) / seg
                pygame.draw.line(surf, color,
                                 (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                                 (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), width)
            d += step
        s += seg


def dotted_circle(surf, color, center, r, spacing=5):
    n = max(8, int(2 * math.pi * r / spacing))
    for k in range(n):
        a = 2 * math.pi * k / n
        surf.fill(color, (int(center[0] + r * math.cos(a)), int(center[1] + r * math.sin(a)), 1, 1))


def _interp(x, xs, ys):
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return ys[-1]


class Renderer:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy RL")
        self.screen = pygame.display.set_mode((SCREEN_W, HEIGHT))
        self.f = font(14)
        self.f_bold = font(14, True)
        self.f_small = font(12)
        self.f_huge = font(40, True)
        self.lh = self.f.get_linesize()
        self.cw = self.f.size("0")[0]
        self.backdrop = self._make_backdrop()
        self.chart = None
        self.chart_key = None

    def _make_backdrop(self):
        s = pygame.Surface((WIDTH, HEIGHT))
        s.fill(BG)
        step = 40
        for x in range(step, WIDTH, step):
            for y in range(step, HEIGHT, step):
                pygame.draw.line(s, FAINT, (x - 3, y), (x + 3, y))
                pygame.draw.line(s, FAINT, (x, y - 3), (x, y + 3))
        pygame.draw.line(s, LINE, (0, GROUND), (WIDTH, GROUND))
        return s

    # ---- Drawing helpers ----
    def _text(self, surf, s, pos, color=TEXT, f=None, anchor="topleft", bg=None):
        # bg=TEXT with color=BG gives reverse video
        t = (f or self.f).render(s, True, color, bg)
        r = t.get_rect(**{anchor: pos})
        surf.blit(t, r)
        return r

    def _box(self, surf, rect, title=None):
        surf.fill(BG, rect)
        pygame.draw.rect(surf, LINE, rect, 1)
        if title:
            self._text(surf, f" {title} ", (rect.x + 8, rect.y), MID, anchor="midleft", bg=BG)

    # ---- Frame ----
    def draw(self, env, stats, info, visuals=True):
        if visuals:
            self._draw_game(env)
        else:
            self._draw_idle(info)
        self._draw_panel(stats, info)
        pygame.display.flip()

    def _draw_game(self, env):
        scr = self.screen
        scr.set_clip(pygame.Rect(0, 0, WIDTH, HEIGHT))
        scr.blit(self.backdrop, (0, 0))

        for p in env.pipes:
            x = int(p.x)
            top, bottom = int(p.top), int(p.bottom)
            pygame.draw.rect(scr, LINE, (x, 0, PIPE_W, top), 1)
            pygame.draw.rect(scr, LINE, (x, bottom, PIPE_W, GROUND - bottom), 1)

        target = env.next_pipe()
        aim = (int(target.x + PIPE_W), int(target.gap_y))
        dashed_lines(scr, DIM, [(BIRD_X, int(env.y)), aim], (4, 4))
        dotted_circle(scr, DIM, aim, 5, 3)

        # triangle nose points along the tilt, the two back corners are offset sideways by W
        tilt = max(-30, min(80, env.vel * 6))
        ux, uy = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        L, W = 15, 7
        cx, cy = BIRD_X, int(env.y)
        pts = [(cx + ux * L, cy + uy * L),
               (cx - ux * L * 0.6 - uy * W, cy - uy * L * 0.6 + ux * W),
               (cx - ux * L * 0.6 + uy * W, cy - uy * L * 0.6 - ux * W)]
        pygame.draw.polygon(scr, BRIGHT if env.last_action == 1 else TEXT, pts)

        txt = self.f_huge.render(str(env.score), True, BRIGHT)
        scr.blit(txt, txt.get_rect(midtop=(WIDTH // 2, 16)))

        self._draw_hud(env, tilt)
        scr.set_clip(None)

    def _draw_hud(self, env, tilt):
        scr = self.screen

        rows = [("frame", f"{env.frames}"),
                ("score", f"{env.score}"),
                ("velocity", f"{env.vel:.2f}"),
                ("height", f"{env.y:.0f}/{GROUND}")]
        box = pygame.Rect(10, 0, 20 * self.cw + 20, self.lh * len(rows) + 18)
        box.bottom = HEIGHT - 10
        self._box(scr, box, "bird")
        for k, (name, value) in enumerate(rows):
            y = box.y + 11 + self.lh * k
            self._text(scr, name, (box.x + 10, y), DIM)
            self._text(scr, value, (box.right - 10, y), TEXT, anchor="topright")

        box = pygame.Rect(0, 0, 96, 104)
        box.bottomright = (WIDTH - 10, HEIGHT - 10)
        self._box(scr, box, "tilt")
        c = (box.centerx, box.y + 46)
        dotted_circle(scr, LINE, c, 30, 4)
        rad = math.radians(tilt)
        tip = (c[0] + math.cos(rad) * 30, c[1] + math.sin(rad) * 30)
        pygame.draw.line(scr, TEXT, c, tip)
        scr.fill(BRIGHT, (tip[0] - 1, tip[1] - 1, 3, 3))
        self._text(scr, f"{tilt:+.0f}°", (c[0], box.bottom - 6), DIM, f=self.f_small, anchor="midbottom")

    def _draw_idle(self, info):
        scr = self.screen
        scr.fill(BG, (0, 0, WIDTH, HEIGHT))

        rows = [("-- visuals off --", TEXT),
                ("training at full speed", DIM),
                (f"{info['sps']:,.0f} steps/s", TEXT),
                ("", DIM),
                ("press v to watch again", DIM)]
        x = WIDTH // 2 - max(self.f.size(line)[0] for line, _ in rows) // 2
        y = HEIGHT // 2 - 2 * self.lh
        for k, (line, col) in enumerate(rows):
            r = self._text(scr, line, (x, y + k * self.lh), col)

        # blinking cursor after the last line, ~530ms like a terminal
        if (pygame.time.get_ticks() // 530) % 2 == 0:
            scr.fill(TEXT, (r.right + 2, r.y + 2, self.cw, self.lh - 4))

    # ---- Stats panel ----
    def _draw_panel(self, stats, info):
        scr = self.screen
        x0 = WIDTH
        pad = 16
        scr.fill(BG, (x0, 0, PANEL_W, HEIGHT))
        pygame.draw.line(scr, LINE, (x0, 0), (x0, HEIGHT))

        r = self._text(scr, "flappy_rl", (x0 + pad, 12), BRIGHT, f=self.f_bold)
        r = self._text(scr, "/live", (r.right, 12), DIM)
        self._text(scr, f" {info['mode']} ", (r.right + 14, 12), BG, f=self.f_bold, bg=TEXT)
        self._text(scr, f"speed {info['speed']}", (SCREEN_W - pad, 12), DIM, anchor="topright")

        cells = [
            ("episode", f"{stats.episodes + 1:,}"),
            ("score", f"{info['score']:,}"),
            ("best", f"{stats.best:,}"),
            ("last", f"{stats.scores[-1]:,}" if stats.scores else "-"),
            (f"avg{stats.window}", f"{stats.average():.1f}"),
            (f"success>={stats.goal}", f"{stats.success_rate():.0f}%"),
            ("epsilon", f"{info['epsilon']:.4f}"),
            ("states", f"{info['states']:,}"),
        ]
        box = pygame.Rect(x0 + pad, 44, PANEL_W - 2 * pad, self.lh * 4 + 22)
        self._box(scr, box, "status")
        half = (box.w - 20) // 2
        for i, (label, value) in enumerate(cells):
            cx = box.x + 10 + (i % 2) * (half + 10)
            cy = box.y + 12 + (i // 2) * self.lh
            self._text(scr, label, (cx, cy), DIM)
            self._text(scr, value, (cx + half - 10, cy), TEXT, anchor="topright")

        top = box.bottom + 14
        key = info["chart_tick"]
        if self.chart is None or key != self.chart_key:
            self.chart = self._make_charts(stats, PANEL_W - pad * 2, HEIGHT - top - 2 * self.lh - 18)
            self.chart_key = key
        scr.blit(self.chart, (x0 + pad, top))

        keys = [("spc", "pause"), ("up/dn", "speed"), ("v", "visuals"), ("d", "demo"),
                ("s", "save"), ("r", "reset"), ("esc", "quit")]
        x, y = x0 + pad, HEIGHT - 10 - self.lh
        for name, what in keys:
            r = self._text(scr, f" {name} ", (x, y), BG, f=self.f_small, bg=MID)
            r = self._text(scr, " " + what, (r.right, y), DIM, f=self.f_small)
            x = r.right + 10

        if info.get("toast"):
            self._text(scr, f" {info['toast']} ", (WIDTH // 2, 10), BG, anchor="midtop", bg=TEXT)

    def _make_charts(self, stats, w, h):
        surf = pygame.Surface((w, h))
        surf.fill(BG)
        gap = 12
        h1 = int((h - gap) * 0.6)
        h2 = h - gap - h1
        self._chart(surf, pygame.Rect(0, 0, w, h1), "score / episode", stats.scores,
                    [("avg", BRIGHT, None), ("best", MID, (4, 4))], stats.averages,
                    nice_max(stats.best), best=stats.best)
        self._chart(surf, pygame.Rect(0, h1 + gap, w, h2), f"success rate (>= {stats.goal})", None,
                    [("rate", TEXT, None)], stats.rates, 100, percent=True)
        return surf

    def _chart(self, surf, rect, title, points, legend, line, top, best=None, percent=False):
        # push the box down half a row so the title has room to sit on the border
        box = pygame.Rect(rect.x, rect.y + self.lh // 2, rect.w, rect.h - self.lh // 2)
        self._box(surf, box, title)

        # legend goes right to left along the top border: name, then a sample of the line style
        lx = box.right - 10
        for name, col, dash in reversed(legend):
            t = self.f_small.render(f" {name} ", True, DIM, BG)
            lx -= t.get_width()
            surf.blit(t, t.get_rect(midleft=(lx, box.y)))
            lx -= 16
            surf.fill(BG, (lx - 2, box.y - 1, 20, 3))
            dashed_lines(surf, col, [(lx, box.y), (lx + 16, box.y)], dash)
            lx -= 6

        plot = pygame.Rect(box.x + 48, box.y + 16, box.w - 62, box.h - 28 - self.lh)

        for i in range(5):
            v = top * i / 4
            y = int(plot.bottom - plot.h * i / 4)
            for x in range(plot.x, plot.right, 3):
                surf.fill(FAINT, (x, y, 1, 1))
            lab = f"{v:.0f}%" if percent else (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
            self._text(surf, lab, (plot.x - 8, y), DIM, f=self.f_small, anchor="midright")

        n = len(line)
        if n == 0:
            self._text(surf, "waiting for first episode_", plot.center, DIM, f=self.f_small, anchor="center")
            return

        self._text(surf, "1", (plot.x, plot.bottom + 6), DIM, f=self.f_small)
        self._text(surf, f"{n:,}", (plot.right, plot.bottom + 6), DIM, f=self.f_small, anchor="topright")
        self._text(surf, "episode", (plot.centerx, plot.bottom + 6), DIM, f=self.f_small, anchor="midtop")

        def sx(i):
            return plot.x + (plot.w * i / (n - 1) if n > 1 else plot.w / 2)

        def sy(v):
            return plot.bottom - plot.h * min(v, top) / top

        cols = max(1, plot.w)

        # more episodes than pixels -> one min..max bar per pixel column instead of dots
        if points is not None:
            if n <= cols:
                for i, v in enumerate(points):
                    surf.fill(DIM, (int(sx(i)) - 1, int(sy(v)) - 1, 2, 2))
            else:
                for c in range(cols):
                    a = c * n // cols
                    b = max(a + 1, (c + 1) * n // cols)
                    seg = points[a:b]
                    pygame.draw.line(surf, DIM, (plot.x + c, sy(min(seg))), (plot.x + c, sy(max(seg))))

        if best:
            y = sy(best)
            dashed_lines(surf, MID, [(plot.x, y), (plot.right, y)], (4, 4))

        _, col, dash = legend[0]
        if n <= cols:
            pts = [(sx(i), sy(v)) for i, v in enumerate(line)]
        else:
            pts = [(plot.x + c, sy(line[min(n - 1, (c + 1) * n // cols - 1)])) for c in range(cols)]

        if len(pts) > 1:
            if percent:
                # hatch under the curve, 1px line every 3px
                xs, ys = zip(*pts)
                for x in range(int(xs[0]), int(xs[-1]) + 1, 3):
                    pygame.draw.line(surf, FAINT, (x, _interp(x, xs, ys)), (x, plot.bottom))
            dashed_lines(surf, col, pts, dash)
        surf.fill(col, (int(pts[-1][0]) - 1, int(pts[-1][1]) - 1, 3, 3))
