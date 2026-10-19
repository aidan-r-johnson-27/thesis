import math
import random

import pygame

from game import BIRD_R, BIRD_X, GROUND, HEIGHT, PIPE_GAP, PIPE_SPEED, PIPE_W, WIDTH

# ---- Screen layout ----
# The window is the game column on the left plus a stats panel on the right,
# sized so both halves land on whole pixels.
PANEL_W = 560
SCREEN_W = WIDTH + PANEL_W

# ---- Game palette (RGB) ----
SKY_TOP = (78, 173, 214)
SKY_BOTTOM = (178, 227, 238)
PIPE = (94, 186, 72)
PIPE_DARK = (58, 132, 44)
PIPE_LIGHT = (150, 222, 118)
GROUND_TOP = (222, 216, 148)
GROUND_DIRT = (214, 186, 110)
GRASS = (120, 196, 70)
BIRD = (250, 204, 40)
BIRD_DARK = (222, 150, 20)
BEAK = (245, 110, 40)

# ---- Panel palette (dark UI) ----
BG = (17, 20, 27)
CARD = (27, 31, 41)
GRID = (42, 47, 60)
TEXT = (230, 233, 240)
MUTED = (132, 140, 158)
ACCENT = (255, 184, 56)
DOTS = (98, 130, 190)
GOOD = (88, 204, 132)
BAD = (236, 96, 96)


def font(size, bold=False):
    # Build a TTF font, trying nice system fonts first, falling back to the
    # default pygame font (the +6 offsets its smaller default size).
    for name in ("inter", "dejavusans", "liberationsans", "arial"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size + 6)


def nice_max(v):
    # Round an axis maximum up to a "nice" 1/2/2.5/5 x 10^k number so chart
    # gridlines carry clean labels; small values floor at 4 to keep the same
    # tick spacing when a run is still young.
    if v <= 4:
        return 4
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


class Renderer:
    # Owns the window and draws the game view + stats panel each frame.

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy RL")
        self.screen = pygame.display.set_mode((SCREEN_W, HEIGHT))
        self.f_huge = font(48, True)
        self.f_big = font(22, True)
        self.f_med = font(15, True)
        self.f_small = font(12)
        self.f_tiny = font(11)
        self.sky = self._make_sky()
        self.bird = self._make_bird()
        # Clouds drift slowly; the fixed seed keeps the layout consistent.
        rng = random.Random(3)
        self.clouds = [(rng.uniform(0, WIDTH), rng.uniform(40, 300), rng.uniform(0.6, 1.3)) for _ in range(6)]
        self.scroll = 0.0
        self.chart = None       # Cached composite chart surface.
        self.chart_key = None   # Redraw only when this key (episode tick) changes.

    def _make_sky(self):
        # Pre-render the vertical sky-gradient as a reusable surface.
        s = pygame.Surface((WIDTH, GROUND))
        for y in range(GROUND):
            t = y / GROUND
            c = [int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t) for i in range(3)]
            pygame.draw.line(s, c, (0, y), (WIDTH, y))
        return s

    def _make_bird(self):
        # Pre-render the bird sprite (body, wing, eye, beak) once.
        r = BIRD_R
        s = pygame.Surface((r * 4, r * 3), pygame.SRCALPHA)
        cx, cy = r * 2, int(r * 1.5)
        pygame.draw.ellipse(s, BIRD_DARK, (cx - r - 1, cy - r, 2 * r + 4, 2 * r + 1))
        pygame.draw.ellipse(s, BIRD, (cx - r, cy - r, 2 * r + 2, 2 * r - 1))
        pygame.draw.ellipse(s, (255, 244, 200), (cx - r + 1, cy - 1, r + 2, r - 3))
        pygame.draw.ellipse(s, (255, 255, 255), (cx + 2, cy - r + 2, 10, 10))
        pygame.draw.circle(s, (20, 20, 20), (cx + 8, cy - r + 7), 2)
        pygame.draw.polygon(s, BEAK, [(cx + r - 2, cy - 1), (cx + r + 9, cy + 2), (cx + r - 2, cy + 6)])
        return s

    def draw(self, env, stats, info, visuals=True):
        # Draw one frame: game view (or placeholder), stats panel, flip.
        if visuals:
            self._draw_game(env)
        else:
            self._draw_idle(info)
        self._draw_panel(stats, info)
        pygame.display.flip()

    def _draw_cloud(self, x, y, k):
        # Draw a fluffy cloud made of overlapping circles, scaled by k.
        for dx, dy, r in ((0, 0, 18), (18, -8, 22), (38, 0, 18), (20, 6, 18)):
            pygame.draw.circle(self.screen, (245, 250, 252), (int(x + dx * k), int(y + dy * k)), int(r * k))

    def _draw_game(self, env):
        # Draw the actual game frame: sky, clouds, pipes, ground, bird, score.
        scr = self.screen
        scr.set_clip(pygame.Rect(0, 0, WIDTH, HEIGHT))
        scr.blit(self.sky, (0, 0))
        self.scroll = env.frames * PIPE_SPEED
        for x, y, k in self.clouds:
            cx = (x - self.scroll * 0.15 * k) % (WIDTH + 120) - 60
            self._draw_cloud(cx, y, k)

        # Pipes with a light edge, dark rim and a wider "cap" at each end.
        for p in env.pipes:
            x = int(p.x)
            top, bottom = int(p.top), int(p.bottom)
            for rect in (pygame.Rect(x, 0, PIPE_W, top), pygame.Rect(x, bottom, PIPE_W, GROUND - bottom)):
                pygame.draw.rect(scr, PIPE, rect)
                pygame.draw.rect(scr, PIPE_LIGHT, (rect.x + 6, rect.y, 8, rect.h))
                pygame.draw.rect(scr, PIPE_DARK, (rect.right - 10, rect.y, 10, rect.h))
            for cy in (top - 24, bottom):
                cap = pygame.Rect(x - 5, cy, PIPE_W + 10, 24)
                pygame.draw.rect(scr, PIPE, cap)
                pygame.draw.rect(scr, PIPE_LIGHT, (cap.x + 6, cap.y, 8, cap.h))
                pygame.draw.rect(scr, PIPE_DARK, cap, 3)

        # Guide line showing where the bird aims at the next gap.
        target = env.next_pipe()
        line = pygame.Surface((WIDTH, GROUND), pygame.SRCALPHA)
        pygame.draw.line(line, (255, 255, 255, 90), (BIRD_X, int(env.y)), (int(target.x + PIPE_W), int(target.gap_y)), 2)
        pygame.draw.circle(line, (255, 255, 255, 120), (int(target.x + PIPE_W), int(target.gap_y)), 5)
        scr.blit(line, (0, 0))

        # Scrolling ground with grass tufts that track the world scroll.
        pygame.draw.rect(scr, GROUND_DIRT, (0, GROUND, WIDTH, HEIGHT - GROUND))
        pygame.draw.rect(scr, GRASS, (0, GROUND, WIDTH, 12))
        off = int(self.scroll) % 24
        for sx in range(-off, WIDTH + 24, 24):
            pygame.draw.polygon(scr, (98, 170, 56), [(sx, GROUND + 12), (sx + 12, GROUND + 12), (sx + 6, GROUND)])
        pygame.draw.line(scr, (90, 70, 40), (0, GROUND), (WIDTH, GROUND), 2)
        pygame.draw.rect(scr, GROUND_TOP, (0, GROUND + 12, WIDTH, 6))

        # Bird tilts with velocity; a flap shows a little "wing puff" dot.
        angle = max(-30, min(80, env.vel * 6))
        bird = pygame.transform.rotate(self.bird, -angle)
        scr.blit(bird, bird.get_rect(center=(BIRD_X, int(env.y))))
        if env.last_action == 1:
            pygame.draw.circle(scr, (255, 255, 255), (BIRD_X - BIRD_R - 6, int(env.y) + 4), 3)

        # Score with a drop shadow.
        txt = self.f_huge.render(str(env.score), True, (255, 255, 255))
        sh = self.f_huge.render(str(env.score), True, (40, 40, 40))
        r = txt.get_rect(midtop=(WIDTH // 2, 28))
        scr.blit(sh, r.move(3, 3))
        scr.blit(txt, r)
        scr.set_clip(None)

    def _draw_idle(self, info):
        # Placeholder screen shown when visuals are toggled off (V).
        scr = self.screen
        pygame.draw.rect(scr, (12, 14, 19), (0, 0, WIDTH, HEIGHT))
        t = pygame.time.get_ticks() / 1000
        # Three pulsing dots give a "still alive" cue.
        for i in range(3):
            a = int(120 + 120 * math.sin(t * 4 - i * 0.7))
            pygame.draw.circle(scr, (a, a // 2 + 60, 40), (WIDTH // 2 - 24 + i * 24, HEIGHT // 2 - 40), 7)
        self._center(self.f_big, "Visuals off", TEXT, HEIGHT // 2)
        self._center(self.f_small, "training at full speed", MUTED, HEIGHT // 2 + 28)
        self._center(self.f_small, f"{info['sps']:,.0f} steps/s", MUTED, HEIGHT // 2 + 48)
        self._center(self.f_small, "press V to watch again", MUTED, HEIGHT // 2 + 90)

    def _center(self, f, text, color, y, x=WIDTH // 2):
        # Blit `text` centred at (x, y) on the screen.
        s = f.render(text, True, color)
        self.screen.blit(s, s.get_rect(center=(x, y)))

    def _draw_panel(self, stats, info):
        # Right-hand column: title + mode badge, stat cards, charts, help.
        scr = self.screen
        x0 = WIDTH
        pygame.draw.rect(scr, BG, (x0, 0, PANEL_W, HEIGHT))
        pad = 18
        title = self.f_big.render("Training", True, TEXT)
        scr.blit(title, (x0 + pad, 14))
        mode = info["mode"]
        mcol = {"TRAINING": GOOD, "PAUSED": ACCENT, "DEMO": DOTS}.get(mode, MUTED)
        badge = self.f_tiny.render(mode, True, BG)
        br = badge.get_rect().inflate(14, 6)
        br.topleft = (x0 + pad + title.get_width() + 12, 19)
        pygame.draw.rect(scr, mcol, br, border_radius=8)
        scr.blit(badge, badge.get_rect(center=br.center))
        sp = self.f_small.render(f"speed {info['speed']}", True, MUTED)
        scr.blit(sp, sp.get_rect(topright=(SCREEN_W - pad, 22)))

        # 2x4 grid of small stat cards.
        cards = [
            ("Episode", f"{stats.episodes + 1:,}"),
            ("Score", f"{info['score']:,}"),
            ("Best", f"{stats.best:,}"),
            ("Last", f"{stats.scores[-1]:,}" if stats.scores else "-"),
            (f"Avg ({stats.window})", f"{stats.average():.1f}"),
            (f"Success ≥{stats.goal}", f"{stats.success_rate():.0f}%"),
            ("Epsilon", f"{info['epsilon']:.4f}"),
            ("States", f"{info['states']:,}"),
        ]
        cw = (PANEL_W - pad * 2 - 3 * 8) // 4
        for i, (label, value) in enumerate(cards):
            cx = x0 + pad + (i % 4) * (cw + 8)
            cy = 50 + (i // 4) * 58
            pygame.draw.rect(scr, CARD, (cx, cy, cw, 50), border_radius=8)
            scr.blit(self.f_tiny.render(label, True, MUTED), (cx + 10, cy + 7))
            scr.blit(self.f_med.render(value, True, TEXT), (cx + 10, cy + 24))

        # Charts are rebuilt only when a new episode arrived.
        key = info["chart_tick"]
        if self.chart is None or key != self.chart_key:
            self.chart = self._make_charts(stats, PANEL_W - pad * 2, 380)
            self.chart_key = key
        scr.blit(self.chart, (x0 + pad, 172))

        help_text = "SPACE pause   ↑/↓ speed   V visuals   D demo   S save   R reset   ESC quit"
        h = self.f_tiny.render(help_text, True, MUTED)
        scr.blit(h, h.get_rect(midbottom=(x0 + PANEL_W // 2, HEIGHT - 10)))
        if info.get("toast"):
            t = self.f_small.render(info["toast"], True, BG)
            tr = t.get_rect().inflate(18, 8)
            tr.midbottom = (WIDTH // 2, HEIGHT - 72)
            pygame.draw.rect(scr, ACCENT, tr, border_radius=10)
            scr.blit(t, t.get_rect(center=tr.center))

    def _make_charts(self, stats, w, h):
        # Composite the two chart panels (score-per-episode, success rate).
        surf = pygame.Surface((w, h))
        surf.fill(BG)
        gap = 14
        h1 = int((h - gap) * 0.62)
        h2 = h - gap - h1
        top = nice_max(stats.best)
        self._chart(surf, pygame.Rect(0, 0, w, h1), "Score per episode", stats.scores, stats.averages, top,
                    [("score", DOTS), (f"avg of last {stats.window}", ACCENT), ("best", GOOD)], stats.best)
        self._chart(surf, pygame.Rect(0, h1 + gap, w, h2), f"Success rate  (score ≥ {stats.goal}, last {stats.window})",
                    None, stats.rates, 100, [("success %", GOOD)], None, percent=True)
        return surf

    def _chart(self, surf, rect, title, points, line, top, legend, best, percent=False):
        # `points` are individual per-episode dots, downsampled to one
        # vertical bar per pixel column once they outnumber the width;
        # `line` is the rolling average (or success rate) drawn connected;
        # `best` marks the best score as a dashed guideline.
        pygame.draw.rect(surf, CARD, rect, border_radius=10)
        surf.blit(self.f_small.render(title, True, TEXT), (rect.x + 12, rect.y + 9))
        # Legend badges, right-aligned.
        lx = rect.right - 12
        for name, col in reversed(legend):
            t = self.f_tiny.render(name, True, MUTED)
            lx -= t.get_width()
            surf.blit(t, (lx, rect.y + 11))
            lx -= 14
            pygame.draw.rect(surf, col, (lx, rect.y + 15, 9, 4), border_radius=2)
            lx -= 12
        plot = pygame.Rect(rect.x + 46, rect.y + 34, rect.w - 60, rect.h - 58)
        # Horizontal gridlines with "nice" labels.
        for i in range(5):
            v = top * i / 4
            y = plot.bottom - plot.h * i / 4
            pygame.draw.line(surf, GRID, (plot.x, y), (plot.right, y))
            lab = f"{v:.0f}%" if percent else (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
            t = self.f_tiny.render(lab, True, MUTED)
            surf.blit(t, t.get_rect(midright=(plot.x - 6, y)))
        n = len(line)
        if n == 0:
            self._center_on(surf, "waiting for first episode…", plot.center)
            return
        # Episode axis labels (sparse, to avoid clutter).
        for i in range(5):
            ep = 1 + int((n - 1) * i / 4) if n > 1 else 1
            x = plot.x + plot.w * i / 4
            t = self.f_tiny.render(f"{ep:,}", True, MUTED)
            anchor = {0: "topleft", 4: "topright"}.get(i, "midtop")
            surf.blit(t, t.get_rect(**{anchor: (x, plot.bottom + 5)}))

        # X/Y mapping helpers.
        def sx(i):
            return plot.x + (plot.w * i / (n - 1) if n > 1 else plot.w / 2)

        def sy(v):
            return plot.bottom - plot.h * min(v, top) / top

        cols = max(1, plot.w)
        # Per-episode scatter; once denser than the width, draw one vertical
        # "min..max" bar per pixel column instead.
        if points is not None:
            if n <= cols:
                for i, v in enumerate(points):
                    pygame.draw.circle(surf, DOTS, (sx(i), sy(v)), 2 if n < 300 else 1)
            else:
                for c in range(cols):
                    a = c * n // cols
                    b = max(a + 1, (c + 1) * n // cols)
                    seg = points[a:b]
                    x = plot.x + c
                    pygame.draw.line(surf, DOTS, (x, sy(min(seg))), (x, sy(max(seg))))
        # Dashed "best score" guideline.
        if best is not None and best > 0:
            y = sy(best)
            for x in range(plot.x, plot.right, 8):
                pygame.draw.line(surf, GOOD, (x, y), (min(x + 4, plot.right), y), 2)
        # Rolling average / success-rate line, downsampled if needed.
        col = legend[1][1] if points is not None else legend[0][1]
        if n <= cols:
            pts = [(sx(i), sy(v)) for i, v in enumerate(line)]
        else:
            pts = [(plot.x + c, sy(line[min(n - 1, (c + 1) * n // cols - 1)])) for c in range(cols)]
        if len(pts) > 1:
            if percent:
                # Soft translucent fill under the success-rate curve.
                fill = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                pygame.draw.polygon(fill, (*col, 40), [(pts[0][0], plot.bottom), *pts, (pts[-1][0], plot.bottom)])
                surf.blit(fill, (0, 0))
            pygame.draw.lines(surf, col, False, pts, 2)
        pygame.draw.circle(surf, col, pts[-1], 4)

    def _center_on(self, surf, text, pos):
        # Blit `text` centred at `pos` on a chart surface.
        t = self.f_small.render(text, True, MUTED)
        surf.blit(t, t.get_rect(center=pos))