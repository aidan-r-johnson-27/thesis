import argparse
import os
import time

from agent import QAgent, Stats, load, save
from game import FlappyEnv, MAX_SCORE

# Selectable simulation speeds in frames per redraw; 0 means "max" (as fast
# as the machine allows) and is only reachable via the speed-up keys.
SPEEDS = [1, 2, 4, 8, 16, 32, 100, 500, 0]
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Flappy Bird reinforcement learning")
    p.add_argument("--headless", action="store_true", help="train without a window")
    p.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = forever in visual mode, 5000 headless)")
    p.add_argument("--demo", action="store_true", help="start in demo mode (greedy, no learning)")
    p.add_argument("--fresh", action="store_true", help="ignore the saved model and start from scratch")
    p.add_argument("--model", default=os.path.join(HERE, "model.json"), help="where to load/save the model")
    p.add_argument("--goal", type=int, default=50, help="score that counts as a success")
    p.add_argument("--max-score", type=int, default=MAX_SCORE, help="end an episode once this score is reached")
    p.add_argument("--speed", type=int, default=1, choices=[s for s in SPEEDS if s], help="starting simulation speed")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def setup(args):
    # Fresh env and agent; unless --fresh, fold the saved model + stats in.
    agent = QAgent()
    stats = Stats(goal=args.goal)
    loaded = False
    if not args.fresh:
        loaded = load(args.model, agent, stats)
    return FlappyEnv(seed=args.seed, max_score=args.max_score), agent, stats, loaded


def finish_episode(env, agent, stats, learning):
    # Wrap up an episode: in learning mode the trajectory is replayed into
    # the table and the score recorded, in demo mode it is thrown away.
    # Either way the env is reset for the next episode.
    if learning:
        agent.learn()
        stats.add(env.score)
    else:
        agent.discard()
    env.reset()


def run_headless(args):
    # Train `--episodes` episodes with no window: a progress line every 100
    # episodes, an autosave every 500, Ctrl+C to stop early.
    env, agent, stats, loaded = setup(args)
    total = args.episodes or 5000
    print(f"{'resumed' if loaded else 'new'} model: {args.model} ({stats.episodes} episodes so far)")
    print(f"training {total} episodes headless, ctrl+c to stop early")
    start = time.time()
    steps = 0
    target = stats.episodes + total
    state = env.reset()
    try:
        while stats.episodes < target:
            action = agent.act(state)
            nxt, reward, done = env.step(action)
            agent.remember(state, action, reward, nxt, done)
            state = nxt
            steps += 1
            if done:
                finish_episode(env, agent, stats, True)
                state = env.state()
                if stats.episodes % 100 == 0:
                    rate = steps / max(1e-9, time.time() - start)
                    print(f"ep {stats.episodes:>7,}  avg {stats.average():>7.1f}  best {stats.best:>5}  "
                          f"success {stats.success_rate():>5.1f}%  states {len(agent.q):>6,}  {rate:>9,.0f} steps/s")
                if stats.episodes % 500 == 0:
                    save(args.model, agent, stats)
    except KeyboardInterrupt:
        print("\nstopped")
    save(args.model, agent, stats)
    print(f"saved {args.model}  episodes {stats.episodes:,}  best {stats.best}  avg {stats.average():.1f}")
    print("watch it with: ./run.sh")


def run_visual(args):
    # Windowed mode: the game view plus the live stats panel. pygame is
    # imported lazily so the headless path never depends on a display.
    import pygame

    from render import Renderer

    env, agent, stats, loaded = setup(args)
    ui = Renderer()
    clock = pygame.time.Clock()
    # Loop state: `speed_i` indexes SPEEDS (arrow keys walk it), `visuals`
    # is the V toggle for the game view, `learning` flips demo/training on
    # D, and chart_tick only bumps when a fresh episode lands so the cached
    # chart surface knows it's time to redraw.
    speed_i = SPEEDS.index(args.speed)
    visuals = True
    paused = False
    learning = not args.demo
    toast, toast_until = ("resumed saved model" if loaded else ""), time.time() + 2.5
    chart_tick = 0
    last_chart = 0.0
    chart_eps = -1
    sps, sps_steps, sps_time = 0.0, 0, time.time()
    target = stats.episodes + args.episodes if args.episodes else None
    state = env.state()
    running = True

    def notify(msg):
        # Corner toast that lingers for 2 seconds.
        nonlocal toast, toast_until
        toast, toast_until = msg, time.time() + 2.0

    while running:
        # ---- Input ----
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_EQUALS, pygame.K_PLUS):
                    speed_i = min(len(SPEEDS) - 1, speed_i + 1)
                elif e.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                    speed_i = max(0, speed_i - 1)
                elif e.key == pygame.K_v:
                    visuals = not visuals
                elif e.key == pygame.K_d:
                    learning = not learning
                    agent.discard()
                    env.reset()
                    state = env.state()
                    notify("demo: best policy, no learning" if not learning else "training resumed")
                elif e.key == pygame.K_s:
                    save(args.model, agent, stats)
                    notify("saved")
                elif e.key == pygame.K_r:
                    env = FlappyEnv(seed=args.seed, max_score=args.max_score)
                    agent = QAgent()
                    stats = Stats(goal=args.goal)
                    learning = True
                    state = env.state()
                    chart_tick = 0
                    chart_eps = -1
                    last_chart = 0.0
                    sps, sps_steps, sps_time = 0.0, 0, time.time()
                    notify("training reset")

        # ---- Simulation ----
        # With visuals off the speed is effectively "max" (0), so each loop
        # iteration advances as many frames as fit in a 14 ms budget.
        if not paused:
            speed = SPEEDS[speed_i] if visuals else 0
            deadline = time.perf_counter() + 0.014
            n = 0
            # Step frames until one of the stops trips: the requested stepped
            # speed, a whole episode at the slowest speeds (keeps the view
            # from stalling mid-run), the 14 ms budget under "max", or a
            # reached episode target.
            while True:
                action = agent.act(state, greedy=not learning)
                nxt, reward, done = env.step(action)
                if learning:
                    agent.remember(state, action, reward, nxt, done)
                state = nxt
                n += 1
                if done:
                    finish_episode(env, agent, stats, learning)
                    state = env.state()
                    if learning and stats.episodes % 100 == 0:
                        save(args.model, agent, stats)
                    if target and stats.episodes >= target:
                        running = False
                        break
                    if speed and speed <= 4:
                        break
                if speed and n >= speed:
                    break
                if time.perf_counter() > deadline:
                    break
            sps_steps += n

        # ---- Status refresh ----
        # Steps/second is rolled over a half-second window, and the chart
        # redraw flag only fires once per fresh episode with a small debounce.
        now = time.time()
        if now - sps_time >= 0.5:
            sps = sps_steps / (now - sps_time)
            sps_steps, sps_time = 0, now
        if stats.episodes != chart_eps and now - last_chart > 0.1:
            chart_tick += 1
            chart_eps = stats.episodes
            last_chart = now

        mode = "PAUSED" if paused else ("TRAINING" if learning else "DEMO")
        speed_label = "max" if not visuals or SPEEDS[speed_i] == 0 else f"{SPEEDS[speed_i]}x"
        info = {
            "mode": mode,
            "speed": speed_label,
            "score": env.score,
            "epsilon": agent.epsilon,
            "states": len(agent.q),
            "sps": sps,
            "chart_tick": chart_tick,
            "toast": toast if now < toast_until else "",
        }
        ui.draw(env, stats, info, visuals)
        clock.tick(60 if visuals else 0)

    save(args.model, agent, stats)
    pygame.quit()
    print(f"saved {args.model}  episodes {stats.episodes:,}  best {stats.best}")


def main():
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_visual(args)


if __name__ == "__main__":
    main()