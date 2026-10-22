import argparse

import matplotlib

# draw straight to a file, no window needed
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from boids_rl.config import add_config_args
from boids_rl.env import BoidsEnv, polarization
from boids_rl.eval import load_run, run_episode


def record_episode(model, cfg, seed):
    env = BoidsEnv(cfg.env, cfg.reward, seed=seed, auto_reset=False)
    frames = []
    for _ in run_episode(model, env):
        frames.append((env.pos.copy(), env.vel.copy(), env.pred_pos.copy()))
    return frames


def render_episode(model, cfg, path, seed=None, fps=20):
    if seed is None:
        seed = cfg.eval.seed
    frames = record_episode(model, cfg, seed)
    size = cfg.env.world_size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    pos, vel, pred_pos = frames[0]
    arrows = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color="tab:blue",
                       angles="xy", scale_units="xy", scale=cfg.env.max_speed / 1.5, width=0.006)
    predators = ax.scatter(pred_pos[:, 0], pred_pos[:, 1], s=90, color="tab:red", marker="D")
    title = ax.set_title("")

    def draw_frame(i):
        pos, vel, pred_pos = frames[i]
        arrows.set_offsets(pos)
        arrows.set_UVC(vel[:, 0], vel[:, 1])
        predators.set_offsets(pred_pos)
        title.set_text(f"step {i + 1}/{len(frames)}   polarization {polarization(vel):.2f}")
        return arrows, predators, title

    anim = FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / fps, blit=False)
    if path.endswith(".mp4"):
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate one deterministic episode.")
    parser.add_argument("run_dir")
    parser.add_argument("out", help="output path, .gif or .mp4 (mp4 needs ffmpeg)")
    parser.add_argument("--checkpoint", help="default: <run_dir>/checkpoints/latest.pt")
    parser.add_argument("--seed", type=int, help="episode seed (default: eval.seed from the config)")
    parser.add_argument("--fps", type=int, default=20)
    add_config_args(parser)
    args = parser.parse_args()
    cfg, model = load_run(args.run_dir, args.checkpoint, args)
    render_episode(model, cfg, args.out, seed=args.seed, fps=args.fps)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
