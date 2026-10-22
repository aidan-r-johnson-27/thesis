import argparse
import json
from dataclasses import asdict, dataclass, field, fields

# every field below is also a command line flag: --<section>.<field>, e.g. --env.n_agents 32


@dataclass
class EnvConfig:
    n_agents: int = 60             # boids per world
    world_size: float = 64.0        # side of the square world, which wraps around at the edges
    dt: float = 0.1
    max_speed: float = 2.0
    min_speed: float = 0.5          # boids can't hover; 0 turns this off
    max_accel: float = 2.0          # actions are clipped to this
    k_neighbors: int = 4            # how many neighbours each boid observes
    neighbor_radius: float = 4.0    # how far a boid can see
    max_steps: int = 200            # world episode length, then it's cut off

    # predators are scripted, not learned: each one chases the nearest boid
    n_predators: int = 16           # one per 16 boids at the default flock size
    predator_max_speed: float = 2.4  # faster than a boid, so a lone boid can't just outrun it
    predator_max_accel: float = 3.0
    catch_radius: float = 0.5        # attacks once its target is this close
    predator_cooldown: int = 20      # steps it waits after every attack, hit or miss
    # confusion effect: an attack hits with probability 1 / (1 + confusion * m), where m is how
    # many other boids are within confusion_radius of the target; 0 turns it off
    confusion: float = 1.0
    confusion_radius: float = 2.0

    collision_radius: float = 0.25  # two boids closer than this both die; 0 turns collisions off


@dataclass
class RewardConfig:
    # reward = sum of w_<term> * term (the terms are in rewards.py); a weight of 0 removes that term
    # the default task is just survival, cohesion/alignment are only for the "reynolds" ablation
    w_caught: float = 1.0
    w_collision: float = 1.0
    w_cohesion: float = 0.0
    w_alignment: float = 0.0


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 4096      # a rollout is rollout_steps * n_envs * n_agents (131k) rows
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    init_log_std: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 0
    total_env_steps: int = 400_000  # summed over envs; every env step gives n_agents samples
    rollout_steps: int = 128        # steps per env per rollout
    n_envs: int = 4
    run_dir: str = "runs/default"
    checkpoint_every: int = 50      # in PPO updates


@dataclass
class EvalConfig:
    n_episodes: int = 20
    seed: int = 10_000              # far from the training seeds so eval episodes are new ones


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, saved):
        cfg = cls()
        for section in fields(cls):
            section_cfg = getattr(cfg, section.name)
            for name, value in saved.get(section.name, {}).items():
                # skip settings that don't exist anymore so old config.json files still load
                if hasattr(section_cfg, name):
                    setattr(section_cfg, name, value)
        return cfg

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class ExperimentConfig:
    # settings for experiments.py
    name: str = "ablation"          # results go to <results_dir>/<name>/
    results_dir: str = "results"
    ablations: tuple = ("survival", "no_confusion", "no_predator", "no_collisions", "reynolds")
    seeds: tuple = (0, 1, 2)
    workers: int = 1                # runs trained in parallel
    eval_flock_sizes: tuple = (8, 32, 64)  # extra evals at other flock sizes, to test generalization


# each ablation changes a few "<section>.<field>" settings and leaves the rest as the base config
ABLATIONS = {
    "survival": {},
    "no_confusion": {"env.confusion": 0.0},
    "no_predator": {"env.n_predators": 0},
    "no_collisions": {"env.collision_radius": 0.0},
    # the original setup: no predator, no deaths, reward is cohesion + alignment
    "reynolds": {"env.n_predators": 0, "env.collision_radius": 0.0, "reward.w_caught": 0.0,
                 "reward.w_collision": 0.0, "reward.w_cohesion": 0.5, "reward.w_alignment": 0.5},
}


def apply_overrides(cfg, overrides):
    # overrides look like {"env.n_agents": 32}
    for key, value in overrides.items():
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, value)
    return cfg


def parse_bool(text):
    if text.lower() in ("1", "true", "yes"):
        return True
    if text.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def add_dataclass_args(parser, prefix, obj):
    group = parser.add_argument_group(prefix)
    for f in fields(obj):
        default = getattr(obj, f.name)
        flag = f"--{prefix}.{f.name}"
        help_text = f"default: {default}"
        # defaults are None so I can tell later which flags were actually given
        if isinstance(default, tuple):
            item_type = type(default[0]) if default else str
            group.add_argument(flag, type=item_type, nargs="+", default=None, help=help_text)
        elif isinstance(default, bool):
            group.add_argument(flag, type=parse_bool, default=None, help=help_text)
        else:
            group.add_argument(flag, type=type(default), default=None, help=help_text)


def apply_dataclass_args(obj, args, prefix):
    for f in fields(obj):
        value = getattr(args, f"{prefix}.{f.name}", None)
        if value is None:
            continue
        if isinstance(value, list):
            value = tuple(value)
        setattr(obj, f.name, value)
    return obj


def add_config_args(parser):
    defaults = Config()
    for section in fields(Config):
        add_dataclass_args(parser, section.name, getattr(defaults, section.name))


def apply_config_args(cfg, args):
    for section in fields(Config):
        apply_dataclass_args(getattr(cfg, section.name), args, section.name)
    return cfg
