"""Steps 3, 5, 6: train PPO, optionally with your own reward wrapper.

    python train.py                          # built-in reward
    python train.py --reward time            # + small per-frame fee
    python train.py --reward hover           # broken on purpose
    python train.py --reward fixed           # the fix
    python train.py --reward hover --seed 2  # try another run (training is random)

Watch ep_rew_mean in the printout: it should climb. (With a wrapper on, that
number includes your bonus, so judge the model with evaluate.py instead.)
"""
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rewards import REWARDS, make_env

p = argparse.ArgumentParser()
p.add_argument("--reward", choices=REWARDS, default="default")
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# 16 copies of the game run side by side so it learns faster.
env = make_vec_env(lambda: make_env(args.reward), n_envs=16, seed=args.seed)

# Settings from the Hugging Face Deep RL course, Unit 1. They reliably solve it.
model = PPO(
    "MlpPolicy", env,
    n_steps=1024, batch_size=64, n_epochs=4,
    gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    seed=args.seed, verbose=1,
)

start = time.time()
model.learn(total_timesteps=args.steps)
print(f"trained {args.steps:,} steps in {(time.time() - start) / 60:.1f} min")

name = f"ppo_{args.reward}_seed{args.seed}"
model.save(name)
print(f"saved {name}.zip")
