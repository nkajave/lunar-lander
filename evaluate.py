"""Step 4 and "How you know it works": score a model, watch it, record it.

    python evaluate.py ppo_default_seed0              # score over 20 attempts
    python evaluate.py ppo_default_seed0 --watch      # open the game window
    python evaluate.py ppo_hover_seed0 --record       # save MP4 clips to videos/

Scoring always uses the plain game (wrapper OFF), so your bonus can't flatter it.
An average of 200+ means solved. "Frames" and "landed" are how you spot
hovering: a hover-hacker runs near 1,000 frames and rarely lands.
"""
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--watch", action="store_true")
p.add_argument("--record", action="store_true")
args = p.parse_args()

model = PPO.load(args.model)

if args.record:
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, "videos", name_prefix=args.model.removesuffix(".zip"),
        episode_trigger=lambda i: i < 3,  # first 3 attempts
    )
else:
    env = gym.make("LunarLander-v3", render_mode="human" if args.watch else None)

scores, lengths, landed = [], [], 0
for i in range(args.episodes):
    obs, _ = env.reset(seed=1000 + i)
    total, frames, done = 0.0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        frames += 1
        done = terminated or truncated
    # Landing ends the game with the final +100; a crash ends it with -100.
    if terminated and reward > 0:
        landed += 1
    scores.append(total)
    lengths.append(frames)
    print(f"attempt {i + 1:2d}: score {total:7.1f}  frames {frames:4d}")

env.close()
print(f"\naverage score {np.mean(scores):.1f} ± {np.std(scores):.1f}"
      f"  (200+ = solved)")
print(f"average frames {np.mean(lengths):.0f}   landed {landed}/{args.episodes}")
