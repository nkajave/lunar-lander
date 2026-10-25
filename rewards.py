"""Reward wrappers: thin layers between the game and PPO.

Each one takes the built-in reward, adds its own terms, and passes the total on.
Pick one by name with --reward when training (see REWARDS at the bottom).
"""
import math

import gymnasium as gym


def upright_and_slow(obs):
    """True when the rocket is nearly level and barely moving."""
    vx, vy, angle = obs[2], obs[3], obs[4]
    return abs(angle) < 0.1 and math.hypot(vx, vy) < 0.2


class TimePenalty(gym.Wrapper):
    """Step 5: a tiny charge every frame, so dawdling costs something."""

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverBonus(gym.Wrapper):
    """Step 6, the BROKEN version: +1 for every frame upright and slow.

    A rocket hanging in the air can farm up to 1,000 points before the
    1,000-frame cutoff, and landing ends the payments. Expect hovering.
    """

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if upright_and_slow(obs):
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class FixedBonus(gym.Wrapper):
    """Step 6, the FIX: pay for *getting* upright and slow, not for staying there.

    The bonus is the change in a 0/1 "upright and slow" score between frames
    (like the built-in reward does with distance and speed). Summed over an
    attempt it can never exceed 1, so there's nothing to farm. A small fee per
    frame makes waiting cost something.
    """

    def __init__(self, env, bonus=1.0, fee=0.05):
        super().__init__(env)
        self.bonus, self.fee = bonus, fee
        self.prev = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = float(upright_and_slow(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = float(upright_and_slow(obs))
        reward += self.bonus * (now - self.prev) - self.fee
        self.prev = now
        return obs, reward, terminated, truncated, info


REWARDS = {
    "default": None,  # built-in reward only
    "time": TimePenalty,
    "hover": HoverBonus,
    "fixed": FixedBonus,
}


def make_env(reward="default", render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    wrapper = REWARDS[reward]
    return wrapper(env) if wrapper else env
