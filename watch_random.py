"""Step 2: watch a rocket with random actions tumble and crash.

    python watch_random.py
"""
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)
names = ["x", "y", "vx", "vy", "angle", "spin", "leg_L", "leg_R"]

for attempt in range(3):
    total, done = 0.0, False
    while not done:
        action = env.action_space.sample()  # 0 nothing, 1 left, 2 main, 3 right
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"attempt {attempt + 1}: score {total:.1f}")
    print("  last observation:", {n: round(float(v), 2) for n, v in zip(names, obs)})
    obs, info = env.reset()

env.close()
