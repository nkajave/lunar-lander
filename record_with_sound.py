"""Record a model's attempts as MP4s WITH sound.

Lunar Lander has no audio, so this script makes its own from what happens
each frame: a rumble while the main engine fires, a hiss for the side
engines, a thud when a leg touches down, a chime for a safe landing and an
explosion for a crash.

    python record_with_sound.py ppo_default_seed0
    python record_with_sound.py ppo_hover_seed0 --episodes 2

Clips go to videos/<model>-sound-<n>.mp4. Needs nothing beyond requirements:
the ffmpeg program comes bundled with moviepy (via imageio-ffmpeg).
"""
import argparse
import os
import subprocess
import wave

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
from stable_baselines3 import PPO

RATE = 44_100          # audio samples per second
FPS = 50               # Lunar Lander runs at 50 frames a second
SPF = RATE // FPS      # audio samples per game frame (882)
TAIL = FPS * 2         # hold the last picture 2 s so the ending sound plays out
rng = np.random.default_rng(0)


def lowpass(x, strength):
    """Simple smoothing filter: higher strength = deeper, muffled sound."""
    y = np.empty_like(x)
    acc = 0.0
    a = 1.0 / strength
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def envelope(on_per_frame, attack=0.3):
    """Turn a per-frame on/off list into a smooth per-sample volume curve."""
    target = np.repeat(np.asarray(on_per_frame, float), SPF)
    return lowpass(target, strength=SPF * attack)


def tone(freq, seconds, decay):
    t = np.arange(int(RATE * seconds)) / RATE
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)


def make_audio(actions, touchdowns, ending, n_frames):
    n = n_frames * SPF
    out = np.zeros(n)

    # Main engine: deep rumbling noise while action == 2.
    main = [1.0 if a == 2 else 0.0 for a in actions] + [0.0] * (n_frames - len(actions))
    rumble = lowpass(rng.standard_normal(n), 25)
    out += 2.5 * rumble * envelope(main)

    # Side engines: lighter hiss while action is 1 or 3.
    side = [1.0 if a in (1, 3) else 0.0 for a in actions] + [0.0] * (n_frames - len(actions))
    hiss = lowpass(rng.standard_normal(n), 4)
    out += 0.12 * hiss * envelope(side, attack=0.15)

    # Leg touchdowns: short low thud.
    thud = tone(90, 0.25, 18)
    for f in touchdowns:
        s = f * SPF
        out[s:s + len(thud)] += 0.5 * thud[: n - s]

    # Ending: chime for a landing, explosion for a crash.
    s = len(actions) * SPF
    if ending == "landed":
        chime = np.concatenate([tone(660, 0.18, 6), tone(880, 0.6, 4)])
        out[s:s + len(chime)] += 0.35 * chime[: n - s]
    elif ending == "crashed":
        length = min(int(RATE * 1.5), n - s)
        t = np.arange(length) / RATE
        boom = lowpass(rng.standard_normal(length), 12) * np.exp(-t * 3)
        out[s:s + length] += 3.0 * boom + 0.6 * tone(55, length / RATE, 4)

    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9 * 32767).astype(np.int16)


def record(model, env, seed, path):
    obs, _ = env.reset(seed=seed)
    frames, actions, touchdowns = [env.render()], [], []
    legs_before = (0, 0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(int(action))
        frames.append(env.render())
        legs = (int(obs[6]), int(obs[7]))
        if any(now and not before for now, before in zip(legs, legs_before)):
            touchdowns.append(len(actions))
        legs_before = legs
        done = terminated or truncated

    if terminated:
        ending = "landed" if reward > 0 else "crashed"
    else:
        ending = "timed out"
    frames += [frames[-1]] * TAIL

    wav_path = path.replace(".mp4", ".wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(make_audio(actions, touchdowns, ending, len(frames)).tobytes())

    h, wd, _ = frames[0].shape
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wd}x{h}", "-r", str(FPS), "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(wav_path)
    return ending, len(actions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    model = PPO.load(args.model)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    name = os.path.basename(args.model).removesuffix(".zip")
    os.makedirs("videos", exist_ok=True)
    for i in range(args.episodes):
        path = f"videos/{name}-sound-{i}.mp4"
        ending, frames = record(model, env, seed=1000 + i, path=path)
        print(f"{path}: {ending} after {frames} frames")
    env.close()
