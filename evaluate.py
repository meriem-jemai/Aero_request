
"""Evaluate a trained agent: compare with random play, plot the reward curve, record a GIF."""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import Aerorequest  # noqa: F401  (registers the environments)

ENV_ID = "Aerorequest_features_v0"


def run_episode(env, policy, seed, record=False, max_frames=240):
    """Play one episode. Returns (targets reached, crashed, frames)."""
    obs, _ = env.reset(seed=seed)
    frames, steps = [], 0
    while True:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        steps += 1
        if record and steps % 2 == 0 and len(frames) < max_frames:
            frames.append(env.render()[::2, ::2])  # half resolution keeps the GIF small
        if terminated or truncated:
            return env.unwrapped.score // 10, terminated, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/ppo")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = PPO.load(os.path.join(args.run, "model"))
    vecnorm = VecNormalize.load(
        os.path.join(args.run, "vecnormalize.pkl"), DummyVecEnv([lambda: gym.make(ENV_ID)])
    )
    vecnorm.training = False
    env = gym.make(ENV_ID, render_mode="rgb_array")

    def agent(obs):
        action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        return int(action)

    def random_policy(obs):
        return env.action_space.sample()

    # 1) Compare trained agent with random play
    for name, policy in [("PPO agent", agent), ("Random", random_policy)]:
        targets, crashes = [], 0
        for i in range(args.episodes):
            reached, crashed, _ = run_episode(env, policy, seed=1000 + i)
            targets.append(reached)
            crashes += int(crashed)
        print(f"{name:10s} {np.mean(targets):5.1f} ± {np.std(targets):4.1f} targets per episode, "
              f"crashed in {crashes}/{args.episodes} episodes")

    # 2) Reward curve from training logs
    df = load_results(os.path.join(args.run, "monitor"))
    plt.figure(figsize=(7, 4))
    plt.plot(df["l"].cumsum(), df["r"].rolling(100, min_periods=1).mean())
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward (rolling mean)")
    plt.title("PPO on Aerorequest_features_v0")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "reward_curve.png"), dpi=150)

    # 3) GIF of the trained agent
    _, _, frames = run_episode(env, agent, seed=0, record=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(os.path.join(args.out, "agent.gif"), save_all=True,
                   append_images=images[1:], duration=83, loop=0)

    print("Saved reward_curve.png and agent.gif to", args.out)


if __name__ == "__main__":
    main()