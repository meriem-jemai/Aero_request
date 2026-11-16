"""Train a PPO agent on Aero_request (feature observations)."""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import Aerorequest  # noqa: F401  (registers the environments)

ENV_ID = "Aerorequest_features_v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/ppo")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = make_vec_env(
        ENV_ID,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=os.path.join(args.out, "monitor"),
    )
    # Positions are in pixels (0-650), so normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MultiInputPolicy", env, n_steps=1024, batch_size=256, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    model.save(os.path.join(args.out, "model"))
    env.save(os.path.join(args.out, "vecnormalize.pkl"))
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
