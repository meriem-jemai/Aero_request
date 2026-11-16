import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import Aerorequest  # noqa: F401  (registers the environments)

ENV_IDS = ["Aerorequest_pixels_v0", "Aerorequest_features_v0"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_passes_gymnasium_checker(env_id):
    env = gym.make(env_id).unwrapped
    check_env(env, skip_render_check=True)


def test_pixel_observation_shape():
    env = gym.make("Aerorequest_pixels_v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8


def test_same_seed_same_target():
    env = gym.make("Aerorequest_features_v0")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.array_equal(obs1["target_position"], obs2["target_position"])


def test_episode_ends_with_random_actions():
    env = gym.make("Aerorequest_features_v0")
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(5000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert terminated or truncated
