"""Gymnasium wrapper around the drone game."""
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .game import FPS, HEIGHT, WIDTH, DroneGame


class Aerorequest(gym.Env):
    """
    Custom Gymnasium environment: control a drone to reach targets.

    Parameters:
        render_mode (str): 'human' or 'rgb_array'.
        obs_type (str): 'pixels' (64x64 RGB image) or 'features' (state vector).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, obs_type=None):
        super().__init__()
        if obs_type not in ("pixels", "features"):
            raise ValueError("Invalid observation type. Choose 'pixels' or 'features'.")
        self.obs_type = obs_type
        self.render_mode = render_mode

        pygame.init()
        self.game = DroneGame()
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(4)
        if obs_type == "pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        else:
            inf = np.inf
            self.observation_space = spaces.Dict({
                "player_position": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
                "player_velocity": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
                "player_angle": spaces.Box(low=-inf, high=inf, shape=(1,), dtype=np.float32),
                "angular_velocity": spaces.Box(low=-inf, high=inf, shape=(1,), dtype=np.float32),
                "target_position": spaces.Box(low=0, high=max(WIDTH, HEIGHT), shape=(2,), dtype=np.float32),
            })

    @property
    def score(self):
        return self.game.score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(rng=self.np_random)
        self.prev_dist = self.game.distance_to_target()

        if self.render_mode == "human":
            self._render_human()
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Returns:
            observation, reward, terminated, truncated, info
        """
        dist, reached, crashed = self.game.step(action)

        # Reward: progress towards the target, +10 for reaching it, -10 for crashing
        reward = (self.prev_dist - dist) / 100
        self.prev_dist = dist
        if reached:
            reward += 10.0
            self.prev_dist = self.game.distance_to_target()
        if crashed:
            reward -= 10.0

        terminated = crashed
        truncated = self.game.time >= self.game.time_limit

        if self.render_mode == "human":
            self._render_human()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_human()

    def _render_frame(self):
        """Return the current frame as an RGB array of shape (height, width, 3)."""
        canvas = pygame.Surface((WIDTH, HEIGHT))
        self.game.draw(canvas)
        return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def _render_human(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Aerorequest Environment")
            self.clock = pygame.time.Clock()
        self.game.draw(self.screen)
        pygame.display.update()
        self.clock.tick(FPS)

    def _get_obs(self):
        if self.obs_type == "pixels":
            frame = self._render_frame()  # (height, width, 3)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            small = pygame.transform.scale(surface, (64, 64))
            return pygame.surfarray.array3d(small).swapaxes(0, 1)

        g = self.game
        return {
            "player_position": np.array([g.x, g.y], dtype=np.float32),
            "player_velocity": np.array([g.vx, g.vy], dtype=np.float32),
            "player_angle": np.array([g.angle], dtype=np.float32),
            "angular_velocity": np.array([g.angular_speed], dtype=np.float32),
            "target_position": np.array(g.target, dtype=np.float32),
        }

    def _get_info(self):
        return {"distance": self.game.distance_to_target(), "score": self.game.score}

    def close(self):
        pygame.quit()