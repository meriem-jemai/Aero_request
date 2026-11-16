"""
Core drone game: physics, targets and drawing.

Has no Gymnasium dependency, so it can run on its own,
e.g. as a playable browser game.
"""
import os
import random
from math import cos, pi, sin, sqrt

import pygame

RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

WIDTH, HEIGHT = 650, 650
FPS = 24
BACKGROUND = (131, 176, 181)
FLASH_COLOR = (255, 0, 0)


class _PythonRng:
    """Minimal random generator with the same interface as numpy's Generator.integers."""

    def integers(self, low, high):
        return random.randrange(low, high)


class DroneGame:
    """A 2D drone with two thrusters that has to reach targets without leaving the screen."""

    gravity = 0.08
    mass = 1
    arm = 25  # length of the drone arm
    thruster_mean = 0.04
    thruster_amplitude = 0.04
    diff_amplitude = 0.003
    target_radius = 50
    time_limit = 100  # seconds
    flash_duration = 0.5  # seconds

    def __init__(self, rng=None):
        self.rng = rng or _PythonRng()
        self._sprites = None
        self._font = None
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, rng=None):
        """Put the drone back in the centre and spawn a new target."""
        if rng is not None:
            self.rng = rng
        self.x, self.y = WIDTH / 2, HEIGHT / 2
        self.vx, self.vy = 0.0, 0.0
        self.angle, self.angular_speed = 0.0, 0.0
        self.score = 0
        self.time = 0.0
        self.crashed = False
        self.flash_timer = 0.0
        self.cloud_x = [150.0, 400.0]
        self.spawn_target()

    def spawn_target(self):
        self.target = (
            int(self.rng.integers(100, WIDTH - 100)),
            int(self.rng.integers(100, HEIGHT - 100)),
        )

    def distance_to_target(self):
        return sqrt((self.x - self.target[0]) ** 2 + (self.y - self.target[1]) ** 2)

    # ---------------------------------------------------------------- physics
    def step(self, action):
        """
        Advance the game by one frame.

        Actions: 0 = thrust up, 1 = thrust down, 2 = rotate left, 3 = rotate right,
        anything else = default thrust.

        Returns:
            (distance, reached, crashed): distance to the target active during this step,
            whether it was reached, and whether the drone left the screen.
        """
        left = right = self.thruster_mean
        if action == 0:
            left = right = self.thruster_mean + self.thruster_amplitude
        elif action == 1:
            left = right = self.thruster_mean - self.thruster_amplitude
        elif action == 2:
            left = self.thruster_mean - self.diff_amplitude
        elif action == 3:
            right = self.thruster_mean - self.diff_amplitude

        # Newton's laws
        angle_rad = self.angle * pi / 180
        self.vx += -(left + right) * sin(angle_rad) / self.mass
        self.vy += self.gravity - (left + right) * cos(angle_rad) / self.mass
        self.angular_speed += self.arm * (right - left) / self.mass

        self.x += self.vx
        self.y += self.vy
        self.angle += self.angular_speed

        # Background animation and effects
        self.cloud_x[0] += 0.3
        if self.cloud_x[0] > WIDTH:
            self.cloud_x[0] = -200
        self.cloud_x[1] -= 0.2
        if self.cloud_x[1] < -150:
            self.cloud_x[1] = WIDTH
        if self.flash_timer > 0:
            self.flash_timer -= 1 / FPS

        # Target
        distance = self.distance_to_target()
        reached = distance < self.target_radius
        if reached:
            self.score += 10
            self.spawn_target()

        # Crash
        if not (0 <= self.x <= WIDTH and 0 <= self.y <= HEIGHT):
            self.crashed = True
            self.flash_timer = self.flash_duration

        self.time += 1 / FPS
        return distance, reached, self.crashed

    # ---------------------------------------------------------------- drawing
    def _load_assets(self):
        def load(name, size):
            return pygame.transform.scale(pygame.image.load(os.path.join(RESOURCES_DIR, name)), size)

        self._sprites = {
            "drone": load("drone1.png", (80, 24)),
            "target": load("target1.png", (50, 50)),
            "sun": load("sun.png", (300, 300)),
            "cloud1": load("cloud1.png", (200, 100)),
            "cloud2": load("cloud2.png", (150, 75)),
        }
        self._font = pygame.font.Font(None, 36)

    def draw(self, surface):
        """Draw the current state onto a pygame surface."""
        if self._sprites is None:
            self._load_assets()
        s = self._sprites

        surface.fill(BACKGROUND)
        surface.blit(s["cloud1"], (self.cloud_x[0], 200))
        surface.blit(s["cloud2"], (self.cloud_x[1], 500))
        surface.blit(s["sun"], (WIDTH - 150, -100))

        drone = pygame.transform.rotate(s["drone"], self.angle)
        surface.blit(drone, (int(self.x - drone.get_width() / 2), int(self.y - drone.get_height() / 2)))
        surface.blit(s["target"], (self.target[0] - 25, self.target[1] - 25))

        surface.blit(self._font.render(f"Time: {int(self.time)}", True, (255, 255, 255)), (20, 20))
        surface.blit(self._font.render(f"Score: {self.score}", True, (255, 255, 255)), (20, 60))

        if self.flash_timer > 0:
            for rect in [(0, 0, WIDTH, 10), (0, 0, 10, HEIGHT), (0, HEIGHT - 10, WIDTH, 10), (WIDTH - 10, 0, 10, HEIGHT)]:
                pygame.draw.rect(surface, FLASH_COLOR, rect)