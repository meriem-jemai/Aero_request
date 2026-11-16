"""Fly the drone yourself: W = thrust up, S = thrust down, A/D = rotate."""
import gymnasium as gym
import pygame

import Aerorequest  # noqa: F401  (registers the environments)

KEYS = {pygame.K_w: 0, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 3}
NO_INPUT = -1  # keeps default thrust when no key is pressed


def main():
    env = gym.make("Aerorequest_features_v0", render_mode="human")
    env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pressed = pygame.key.get_pressed()
        action = next((a for key, a in KEYS.items() if pressed[key]), NO_INPUT)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
