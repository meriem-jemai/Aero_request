"""Browser version of Aero_request (built with pygbag)."""
import asyncio

import pygame

from game import FPS, HEIGHT, WIDTH, DroneGame

KEYS = {
    pygame.K_w: 0, pygame.K_UP: 0,
    pygame.K_s: 1, pygame.K_DOWN: 1,
    pygame.K_a: 2, pygame.K_LEFT: 2,
    pygame.K_d: 3, pygame.K_RIGHT: 3,
}
RESTART_DELAY = 1.5  # seconds after a crash


async def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aero_request")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)

    game = DroneGame()
    wait = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        pressed = pygame.key.get_pressed()
        action = next((a for key, a in KEYS.items() if pressed[key]), -1)

        over = game.crashed or game.time >= game.time_limit
        if over:
            wait += 1 / FPS
            if wait > RESTART_DELAY:
                game.reset()
                wait = 0.0
        else:
            game.step(action)

        game.draw(screen)
        if over:
            text = font.render(f"Game over! Score: {game.score}", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(0)  # lets the browser breathe between frames


asyncio.run(main())