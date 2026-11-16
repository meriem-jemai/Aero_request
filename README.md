# Aero_request

![tests](https://github.com/meriem-jemai/Aero_request/actions/workflows/tests.yml/badge.svg)

A custom reinforcement learning environment built with **Gymnasium** and **Pygame**.
A 2D drone with two thrusters has to fly to randomly placed targets without leaving the screen.
The physics (gravity, thrust, momentum, rotation) make it a small but non-trivial control problem.

![Demo](https://github.com/meriemjma/Aero_request/assets/130979570/c5ec806f-6e75-4697-847e-dca72f3dfa5c)

## Features

- Two observation modes: **raw pixels** (64×64 RGB) or **state features**
- Shaped reward for reaching targets and avoiding crashes
- Seeding support for reproducible episodes
- Passes Gymnasium's official `check_env`, tested with pytest and CI

## Installation

```bash
git clone https://github.com/meriem-jemai/Aero_request.git
cd Aero_request
pip install -e .
```

## Quick start

```python
import gymnasium as gym
import Aerorequest  # registers the environments

env = gym.make("Aerorequest_features_v0")   # or "Aerorequest_pixels_v0"
obs, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Environment details

| | |
|---|---|
| **Actions** | `Discrete(4)`: 0 thrust up, 1 thrust down, 2 rotate left, 3 rotate right |
| **Pixel observation** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Feature observation** | drone position, velocity, angle, angular velocity, target position |
| **Reward** | +progress towards the target, **+10** for reaching it, **−10** for crashing |
| **Termination** | drone leaves the screen |
| **Truncation** | time limit (100 seconds at 24 FPS) |

## Play it yourself

```bash
python play.py
```

**W** thrust up · **S** thrust down · **A / D** rotate

## Run the tests

```bash
pip install pytest
pytest -q
```

## Project structure

```
Aerorequest/
├── __init__.py          # environment registration
└── env/
    ├── aerorequest.py   # environment logic, physics, rendering
    ├── Player.py        # drone state
    └── resources/       # sprites
tests/test_env.py        # pytest suite
play.py                  # keyboard control
```
