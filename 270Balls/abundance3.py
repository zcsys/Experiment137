#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import jit
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time
import numpy as np

# Resolutions
FULL_HD_WIDTH = 1920
FULL_HD_HEIGHT = 1080
GRID_WIDTH = 192  # 1920 // 10
GRID_HEIGHT = 108  # 1080 // 10
CHANNELS = 3

# Ball parameters
N_BALLS = 10
BALL_RADIUS = 50  # in actual pixels
DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1),
             (-1, 0), (-1, -1), (0, -1), (1, -1)]

class Ball:
    def __init__(self, x, y):
        self.x = int(x)  # ensure integer pixel coordinates
        self.y = int(y)
        self.radius = BALL_RADIUS
        self.circle = None

    def move(self):
        dx, dy = DIRECTIONS[np.random.randint(len(DIRECTIONS))]
        self.x = (self.x + dx) % FULL_HD_WIDTH
        self.y = (self.y + dy) % FULL_HD_HEIGHT

@jit
def diffuse_step(grid, diffusion_rate=0.1):
    padded = jnp.pad(grid, ((1, 1), (1, 1), (0, 0)), mode='wrap')
    laplacian = (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:] -
        4 * padded[1:-1, 1:-1]
    )
    return jnp.clip(grid + diffusion_rate * laplacian, 0, 1)

def initialize_grid():
    key = jax.random.PRNGKey(0)
    grid = jnp.zeros((GRID_HEIGHT, GRID_WIDTH, CHANNELS))
    for channel in range(CHANNELS):
        random_values = jax.random.uniform(key, (GRID_HEIGHT, GRID_WIDTH))
        key, _ = jax.random.split(key)
        channel_mask = random_values < 0.3
        grid = grid.at[:,:,channel].set(channel_mask.astype(float))
    return grid

def initialize_balls():
    balls = []
    for _ in range(N_BALLS):
        x = np.random.randint(BALL_RADIUS, FULL_HD_WIDTH - BALL_RADIUS)
        y = np.random.randint(BALL_RADIUS, FULL_HD_HEIGHT - BALL_RADIUS)
        balls.append(Ball(x, y))
    return balls

def draw_balls(grid, balls):
    # Create a display array at full resolution
    display = np.zeros((FULL_HD_HEIGHT, FULL_HD_WIDTH, CHANNELS))

    # Upscale the diffusion grid
    for i in range(CHANNELS):
        display[:,:,i] = np.kron(grid[:,:,i], np.ones((10,10)))

    # Draw balls
    for ball in balls:
        y, x = np.ogrid[-ball.radius:ball.radius+1, -ball.radius:ball.radius+1]
        mask = x*x + y*y <= ball.radius*ball.radius

        # Get ball position and handle wrapping
        y_pos = np.arange(ball.y - ball.radius, ball.y + ball.radius + 1) % FULL_HD_HEIGHT
        x_pos = np.arange(ball.x - ball.radius, ball.x + ball.radius + 1) % FULL_HD_WIDTH

        # Create meshgrid of positions
        X, Y = np.meshgrid(x_pos, y_pos)

        # Set white color where mask is True
        display[Y[mask], X[mask]] = 1.0

    return display

if __name__ == "__main__":
    fig = plt.figure(figsize=(FULL_HD_WIDTH/100, FULL_HD_HEIGHT/100), dpi=100)

    # Set up axes
    ax_main = plt.Axes(fig, [0., 0.1, 1., 0.9])
    ax_main.set_axis_off()
    fig.add_axes(ax_main)

    ax_time = plt.Axes(fig, [0., 0., 1., 0.05])
    ax_time.set_axis_off()
    fig.add_axes(ax_time)

    time_text = ax_time.text(0.05, 0.5, '', fontsize=10)

    # Initialize
    grid = initialize_grid()
    balls = initialize_balls()
    display = draw_balls(grid, balls)
    img = ax_main.imshow(display, interpolation='nearest')

    frame_count = 0
    total_time = 0
    last_time = time.time()

    def update(frame):
        global grid, frame_count, total_time, last_time

        start_time = time.time()

        # Update diffusion
        grid = diffuse_step(grid)

        # Move balls
        for ball in balls:
            ball.move()

        # Create display with both diffusion and balls
        display = draw_balls(grid, balls)
        img.set_array(display)

        step_time = time.time() - start_time
        frame_count += 1
        total_time += step_time
        fps = 1.0 / (time.time() - last_time) if frame_count > 1 else 0
        last_time = time.time()

        timing_info = (
            f'Step: {frame_count:5d} | '
            f'Last step: {step_time*1000:6.2f} ms | '
            f'Avg step: {(total_time/frame_count)*1000:6.2f} ms | '
            f'FPS: {fps:6.1f}'
        )
        time_text.set_text(timing_info)

        return [img, time_text]

    anim = FuncAnimation(
        fig,
        update,
        cache_frame_data=False,
        interval=50
    )

    plt.show()
