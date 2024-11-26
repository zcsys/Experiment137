import torch
import torch.nn.functional as F
import pygame
import numpy as np
import random
import time
from base_vars import *
from helpers import *

class Grid:
    def __init__(self, cell_size=10, feature_size=3, diffusion_rate=0.01):
        self.cell_size = cell_size
        self.feature_size = feature_size
        self.grid_x = SIMUL_WIDTH // cell_size
        self.grid_y = SIMUL_HEIGHT // cell_size

        print(f"Grid dimensions: {self.grid_x}x{self.grid_y}")
        print(f"Screen dimensions: {SIMUL_WIDTH}x{SIMUL_HEIGHT}")

        # Initialize with correct dimensions (NCHW format)
        self.grid = torch.zeros((1, feature_size, self.grid_y, self.grid_x), dtype=torch.float32)
        self.diffusion_rate = diffusion_rate
        self.last_step_time = 0

        self.kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel = self.kernel.repeat(feature_size, 1, 1, 1)

        # Initialize random values
        for channel in range(3):
            indices = torch.randint(0, self.grid_x * self.grid_y, (10000,))
            x_indices = indices % self.grid_x
            y_indices = indices // self.grid_x
            self.grid[0, channel, y_indices, x_indices] = 255.

    def circular_pad(self, x, pad):
        return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)

    def diffuse(self):
        start_time = time.time()
        padded = self.grid
        padded = self.circular_pad(padded, 1)
        padded = self.circular_pad(padded.transpose(-1, -2), 1).transpose(-1, -2)
        laplacian = F.conv2d(
            padded,
            self.kernel,
            padding=0,
            groups=self.feature_size
        )
        self.grid += self.diffusion_rate * laplacian
        self.last_step_time = (time.time() - start_time) * 1000

def draw_grid_optimized(grid, surface):
    # Get grid data and transpose to (H, W, C)
    grid_data = grid.grid[0].permute(1, 2, 0).numpy()
    grid_data = grid_data.clip(0, 255).astype(np.uint8)

    # Create upscaled array
    upscaled = np.repeat(
        np.repeat(grid_data, grid.cell_size, axis=0),
        grid.cell_size, axis=1
    )

    # Ensure dimensions match exactly
    if upscaled.shape != (SIMUL_HEIGHT, SIMUL_WIDTH, 3):
        print(f"Warning: Array shape mismatch. Got {upscaled.shape}, expected {(SIMUL_HEIGHT, SIMUL_WIDTH, 3)}")
        # Crop or pad if necessary
        upscaled = upscaled[:SIMUL_HEIGHT, :SIMUL_WIDTH, :]

    # Use surfarray.pixels3d for direct surface manipulation
    surface_array = pygame.surfarray.pixels3d(surface)
    surface_array[:] = upscaled.transpose(1, 0, 2)  # Transpose for pygame's format
    del surface_array  # Release the surface lock

def main():
    pygame.init()
    screen = pygame.display.set_mode((SIMUL_WIDTH, SIMUL_HEIGHT))
    clock = pygame.time.Clock()
    grid = Grid()
    running = True
    font = pygame.font.Font(None, 36)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        start_time = time.time()
        grid.diffuse()

        screen.fill((0, 0, 0))
        draw_grid_optimized(grid, screen)

        time_text = font.render(f"Step time: {grid.last_step_time:.2f}ms", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))

        total_step_time = (time.time() - start_time) * 1000
        step_text = font.render(f"Total step: {total_step_time:.2f}ms", True, (255, 255, 255))
        screen.blit(step_text, (10, 50))

        pygame.display.flip()
        #clock.tick(60)  # Uncomment to limit framerate

    pygame.quit()

if __name__ == "__main__":
    main()
