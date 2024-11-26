import torch
import pygame
import json
import numpy as np
from base_vars import *
from helpers import *

class Grid:
    def __init__(self, cell_size = 30, feature_dim = 3, diffusion_rate = 0.001,
                 saved_state = None):
        self.cell_size = cell_size
        self.feature_dim = feature_dim
        self.grid_x = SIMUL_WIDTH // cell_size
        self.grid_y = SIMUL_HEIGHT // cell_size
        self.diffusion_rate = diffusion_rate

        # Laplacian kernel
        self.kernel = torch.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ],
            dtype = torch.float32
        ).view(1, 1, 3, 3).repeat(feature_dim, 1, 1, 1)

        if saved_state:
            self.grid = torch.tensor(saved_state, dtype = torch.float32)
            return

        # Initialize with correct dimensions (NCHW format)
        self.grid = torch.zeros((1, feature_dim, self.grid_y, self.grid_x),
                                dtype = torch.float32)

        # Initialize random values
        for channel in range(3):
            indices = torch.randint(0, self.grid_x * self.grid_y, (1000,))
            x_indices = indices % self.grid_x
            y_indices = indices // self.grid_x
            self.grid[0, channel, y_indices, x_indices] = 255.

    def circular_pad(self, x, pad):
        return torch.cat([x[..., -pad:], x, x[..., :pad]], dim = -1)

    def diffuse(self):
        padded = self.circular_pad(self.grid, 1)
        padded = self.circular_pad(
            padded.transpose(-1, -2),
            1
        ).transpose(-1, -2)
        laplacian = torch.nn.functional.conv2d(
            padded,
            self.kernel,
            padding = 0,
            groups = self.feature_dim
        )
        self.grid += self.diffusion_rate * laplacian

    def draw(self, surface):
        # Get grid data and transpose to (H, W, C)
        grid_data = self.grid[0].permute(1, 2, 0).numpy().astype(np.uint8)

        # Create upscaled array
        upscaled = np.repeat(
            np.repeat(grid_data, self.cell_size, axis = 0),
            self.cell_size,
            axis = 1
        )

        # Use surfarray.pixels3d for direct surface manipulation
        surface_array = pygame.surfarray.pixels3d(
            surface.subsurface((0, 0, SIMUL_WIDTH, SIMUL_HEIGHT))
        )
        surface_array[:] = upscaled.transpose(1, 0, 2)  # For pygame's format
