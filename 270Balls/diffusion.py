import torch
import torch.nn.functional as F
import pygame
import json
import numpy as np
from base_vars import *
from helpers import *

class Grid:
    def __init__(self, cell_size = 10, feature_dim = 3, diffusion_rate = 0.001,
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

        # Sobel kernels
        self.kernel_x = torch.tensor(
            [[-1, 0, 1]],
            dtype = torch.float32
        ).view(1, 1, 1, 3).repeat(self.feature_dim, 1, 1, 1)
        self.kernel_y = torch.tensor(
            [[-1], [0], [1]],
            dtype = torch.float32
        ).view(1, 1, 3, 1).repeat(self.feature_dim, 1, 1, 1)

        if saved_state:
            self.grid = torch.tensor(saved_state, dtype = torch.float32)
            return

        # Initialize with correct dimensions (NCHW format)
        self.grid = torch.zeros((1, feature_dim, self.grid_y, self.grid_x),
                                dtype = torch.float32)

        # Initialize random values
        for channel in range(3):
            indices = torch.randint(0, self.grid_x * self.grid_y, (10000,))
            x_indices = indices % self.grid_x
            y_indices = indices // self.grid_x
            self.grid[0, channel, y_indices, x_indices] = 255.

    def gradient(self):
        self.padded = F.pad(self.grid, (1, 1, 1, 1), mode = "replicate")
        grad_x = F.conv2d(
            self.padded,
            self.kernel_x,
            padding = 0,
            groups = self.feature_dim
        )
        grad_y = F.conv2d(
            self.padded,
            self.kernel_y,
            padding = 0,
            groups = self.feature_dim
        )
        return grad_x, grad_y

    def diffuse(self, force_field = None):
        laplacian = F.conv2d(
            self.padded,
            self.kernel,
            padding = 0,
            groups = self.feature_dim
        )
        self.grid += self.diffusion_rate * laplacian
        return self.apply_forces(force_field) if force_field is not None else 0.

    def apply_forces(self, force_field):
        # For x direction
        x_pos = torch.clamp(force_field[0], min = 0)
        x_neg = torch.clamp(force_field[0], max = 0)

        # Apply x movements
        self.grid[0] -= x_pos + x_neg
        self.grid[0] += x_pos.roll(1, dims = -1)  # Move right
        self.grid[0] += x_neg.roll(-1, dims = -1) # Move left

        # For y direction
        y_pos = torch.clamp(force_field[1], min = 0)
        y_neg = torch.clamp(force_field[1], max = 0)

        # Apply y movements
        self.grid[0] -= y_pos + y_neg
        self.grid[0] += y_pos.roll(1, dims = -2)  # Move down
        self.grid[0] += y_neg.roll(-1, dims = -2) # Move up

        # Check mass conservation and return the excess
        total_before = self.grid[0].sum()
        self.grid[0] = torch.clamp(self.grid[0], 0, 255)
        return (self.grid[0].sum() - total_before).item()

    def draw(self, surface):
        pygame.surfarray.blit_array(
            surface.subsurface((0, 0, SIMUL_WIDTH, SIMUL_HEIGHT)),
            torch.repeat_interleave(
                torch.repeat_interleave(
                    self.grid[0].permute(1, 2, 0),
                    self.cell_size,
                    dim = 0
                ),
                self.cell_size,
                dim = 1
            ).permute(1, 0, 2).numpy().astype(np.uint8)
        )
