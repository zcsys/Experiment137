import pygame
import torch
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
GRID_SIZE = 40
CELL_WIDTH = WINDOW_WIDTH // GRID_SIZE
CELL_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Setup display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("2D Grid Visualization")

# Create grid
grid = torch.rand((GRID_SIZE, GRID_SIZE), dtype=torch.float32)

def update_grid():
    noise = torch.sin(torch.rand_like(grid) * 0.1) * 0.01
    grid.add_(noise)
    grid.clamp_(0.0, 1.0)

def render():
    screen.fill((26, 26, 26))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            intensity = int(grid[i, j].item() * 255)
            color = (intensity, intensity, 0)
            pygame.draw.rect(screen, color, (i * CELL_WIDTH, j * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
    pygame.display.flip()

if __name__ == "__main__":
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        update_grid()
        render()
        clock.tick(60)

    pygame.quit()
