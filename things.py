import torch
import pygame
import random
from simulation import SCREEN_WIDTH, SCREEN_HEIGHT, MENU_WIDTH

CELL_SIZE = 50
SPEED_CONSTANT = 1.

def default_action_function():
    if random.random() < 0.5:
        return torch.tensor(random.uniform(0, 2 * torch.pi))  # Random angle
    return torch.tensor(float('nan'))  # No movement

def generate_unique_positions(N):
    positions = torch.rand((N, 2))
    positions[:, 0] *= SCREEN_WIDTH - MENU_WIDTH
    positions[:, 1] *= SCREEN_HEIGHT

    while positions.unique(dim = 0).size(0) < N:
        duplicates = positions.unique(dim = 0, return_counts = True)[1] > 1
        positions[duplicates] = torch.rand((duplicates.sum(), 2))
        positions[duplicates, 0] *= SCREEN_WIDTH - MENU_WIDTH
        positions[duplicates, 1] *= SCREEN_HEIGHT

    return positions

class Things:
    def __init__(self, action_functions):
        self.action_functions = action_functions
        self.num_things = len(action_functions)
        self.positions = generate_unique_positions(self.num_things)
        self.handle_boundary_checks()
        self.resolve_overlaps()

    def update_positions(self):
        # Get movement directions (angles or None) from the assigned strategies
        movement_angles = torch.stack(
            [self.action_functions[i]() for i in range(self.num_things)]
        )

        # Mask for valid movements (where angles are not None)
        valid_movements = ~torch.isnan(movement_angles)

        # Calculate dx and dy for valid movements
        dx = SPEED_CONSTANT * torch.cos(movement_angles[valid_movements])
        dy = SPEED_CONSTANT * torch.sin(movement_angles[valid_movements])

        # Apply all valid movements
        self.positions[valid_movements, 0] += dx
        self.positions[valid_movements, 1] += dy

        self.handle_boundary_checks()
        self.resolve_overlaps()

    def handle_boundary_checks(self):
        self.positions[:, 0] = self.positions[:, 0].clamp(
            CELL_SIZE,
            SCREEN_WIDTH - MENU_WIDTH - CELL_SIZE)
        self.positions[:, 1] = self.positions[:, 1].clamp(
            CELL_SIZE,
            SCREEN_HEIGHT - CELL_SIZE
        )

    def resolve_overlaps(self):
        diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diff, dim = 2)

        too_close = (distances < CELL_SIZE * 2) & (distances > 0)
        overlap_amount = CELL_SIZE * 2 - distances
        overlap_amount = torch.where(
            too_close,
            overlap_amount,
            torch.tensor(0.0)
        )

        direction = diff / (distances.unsqueeze(2) + 1e-6)
        move_distance = (0.5 * overlap_amount).unsqueeze(2) * direction

        self.positions += move_distance.sum(dim = 1)
        self.positions -= move_distance.sum(dim = 0)

    def add_thing(self, strategy = None):
        # Add a new thing at a random position
        new_position = torch.rand(1, 2)
        new_position[0, 0] *= SCREEN_WIDTH - MENU_WIDTH
        new_position[0, 1] *= SCREEN_HEIGHT
        self.positions = torch.cat((self.positions, new_position))
        self.action_functions.append(
            strategy if strategy else default_movement_strategy
        )
        self.num_things += 1

    def remove_thing(self, i):
        self.positions = torch.cat((self.positions[:i], self.positions[i + 1:]))
        del self.action_functions[i]
        self.num_things -= 1

    def draw(self, screen):
        for pos in self.positions:
            pygame.draw.circle(screen, (0, 255, 0), (int(pos[0].item()),
                               int(pos[1].item())), CELL_SIZE)

    def get_state(self):
        return {
            'positions': self.positions.tolist()
        }

    def load_state(self, state):
        self.positions = torch.tensor(state['positions'])
