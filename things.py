import torch
import pygame
import random
from base_vars import *
from simulation import Simulation

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
    def __init__(self, thing_types):
        self.thing_types = thing_types
        self.num_things = len(thing_types)
        self.positions = generate_unique_positions(self.num_things)
        self.handle_boundary_checks()
        self.resolve_overlaps()

    def update_positions(self):
        # Get movement directions (angles or None) from the assigned strategies
        movement_angles = torch.stack(
            [THING_TYPES[self.thing_types[i]]["action_function"]()
             for i in range(self.num_things)]
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
        # Get sizes for all things from THING_TYPES dynamically
        sizes = torch.tensor([THING_TYPES[thing]["size"] for thing in self.thing_types])

        # Start and end positions for diagonal lines (or circles for other things)
        start_pos = self.positions - sizes.unsqueeze(1)
        end_pos = self.positions + sizes.unsqueeze(1)

        # Clamp start and end positions to stay within boundaries
        start_pos[:, 0] = start_pos[:, 0].clamp(sizes, SCREEN_WIDTH - MENU_WIDTH - sizes)
        start_pos[:, 1] = start_pos[:, 1].clamp(sizes, SCREEN_HEIGHT - sizes)

        end_pos[:, 0] = end_pos[:, 0].clamp(sizes, SCREEN_WIDTH - MENU_WIDTH - sizes)
        end_pos[:, 1] = end_pos[:, 1].clamp(sizes, SCREEN_HEIGHT - sizes)

        # Recalculate center positions based on clamped start/end points
        self.positions = (start_pos + end_pos) / 2

    def resolve_overlaps(self):
        # Identify which objects are "cells" (green ones)
        is_cell = torch.tensor([thing == "cell" or thing == "controlled_cell" for thing in self.thing_types])

        # Get positions of all cells
        cell_positions = self.positions[is_cell]

        # Get sizes of the cells from THING_TYPES
        cell_sizes = torch.tensor([THING_TYPES["cell"]["size"]] * len(cell_positions))

        # Calculate pairwise differences in positions for cells only
        diff = cell_positions.unsqueeze(1) - cell_positions.unsqueeze(0)

        # Calculate pairwise distances for cells only
        distances = torch.norm(diff, dim=2)

        # Get the overlap threshold dynamically for each pair (sum of their sizes)
        overlap_threshold = cell_sizes.unsqueeze(1) + cell_sizes.unsqueeze(0)

        # Identify overlapping pairs (distance < threshold)
        overlap_mask = (distances < overlap_threshold) & (distances > 0)

        # Calculate overlap amount (how much they're overlapping)
        overlap_amount = overlap_threshold - distances
        overlap_amount = torch.where(overlap_mask, overlap_amount, torch.tensor(0.0))

        # Calculate movement directions to resolve overlap
        direction = diff / (distances.unsqueeze(2) + 1e-6)  # Normalize the direction

        # Calculate the adjustment needed to resolve overlaps
        move_distance = (0.5 * overlap_amount.unsqueeze(2)) * direction

        # Adjust positions: subtract from the current positions and add to others (for cells only)
        self.positions[is_cell] += move_distance.sum(dim=1)
        self.positions[is_cell] -= move_distance.sum(dim=0)

    def add_thing(self, type_name):
        new_position = torch.rand(1, 2)
        new_position[0, 0] *= SCREEN_WIDTH - MENU_WIDTH
        new_position[0, 1] *= SCREEN_HEIGHT
        self.positions = torch.cat((self.positions, new_position))
        self.thing_types.append(
            (type_name, THING_TYPES[type_name]["action_function"])
        )
        self.num_things += 1

        self.handle_boundary_checks()
        self.resolve_overlaps()

    def remove_thing(self, i):
        self.positions = torch.cat((self.positions[:i], self.positions[i + 1:]))
        del self.action_functions[i]
        self.num_things -= 1

    def draw(self, screen):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            color = THING_TYPES[thing_type]["color"]
            size = THING_TYPES[thing_type]["size"]

            if thing_type == "sugar":
                start_pos = (int(pos[0].item() - size),
                             int(pos[1].item() + size))
                end_pos = (int(pos[0].item() + size),
                           int(pos[1].item() - size))
                pygame.draw.line(screen, color, start_pos, end_pos, width = 1)
            else:
                pygame.draw.circle(screen, color, (int(pos[0].item()),
                                   int(pos[1].item())), size)

    def get_state(self):
        return {
            'positions': self.positions.tolist(),
            'types': [thing_type for thing_type, _ in self.thing_types]
        }

    def load_state(self, state):
        self.positions = torch.tensor(state['positions'])
        self.thing_types = [(thing_type, default_action_function)
                            for thing_type in state['types']]
