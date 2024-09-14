import torch
import pygame
import random
from base_vars import *
from simulation import Simulation

def generate_unique_positions(N):
    # Generate random variables (positions) initially on CPU
    positions = torch.rand((N, 2))  # Default dtype is float32, device is CPU

    # Apply scaling and transformation to positions
    positions[:, 0] *= SCREEN_WIDTH - MENU_WIDTH
    positions[:, 1] *= SCREEN_HEIGHT

    # Ensure unique positions by replacing duplicates
    while positions.unique(dim=0).size(0) < N:
        duplicates = positions.unique(dim=0, return_counts=True)[1] > 1
        new_randoms = torch.rand((duplicates.sum(), 2))  # Generate new randoms
        positions[duplicates] = new_randoms
        positions[duplicates, 0] *= SCREEN_WIDTH - MENU_WIDTH
        positions[duplicates, 1] *= SCREEN_HEIGHT

    return positions

class Things:
    def __init__(self, thing_types):
        self.thing_types = thing_types
        self.num_things = len(thing_types)
        self.positions = generate_unique_positions(self.num_things)
        self.energies = torch.tensor(
            [INITIAL_ENERGY if thing_type != "sugar" else 0.
             for thing_type in self.thing_types]
        )
        self.handle_boundary_checks()
        self.resolve_overlaps()
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)

    def update_positions(self, controlled_direction):
        # Get movement directions (angles or None) from the assigned strategies
        movement_angles = torch.stack(
            [get_controlled_action(controlled_direction)] +
            [THING_TYPES[self.thing_types[i]]["action_function"]()
             for i in range(1, self.num_things)]
        )

        is_sugar = torch.tensor([thing_type == "sugar"
                                 for thing_type in self.thing_types])

        # Mask for valid movements
        valid_movements = (
            ~torch.isnan(movement_angles) &
            (is_sugar | (self.energies > 0))
        )

        # Calculate dx and dy for valid movements
        dx = SPEED_CONSTANT * torch.cos(movement_angles[valid_movements])
        dy = SPEED_CONSTANT * torch.sin(movement_angles[valid_movements])

        # Apply all valid movements
        self.positions[valid_movements, 0] += dx
        self.positions[valid_movements, 1] += dy

        self.energies[valid_movements & ~is_sugar] -= SPEED_CONSTANT

        self.handle_boundary_checks()
        self.resolve_overlaps()

    def handle_boundary_checks(self):
        # Get sizes for all things from THING_TYPES dynamically
        sizes = torch.tensor([THING_TYPES[thing]["size"] for thing in self.thing_types])

        # Adjust positions to allow cells to reach the very edge of the screen
        self.positions[:, 0] = self.positions[:, 0].clamp(min=sizes, max=SCREEN_WIDTH - MENU_WIDTH - sizes)
        self.positions[:, 1] = self.positions[:, 1].clamp(min=sizes, max=SCREEN_HEIGHT - sizes)

    def resolve_overlaps(self):
        # Get sizes of all things
        sizes = torch.tensor([THING_TYPES[thing]["size"] for thing in self.thing_types])

        # Compute pairwise differences in positions for all things
        diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)

        # Compute pairwise distances for all things
        distances = torch.norm(diff, dim=2)

        # Compute the overlap threshold (sum of sizes) for each pair
        overlap_threshold = sizes.unsqueeze(1) + sizes.unsqueeze(0)

        # Identify which pairs are closer than the overlap threshold
        overlap_mask = (distances < overlap_threshold) & (distances > 0)

        # Get the indices of all overlapping pairs
        overlap_indices = overlap_mask.nonzero(as_tuple=False)

        # Filter out duplicates by ensuring only (i, j) pairs where i < j are processed
        overlap_indices = overlap_indices[overlap_indices[:, 0] < overlap_indices[:, 1]]

        # Get the types of the things involved in the overlaps
        types_i = [self.thing_types[i] for i in overlap_indices[:, 0]]
        types_j = [self.thing_types[j] for j in overlap_indices[:, 1]]

        # Handle cell-cell repulsion
        cell_cell_mask = [(t_i != "sugar" and t_j != "sugar") for t_i, t_j in zip(types_i, types_j)]
        cell_cell_indices = overlap_indices[cell_cell_mask]

        if len(cell_cell_indices) > 0:
            overlap_amounts = overlap_threshold[cell_cell_indices[:, 0], cell_cell_indices[:, 1]] - distances[cell_cell_indices[:, 0], cell_cell_indices[:, 1]]
            directions = diff[cell_cell_indices[:, 0], cell_cell_indices[:, 1]] / (distances[cell_cell_indices[:, 0], cell_cell_indices[:, 1]].unsqueeze(1) + 1e-6)
            move_distances = 0.5 * overlap_amounts.unsqueeze(1) * directions

            self.positions[cell_cell_indices[:, 0]] += move_distances
            self.positions[cell_cell_indices[:, 1]] -= move_distances

        # Handle cell-sugar interaction (eating) with energy splitting
        cell_sugar_mask = [(t_i == "sugar" and t_j != "sugar") or (t_i != "sugar" and t_j == "sugar") for t_i, t_j in zip(types_i, types_j)]
        cell_sugar_indices = overlap_indices[cell_sugar_mask]

        sugars_to_remove = set()  # Use a set to track sugars to be removed

        if len(cell_sugar_indices) > 0:
            # For each sugar, gather the cells that are close to it
            sugars_eaten = {}
            for pair in cell_sugar_indices:
                sugar_idx = pair[0] if self.thing_types[pair[0]] == "sugar" else pair[1]
                cell_idx = pair[1] if self.thing_types[pair[0]] == "sugar" else pair[0]

                if sugar_idx not in sugars_eaten:
                    sugars_eaten[sugar_idx] = []
                sugars_eaten[sugar_idx].append(cell_idx)

            # Distribute the energy evenly to all cells involved (no removal yet)
            for sugar_idx, cell_indices in sugars_eaten.items():
                # Split the energy among all cells that are close to this sugar
                energy_per_cell = 1000 / len(cell_indices)
                for cell_idx in cell_indices:
                    self.energies[cell_idx] += energy_per_cell

                # Mark this sugar for removal after energy distribution
                sugars_to_remove.add(sugar_idx)

        # Remove all sugars marked for removal after all energy updates
        if len(sugars_to_remove) > 0:
            sugars_to_remove = sorted(sugars_to_remove)

            # Remove sugars from positions, types, and energies
            self.positions = torch.stack([self.positions[i] for i in range(self.num_things) if i not in sugars_to_remove])
            self.thing_types = [self.thing_types[i] for i in range(self.num_things) if i not in sugars_to_remove]
            self.energies = torch.stack([self.energies[i] for i in range(self.num_things) if i not in sugars_to_remove])

            # Update the number of things
            self.num_things = len(self.thing_types)

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
        del self.thing_types[i]
        self.energies = torch.cat((self.energies[:i], self.energies[i + 1:]))
        self.num_things -= 1

    def draw(self, screen):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = THING_TYPES[thing_type]["color"]
            size = THING_TYPES[thing_type]["size"]

            if thing_type == "sugar":
                size /= SQRT2
                start_pos = (int(pos[0].item() - size),
                             int(pos[1].item() + size))
                end_pos = (int(pos[0].item() + size),
                           int(pos[1].item() - size))
                pygame.draw.line(screen, thing_color, start_pos, end_pos, width = 2)
            else:
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)

                energy_text = self.font.render(
                    f"{int(self.energies[i].item())}", True, (0, 0, 0)
                )
                text_rect = energy_text.get_rect(center = (int(pos[0].item()),
                                                           int(pos[1].item())))
                screen.blit(energy_text, text_rect)

    def get_state(self):
        return {
            'positions': self.positions.tolist(),
            'types': self.thing_types,
            'energies': self.energies.tolist()
        }

    def load_state(self, state):
        self.positions = torch.tensor(state['positions'])
        self.thing_types = state['types']
        self.energies = torch.tensor(state['energies'])
        self.num_things = len(self.positions)
