import torch
import pygame
import random
from base_vars import *
from action_functions import *
from thing_types import THING_TYPES

def generate_position(size,
                      width = SCREEN_WIDTH - MENU_WIDTH,
                      height = SCREEN_HEIGHT):
    return torch.tensor(
        [
            random.randint(int(size), int(width - size)),
            random.randint(int(size), int(height - size))
        ],
        dtype = torch.float32
    ).unsqueeze(0)

def add_positions(sizes,
                  existing_sizes = torch.empty(0),
                  existing_positions = torch.empty((0, 2)),
                  width = SCREEN_WIDTH - MENU_WIDTH,
                  height = SCREEN_HEIGHT):
    existing_N = len(existing_positions)
    total_N = len(sizes) + existing_N

    positions = existing_positions
    sizes = torch.cat((existing_sizes, sizes), dim = 0)

    i = existing_N
    while i < total_N:
        new_position = torch.tensor(
            [
                random.randint(int(sizes[i]), int(width - sizes[i])),
                random.randint(int(sizes[i]), int(height - sizes[i]))
            ],
            dtype = torch.float32
        ).unsqueeze(0)

        distances = torch.norm(positions - new_position, dim = 1)

        overlap = False
        for j, distance in enumerate(distances):
            if distance < sizes[i] + sizes[j]:
                overlap = True
                break
        if overlap:
            continue

        positions = torch.cat((positions, new_position), dim = 0)
        i += 1

    return sizes, positions

class Things:
    def __init__(self, thing_types):
        self.thing_types = thing_types
        self.sizes = torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        _, self.positions = add_positions(self.sizes)
        self.energies = torch.tensor(
            [THING_TYPES[x]["initial_energy"] for x in thing_types]
        )
        self.num_things = len(thing_types)
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

    def update_positions(self):
        movement_tensor = torch.tensor(
            [THING_TYPES[thing_type]["action_function"]()
             for thing_type in self.thing_types]
        )

        overlap = torch.tensor(
            [THING_TYPES[thing_type]["overlap"]
            for thing_type in self.thing_types]
        )

        provisional_positions = self.positions + movement_tensor

        # Prevent moving beyond the edges
        provisional_positions = torch.stack(
            [
                torch.clamp(
                    provisional_positions[:, 0],
                    min = self.sizes,
                    max = SCREEN_WIDTH - MENU_WIDTH - self.sizes
                ),
                torch.clamp(
                    provisional_positions[:, 1],
                    min = self.sizes,
                    max = SCREEN_HEIGHT - self.sizes
                )
            ],
            dim = 1
        )

        # Prevent overlaps
        diffs = provisional_positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim = 2)
        size_sums = self.sizes.unsqueeze(1) + self.sizes.unsqueeze(0)
        overlap_mask = distances < size_sums
        overlap_mask.fill_diagonal_(False)
        pairwise_overlap = overlap.unsqueeze(1) | overlap.unsqueeze(0)
        stoppable_mask = torch.logical_and(overlap_mask, ~pairwise_overlap)
        overlap_detected = stoppable_mask.any(dim = 1)
        final_apply_mask = torch.logical_or(overlap,
                                            ~overlap_detected).unsqueeze(1)

        # Apply the movements
        self.positions = torch.where(
            final_apply_mask,
            provisional_positions,
            self.positions
        )

        #self.energies[valid_movements & ~is_sugar] -= SPEED_CONSTANT

    def add_things(self, types):
        self.types += types
        self.sizes, self.positions = add_positions(
            [THING_TYPES[x]["size"] for x in types],
            self.sizes,
            self.positions
        )
        self.energies = torch.cat(
            (
                self.energies,
                torch.tensor([THING_TYPES[x]["initial_energy"] for x in types])
            ),
            dim = 0
        )
        self.num_things += len(types)

    def remove_things(self, indices):
        mask = torch.ones(self.num_things, dtype = torch.bool)
        mask[indices] = False
        self.thing_types = [
            thing
            for i, thing in enumerate(self.thing_types)
            if i not in set(indices)
        ]
        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.energies = self.energies[mask]
        self.num_things = mask.sum().item()

    def draw(self, screen):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = THING_TYPES[thing_type]["color"]
            size = THING_TYPES[thing_type]["size"]

            if THING_TYPES[thing_type]["draw_as"] == "line":
                size /= SQRT2
                start_pos = (int(pos[0].item() - size),
                             int(pos[1].item() + size))
                end_pos = (int(pos[0].item() + size),
                           int(pos[1].item() - size))
                pygame.draw.line(
                    screen, thing_color, start_pos, end_pos, width = 2
                )
            else:
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
                energy_text = self.font.render(
                    f"{self.energies[i].item() / 1000:.1f}k", True, (0, 0, 0)
                )
                text_rect = energy_text.get_rect(center = (int(pos[0].item()),
                                                           int(pos[1].item())))
                screen.blit(energy_text, text_rect)

    def get_state(self):
        return {
            'types': self.thing_types,
            'positions': self.positions.tolist(),
            'energies': self.energies.tolist()
        }

    def load_state(self, state):
        self.thing_types = state['types']
        self.sizes = torch.tensor(
            [THING_TYPES[x]["size"] for x in self.thing_types]
        )
        self.positions = torch.tensor(state['positions'])
        self.energies = torch.tensor(state['energies'])
        self.num_things = len(self.positions)
