import torch
import pygame
import random
from base_vars import *
from action_functions import *
from thing_types import THING_TYPES
from simulation import draw_dashed_circle

def add_positions(sizes,
                  existing_sizes = torch.empty(0),
                  existing_positions = torch.empty((0, 2)),
                  width = SIMUL_WIDTH,
                  height = SIMUL_HEIGHT):
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
        self.N = len(thing_types)
        self.E = 0.
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 16)

        self.Pop = sum(1 for thing in self.thing_types if thing == "cell")
        self.genomes = torch.zeros((self.Pop, 16))
        self.lineages = [[] for _ in range(self.Pop)]
        self.apply_genomes()

        self.diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        self.movement_tensor = torch.tensor([[0., 0.] for _ in range(self.N)])
        self.sensory_inputs()

    def apply_genomes(self):
        # Monad26 neurogenetics
        self.weights_i_1 = self.genomes[:, 0:4].T
        self.weights_1_o = self.genomes[:, 4:8].T
        self.biases_i_1 = self.genomes[:, 8:12]
        self.biases_1_o = self.genomes[:, 12:14]

    def mutate(self, i, probability = 0.1, strength = 1.):
        genomes_after_mutation = self.genomes.clone()
        genome_to_mutate = genomes_after_mutation[i]
        mutation_mask = torch.rand_like(genome_to_mutate) < probability
        mutations = torch.rand_like(genome_to_mutate) * 2 - 1
        genome_to_mutate += mutation_mask * mutations * strength
        if mutation_mask.any():
            print(f"Original genome {i}: {self.genomes[i].tolist()}")
            print(f"Mutated genome {i}: {genomes_after_mutation.tolist()}")
            print("========")
        return genomes_after_mutation

    def sensory_inputs(self):
        # Tensor masks that will be useful
        self_mask = torch.eye(self.N, dtype = torch.bool)
        sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )
        cell_mask = torch.tensor(
            [thing_type == "cell" for thing_type in self.thing_types]
        )

        # For each non-sugar, there's a vector pointing towards the center of
        # the universe, with increasing magnitude as the thing gets closer to
        # edges. This is the first input vector for each particle.
        midpoint = torch.tensor([SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2])
        col1 = (1 - self.positions[cell_mask] / midpoint)

        # For each non-sugar, the combined effect of sugar particles in their
        # vicinity is calculated. This is the second input vector for particles.
        distances = torch.norm(self.diffs, dim = 2)
        in_sight = ((distances <= SIGHT) & ~self_mask) & sugar_mask.unsqueeze(0)
        effect_of_sugars = torch.where(
            in_sight,
            -1. / (distances + 1e-7),
            torch.tensor(0.)
        )
        normalized_diffs = self.diffs / (torch.norm(self.diffs, dim = 2,
                                         keepdim = True) + 1e-7)
        col2 = (
            normalized_diffs * effect_of_sugars.unsqueeze(2)
        )[cell_mask].sum(dim = 1) * 20.

        # Combine the inputs to create (N, 2, 2)-shaped final input tensor
        self.input_vectors = torch.stack([col1, col2], dim = 1)

    def neural_action(self):
        layer_1 = (torch.matmul(self.weights_i_1.view(self.Pop, 2, 2),
                   self.input_vectors) + self.biases_i_1.view(self.Pop, 2, 2))
        layer_1 = torch.tanh(layer_1)
        layer_o = (torch.matmul(self.weights_1_o.view(self.Pop, 2, 2),
                   layer_1) + self.biases_1_o.view(self.Pop, 2, 1))
        layer_o = torch.tanh(layer_o)
        return layer_o

    def neural_action_placeholder(self):
        self.sensory_inputs()

        values = torch.tensor([-1, 0, 1], dtype = torch.float32)
        weights = torch.tensor([1, 3, 1], dtype = torch.float32)
        indices = torch.multinomial(weights, (self.N - 1) * 2,
                                    replacement = True)
        indices = indices.view(self.N - 1, 2)
        return values[indices]

    def final_action(self):
        print(self.neural_action())
        self.movement_tensor = torch.cat(
            (
                torch.tensor(
                    controlled_action(),
                    dtype = torch.float32
                ).unsqueeze(0),
                self.neural_action()
            ),
            dim = 0
        )

        self.update_positions()

    def update_positions(self):
        overlap = torch.tensor(
            [THING_TYPES[thing_type]["overlap"]
            for thing_type in self.thing_types]
        )

        provisional_positions = self.positions + self.movement_tensor

        # Prevent moving beyond the edges
        provisional_positions = torch.stack(
            [
                torch.clamp(
                    provisional_positions[:, 0],
                    min = self.sizes,
                    max = SIMUL_WIDTH - self.sizes
                ),
                torch.clamp(
                    provisional_positions[:, 1],
                    min = self.sizes,
                    max = SIMUL_HEIGHT - self.sizes
                )
            ],
            dim = 1
        )

        # Detect collisions
        diffs = provisional_positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim = 2)
        size_sums = self.sizes.unsqueeze(1) + self.sizes.unsqueeze(0)
        overlap_mask = (distances < size_sums).fill_diagonal_(False)

        # Prevent overlaps
        pairwise_overlap = overlap.unsqueeze(1) | overlap.unsqueeze(0)
        stoppable_mask = torch.logical_and(overlap_mask, ~pairwise_overlap)
        overlap_detected = stoppable_mask.any(dim = 1)
        final_apply_mask = torch.logical_or(overlap,
                                            ~overlap_detected).unsqueeze(1)

        # Allow movement only if there's enough energy or if type is 'sugar'
        movement_magnitudes = torch.diag(distances)
        energy_mask = torch.gt(self.energies, movement_magnitudes).unsqueeze(1)
        sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )
        move_condition_mask = torch.logical_or(energy_mask,
                                               sugar_mask.unsqueeze(1))
        final_apply_mask = torch.logical_and(final_apply_mask,
                                             move_condition_mask)

        # Apply the movements
        self.positions = torch.where(
            final_apply_mask,
            provisional_positions,
            self.positions
        )

        # Reduce energies
        actual_magnitudes = torch.where(
            final_apply_mask.squeeze(),
            movement_magnitudes,
            torch.tensor(0.)
        )[~sugar_mask]
        self.energies[~sugar_mask] -= actual_magnitudes
        self.E += actual_magnitudes.sum().item()

        # Handle sugar vs cell collisions
        sugar_vs_cell = (
            overlap_mask &
            sugar_mask.unsqueeze(1) &
            ~sugar_mask.unsqueeze(0)
        )

        if sugar_vs_cell.any():
            sugar_idx, cell_idx = sugar_vs_cell.nonzero(as_tuple = True)
            energy_per_non_sugar = 1000 / sugar_vs_cell.sum(dim = 1)[sugar_idx]
            self.energies = self.energies.scatter_add(
                0,
                cell_idx,
                energy_per_non_sugar
            )
            self.remove_things(sugar_idx.tolist())

        # Update diffs
        self.diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)

    def cell_division(self):
        pass

    def add_things(self, types):
        self.thing_types += types
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES[x]["size"] for x in types]),
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
        self.N += len(types)
        self.Pop += sum([1 for thing_type in types if thing_type == "cell"])

    def remove_things(self, indices):
        self.thing_types = [
            thing_type
            for i, thing_type in enumerate(self.thing_types)
            if i not in set(indices)
        ]
        mask = torch.ones(self.N, dtype = torch.bool)
        mask[indices] = False
        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.energies = self.energies[mask]
        self.N = mask.sum().item()
        self.Pop = sum([1 for thing_type in self.thing_types
                        if thing_type == "cell"])

    def draw(self, screen, show_sight = False, show_forces = False):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = THING_TYPES[thing_type]["color"]
            size = THING_TYPES[thing_type]["size"]

            pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                               int(pos[1].item())), size)
            if thing_type != "sugar":
                energy_text = self.energies[i].item()
                if energy_text < 1000:
                    energy_text = str(int(energy_text))
                elif energy_text < 10000:
                    energy_text = f"{int(energy_text / 100) / 10:.1f}k"
                else:
                    energy_text = f"{int(energy_text / 1000)}k"
                energy_text = self.font.render(energy_text, True, (0, 0, 0))
                text_rect = energy_text.get_rect(center = (int(pos[0].item()),
                                                           int(pos[1].item())))
                screen.blit(energy_text, text_rect)

            if show_sight and thing_type != "sugar":
                draw_dashed_circle(screen, GREEN, (int(pos[0].item()),
                                   int(pos[1].item())), SIGHT)

            if show_forces and thing_type != "sugar":
                input_vector_1 = self.input_vectors[i, 0]
                input_vector_2 = self.input_vectors[i, 1]
                movement_vector = self.movement_tensor[i]

                end_pos_1 = pos + input_vector_1 * 50
                end_pos_2 = pos + input_vector_2 * 50
                end_pos_3 = pos + movement_vector * 50

                pygame.draw.line(screen, RED, (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_1[0].item()),
                                 int(end_pos_1[1].item())), 1)
                pygame.draw.line(screen, CYAN, (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_2[0].item()),
                                 int(end_pos_2[1].item())), 1)
                pygame.draw.line(screen, WHITE, (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_3[0].item()),
                                 int(end_pos_3[1].item())), 2)

    def get_state(self):
        return {
            'types': self.thing_types,
            'positions': self.positions.tolist(),
            'energies': self.energies.tolist(),
            'E': self.E,
            'genomes': self.genomes.tolist(),
            'lineages': self.lineages
        }

    def load_state(self, state):
        self.thing_types = state['types']
        self.sizes = torch.tensor(
            [THING_TYPES[x]["size"] for x in self.thing_types]
        )
        self.positions = torch.tensor(state['positions'])
        self.energies = torch.tensor(state['energies'])
        self.N = len(self.positions)
        self.E = state['E']
        self.genomes = torch.tensor(state['genomes'])
        self.lineages = state['lineages']
