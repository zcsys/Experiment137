import torch
import pygame
import random
from base_vars import *
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
        # Main attributes
        self.thing_types = thing_types
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        )
        self.energies = torch.tensor(
            [THING_TYPES[x]["initial_energy"] for x in thing_types]
        )
        self.N = len(thing_types)
        self.E = 0.
        self.calc_state()

        self.last_movement_was_successful = torch.ones((self.Pop, 1),
                                                       dtype = bool)
        #self.genomes = torch.zeros((self.Pop, 30)) # GENOME210_0
        self.genomes = torch.rand((self.Pop, 34)) * 2 - 1
        self.lineages = []
        self.apply_genomes()
        self.sensory_inputs()

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 16)

    def calc_state(self):
        # Tensor masks that will be useful
        self.self_mask = torch.eye(self.N, dtype = torch.bool)
        self.sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )
        self.cell_mask = torch.tensor(
            [thing_type == "cell" for thing_type in self.thing_types]
        )

        # Calculate state vars
        self.diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        self.Pop = self.cell_mask.sum().item()

    def apply_genomes(self):
        # Monad211 neurogenetics
        self.weights_i_1 = self.genomes[:, 0:20].view(self.Pop, 4, 5)
        self.weights_1_o = self.genomes[:, 20:28].view(self.Pop, 2, 4)
        self.biases_i_1 = self.genomes[:, 28:32].view(self.Pop, 4, 1)
        self.biases_1_o = self.genomes[:, 32:34].view(self.Pop, 2, 1)

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
        # For each non-sugar, there's a vector pointing towards the center of
        # the universe, with increasing magnitude as the thing gets closer to
        # edges. This is the first input vector for each particle.
        midpoint = torch.tensor([SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2])
        col1 = (1 - self.positions[self.cell_mask] / midpoint)

        # For each non-sugar, the combined effect of sugar particles in their
        # vicinity is calculated. This is the second input vector for particles.
        distances = torch.norm(self.diffs, dim = 2)
        in_sight = (
            (distances <= SIGHT) &
            ~self.self_mask &
            self.sugar_mask.unsqueeze(0)
        )
        effect_of_sugars = torch.where(
            in_sight,
            -1. / (distances + 1e-7),
            torch.tensor(0.)
        )
        normalized_diffs = self.diffs / (torch.norm(self.diffs, dim = 2,
                                         keepdim = True) + 1e-7)
        col2 = (
            normalized_diffs * effect_of_sugars.unsqueeze(2)
        )[self.cell_mask].sum(dim = 1) * 20.

        # Combine the inputs to create (Pop, 5, 1)-shaped final input tensor
        self.input_vectors = torch.cat(
            [
                col1,
                col2,
                self.last_movement_was_successful
            ],
            dim = 1
        ).view(self.Pop, 5, 1)

    def neural_action(self):
        input_tensor = self.input_vectors
        layer_1 = torch.tanh((torch.bmm(self.weights_i_1, input_tensor) +
                   self.biases_i_1))
        return torch.tanh((torch.bmm(self.weights_1_o, layer_1) +
                   self.biases_1_o)).view(self.Pop, 2)

    def random_action(self):
        numberOf_sugars = self.sugar_mask.sum().item()
        values = torch.tensor([-1, 0, 1], dtype = torch.float32)
        weights = torch.tensor([1, 3, 1], dtype = torch.float32)
        indices = torch.multinomial(
            weights,
            numberOf_sugars * 2,
            replacement = True
        )
        indices = indices.view(numberOf_sugars, 2)
        return values[indices]

    def controlled_action(self):
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            dx += -1
        if keys[pygame.K_RIGHT]:
            dx += 1
        if keys[pygame.K_UP]:
            dy += -1
        if keys[pygame.K_DOWN]:
            dy += 1
        return torch.tensor([dx, dy], dtype = torch.float32)

    def final_action(self):
        self.sensory_inputs()

        self.movement_tensor = torch.tensor([[0., 0.] for _ in range(self.N)])

        self.movement_tensor[self.thing_types.index("controlled_cell")] = (
            self.controlled_action()
        )
        self.movement_tensor[self.cell_mask] = self.neural_action()
        self.movement_tensor[self.sugar_mask] = self.random_action()

        self.update_positions()
        self.calc_state()

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

        self.last_movement_was_successful = torch.where(
            final_apply_mask,
            True,
            False
        )[self.cell_mask]

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

    def draw(self, screen, show_sight = False, show_forces = False):
        masked_indices = torch.nonzero(self.cell_mask,
                                       as_tuple = False).squeeze()

        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = THING_TYPES[thing_type]["color"]
            size = self.sizes[i].item()

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

            if show_sight and thing_type == "cell":
                draw_dashed_circle(screen, GREEN, (int(pos[0].item()),
                                   int(pos[1].item())), SIGHT)

            if show_forces and thing_type == "cell":
                idx = (masked_indices == i).nonzero(as_tuple = False).item()

                input_vector_1 = self.input_vectors[idx, 0]
                input_vector_2 = self.input_vectors[idx, 1]
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
