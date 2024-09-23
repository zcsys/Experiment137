import torch
import pygame
import random
import math
from base_vars import *
from simulation import draw_dashed_circle

def add_positions(sizes,
                  existing_sizes = torch.empty(0),
                  existing_positions = torch.empty((0, 2)),
                  width = SIMUL_WIDTH,
                  height = SIMUL_HEIGHT):
    existing_N = len(existing_positions)
    total_N = existing_N + len(sizes)

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

        distances = torch.norm(new_position - positions, dim = 1)
        if (distances < sizes[i] + sizes[:i]).any():
            continue

        positions = torch.cat((positions, new_position), dim = 0)
        i += 1

    return sizes, positions

def remove_element(tensor, i):
    return torch.cat((tensor[:i], tensor[i + 1:]), dim = 0)

class Things:
    def __init__(self, thing_types):
        # Main attributes
        self.thing_types = thing_types
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        )

        # Initialize tensor masks
        self.cell_mask = torch.tensor(
            [thing_type == "cell" or thing_type == "controlled_cell"
             for thing_type in self.thing_types]
        )
        self.sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )

        # Initialize state vars
        self.E = 0.
        self.N = len(self.thing_types)
        self.Pop = self.cell_mask.sum().item()
        self.energies = torch.tensor(
            [THING_TYPES[thing_type]["initial_energy"]
            for thing_type in thing_types]
        )

        # Initialize genomes and lineages
        self.genomes = torch.zeros((self.Pop, 34)) # GENOME211_0
        self.lineages = [[0] for _ in range(self.Pop)]
        self.apply_genomes()

        # Initialize sensory input data
        self.last_movement_was_successful = torch.ones(
            self.Pop,
            dtype = bool
        ).unsqueeze(1)
        self.sensory_inputs()

        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

    def apply_genomes(self):
        # Monad211 neurogenetics
        self.weights_i_1 = self.genomes[:, 0:20].view(self.Pop, 4, 5)
        self.weights_1_o = self.genomes[:, 20:28].view(self.Pop, 2, 4)
        self.biases_i_1 = self.genomes[:, 28:32].view(self.Pop, 4, 1)
        self.biases_1_o = self.genomes[:, 32:34].view(self.Pop, 2, 1)

    def mutate(self, i, probability = 0.1, strength = 1., show = False):
        mutated_genome = self.genomes[i].clone()
        mutation_mask = torch.rand_like(mutated_genome) < probability
        mutations = torch.rand_like(mutated_genome) * 2 - 1
        mutated_genome += mutation_mask * mutations * strength
        if mutation_mask.any() and show:
            print(f"Original genome {i}: {self.genomes[i].tolist()}")
            print(f"Mutated genome {i}: {mutated_genome.tolist()}")
            print("========")
        return mutated_genome

    def sensory_inputs(self):
        # For each non-sugar, there's a vector pointing towards the center of
        # the universe, with increasing magnitude as the thing gets closer to
        # edges. This is the first input vector for each particle.
        if self.cell_mask.any():
            midpoint = torch.tensor([SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2])
            col1 = (1 - self.positions[self.cell_mask] / midpoint)
        else:
            col1 = torch.zeros((self.Pop, 2))

        # For each non-sugar, the combined effect of sugar particles in their
        # vicinity is calculated. This is the second input vector for particles.
        if self.cell_mask.any() and self.sugar_mask.any():
            self.diffs = (self.positions[self.sugar_mask].unsqueeze(0) -
                          self.positions[self.cell_mask].unsqueeze(1))
            distances = torch.norm(self.diffs, dim = 2)
            col2 = torch.where(
                (distances <= SIGHT).unsqueeze(2),
                self.diffs / (distances.unsqueeze(2) + 1e-7) ** 2,
                torch.tensor([0., 0.])
            ).sum(dim = 1) * 21.
        else:
            col2 = torch.zeros((self.Pop, 2))

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
        if numberOf_sugars == 0:
            return
        values = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                              dtype = torch.float32)
        weights = torch.tensor([1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1],
                               dtype = torch.float32)
        indices = torch.multinomial(
            weights,
            numberOf_sugars * 2,
            replacement = True
        ).view(numberOf_sugars, 2)
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
        if keys[pygame.K_SPACE]:
            self.cell_division(0)
        return torch.tensor([dx, dy], dtype = torch.float32)

    def final_action(self):
        self.sensory_inputs()
        if self.N > 0:
            self.movement_tensor = torch.tensor([[0., 0.]
                                                 for _ in range(self.N)])
        if self.cell_mask.any():
            self.movement_tensor[self.cell_mask] = self.neural_action()
        if "controlled_cell" in self.thing_types:
            self.movement_tensor[0] = self.controlled_action()
        if self.sugar_mask.any():
            self.movement_tensor[self.sugar_mask] = self.random_action()
        self.update_positions()

    def update_positions(self):
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
        cell_vs_cell = (
            self.cell_mask.unsqueeze(1) &
            self.cell_mask.unsqueeze(0) &
            overlap_mask
        ).any(dim = 1)

        # Allow movement only if there's enough energy or if type is 'sugar'
        movement_magnitudes = torch.diag(distances)
        final_apply_mask = (
            (self.energies > movement_magnitudes) &
            ~cell_vs_cell |
            self.sugar_mask
        )

        # Apply the movements
        self.positions = torch.where(
            final_apply_mask.unsqueeze(1),
            provisional_positions,
            self.positions
        )

        self.last_movement_was_successful = final_apply_mask[
            self.cell_mask
        ].unsqueeze(1)

        # Reduce energies from cells and give to system
        actual_magnitudes = torch.where(
            final_apply_mask & self.cell_mask,
            movement_magnitudes,
            torch.tensor(0.)
        )
        self.energies -= actual_magnitudes
        self.E += actual_magnitudes.sum().item()

        # Handle sugar vs cell collisions
        sugar_vs_cell = (
            overlap_mask &
            self.sugar_mask.unsqueeze(1) &
            self.cell_mask.unsqueeze(0)
        )

        if sugar_vs_cell.any():
            sugar_idx, cell_idx = sugar_vs_cell.nonzero(as_tuple = True)
            energy_per_non_sugar = 1000 / sugar_vs_cell.sum(dim = 1)[sugar_idx]
            self.energies = self.energies.scatter_add(
                0,
                cell_idx,
                energy_per_non_sugar
            )
            self.remove_sugars(sugar_idx.tolist())

    def cell_division(self, i):
        # See if division is possible
        thing_type = self.thing_types[i]
        if thing_type == "controlled_cell":
            thing_type = "cell"
        initial_energy = torch.tensor(THING_TYPES[thing_type]["initial_energy"])
        if self.energies[i] < 2 * initial_energy:
            return 0
        size = THING_TYPES[thing_type]["size"]
        x, y = tuple(self.positions[i].tolist())
        angle = random.random() * 2 * math.pi
        new_position = torch.tensor([
            x + math.cos(angle) * (size + 1) * 2,
            y + math.sin(angle) * (size + 1) * 2
        ])
        distances = torch.norm(
            self.positions[self.cell_mask] - new_position, dim = 1
        )
        if (new_position[0] < size or new_position[0] > SIMUL_WIDTH - size or
            new_position[1] < size or new_position[1] > SIMUL_HEIGHT - size or
            (distances < self.sizes[self.cell_mask] + size).any()):
            return 0

        self.thing_types += [thing_type]
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(size).unsqueeze(0)
            ),
            dim = 0
        )
        self.positions = torch.cat(
            (
                self.positions,
                new_position.unsqueeze(0)
            ),
            dim = 0
        )
        self.energies[i] -= initial_energy
        self.energies = torch.cat(
            (
                self.energies,
                initial_energy.unsqueeze(0)
            ),
            dim = 0
        )
        self.movement_tensor = torch.cat(
            (
                self.movement_tensor,
                torch.tensor([[0., 0.]])
            ),
            dim = 0
        )
        self.last_movement_was_successful = torch.cat(
            (
                self.last_movement_was_successful,
                torch.tensor([[True]])
            ),
            dim = 0
        )

        i = self.cell_mask[:i].sum().item()
        genome = self.mutate(i)
        self.genomes = torch.cat(
            (
                self.genomes,
                genome.unsqueeze(0)
            ),
            dim = 0
        )
        self.weights_i_1 = torch.cat(
            (
                self.weights_i_1,
                genome[0:20].view(1, 4, 5)
            ),
            dim = 0
        )
        self.weights_1_o = torch.cat(
            (
                self.weights_1_o,
                genome[20:28].view(1, 2, 4)
            ),
            dim = 0
        )
        self.biases_i_1 = torch.cat(
            (
                self.biases_i_1,
                genome[28:32].view(1, 4, 1)
            ),
            dim = 0
        )
        self.biases_1_o = torch.cat(
            (
                self.biases_1_o,
                genome[32:34].view(1, 2, 1)
            ),
            dim = 0
        )
        if not genome is self.genomes[i]:
            new_lineage = self.lineages[i] + [0]
            while True:
                new_lineage[-1] += 1
                if new_lineage not in self.lineages:
                    break
            self.lineages += [new_lineage]

        self.cell_mask = torch.cat(
            (
                self.cell_mask,
                torch.tensor([True])
            ),
            dim = 0
        )
        self.sugar_mask = torch.cat(
            (
                self.sugar_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.N += 1
        self.Pop += 1

        return 1

    def cell_death(self, indices):
        # Remove cell-only attributes
        for i in indices:
            # Get cell-only index from general index
            idx = self.cell_mask[:i].sum().item()

            # Remove attributes
            self.last_movement_was_successful = remove_element(
                self.last_movement_was_successful, idx
            )

        # Remove universal attributes
        for i in indices:
            # Update main attributes
            del self.thing_types[i]
            self.sizes = remove_element(self.sizes, i)
            self.positions = remove_element(self.positions, i)
            self.energies = remove_element(self.energies, i)

            # Update genomics
            self.genomes = remove_element(self.genomes, i)
            del self.lineages[i]

            # Update state vars
            self.cell_mask = remove_element(self.cell_mask, i)
            self.sugar_mask = remove_element(self.sugar_mask, i)

        # Update collective state vars
        self.N -= len(indices)
        self.Pop -= len(indices)

        self.apply_genomes()

    def add_sugars(self, N):
        self.thing_types += ["sugar" for _ in range(N)]
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES["sugar"]["size"] for _ in range(N)]),
            self.sizes,
            self.positions
        )
        self.N += N
        self.energies = torch.cat(
            (
                self.energies,
                torch.tensor([THING_TYPES["sugar"]["initial_energy"]
                              for _ in range(N)])
            ),
            dim = 0
        )
        self.cell_mask = torch.cat(
            (
                self.cell_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.sugar_mask = torch.cat(
            (
                self.sugar_mask,
                torch.ones(N, dtype = torch.bool)
            ),
            dim = 0
        )

    def remove_sugars(self, indices):
        self.thing_types = [
            thing_type
            for i, thing_type in enumerate(self.thing_types)
            if i not in set(indices)
        ]

        mask = torch.ones(self.N, dtype = torch.bool)
        mask[indices] = False
        self.N = mask.sum().item()

        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.energies = self.energies[mask]

        self.cell_mask = self.cell_mask[mask]
        self.sugar_mask = self.sugar_mask[mask]
        self.Pop = self.cell_mask.sum().item()

    def draw(self, screen, show_energy = True, show_sight = False,
             show_forces = False):
        masked_indices = torch.nonzero(self.cell_mask,
                                       as_tuple = False).squeeze()

        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = THING_TYPES[thing_type]["color"]
            size = self.sizes[i].item()

            pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                               int(pos[1].item())), size)

            if show_energy and thing_type != "sugar":
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
                idx = self.cell_mask[:i].sum().item()
                if idx >= len(self.input_vectors):
                    return
                input_vector_1 = self.input_vectors[idx, 0:2].squeeze(1)
                input_vector_2 = self.input_vectors[idx, 2:4].squeeze(1)
                movement_vector = self.movement_tensor[i]

                end_pos_1 = pos + input_vector_1 * 50
                end_pos_2 = pos + input_vector_2 * 50
                end_pos_3 = pos + movement_vector * 20

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
