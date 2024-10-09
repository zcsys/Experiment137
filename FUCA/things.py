import torch
import pygame
import random
import math
import json
from base_vars import *
from helpers import *
from simulation import draw_dashed_circle

class Things:
    def __init__(self, thing_types = None, state_file = None):
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

        # Initialize system heat
        self.heat = 21

        if state_file:
            self.load_state(state_file)
            return

        # Main attributes
        self.thing_types = thing_types
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        )

        # Initialize tensor masks
        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )

        # Initialize state vars
        self.E = 0.
        self.N = len(self.thing_types)
        self.Pop = self.monad_mask.sum().item()
        self.energies = torch.tensor(
            [THING_TYPES[thing_type]["initial_energy"]
            for thing_type in thing_types]
        )
        self.colors = [THING_TYPES[x]["color"] for x in self.thing_types]
        self.hidden_1 = torch.zeros((self.Pop, 8, 1), dtype = torch.float32)
        self.hidden_2 = torch.zeros((self.Pop, 8, 1), dtype = torch.float32)
        self.boxes = get_box(self.positions)

        # Initialize genomes and lineages
        self.genomes = torch.tensor(GENOME429_0, dtype = torch.float32).repeat(
            self.Pop, 1
        )
        self.lineages = [[0] for _ in range(self.Pop)]
        self.apply_genomes()

        # Initialize the monad messages
        self.messages = torch.zeros(
            (self.Pop, 1),
            dtype = torch.float32
        )

        # Initialize sensory input data
        self.last_movement_was_successful = torch.ones(
            self.Pop,
            dtype = torch.bool
        ).unsqueeze(1)
        self.incoming_messages = torch.zeros(
            (self.Pop, 3),
            dtype = torch.float32
        )
        self.sensory_inputs()

    def from_general_to_monad_idx(self, i):
        return self.monad_mask[:i].sum().item()

    def from_monad_to_general_idx(self, i):
        return torch.nonzero(self.monad_mask)[i].item()

    def get_generation(self, i):
        return self.lineages[i][0] + len(self.lineages[i])

    def apply_genomes(self):
        """Monad429 neurogenetics"""
        # Layer 1
        self.weights_i_1 = self.genomes[:, 0:72].view(self.Pop, 8, 9)
        self.weights_h_1 = self.genomes[:, 72:136].view(self.Pop, 8, 8)
        self.biases_i_1 = self.genomes[:, 136:144].view(self.Pop, 8, 1)

        # Layer 2
        self.weights_1_2 = self.genomes[:, 144:208].view(self.Pop, 8, 8)
        self.weights_h_2 = self.genomes[:, 208:272].view(self.Pop, 8, 8)
        self.biases_1_2 = self.genomes[:, 272:280].view(self.Pop, 8, 1)

        # Output layer
        self.weights_2_o = self.genomes[:, 280:312].view(self.Pop, 4, 8)
        self.biases_2_o = self.genomes[:, 312:316].view(self.Pop, 4, 1)

    def mutate(self, i, show = False):
        probability = 0.1
        strength = 1.

        original_genome = self.genomes[i].clone()
        n = int(len(original_genome) / 2)
        coding_part = original_genome[:n]
        regulatory_part = original_genome[n:].int()
        genome_to_mutate = coding_part[regulatory_part]

        mutation_mask = torch.rand_like(genome_to_mutate) < probability
        mutations = torch.rand_like(genome_to_mutate) * 2 - 1
        coding_part[regulatory_part] = (
            genome_to_mutate + mutation_mask * mutations * strength
        )

        if mutation_mask.any() and show:
            pass

        return torch.cat((coding_part, regulatory_part.float()), dim = 0)

    def sensory_inputs(self):
        # For each non-sugar, there's a vector pointing towards the center of
        # the universe, with increasing magnitude as the thing gets closer to
        # edges. This is the first input vector for each particle.
        if self.monad_mask.any():
            midpoint = torch.tensor([SIMUL_WIDTH / 2, SIMUL_HEIGHT / 2])
            col1 = (1 - self.positions[self.monad_mask] / midpoint)
        else:
            col1 = torch.zeros((self.Pop, 2))

        # For each non-sugar, the combined effect of sugar particles in their
        # vicinity is calculated. This is the second input vector for particles.
        if self.monad_mask.any() and self.sugar_mask.any():
            self.diffs = (self.positions[self.sugar_mask].unsqueeze(0) -
                          self.positions[self.monad_mask].unsqueeze(1))
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
                self.last_movement_was_successful,
                (self.energies[self.monad_mask] / 10000).unsqueeze(1),
                self.incoming_messages
            ],
            dim = 1
        ).view(self.Pop, 9, 1)

    def neural_action(self):
        self.hidden_1 = torch.tanh(
            torch.bmm(self.weights_i_1, self.input_vectors) +
            torch.bmm(self.weights_h_1, self.hidden_1) +
            self.biases_i_1
        )

        self.hidden_2 = torch.tanh(
            torch.bmm(self.weights_1_2, self.hidden_1) +
            torch.bmm(self.weights_h_2, self.hidden_2) +
            self.biases_1_2
        )

        return torch.tanh(
            torch.bmm(self.weights_2_o, self.hidden_2) +
            self.biases_2_o
        ).view(self.Pop, 4)

    def random_action(self):
        numberOf_sugars = self.sugar_mask.sum().item()
        if self.heat == 0:
            return torch.tensor([[0, 0] for _ in range(numberOf_sugars)],
                                dtype = torch.float32)
        values = (torch.tensor(list(range(self.heat)), dtype = torch.float32) -
                  (self.heat - 1) / 2)
        weights = torch.ones(self.heat, dtype = torch.float32)
        indices = torch.multinomial(
            weights,
            numberOf_sugars * 2,
            replacement = True
        ).view(numberOf_sugars, 2)
        return values[indices]

    def final_action(self):
        # Update sensory inputs
        self.sensory_inputs()

        # Initialize the movement tensor for this step
        if self.N > 0:
            self.movement_tensor = torch.tensor([[0., 0.]
                                                 for _ in range(self.N)])

        # monad actions
        if self.monad_mask.any():
            # Get output tensor
            neural_action = self.neural_action()

            # Apply movements
            self.movement_tensor[self.monad_mask] = neural_action[:, :2]

            # Broadcast messages
            self.messages = neural_action[:, 3]

            # Apply fissions
            random_gen = torch.rand(self.Pop)
            to_divide = neural_action[:, 2] > random_gen
            for i in to_divide.nonzero():
                self.monad_division(self.from_monad_to_general_idx(i))

        # Sugar movements
        if self.sugar_mask.any():
            self.movement_tensor[self.sugar_mask] = self.random_action()

        # Apply changes in positions
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
        monad_vs_monad = (
            self.monad_mask.unsqueeze(1) &
            self.monad_mask.unsqueeze(0) &
            overlap_mask
        ).any(dim = 1)

        # Allow movement only if there's enough energy or if type is 'sugar'
        movement_magnitudes = torch.diag(distances)
        final_apply_mask = (
            (self.energies > movement_magnitudes) &
            ~monad_vs_monad |
            self.sugar_mask
        )

        # Apply the movements
        self.positions = torch.where(
            final_apply_mask.unsqueeze(1),
            provisional_positions,
            self.positions
        )

        # Update the LMWS input neuron
        self.last_movement_was_successful = final_apply_mask[
            self.monad_mask
        ].unsqueeze(1)

        # Reduce energies from monads and give to system
        actual_magnitudes = torch.where(
            final_apply_mask & self.monad_mask,
            movement_magnitudes,
            torch.tensor(0.)
        )
        self.energies -= actual_magnitudes
        # self.E += actual_magnitudes.sum().item() # Works with Rules(0)

        # Handle sugar vs monad collisions
        sugar_vs_monad = (
            overlap_mask &
            self.sugar_mask.unsqueeze(1) &
            self.monad_mask.unsqueeze(0)
        )

        if sugar_vs_monad.any():
            sugar_idx, monad_idx = sugar_vs_monad.nonzero(as_tuple = True)
            energy_per_non_sugar = (
                SUGAR_ENERGY / sugar_vs_monad.sum(dim = 1)[sugar_idx]
            )
            self.energies = self.energies.scatter_add(
                0,
                monad_idx,
                energy_per_non_sugar
            )
            self.remove_sugars(unique(sugar_idx.tolist()))

        # Deliver messages
        c_pos = self.positions[self.monad_mask]
        c_diffs = c_pos.unsqueeze(0) - c_pos.unsqueeze(1)
        c_dist = torch.norm(c_diffs, dim = 2)
        in_sight_mask = (c_dist < SIGHT).fill_diagonal_(False).int()

        self.incoming_messages = torch.zeros(
            (self.Pop, 3),
            dtype = torch.float32
        )

        if in_sight_mask.any():
            self.recipients = torch.unique(in_sight_mask.nonzero())
            self.first_senders = torch.argmax(
                in_sight_mask[self.recipients], dim = 1
            )
            direction = (
                (
                    c_diffs[self.recipients, self.first_senders] /
                    (
                        c_dist[self.recipients, self.first_senders].unsqueeze(1)
                        + 1e-7
                    )
                )
            )
            self.incoming_messages[self.recipients] = torch.cat(
                (
                    direction,
                    self.messages[self.first_senders].unsqueeze(1),
                ),
                dim = 1
            )

    def monad_division(self, i):
        # Set out main attributes and see if division is possible
        thing_type = self.thing_types[i]
        initial_energy = self.energies[i] / 2
        if (initial_energy <
            torch.tensor(THING_TYPES[thing_type]["initial_energy"])):
            return 0
        # print("Monad division at energy", int(initial_energy.item()))
        size = THING_TYPES[thing_type]["size"]
        x, y = tuple(self.positions[i].tolist())
        angle = random.random() * 2 * math.pi
        new_position = torch.tensor([
            x + math.cos(angle) * (size + 1) * 2,
            y + math.sin(angle) * (size + 1) * 2
        ])
        distances = torch.norm(
            self.positions[self.monad_mask] - new_position, dim = 1
        )
        if (new_position[0] < size or new_position[0] > SIMUL_WIDTH - size or
            new_position[1] < size or new_position[1] > SIMUL_HEIGHT - size or
            (distances < self.sizes[self.monad_mask] + size).any()):
            return 0

        # Create a new set of attributes
        self.thing_types.append(thing_type)
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
        self.boxes = torch.cat(
            (
                self.boxes,
                get_box(new_position.unsqueeze(0))
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
        self.messages = torch.cat(
            (
                self.messages,
                torch.tensor([0.])
            ),
            dim = 0
        )
        self.incoming_messages = torch.cat(
            (
                self.incoming_messages,
                torch.zeros((1, 3), dtype = torch.float32)
            ),
            dim = 0
        )

        # Mutate the old genome & apply the new genome
        idx = self.from_general_to_monad_idx(i)
        genome = self.mutate(idx)
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
                genome[0:72].view(1, 8, 9)
            ),
            dim = 0
        )
        self.weights_h_1 = torch.cat(
            (
                self.weights_h_1,
                genome[72:136].view(1, 8, 8)
            ),
            dim = 0
        )
        self.biases_i_1 = torch.cat(
            (
                self.biases_i_1,
                genome[136:144].view(1, 8, 1)
            ),
            dim = 0
        )
        self.weights_1_2 = torch.cat(
            (
                self.weights_1_2,
                genome[144:208].view(1, 8, 8)
            ),
            dim = 0
        )
        self.weights_h_2 = torch.cat(
            (
                self.weights_h_2,
                genome[208:272].view(1, 8, 8)
            ),
            dim = 0
        )
        self.biases_1_2 = torch.cat(
            (
                self.biases_1_2,
                genome[272:280].view(1, 8, 1)
            ),
            dim = 0
        )
        self.weights_2_o = torch.cat(
            (
                self.weights_2_o,
                genome[280:312].view(1, 4, 8)
            ),
            dim = 0
        )
        self.biases_2_o = torch.cat(
            (
                self.biases_2_o,
                genome[312:316].view(1, 4, 1)
            ),
            dim = 0
        )
        self.hidden_1 = torch.cat(
            (
                self.hidden_1,
                torch.zeros((1, 8, 1), dtype = torch.float32)
            ),
            dim = 0
        )
        self.hidden_2 = torch.cat(
            (
                self.hidden_2,
                torch.zeros((1, 8, 1), dtype = torch.float32)
            ),
            dim = 0
        )
        if genome is self.genomes[idx]:
            self.lineages.append(self.lineages[idx])
            self.colors.append(self.color[i])
        else:
            new_lineage = self.lineages[idx] + [0]
            while True:
                new_lineage[-1] += 1
                if new_lineage not in self.lineages:
                    break
            self.lineages.append(new_lineage)
            self.colors.append(get_color_by_genome(genome))
            # print(new_lineage)

        # Update state vars
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
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

    def monad_death(self, indices):
        for i in indices[::-1]:
            # Remove monad-only attributes
            self.last_movement_was_successful = remove_element(
                self.last_movement_was_successful, i
            )
            self.genomes = remove_element(self.genomes, i)
            self.hidden_1 = remove_element(self.hidden_1, i)
            self.hidden_2 = remove_element(self.hidden_2, i)
            self.messages = remove_element(self.messages, i)
            self.incoming_messages = remove_element(self.incoming_messages, i)
            del self.lineages[i]

            # Get general index to remove universal attributes
            idx = self.from_monad_to_general_idx(i)

            # Update main attributes
            del self.thing_types[idx]
            del self.colors[idx]
            self.sizes = remove_element(self.sizes, idx)
            self.positions = remove_element(self.positions, idx)
            self.energies = remove_element(self.energies, idx)
            self.boxes = remove_element(self.boxes, idx)

            # Update state vars
            self.monad_mask = remove_element(self.monad_mask, idx)
            self.sugar_mask = remove_element(self.sugar_mask, idx)

        # Update collective state vars
        self.N -= len(indices)
        self.Pop -= len(indices)

        self.apply_genomes()

    def add_sugars(self, N):
        for _ in range(N):
            self.thing_types.append("sugar")
            self.colors.append(THING_TYPES["sugar"]["color"])
        self.sizes, self.positions = add_positions(
            torch.tensor([THING_TYPES["sugar"]["size"] for _ in range(N)]),
            self.sizes,
            self.positions
        )
        self.boxes = torch.cat(
            (
                self.boxes,
                get_box(self.positions[:N])
            ),
            dim = 0
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
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
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
        for i in indices[::-1]:
            del self.thing_types[i]
            del self.colors[i]

        mask = torch.ones(self.N, dtype = torch.bool)
        mask[indices] = False
        self.N = mask.sum().item()

        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.energies = self.energies[mask]
        self.boxes = self.boxes[mask]

        self.monad_mask = self.monad_mask[mask]
        self.sugar_mask = self.sugar_mask[mask]
        self.Pop = self.monad_mask.sum().item()

    def draw(self, screen, show_info = True, show_sight = False,
             show_forces = True, show_communication = True):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = self.colors[i]
            size = self.sizes[i].item()
            idx = self.from_general_to_monad_idx(i)

            if thing_type == "sugar":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
            elif thing_type == "monad":
                nucleus_size = THING_TYPES["monad"]["nucleus_size"]
                draw_dashed_circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), nucleus_size)

            if show_info and thing_type == "monad":
                # Show energy
                energy_text = self.energies[i].item()
                if energy_text < 1000:
                    energy_text = str(int(energy_text))
                elif energy_text < 10000:
                    energy_text = f"{int(energy_text / 100) / 10:.1f}k"
                else:
                    energy_text = f"{int(energy_text / 1000)}k"
                energy_text = self.font.render(energy_text, True, colors["Z"])
                energy_rect = energy_text.get_rect(
                    center = (
                        int(pos[0].item()),
                        int(pos[1].item() - 2 * nucleus_size)
                    )
                )
                screen.blit(energy_text, energy_rect)

                # Show message
                message_text = float_msg_to_str(self.messages[idx].item())
                message_text = self.font.render(message_text, True, colors["Z"])
                message_rect = message_text.get_rect(
                    center = (
                        int(pos[0].item()),
                        int(pos[1].item() + 2 * nucleus_size)
                    )
                )
                screen.blit(message_text, message_rect)

            if show_sight and thing_type == "monad":
                draw_dashed_circle(screen, self.colors[i], (int(pos[0].item()),
                                   int(pos[1].item())), SIGHT)

            try:
                input_vector_1 = self.input_vectors[idx, 0:2].squeeze(1)
                input_vector_2 = self.input_vectors[idx, 2:4].squeeze(1)
                movement_vector = self.movement_tensor[i]
            except:
                show_forces = False
            if show_forces and thing_type == "monad":
                input_vector_1 /= (torch.norm(input_vector_1, dim = 0) + 1e-7)
                input_vector_2 /= (torch.norm(input_vector_2, dim = 0) + 1e-7)
                movement_vector /= (torch.norm(movement_vector, dim = 0) + 1e-7)

                end_pos_1 = pos + input_vector_1 * self.sizes[i]
                end_pos_2 = pos + input_vector_2 * self.sizes[i]
                end_pos_3 = pos - movement_vector * self.sizes[i] * 2

                pygame.draw.line(screen, colors["R"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_1[0].item()),
                                 int(end_pos_1[1].item())), 1)
                pygame.draw.line(screen, colors["H"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_2[0].item()),
                                 int(end_pos_2[1].item())), 1)
                pygame.draw.line(screen, colors["Z"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_3[0].item()),
                                 int(end_pos_3[1].item())), 3)

        # Draw communication network
        try:
            first_senders = self.first_senders.tolist()
            recipients = self.recipients.tolist()

            if recipients[-1] >= self.Pop or first_senders[-1] >= self.Pop:
                show_communication = False
        except:
            show_communication = False

        if show_communication:
            for sender, recipient in zip(first_senders, recipients):
                recipient_pos = self.positions[
                    self.from_monad_to_general_idx(recipient)
                ].tolist()

                sender_pos = self.positions[
                    self.from_monad_to_general_idx(sender)
                ].tolist()

                pygame.draw.line(
                    screen,
                    colors["T"],
                    (
                        int(recipient_pos[0]),
                        int(recipient_pos[1])
                    ),
                    (
                        int(sender_pos[0]),
                        int(sender_pos[1])
                    ),
                    1
                )

    def get_state(self):
        return {
            'types': self.thing_types,
            'positions': self.positions.tolist(),
            'energies': self.energies.tolist(),
            'E': self.E,
            'genomes': self.genomes.tolist(),
            'lineages': self.lineages,
            'colors': self.colors,
            'LMWS': self.last_movement_was_successful.tolist(),
            'hidden_1': self.hidden_1.tolist(),
            'hidden_2': self.hidden_2.tolist(),
            'messages': self.messages.tolist(),
            'incoming': self.incoming_messages.tolist()
        }

    def load_state(self, state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)["things_state"]

        self.thing_types = state['types']
        self.sizes = torch.tensor(
            [THING_TYPES[x]["size"] for x in self.thing_types]
        )
        self.positions = torch.tensor(state['positions'])
        self.boxes = get_box(self.positions)
        self.energies = torch.tensor(state['energies'])
        self.N = len(self.positions)
        self.E = state['E']
        self.genomes = torch.tensor(state['genomes'])
        self.lineages = state['lineages']
        self.colors = state['colors']
        self.last_movement_was_successful = torch.tensor(state['LMWS'])
        self.hidden_1 = torch.tensor(state['hidden_1'])
        self.hidden_2 = torch.tensor(state['hidden_2'])
        self.messages = torch.tensor(state['messages'])
        self.incoming_messages = torch.tensor(state['incoming'])

        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.sugar_mask = torch.tensor(
            [thing_type == "sugar" for thing_type in self.thing_types]
        )
        self.Pop = self.monad_mask.sum().item()

        self.apply_genomes()
        self.sensory_inputs()

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)
