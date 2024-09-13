import math
import random
import struct
import os
import numpy as np
from uuid import uuid4

WINDOW_SIZE = [1920, 1080]
ATOMS = []
COL_BLUE = (0, 0, 255)
COL_YELLOW = (255, 255, 0)
COL_RED = (255, 0, 0)
COL_GREEN = (0, 80, 0)
COL_WHITE = (255, 255, 255)
DEFAULT_GENOME = [0] + ["00000000" for _ in range(8060)]
HEX_MAX = 4294967295
EPOCH = [0]
STEP = [0]

def randomxy(size):
    x = round(random.random() * WINDOW_SIZE[0])
    x = max(x, size)
    x = min(x, WINDOW_SIZE[0] - size)
    y = round(random.random() * WINDOW_SIZE[1])
    y = max(y, size)
    y = min(y, WINDOW_SIZE[1] - size)
    return (x, y)

def random_genome(genes = 8060):
    weights_and_biases = np.random.randn(genes)
    genome = [0] + [float_to_hex(w) for w in weights_and_biases]
    return genome

def float_to_hex(value):
    # Convert a float to its binary representation and then to a hexadecimal
    # string. '!f' means network order (big-endian) float
    packed_value = struct.pack('!f', value)
    return packed_value.hex()

def hex_to_float(hex_str):
    # Convert the hexadecimal string to its binary form and then to a float
    packed_value = bytes.fromhex(hex_str)
    return struct.unpack('!f', packed_value)[0]

def decode_genome(genome):
    # Convert genome into weights and biases
    generation = genome[0]
    weights_and_biases = [hex_to_float(gene) for gene in genome[1:]]

    # Reconstruct the neural network layers
    return(
        generation, # Increases with mutations
        np.array(weights_and_biases[:2601]).reshape(51, 51),        # W1
        np.array(weights_and_biases[2601:5202]).reshape(51, 51),    # W2
        np.array(weights_and_biases[5202:7803]).reshape(51, 51),    # W3
        np.array(weights_and_biases[7803:7905]).reshape(51, 2),     # Wo
        np.array(weights_and_biases[7905:7956]).reshape(1, 51),     # B1
        np.array(weights_and_biases[7956:8007]).reshape(1, 51),     # B2
        np.array(weights_and_biases[8007:8058]).reshape(1, 51),     # B3
        np.array(weights_and_biases[8058:8060]).reshape(1, 2)       # Bo
    )

def create(N, C, S, E = 0., v = None):
    for i in range(N):
        x, y = randomxy(S)
        if v == None:
            v = [0, 0]
        ATOMS.append(
            {
                "x": x,
                "y": y,
                "color": C,
                "size": S,
                "energy": E,
                "velocity": v[:]
            }
        )

def distance(i, j):
    if i == j:
        return (0, 0, 0)

    i_x = ATOMS[i]["x"]
    i_y = ATOMS[i]["y"]
    j_x = ATOMS[j]["x"]
    j_y = ATOMS[j]["y"]

    d = ((j_x - i_x) ** 2 + (j_y - i_y) ** 2) ** 0.5

    if d > 0:
        cos = (j_x - i_x) / d
        sin = (j_y - i_y) / d
    else:
        return (0, 0, 0)

    return (d, sin, cos)

def move_random(max_magnitude = 0.5):
    for this_atom in ATOMS:
        if this_atom["energy"] > 0:
            theta = random.uniform(0, 2 * math.pi)
            impulse_magnitude = min(this_atom["energy"],
                random.uniform(0, max_magnitude))
            this_atom["velocity"][0] += math.cos(theta) * impulse_magnitude
            this_atom["velocity"][1] += math.sin(theta) * impulse_magnitude

            # Burn energy proportional to the impulse magnitude
            this_atom["energy"] -= impulse_magnitude

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def Rules(n):
    # Apply velocities and prevent overflow
    if n == 0:
        edge_friction = 0.5
        for this_atom in ATOMS:
            this_atom["x"] += this_atom["velocity"][0]
            this_atom["y"] += this_atom["velocity"][1]

            if this_atom["x"] <= this_atom["size"]:
                this_atom["x"] = this_atom["size"]
                this_atom["velocity"][0] *= -1 + edge_friction
                this_atom["velocity"][1] *= 1 - edge_friction
            elif this_atom["x"] >= WINDOW_SIZE[0] - this_atom["size"]:
                this_atom["x"] = WINDOW_SIZE[0] - this_atom["size"]
                this_atom["velocity"][0] *= -1 + edge_friction
                this_atom["velocity"][1] *= 1 - edge_friction

            if this_atom["y"] <= this_atom["size"]:
                this_atom["y"] = this_atom["size"]
                this_atom["velocity"][0] *= 1 - edge_friction
                this_atom["velocity"][1] *= -1 + edge_friction
            elif this_atom["y"] >= WINDOW_SIZE[1] - this_atom["size"]:
                this_atom["y"] = WINDOW_SIZE[1] - this_atom["size"]
                this_atom["velocity"][0] *= 1 - edge_friction
                this_atom["velocity"][1] *= -1 + edge_friction

    # Acceleration via color-based forces
    if n == 1:
        k_bb = -100 # blue is induced by blue
        k_by =   10 # blue is induced by yellow
        k_br =    1 # blue is induced by red
        k_bg =    0 # blue is induced by green
        k_yb =   10 # yellow is induced by blue
        k_yy =   -1 # yellow is induced by yellow
        k_yr =    1 # yellow is induced by red
        k_yg =    0 # yellow is induced by green
        k_rb =    1 # red is induced by blue
        k_ry =    1 # red is induced by yellow
        k_rr =    1 # red is induced by red
        k_rg =    0 # red is induced by green
        k_gb =    0 # green is induced by blue
        k_gy =    0 # green is induced by yellow
        k_gr =    0 # green is induced by red
        k_gg =    0 # green is induced by green

        for i in range(len(ATOMS)):
            ATOMS[i]["force"] = [0., 0.]
            for j in range(len(ATOMS)):
                d, sin, cos = distance(i, j)
                i_col = ATOMS[i]["color"]
                j_col = ATOMS[j]["color"]
                i_size = ATOMS[i]["size"]
                j_size = ATOMS[j]["size"]

                if d > i_size + j_size:
                    F = ATOMS[j]["size"] / d ** 2
                else:
                    F = ATOMS[j]["size"] * d / (i_size + j_size) ** 3

                if i_col == COL_BLUE:
                    if j_col == COL_BLUE:
                        F *= k_bb
                    elif j_col == COL_YELLOW:
                        F *= k_by
                    elif j_col == COL_RED:
                        F *= k_br
                    elif j_col == COL_GREEN:
                        F *= k_bg
                elif i_col == COL_YELLOW:
                    if j_col == COL_BLUE:
                        F *= k_yb
                    elif j_col == COL_YELLOW:
                        F *= k_yy
                    elif j_col == COL_RED:
                        F *= k_yr
                    elif j_col == COL_GREEN:
                        F *= k_yg
                elif i_col == COL_RED:
                    if j_col == COL_BLUE:
                        F *= k_rb
                    elif j_col == COL_YELLOW:
                        F *= k_ry
                    elif j_col == COL_RED:
                        F *= k_rr
                    elif j_col == COL_GREEN:
                        F *= k_rg
                if i_col == COL_GREEN:
                    if j_col == COL_BLUE:
                        F *= k_gb
                    elif j_col == COL_YELLOW:
                        F *= k_gy
                    elif j_col == COL_RED:
                        F *= k_gr
                    elif j_col == COL_GREEN:
                        F *= k_gg

                ATOMS[i]["force"][0] += F * cos
                ATOMS[i]["force"][1] += F * sin

            # Apply forces to velocities
            ATOMS[i]["velocity"][0] += ATOMS[i]["force"][0]
            ATOMS[i]["velocity"][1] += ATOMS[i]["force"][1]

    # Apply dead body coefficient on the velocities of particles with no energy
    if n == 2:
        dead_body_coefficient = 0.01
        for this_atom in ATOMS:
            if this_atom["energy"] <= 0:
                this_atom["velocity"][0] *= 1 - dead_body_coefficient
                this_atom["velocity"][1] *= 1 - dead_body_coefficient

    # Green atoms give energy to overlapping blue atoms.
    if n == 3:
        energy_resource = 0.2
        for i in range(len(ATOMS)):
            for j in range(len(ATOMS)):
                if (i == j or ATOMS[i]["color"] != COL_BLUE or
                    ATOMS[j]["color"] != COL_GREEN):
                    continue
                if distance(i, j)[0] < ATOMS[i]["size"] + ATOMS[j]["size"]:
                    ATOMS[i]["energy"] += energy_resource

        # if STEP[0] % 600 == 0:
        #     print("E0:", ATOMS[0]["energy"], "E1:", ATOMS[1]["energy"])

    # Let all particles with energy move a bit randomly,
    # and let all organisms further move according to their inner nature.
    if n == 4:
        move_random(0.1)
        ATOMS[0]["Organism"].move_wise()
        ATOMS[1]["Organism"].move_wise()

    # As all reds get "captured" by blues, the blue with more red wins.
    if n == 5:
        allRedsAreIn = True
        for i in range(len(ATOMS)):
            this_atom = ATOMS[i]
            if this_atom["color"] != COL_RED:
                continue

            if (distance (0, i)[0] > abs(ATOMS[0]["size"] - ATOMS[i]["size"])
                and distance (1, i)[0] > abs(ATOMS[1]["size"] -
                ATOMS[i]["size"])):
                allRedsAreIn = False
                break

        if allRedsAreIn:
            ATOMS[0]["score"] = 0
            ATOMS[1]["score"] = 0
            for i, this_atom in enumerate(ATOMS):
                if this_atom["color"] != COL_RED:
                    continue

                if (distance (0, i)[0] > abs(ATOMS[0]["size"] -
                    ATOMS[i]["size"])):
                    ATOMS[0]["score"] += 1
                if (distance (1, i)[0] > abs(ATOMS[1]["size"] -
                    ATOMS[i]["size"])):
                    ATOMS[1]["score"] += 1

            print("Score 0:", ATOMS[0]["score"])
            print("Score 1:", ATOMS[1]["score"])

            if ATOMS[0]["score"] > ATOMS[1]["score"]:
                EPOCH[0] += 1
                initiate_world([
                    ATOMS[0]["Organism"].genome,
                    mutate(ATOMS[0]["Organism"].genome)]
                )
            elif ATOMS[0]["score"] < ATOMS[1]["score"]:
                EPOCH[0] += 1
                initiate_world([
                    ATOMS[1]["Organism"].genome,
                    mutate(ATOMS[1]["Organism"].genome)]
                )
            else:
                initiate_world([
                    ATOMS[0]["Organism"].genome,
                    ATOMS[1]["Organism"].genome]
                )

    if n == 6:
        if STEP[0] == 2400:
            ATOMS[0]["score"] = 0
            ATOMS[1]["score"] = 0

            for i, this_atom in enumerate(ATOMS):
                if this_atom["color"] != COL_RED:
                    continue

                ATOMS[0]["score"] += distance (0, i)[0]
                ATOMS[1]["score"] += distance (1, i)[0]

            Organism_1 = ATOMS[0]["Organism"]
            Organism_2 = ATOMS[1]["Organism"]

            dump_genome_to_file(
                ATOMS[0]["Organism"].genome,
                ATOMS[0]["score"]
            )
            dump_genome_to_file(
                ATOMS[1]["Organism"].genome,
                ATOMS[1]["score"]
            )

            print("Score 0:", ATOMS[0]["score"])
            print("Score 1:", ATOMS[1]["score"])

            EPOCH[0] += 1

            initiate_world([random_genome(), random_genome()])

def draw_potential_field(window):
    for x in range(0, WINDOW_SIZE[0], 40):
        for y in range(0, WINDOW_SIZE[1], 40):
            force = [0., 0.]
            for atom in ATOMS:
                d_x = atom["x"] - x
                d_y = atom["y"] - y
                d = (d_x ** 2 + d_y ** 2) ** 0.5
                if d > atom["size"]:
                    F = atom["size"] / d ** 2
                    force[0] += F * (d_x / d)
                    force[1] += F * (d_y / d)
            magnitude = (force[0] ** 2 + force[1] ** 2) ** 0.5
            if magnitude > 0:
                pygame.draw.line(
                    window,
                    (0, 255, 0),
                    (x, y),
                    (
                        x + force[0] * 10 / magnitude,
                        y + force[1] * 10 / magnitude
                    ),
                    1
                )

def initiate_world(genomes):
    ATOMS.clear()
    STEP[0] = 0

    create(2, COL_BLUE, 100, 50.)
    create(12, COL_GREEN, 40)
    create(10, COL_RED, 10)
    create(160, COL_YELLOW, 3)

    ATOMS[0]["Organism"] = Organism(0, genome = genomes[0])
    ATOMS[1]["Organism"] = Organism(1, genome = genomes[1])

    # print("Epoch:", EPOCH[0])

# Mutates the genome by flipping random bits in the binary representation of
# floats. The mutation rate is the probability of mutating each gene.
def mutate(genome, mutation_rate = 0.01):
    score = genome[0]
    genome = genome[1]
    new_genome = [genome[0] + 1]
    for gene in genome[1:]:
        while True:
            gene_provisional = gene

            if random.random() < mutation_rate:
                # Convert hex string to binary (byte array)
                packed_value = bytearray(bytes.fromhex(gene))

                # Select a random byte and bit within that byte
                byte_idx = random.randint(0, len(packed_value) - 1)
                bit_idx = random.randint(0, 7)

                # Flip the bit at the selected position
                packed_value[byte_idx] ^= (1 << bit_idx)

                # Convert to hex string and store the mutated gene
                gene_provisional = packed_value.hex()

                if not np.isfinite(hex_to_float(gene_provisional)):
                    continue

            gene = gene_provisional
            break

        # Add the (possibly mutated) gene to the new genome
        new_genome.append(gene)

    return [score, new_genome]

def dump_genome_to_file(genome, score = 137137.15, folder = "genomes/x"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    unique_id = uuid4()
    filename = f"{folder}/{EPOCH[0]}_{genome[0]}_{unique_id}.txt"

    with open(filename, "w") as f:
        f.write(f"{score}\n")
        for gene in genome:
            f.write(f"{gene} ")

    return genome

class Organism:
    def __init__(self, ID = "N/A", mode = "SINGLE_CELL",
        genome = random_genome()):
        self.ID = ID
        self.genome = genome
        self.initialize(mode)

    def initialize(self, mode):
        if mode == "SINGLE_CELL":
            self.atoms = [self.ID]
            self.apply_genome()
        else:
            pass

    def apply_genome(self):
        (
        self.generation,
        self.W1,
        self.W2,
        self.W3,
        self.Wo,
        self.B1,
        self.B2,
        self.B3,
        self.Bo
        ) = decode_genome(self.genome)

    def move_random(self, max_magnitude = 0.5):
        if ATOMS[self.ID]["energy"] > 0:
            theta = random.uniform(0, 2 * math.pi)
            impulse_magnitude = min(ATOMS[self.ID]["energy"],
                random.uniform(0, max_magnitude))
            ATOMS[self.ID]["velocity"][0] += math.cos(theta) * impulse_magnitude
            ATOMS[self.ID]["velocity"][1] += math.sin(theta) * impulse_magnitude

            # Burn energy proportional to the impulse magnitude
            ATOMS[self.ID]["energy"] -= impulse_magnitude

    def sensory_inputs(self):
        input_vector = [
            ATOMS[self.ID]["x"],
            ATOMS[self.ID]["y"],
            ATOMS[self.ID]["energy"],
            ATOMS[self.ID]["velocity"][0],
            ATOMS[self.ID]["velocity"][1]
        ]

        for i in range(len(ATOMS)):
            if (i == self.ID or ATOMS[i]["color"] == COL_YELLOW):
                continue

            d, sin, cos = distance(self.ID, i)

            if d > ATOMS[self.ID]["size"] + ATOMS[i]["size"]:
                F = ATOMS[i]["size"] / d ** 2
            else:
                F = (ATOMS[i]["size"] /
                    (ATOMS[self.ID]["size"] + ATOMS[i]["size"]) ** 2)

            input_vector.append(F * cos)
            input_vector.append(F * sin)

        return input_vector

    def ff(self):
        hidden_1i = np.dot(self.sensory_inputs(), self.W1) + self.B1
        hidden_1o = sigmoid(hidden_1i)
        hidden_2i = np.dot(hidden_1o, self.W2) + self.B2
        hidden_2o = sigmoid(hidden_2i)
        hidden_3i = np.dot(hidden_2o, self.W3) + self.B3
        hidden_3o = sigmoid(hidden_3i)
        hidden_oi = np.dot(hidden_3o, self.Wo) + self.Bo
        return sigmoid(hidden_oi)

    def move_wise(self):
        speed_constant = 0.04
        impulse_vector = self.ff()
        dv_x = impulse_vector[0][0] * speed_constant
        dv_y = impulse_vector[0][1] * speed_constant
        needed_energy = (dv_x ** 2 + dv_y ** 2) ** 0.5
        coeff = 0
        if needed_energy > 0:
            coeff = min(needed_energy,
                ATOMS[self.ID]["energy"]) / needed_energy
        ATOMS[self.ID]["velocity"][0] += dv_x * coeff
        ATOMS[self.ID]["velocity"][1] += dv_y * coeff

        # Burn energy proportional to the impulse magnitude
        ATOMS[self.ID]["energy"] -= (dv_x ** 2 + dv_y ** 2) ** 0.5 * coeff


if __name__ == '__main__':
    import pygame
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Experiment 137.01: Abiogenesis")

    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    run = True
    clock = pygame.time.Clock()

    genome_1 = random_genome()
    """file = "genome_file.txt"
    with open(file, 'rb') as file:
        genes = file.readlines()[-1]
        genome_1 = [gene.decode() for gene in genes.strip().split()]
        genome_1[0] = int(genome_1[0])"""
    genome_2 = random_genome()

    initiate_world([genome_1, genome_2])

    while run:
        STEP[0] += 1
        window.fill((0, 0, 0))
        keys = pygame.key.get_pressed()

        # draw_potential_field(window)

        for this_atom in ATOMS:
            pygame.draw.circle(
                window,
                this_atom["color"],
                (int(this_atom["x"]), int(this_atom["y"])),
                this_atom["size"]
            )

            # Show organism ID.
            if this_atom["color"] == COL_BLUE and "Organism" in this_atom:
                text_surface = font.render(
                    "Gen " + str(this_atom["Organism"].genome[0]),
                    True,
                    COL_WHITE
                )
                window.blit(
                    text_surface,
                    (
                        this_atom["x"] - text_surface.get_width() / 2,
                        this_atom["y"] - text_surface.get_height() / 2
                    )
                )

        pygame.display.flip()

        Rules(0)
        Rules(1)
        Rules(2)
        Rules(3)
        Rules(4)
        # Rules(5)

        """
        # Use this block instead of the next one when you want
        # the user controls to consume energy while moving the organisms.
        speed_constant = 0.1
        if keys[pygame.K_LEFT] and ATOMS[0]["energy"] > 0:
            ATOMS[0]["velocity"][0] -= speed_constant
            ATOMS[0]["energy"] -= speed_constant
        if keys[pygame.K_RIGHT] and ATOMS[0]["energy"] > 0:
            ATOMS[0]["velocity"][0] += speed_constant
            ATOMS[0]["energy"] -= speed_constant
        if keys[pygame.K_UP] and ATOMS[0]["energy"] > 0:
            ATOMS[0]["velocity"][1] -= speed_constant
            ATOMS[0]["energy"] -= speed_constant
        if keys[pygame.K_DOWN] and ATOMS[0]["energy"] > 0:
            ATOMS[0]["velocity"][1] += speed_constant
            ATOMS[0]["energy"] -= speed_constant
        if keys[pygame.K_a] and ATOMS[1]["energy"] > 0:
            ATOMS[1]["velocity"][0] -= speed_constant
            ATOMS[1]["energy"] -= speed_constant
        if keys[pygame.K_d] and ATOMS[1]["energy"] > 0:
            ATOMS[1]["velocity"][0] += speed_constant
            ATOMS[1]["energy"] -= speed_constant
        if keys[pygame.K_w] and ATOMS[1]["energy"] > 0:
            ATOMS[1]["velocity"][1] -= speed_constant
            ATOMS[1]["energy"] -= speed_constant
        if keys[pygame.K_s] and ATOMS[1]["energy"] > 0:
            ATOMS[1]["velocity"][1] += speed_constant
            ATOMS[1]["energy"] -= speed_constant
        """

        speed_constant = 0.1
        if keys[pygame.K_LEFT]:
            ATOMS[0]["velocity"][0] -= speed_constant
        if keys[pygame.K_RIGHT]:
            ATOMS[0]["velocity"][0] += speed_constant
        if keys[pygame.K_UP]:
            ATOMS[0]["velocity"][1] -= speed_constant
        if keys[pygame.K_DOWN]:
            ATOMS[0]["velocity"][1] += speed_constant
        if keys[pygame.K_a]:
            ATOMS[1]["velocity"][0] -= speed_constant
        if keys[pygame.K_d]:
            ATOMS[1]["velocity"][0] += speed_constant
        if keys[pygame.K_w]:
            ATOMS[1]["velocity"][1] -= speed_constant
        if keys[pygame.K_s]:
            ATOMS[1]["velocity"][1] += speed_constant

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    pygame.quit()
