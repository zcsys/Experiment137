import random
import torch
import time
import pickle
import argparse
import datetime
import math
import numpy as np

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MENU_WIDTH = 180
SIMULATION_AREA_WIDTH = SCREEN_WIDTH - MENU_WIDTH

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 200)
LIGHTBLUE = (0, 100, 255)
YELLOW = (170, 170, 0)

THINGS = []
IS_PAUSED = False
STEP = [0, 0, 0, 0]
UNIVERSE = {"bulk energy": 0., "generation": 0}
DISSIPATION_COEFFICIENT = 0.9
YELLOW_FRICTION = 0.01
ENERGY_THRESHOLD = 12000
MATTER_TO_ENERGY = 1000.
MAX_SIGHT = 120
SHOW_DASHED_CIRCLES = False
SHOW_AGGREGATE_FORCE = False
SHOW_GENERATION = False
TOO_MANY_THINGS = 150
POPULATION_SIZE = 10
TOTAL_MASS = 0.
AGING_CONSTANT = 0.0001
PERIOD_TIME = 40
TIME_CHECKPOINT = time.time()
SOUND_ENABLED = True

GENOME15 = [
    0,                                  # Generation
    0., 0., 0., 0., 0., 0.,             # NN weights
    0., 0., 0., 0., 0., 0.,             # NN biases
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # Regulatory genes
]

GENOME05_580 = [
    580,
    6.715152428859048,
    10.754123767536688,
    3.877929697141971,
    19.27460219225479,
    5.588598065400369,
    20.686941010128766,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0
]

def identity(x):
    return x

def distance(i, j):
    if i == j:
        return 0, 0, 0

    i_x, i_y = THINGS[i].position
    j_x, j_y = THINGS[j].position

    d = ((j_x - i_x) ** 2 + (j_y - i_y) ** 2) ** 0.5

    if d > 0:
        sin = (j_y - i_y) / d
        cos = (j_x - i_x) / d
    else:
        return 0, 0, 0

    return d, sin, cos

def generate_thing(mass = None, energy = None, position = None, velocity = None,
    size = None, color = None, dissipation = None):
    if mass is None:
        mass = random.uniform(1, 10)
    if energy is None:
        energy = random.uniform(10, 100)
    if position is None:
        position = (
            random.randint(0, SIMULATION_AREA_WIDTH),
            random.randint(0, SCREEN_HEIGHT)
        )
    if velocity is None:
        velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
    elif isinstance(velocity, (int, float)):
        velocity = (
            velocity * random.uniform(-1, 1),
            velocity * random.uniform(-1, 1)
        )
    if size is None:
        size = random.randint(1, 10)
    if color is None:
        while True:
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            if color != YELLOW and color != LIGHTBLUE:
                break

    dissipation = DISSIPATION_COEFFICIENT
    if color == YELLOW:
        dissipation = 1. - YELLOW_FRICTION

    return Thing(mass, energy, position, velocity, size, color, dissipation)

def mutate(coding_genome, probability = 0.1, strength = 1., regulatory = []):
    mutated_genome = [coding_genome[0]]

    for i, gene in enumerate(coding_genome[1:]):
        if not regulatory or regulatory[i] and random.random() < probability:
            gene += strength * random.uniform(-1, 1)
            print(f"A mutation occurred in the gene {i + 1}.")
        mutated_genome.append(gene)

    if mutated_genome != coding_genome:
        mutated_genome[0] += 1
        print("Original genome:", coding_genome)
        print("Mutated genome:", mutated_genome)
        print("========")
        if mutated_genome[0] > UNIVERSE["generation"]:
            UNIVERSE["generation"] = mutated_genome[0]

    return mutated_genome + regulatory

def toggle_pause():
    global IS_PAUSED
    IS_PAUSED = not IS_PAUSED
    if IS_PAUSED:
        pause_button.text = "Play"
    else:
        pause_button.text = "Pause"

def toggle_sound():
    global SOUND_ENABLED
    SOUND_ENABLED = not SOUND_ENABLED
    if SOUND_ENABLED:
        sound_button.text = "Sound ON"
    else:
        sound_button.text = "Sound OFF"

def draw_dashed_circle(surface, color, center, radius, dash_length = 5):
    angle = 0
    while angle < 360:
        start_x = center[0] + radius * math.cos(math.radians(angle))
        start_y = center[1] + radius * math.sin(math.radians(angle))
        end_angle = angle + dash_length
        end_x = center[0] + radius * math.cos(math.radians(end_angle))
        end_y = center[1] + radius * math.sin(math.radians(end_angle))
        pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 1)
        angle += 2 * dash_length

def draw_info_field(screen, font):
    epoch_text = font.render(f"Epoch: {STEP[-3]}", True, BLACK)
    screen.blit(epoch_text, (SIMULATION_AREA_WIDTH + 10, 130))

    period_text = font.render(f"Period: {STEP[-2]} ({PERIOD_TIME:.0f}\")", True,
        BLACK)
    screen.blit(period_text, (SIMULATION_AREA_WIDTH + 10, 160))

    step_text = font.render(f"Step: {STEP[-1]}", True, BLACK)
    screen.blit(step_text, (SIMULATION_AREA_WIDTH + 10, 190))

    population_text = font.render(f"Pop.: {POPULATION_SIZE}", True, BLACK)
    screen.blit(population_text, (SIMULATION_AREA_WIDTH + 10, 220))

    avgmass_text = font.render(f"Avg. m: {(TOTAL_MASS / POPULATION_SIZE):.2f}",
        True, BLACK)
    screen.blit(avgmass_text, (SIMULATION_AREA_WIDTH + 10, 250))

    totalmass_text = font.render(f"Tot. m: {TOTAL_MASS:.2f}", True, BLACK)
    screen.blit(totalmass_text, (SIMULATION_AREA_WIDTH + 10, 280))

    gen_text = font.render(f"Gen.: {UNIVERSE['generation']}", True, BLACK)
    screen.blit(gen_text, (SIMULATION_AREA_WIDTH + 10, 310))

def generate_wave(frequency, duration, waveform = "sine", sample_rate = 44100,
    amplitude=400):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if waveform == "sine":
        wave = np.sin(2 * np.pi * frequency * t) * amplitude
    elif waveform == "square":
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "sawtooth":
        wave = 2 * amplitude * (t * frequency - np.floor(1/2 + t * frequency))
    elif waveform == "noise":
        wave = np.random.uniform(-1, 1, size=t.shape) * amplitude
    else:
        raise ValueError("Invalid waveform type. Choose 'sine', 'square'," +
            "'sawtooth', or 'noise'.")
    wave = wave.astype(np.int16)
    sound_array = np.array([wave, wave]).T
    sound_array = np.ascontiguousarray(sound_array)
    return pygame.sndarray.make_sound(sound_array)

def save_game_state():
    """
    Example usage of the output file:
        python3 monadology.py --load game_state.pkl
    """
    global THINGS, STEP, UNIVERSE

    game_state = {
        "THINGS": THINGS,
        "STEP": STEP,
        "UNIVERSE": UNIVERSE
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SavedState{timestamp}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(game_state, f)
    print(f"Game state saved to {filename}")

def load_game_state(filename):
    global THINGS, STEP, IS_PAUSED, UNIVERSE

    with open(filename, 'rb') as f:
        game_state = pickle.load(f)

    THINGS = game_state["THINGS"]
    STEP = game_state["STEP"]
    IS_PAUSED = True
    UNIVERSE = game_state["UNIVERSE"]
    print(f"Game state loaded from {filename}")

def toggle_visual(visual_type):
    global SHOW_DASHED_CIRCLES, SHOW_AGGREGATE_FORCE, SHOW_GENERATION
    if visual_type == "dashed_circles":
        SHOW_DASHED_CIRCLES = not SHOW_DASHED_CIRCLES
    elif visual_type == "aggregate_force":
        SHOW_AGGREGATE_FORCE = not SHOW_AGGREGATE_FORCE
    elif visual_type == "generations":
        SHOW_GENERATION = not SHOW_GENERATION

def initialize_world(N = 10):
    global THINGS, STEP, UNIVERSE
    THINGS.clear()
    STEP = [0, 0, 0, 0]
    UNIVERSE = {"bulk energy": 0., "generation": 0}
    """generate_thing(
        mass = 10,
        energy = 1000.,
        size = 10,
        color = LIGHTBLUE,
        velocity = (0., 0.)
    )"""
    for _ in range(N):
        generate_thing(
            mass = 10,
            energy = 1000.,
            size = 10,
            velocity = 0.
        )
    for i in range(len(THINGS)):
        Monad15(i, GENOME15)
    while len(THINGS) < TOO_MANY_THINGS:
        generate_thing(
            mass = 1,
            energy = 1000.,
            size = 1,
            color = YELLOW,
            velocity = 0.
        )

def reset_simulation():
    save_game_state()
    initialize_world()

def step_on():
    global STEP, PERIOD_TIME, TIME_CHECKPOINT
    STEP[-1] += 1
    if STEP[-1] == 2400:
        STEP[-2] += 1
        STEP[-1] = 0
        time_now = time.time()
        PERIOD_TIME = time_now - TIME_CHECKPOINT
        TIME_CHECKPOINT = time_now
    if STEP[-2] == 80:
        STEP[-3] += 1
        STEP[-2] = 0
    if STEP[-3] == 80:
        STEP[-4] += 1
        STEP[-3] = 0

class Button:
    def __init__(self, text, x, y, width, height, color, hover_color,
        clicked_color,  action = None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.clicked_color = clicked_color
        self.action = action
        self.is_pressed = False

    def draw(self, screen, font):
        mouse_pos = pygame.mouse.get_pos()
        if self.is_pressed:
            current_color = self.clicked_color
        elif self.rect.collidepoint(mouse_pos):
            current_color = self.hover_color
        else:
            current_color = self.color

        pygame.draw.rect(screen, current_color, self.rect)

        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center = self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                if self.action:
                    self.action()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_pressed = False

class Thing:
    def __init__(self, mass, energy, position, velocity, size, color,
        dissipation = None):
        self.mass = mass
        self.energy = energy
        self.position = position
        self.velocity = velocity
        self.size = size
        self.color = color
        self.dissipation = dissipation
        THINGS.append(self)

    def apply_velocity(self):
        absolute_max_velocity = 100
        self.velocity = (
            max(
                min(
                    self.velocity[0],
                    absolute_max_velocity
                ),
                -absolute_max_velocity
            ),
            max(
                min(
                    self.velocity[1],
                    absolute_max_velocity
                ),
                -absolute_max_velocity
            )
        )
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )

    def edge_collisions_and_dissipation(self):
        x, y = self.position
        vx, vy = self.velocity

        if x <= self.size:
            x = self.size
            vx *= -1.
        elif x >= SIMULATION_AREA_WIDTH - self.size:
            x = SIMULATION_AREA_WIDTH - self.size
            vx *= -1.

        if y <= self.size:
            y = self.size
            vy *= -1.
        elif y >= SCREEN_HEIGHT - self.size:
            y = SCREEN_HEIGHT - self.size
            vy *= -1.

        self.position = (x, y)
        self.velocity = (vx * self.dissipation, vy * self.dissipation)

    def move_random(self, max_magnitude = 0.1):
        self.velocity = (
            self.velocity[0] + max_magnitude * random.uniform(-1., 1.),
            self.velocity[1] + max_magnitude * random.uniform(-1., 1.)
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.position, self.size)

    def impulse(self, magnitude = None):
        speed_constant = 0.1
        if magnitude is None:
            dvx = speed_constant * random.uniform(-1., 1.)
            dvy = speed_constant * random.uniform(-1., 1.)
        else:
            dvx = speed_constant * magnitude[0]
            dvy = speed_constant * magnitude[1]
        self.velocity = (
            self.velocity[0] + dvx,
            self.velocity[1] + dvy
        )
        energy = (dvx ** 2 + dvy ** 2) ** 0.5 * self.mass
        self.energy -= energy
        # UNIVERSE["bulk energy"] += energy

        if SOUND_ENABLED:
            movement_sound = generate_wave(energy * 1000, 0.1, "sine")
            movement_sound.play()

    def m_to_the_e(self, amount = None):
        MATTER_TO_ENERGY = 1000.
        if amount is None:
            amount = random.uniform(-1., 1.)
        if self.mass > amount and self.energy >= -amount * MATTER_TO_ENERGY:
            self.mass -= amount
            self.energy += amount * MATTER_TO_ENERGY

class Monad15(Thing):
    def __init__(self, i, genome):
        self.mass = THINGS[i].mass
        self.energy = THINGS[i].energy
        self.position = THINGS[i].position
        self.velocity = THINGS[i].velocity
        self.size = THINGS[i].size
        self.color = THINGS[i].color
        self.dissipation = THINGS[i].dissipation

        self.device = torch.device("cpu")

        self.apply_genome(genome)

        THINGS[i] = self

    def sensory_inputs(self):
        self.input_vector = []

        # Self knowledge
        self.input_vector.append([self.mass, self.energy])
        self.input_vector.append([self.velocity[0], self.velocity[1]])

        # The closest walls
        distance_to_walls = [0., 0.]
        d_to_left = max(self.position[0], 1)
        d_to_right = max(SIMULATION_AREA_WIDTH - self.position[0], 1)
        d_to_bottom = max(SCREEN_HEIGHT - self.position[1], 1)
        d_to_top = max(self.position[1], 1)
        if d_to_left < d_to_right:
            distance_to_walls[0] = 1 / d_to_left
        else:
            distance_to_walls[0] = -1 / d_to_right
        if d_to_bottom < d_to_top:
            distance_to_walls[1] = -1 / d_to_bottom
        else:
            distance_to_walls[1] = 1 / d_to_top
        self.input_vector.append(distance_to_walls)

        # Aggregate influence from yellow particles
        aggregate_influence = [0., 0.]
        for j, other_thing in enumerate(THINGS):
            if other_thing.color != YELLOW:
                continue
            d, sin, cos = distance(THINGS.index(self), j)
            if d <= MAX_SIGHT:
                if d > self.size:
                    aggregate_influence[0] += cos / d
                    aggregate_influence[1] += sin / d
                else:
                    aggregate_influence[0] += cos * d / self.size ** 2
                    aggregate_influence[1] += sin * d / self.size ** 2
        self.input_vector.append(aggregate_influence)

        if SHOW_AGGREGATE_FORCE:
            line_end_position = (
                int(self.position[0] + aggregate_influence[0] * 1000),
                int(self.position[1] + aggregate_influence[1] * 1000)
            )

            pygame.draw.line(
                screen,
                (0, 170, 0),
                self.position,
                line_end_position,
                2
            )

            line_end_position = (
                int(self.position[0] + distance_to_walls[0] * 1000),
                int(self.position[1] + distance_to_walls[1] * 1000)
            )

            pygame.draw.line(
                screen,
                (170, 0, 0),
                self.position,
                line_end_position,
                2
            )

        if SHOW_DASHED_CIRCLES:
            draw_dashed_circle(screen, WHITE, self.position, MAX_SIGHT)

    def cell_division(self, energy_to_give):
        if self.energy < energy_to_give:
            return

        mutated_genome = mutate(
            coding_genome = self.genome[0:13],
            regulatory = self.regulatory_genes
        )

        diff = [mutated_genome[i+1] - self.genome[i+1]
            for i in range(len(mutated_genome) - 1)]
        mutated_color = (
            max(min(self.color[0] + int(sum(diff[0:4])) * 100, 255), 100),
            max(min(self.color[1] + int(sum(diff[4:8])) * 100, 255), 100),
            max(min(self.color[2] + int(sum(diff[8:12])) * 100, 255), 100)
        )

        generate_thing(
            mass = 10,
            energy = 1000.,
            position = self.position,
            size = 10,
            color = mutated_color,
            velocity = 10.
        )
        Monad15(-1, mutated_genome)
        self.energy -= energy_to_give

    def apply_genome(self, genome):
        self.genome = genome
        self.generation = genome[0]

        self.input_to_hidden_weights = torch.tensor(
            genome[1:5],
            dtype = torch.float32,
            device = self.device
        ).reshape(2, 2)
        self.input_to_hidden_biases = torch.tensor(
            genome[7:11],
            dtype = torch.float32,
            device = self.device
        ).reshape(2, 2)
        self.hidden_to_output_weights = torch.tensor(
            genome[5:7],
            dtype = torch.float32,
            device = self.device
        ).reshape(1, 2)
        self.hidden_to_output_biases = torch.tensor(
            genome[11:13],
            dtype = torch.float32,
            device = self.device
        ).reshape(1, 2)

        self.regulatory_genes = genome[13:25]

    def feed_forward(self):
        self.action_vector = [
            0.,
            [0., 0.]
        ]

        input_tensor = torch.tensor(
            self.input_vector[2:4],
            dtype = torch.float32,
            device = self.device
        )

        hidden_layer = torch.matmul(self.input_to_hidden_weights,
            input_tensor) + self.input_to_hidden_biases
        hidden_layer = torch.tanh(hidden_layer)

        output_layer = torch.matmul(self.hidden_to_output_weights,
            hidden_layer) + self.hidden_to_output_biases
        output_layer = torch.tanh(output_layer)

        self.action_vector[1] = output_layer[0].tolist()

        if SHOW_AGGREGATE_FORCE:
            line_end_position = (
                int(self.position[0] + self.action_vector[1][0] * 100),
                int(self.position[1] + self.action_vector[1][1] * 100)
            )

            pygame.draw.line(
                screen,
                (255, 255, 170),
                self.position,
                line_end_position,
                2
            )

    def action(self):
        matter_to_energy = self.action_vector[0]
        movement = [x * 10 for x in self.action_vector[1]]

        self.m_to_the_e(matter_to_energy)
        self.impulse(movement)

    def draw(self, screen):
        super().draw(screen)
        if SHOW_GENERATION:
            font = pygame.font.SysFont(None, 12)
            generation_text = font.render(str(self.generation), True, BLACK)
            text_rect = generation_text.get_rect(center = self.position)
            screen.blit(generation_text, text_rect)

def Rules(n, arg1 = None):
    global THINGS, IS_PAUSED, STEP, UNIVERSE, AGING_CONSTANT
    global POPULATION_SIZE, TOTAL_MASS, TOO_MANY_THINGS

    # Calculate distance tensor at each step
    if n == 0:
        positions = torch.tensor(
            [thing.position for thing in THINGS],
            dtype = torch.float32,
            device = "mps")
        distances, sin_theta, cos_theta = calculate_distances(positions)

        # Copy of Rules(2) but using distances[] instead of distance().
        yellows_to_remove = []
        for i, this_thing in enumerate(THINGS):
            for j, other_thing in enumerate(THINGS):
                if i == j:
                    continue
                if (
                    this_thing.color != YELLOW and
                    other_thing.color == YELLOW and
                    distances[i, j] <= this_thing.size and
                    j not in yellows_to_remove
                ):
                    this_thing.energy += other_thing.energy
                    yellows_to_remove.append(j)
        THINGS = [x for i, x in enumerate(THINGS) if i not in yellows_to_remove]

    # Movements created by impulse() and other activities may give energy to
    # system. When threshold reached, the system pops "energy particles".
    if n == 1:
        while UNIVERSE["bulk energy"] >= 1000 and len(THINGS) < TOO_MANY_THINGS:
            generate_thing(
                mass = 0,
                energy = 1000,
                size = 1,
                color = YELLOW,
                velocity = 0.
            )
            UNIVERSE["bulk energy"] -= 1000

    # The "energy particles" are consumed by other things.
    if n == 2:
        yellows_to_remove = []
        for i, this_thing in enumerate(THINGS):
            for j, other_thing in enumerate(THINGS):
                if i == j:
                    continue
                if (
                    this_thing.color != YELLOW and
                    other_thing.color == YELLOW and
                    distance(i, j)[0] <= this_thing.size and
                    j not in yellows_to_remove
                ):
                    this_thing.energy += other_thing.energy
                    yellows_to_remove.append(j)
        THINGS = [x for i, x in enumerate(THINGS) if i not in yellows_to_remove]

    # Processes at each epoch's start
    if n == 3:
        if STEP[-1] == 0:
            if len(THINGS) < TOO_MANY_THINGS:
                for _ in range(100):
                    generate_thing(
                        mass = 0,
                        energy = 1000.,
                        size = 1,
                        color = YELLOW,
                        velocity = 0.1
                    )

            if STEP[-2] != 0: # Skip at the very start
                organisms_to_die = []
                for i, thing in enumerate(THINGS):
                    if i == 0: # immortality to THINGS[0]
                        continue
                    if thing.color != YELLOW:
                        thing_oldmass = thing.mass
                        thing.mass -= 1
                        if thing.mass == 0:
                            # UNIVERSE["bulk energy"] += thing.energy
                            organisms_to_die.append(i)
                        else:
                            # Comment out for easier initial evolution
                            # Uncomment for harder next rounds
                            thing.size *= (thing.mass / thing_oldmass) ** 0.5
                            thing.size = max(5, thing.size)

                THINGS = [x for i, x in enumerate(THINGS)
                    if i not in organisms_to_die]
            elif STEP[-3] != 0 or STEP[-4] != 0:
                    save_game_state()

    # Keyboard control for THINGS[0]
    if n == 4:
        keys = arg1
        dvx, dvy = (0., 0.)
        if THINGS[0].energy > 0:
            if keys[pygame.K_LEFT]:
                dvx -= 1
            if keys[pygame.K_RIGHT]:
                dvx += 1
            if keys[pygame.K_UP]:
                dvy -= 1
            if keys[pygame.K_DOWN]:
                dvy += 1
            if keys[pygame.K_SPACE]:
                THINGS[0].cell_division(11000.)
        THINGS[0].impulse([dvx, dvy])

        if SHOW_AGGREGATE_FORCE:
            line_end_position = (
                int(THINGS[0].position[0] + dvx * 1000),
                int(THINGS[0].position[1] + dvy * 1000)
            )
            pygame.draw.line(
                screen,
                (255, 255, 170),
                THINGS[0].position,
                line_end_position,
                2
            )

    # Population control
    if n == 5:
        masses = [x.mass for x in THINGS if x.color != YELLOW]
        TOTAL_MASS = sum(masses)
        POPULATION_SIZE = len(masses)

        if POPULATION_SIZE <= 20:
            AGING_CONSTANT = 0.0001
        elif POPULATION_SIZE > 20 and POPULATION_SIZE <= 30:
            AGING_CONSTANT = 0.0001 * (POPULATION_SIZE - 20)
        elif POPULATION_SIZE > 30:
            AGING_CONSTANT = 0.001 * (POPULATION_SIZE - 30)

    # Generate energy particles to top up population
    if n == 6:
        while len(THINGS) < TOO_MANY_THINGS:
            generate_thing(
                mass = 0,
                energy = 1000.,
                size = 1,
                color = YELLOW,
                velocity = 0.
            )

    # Autosave
    if n == 7:
        if STEP[-1] == 0 and STEP[-2] == 0 and (STEP[-3] != 0 or STEP[-4] != 0):
            save_game_state()

    # Aging and death
    if n == 8:
        organisms_to_die = []
        for i, thing in enumerate(THINGS):
            if thing.color != YELLOW:
                thing_oldmass = thing.mass
                thing.mass -= AGING_CONSTANT
                if thing.mass <= 0:
                    organisms_to_die.append(i)
                    # UNIVERSE["bulk energy"] += thing.energy
                else:
                    thing.size = max(5, round(thing.mass))

        THINGS = [x for i, x in enumerate(THINGS)
            if i not in organisms_to_die]

if __name__ == '__main__':
    import pygame
    import sys

    parser = argparse.ArgumentParser(
        description = "Experiment 137.02: Monadology"
    )
    parser.add_argument(
        '--load',
        type = str,
        help = 'Filename to load game state from',
        default = None
    )
    args = parser.parse_args()

    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Experiment 137.02: Monadology")
    font = pygame.font.SysFont(None, 24)

    if args.load:
        load_game_state(args.load)
    else:
        initialize_world(10)

    reset_button = Button("Reset", SIMULATION_AREA_WIDTH + 10, 10, 160, 50,
        BLUE, (0, 0, 150), LIGHTBLUE, reset_simulation)
    pause_button = Button("Play" if IS_PAUSED else "Pause",
        SIMULATION_AREA_WIDTH + 10, 70, 160, 50, BLUE, (0, 0, 150), LIGHTBLUE,
        toggle_pause)

    dashed_circle_button = Button("Toggle sight", SIMULATION_AREA_WIDTH + 10,
        340, 160, 50, BLUE, (0, 0, 150), LIGHTBLUE, lambda:
        toggle_visual("dashed_circles"))
    aggregate_force_button = Button("Toggle force", SIMULATION_AREA_WIDTH + 10,
        400, 160, 50, BLUE, (0, 0, 150), LIGHTBLUE, lambda:
        toggle_visual("aggregate_force"))
    generations_button = Button("Generations", SIMULATION_AREA_WIDTH + 10,
        460, 160, 50, BLUE, (0, 0, 150), LIGHTBLUE, lambda:
        toggle_visual("generations"))
    save_button = Button("Save", SIMULATION_AREA_WIDTH + 10, 520, 160, 50, BLUE,
        (0, 0, 150), LIGHTBLUE, save_game_state)
    sound_button = Button("Sound ON", SIMULATION_AREA_WIDTH + 10, 580, 160, 50,
        BLUE, (0, 0, 150), LIGHTBLUE, toggle_sound)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            reset_button.is_clicked(event)
            pause_button.is_clicked(event)
            save_button.is_clicked(event)
            dashed_circle_button.is_clicked(event)
            aggregate_force_button.is_clicked(event)
            generations_button.is_clicked(event)
            sound_button.is_clicked(event)
        keys = pygame.key.get_pressed()

        screen.fill(WHITE)
        pygame.draw.rect(
            screen,
            BLACK,
            (0, 0, SIMULATION_AREA_WIDTH, SCREEN_HEIGHT)
        )
        pygame.draw.rect(
            screen,
            GRAY,
            (SIMULATION_AREA_WIDTH, 0, MENU_WIDTH, SCREEN_HEIGHT)
        )
        reset_button.draw(screen, font)
        pause_button.draw(screen, font)
        draw_info_field(screen, font)
        save_button.draw(screen, font)
        dashed_circle_button.draw(screen, font)
        aggregate_force_button.draw(screen, font)
        generations_button.draw(screen, font)
        sound_button.draw(screen, font)
        for thing in THINGS:
            thing.draw(screen)

        if not IS_PAUSED:
            for thing in THINGS:
                thing.apply_velocity()
                thing.edge_collisions_and_dissipation()
                thing.move_random()
                if (thing.color != YELLOW and thing.color != LIGHTBLUE and
                    thing.energy >= ENERGY_THRESHOLD):
                    thing.cell_division(ENERGY_THRESHOLD - 1000)
                if thing.color != YELLOW:
                    thing.sensory_inputs()
                    thing.feed_forward()
                    thing.action()

            # Rules(0)
            # Rules(1)
            Rules(2)
            # Rules(3)
            # Rules(4, keys)
            Rules(5)
            Rules(6)
            Rules(7)
            Rules(8)

            step_on()

        pygame.display.flip()

    pygame.quit()
    sys.exit()
