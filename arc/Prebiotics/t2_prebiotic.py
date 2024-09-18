import pygame
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Grid size and visualization parameters
CELL_SIZE = 10
WIDTH = 100
HEIGHT = 100

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE))
pygame.display.set_caption("Ant Colony Simulation")

# Ant Agent
class Ant(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.has_food = False

    def move(self):
        # Move randomly on the grid
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        # Simple behavior: move randomly and search for food
        if not self.has_food:
            self.move()
            # Check for food in current location
            if self.pos == self.model.food_pos:
                self.has_food = True

# AntColony Model
class AntColonyModel(Model):
    def __init__(self, width, height, N):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.food_pos = (width // 2, height // 2)  # Food at center

        # Create ants
        for i in range(N):
            ant = Ant(i, self)
            self.schedule.add(ant)
            self.grid.place_agent(ant, (random.randint(0, width - 1), random.randint(0, height - 1)))

    def step(self):
        self.schedule.step()

# Function to draw the grid
def draw_grid(screen, model):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x, y) == model.food_pos:
                pygame.draw.rect(screen, GREEN, rect)  # Draw food as green
            else:
                pygame.draw.rect(screen, WHITE, rect, 1)  # Empty cells

# Function to draw the ants
def draw_ants(screen, model):
    for content, pos in model.grid.coord_iter():
        x, y = pos
        for obj in content:
            if isinstance(obj, Ant):
                color = RED if obj.has_food else BLACK
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)  # Draw ants as red if they have food, otherwise black

# Main simulation loop
def run_simulation():
    # Initialize the model
    model = AntColonyModel(WIDTH, HEIGHT, 100)

    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle events (exit if the user closes the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step the model
        if running:  # Main loop inside an if statement
            model.step()

        # Draw the updated simulation
        screen.fill(WHITE)  # Clear the screen
        draw_grid(screen, model)  # Draw grid
        draw_ants(screen, model)  # Draw ants

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(10)  # Set FPS to 10

    pygame.quit()

# Run the simulation
if __name__ == "__main__":
    run_simulation()
