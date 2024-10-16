import pygame
import numpy as np
from Box2D.b2 import world, polygonShape, dynamicBody, vec2

# Constants
NUM_TRIANGLES = 100
WORLD_WIDTH, WORLD_HEIGHT = 1920, 1080  # Full HD Screen resolution
TRIANGLE_SIZE = 30  # Side length of equilateral triangle
COLORS = ['black', 'blue', 'green', 'cyan', 'red', 'magenta', 'yellow', 'white']
FPS = 60
REPULSION_THRESHOLD = 10  # Distance from edge to start applying repulsive force
REPULSION_STRENGTH = 10  # Strength of repulsive force
COLOR_REPULSION_STRENGTH = 200  # Strength of color repulsive force
COLOR_REPULSION_DISTANCE = 200  # Maximum distance for color repulsion to apply
COLOR_ATTRACTION_STRENGTH = 100  # Strength of color attraction force
COLOR_ATTRACTION_DISTANCE = 200  # Maximum distance for color attraction to apply

# Define color attraction pairs
ATTRACTING_COLOR_PAIRS = {
    'red': 'cyan',
    'cyan': 'red',
    'green': 'magenta',
    'magenta': 'green',
    'blue': 'yellow',
    'yellow': 'blue',
    'black': 'white',
    'white': 'black'
}

# Set up pygame
pygame.init()
screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
clock = pygame.time.Clock()

# Set up the Box2D world
world = world(gravity=(0, 0), doSleep=True)

# Define boundaries of the universe (the screen edges) with elastic restitution and no friction
def create_boundary(x1, y1, x2, y2):
    body = world.CreateStaticBody(position=(0, 0))
    # Friction = 0 to prevent sliding, Restitution = 1 for perfectly elastic collision
    body.CreateEdgeFixture(vertices=[(x1, y1), (x2, y2)], friction=0.0, restitution=1.0)
    return body

# Create screen boundaries
create_boundary(0, 0, WORLD_WIDTH, 0)  # Bottom
create_boundary(0, 0, 0, WORLD_HEIGHT)  # Left
create_boundary(WORLD_WIDTH, 0, WORLD_WIDTH, WORLD_HEIGHT)  # Right
create_boundary(0, WORLD_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT)  # Top

# Function to create a new triangle with random color divisions
def create_triangle(world, pos, size):
    body = world.CreateDynamicBody(position=pos)
    vertices = [(0, 0), (size, 0), (size / 2, np.sqrt(3) / 2 * size)]
    # Friction = 0 to prevent sliding, Restitution = 1 for elastic collision
    body.CreatePolygonFixture(vertices=vertices, density=1.0, friction=0.0, restitution=1.0)

    # Assign random velocities
    body.linearVelocity = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
    body.angularVelocity = np.random.uniform(-5, 5)

    # Assign different colors to the 3 sub-triangles
    triangle_data = {
        "body": body,
        "colors": np.random.choice(COLORS, 3, replace=False)  # 3 unique colors
    }
    return triangle_data

# Create triangles
triangles = []
for _ in range(NUM_TRIANGLES):
    x = np.random.uniform(TRIANGLE_SIZE, WORLD_WIDTH - TRIANGLE_SIZE)
    y = np.random.uniform(TRIANGLE_SIZE, WORLD_HEIGHT - TRIANGLE_SIZE)
    triangles.append(create_triangle(world, (x, y), TRIANGLE_SIZE))

# Function to apply repulsive force when triangles get close to the boundaries
def apply_repulsive_force(triangle):
    body = triangle['body']
    pos = body.position

    # Check proximity to each boundary and apply repulsive force if close enough
    if pos.x < REPULSION_THRESHOLD:  # Left edge
        force = vec2(REPULSION_STRENGTH * (REPULSION_THRESHOLD - pos.x) / REPULSION_THRESHOLD, 0)
        body.ApplyForce(force, body.worldCenter, wake=True)
    elif pos.x > WORLD_WIDTH - REPULSION_THRESHOLD:  # Right edge
        force = vec2(-REPULSION_STRENGTH * (pos.x - (WORLD_WIDTH - REPULSION_THRESHOLD)) / REPULSION_THRESHOLD, 0)
        body.ApplyForce(force, body.worldCenter, wake=True)

    if pos.y < REPULSION_THRESHOLD:  # Bottom edge
        force = vec2(0, REPULSION_STRENGTH * (REPULSION_THRESHOLD - pos.y) / REPULSION_THRESHOLD)
        body.ApplyForce(force, body.worldCenter, wake=True)
    elif pos.y > WORLD_HEIGHT - REPULSION_THRESHOLD:  # Top edge
        force = vec2(0, -REPULSION_STRENGTH * (pos.y - (WORLD_HEIGHT - REPULSION_THRESHOLD)) / REPULSION_THRESHOLD)
        body.ApplyForce(force, body.worldCenter, wake=True)

# Function to apply color-based repulsion force between triangles
def apply_color_repulsion(triangles):
    for i, triangle1 in enumerate(triangles):
        for j, triangle2 in enumerate(triangles):
            if i != j:  # Only compare different triangles
                pos1 = triangle1['body'].position
                pos2 = triangle2['body'].position
                distance = np.linalg.norm([pos1.x - pos2.x, pos1.y - pos2.y])

                if distance < COLOR_REPULSION_DISTANCE:
                    # Check if any of the colors match between triangle1 and triangle2
                    for color1 in triangle1['colors']:
                        if color1 in triangle2['colors']:
                            # Apply repulsive force proportional to the distance between the triangles
                            direction = vec2(pos1.x - pos2.x, pos1.y - pos2.y)
                            direction.Normalize()
                            force = direction * COLOR_REPULSION_STRENGTH * (COLOR_REPULSION_DISTANCE - distance) / COLOR_REPULSION_DISTANCE
                            triangle1['body'].ApplyForce(force, triangle1['body'].worldCenter, wake=True)
                            triangle2['body'].ApplyForce(-force, triangle2['body'].worldCenter, wake=True)

# Function to apply color-based attraction between specific color pairs
def apply_color_attraction(triangles):
    for i, triangle1 in enumerate(triangles):
        for j, triangle2 in enumerate(triangles):
            if i != j:  # Only compare different triangles
                pos1 = triangle1['body'].position
                pos2 = triangle2['body'].position
                distance = np.linalg.norm([pos1.x - pos2.x, pos1.y - pos2.y])

                if distance < COLOR_ATTRACTION_DISTANCE:
                    # Check for attraction between specific color pairs
                    for color1 in triangle1['colors']:
                        for color2 in triangle2['colors']:
                            if ATTRACTING_COLOR_PAIRS.get(color1) == color2:
                                # Apply attractive force inversely proportional to the distance
                                direction = vec2(pos2.x - pos1.x, pos2.y - pos1.y)
                                direction.Normalize()
                                force = direction * COLOR_ATTRACTION_STRENGTH * (COLOR_ATTRACTION_DISTANCE - distance) / COLOR_ATTRACTION_DISTANCE
                                triangle1['body'].ApplyForce(force, triangle1['body'].worldCenter, wake=True)
                                triangle2['body'].ApplyForce(-force, triangle2['body'].worldCenter, wake=True)

# Function to draw a triangle using pygame
def draw_triangle(triangle_data):
    body = triangle_data["body"]
    pos = body.position
    angle = body.angle
    vertices = body.fixtures[0].shape.vertices

    # Rotate and translate vertices
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    vertices = [np.dot(rot_matrix, v) + pos for v in vertices]

    # Convert vertices to pygame coordinates (integer pixels)
    vertices = [(int(v[0]), int(WORLD_HEIGHT - v[1])) for v in vertices]

    # Calculate the centroid of the main triangle to use for sub-triangles
    centroid_x = sum([v[0] for v in vertices]) / 3
    centroid_y = sum([v[1] for v in vertices]) / 3
    centroid = (int(centroid_x), int(centroid_y))

    # Now divide the triangle into smaller sub-triangles for coloring
    for i, color in enumerate(triangle_data["colors"]):
        # Each sub-triangle uses the centroid and two vertices of the original triangle
        small_vertices = [centroid, vertices[i], vertices[(i + 1) % 3]]
        pygame.draw.polygon(screen, pygame.Color(color), small_vertices)

        # Draw the white border for sub-triangles
    for i in range(3):
        small_vertices = [centroid, vertices[i], vertices[(i + 1) % 3]]
        pygame.draw.polygon(screen, pygame.Color("white"), small_vertices, 1)

    # Draw the outline of the main triangle with a white border
    pygame.draw.polygon(screen, pygame.Color("white"), vertices, 1)

# Simulation loop
running = True
time_step = 1.0

while running:
    screen.fill((0, 0, 0))  # Clear screen with black background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Apply repulsive force to each triangle when it gets close to the edges
    for triangle in triangles:
        apply_repulsive_force(triangle)
        draw_triangle(triangle)

    # Apply color-based repulsion between triangles
    apply_color_repulsion(triangles)

    # Apply color-based attraction between triangles
    apply_color_attraction(triangles)

    # Update physics
    world.Step(time_step, 6, 2)

    pygame.display.flip()  # Refresh the screen
    # clock.tick(FPS)

pygame.quit()
