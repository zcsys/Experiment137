import pygame
import numpy as np
import math
from Box2D.b2 import world, dynamicBody, vec2

# Constants
NUM_CIRCLES = 100
WORLD_WIDTH, WORLD_HEIGHT = 1920, 1080  # Full HD Screen resolution
CIRCLE_RADIUS = 10  # Radius of the circle
NUM_SEGMENTS = 3  # Number of segments per circle
COLORS = ['black', 'blue', 'green', 'cyan', 'red', 'magenta', 'yellow', 'white']
FPS = 60
REPULSION_THRESHOLD = 10  # Distance from edge to start applying repulsive force
REPULSION_STRENGTH = 10  # Strength of repulsive force
COLOR_REPULSION_STRENGTH = 100  # Strength of color repulsive force
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

# Define boundaries of the universe (the screen edges)
def create_boundary(x1, y1, x2, y2):
    body = world.CreateStaticBody(position=(0, 0))
    body.CreateEdgeFixture(vertices=[(x1, y1), (x2, y2)], friction=0.0, restitution=1.0)
    return body

# Create screen boundaries
create_boundary(0, 0, WORLD_WIDTH, 0)  # Bottom
create_boundary(0, 0, 0, WORLD_HEIGHT)  # Left
create_boundary(WORLD_WIDTH, 0, WORLD_WIDTH, WORLD_HEIGHT)  # Right
create_boundary(0, WORLD_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT)  # Top

# Function to create a new circle with random color divisions
def create_circle(world, pos, radius, num_segments):
    body = world.CreateDynamicBody(position=pos)
    body.CreateCircleFixture(radius=radius, density=1.0, friction=0.0, restitution=1.0)

    # Assign random velocities
    body.linearVelocity = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
    body.angularVelocity = np.random.uniform(-0.1, 0.1)

    # Assign different colors to each segment of the circle
    circle_data = {
        "body": body,
        "colors": np.random.choice(COLORS, num_segments, replace=False),  # Unique colors per segment
        "radius": radius,
        "num_segments": num_segments
    }
    return circle_data

# Create circles
circles = []
for _ in range(NUM_CIRCLES):
    x = np.random.uniform(CIRCLE_RADIUS, WORLD_WIDTH - CIRCLE_RADIUS)
    y = np.random.uniform(CIRCLE_RADIUS, WORLD_HEIGHT - CIRCLE_RADIUS)
    circles.append(create_circle(world, (x, y), CIRCLE_RADIUS, NUM_SEGMENTS))

# Function to draw a section of the circle
def draw_segment(surface, color, center, radius, start_angle, end_angle, rotation_angle):
    points = [center]
    for angle in range(int(start_angle), int(end_angle) + 1):
        rotated_angle = angle + rotation_angle  # Apply rotation to the angle
        x = center[0] + int(radius * math.cos(math.radians(rotated_angle)))
        y = center[1] + int(radius * math.sin(math.radians(rotated_angle)))
        points.append((x, y))
    pygame.draw.polygon(surface, color, points)

# Function to draw a circle with segments
def draw_circle(circle_data):
    body = circle_data["body"]
    pos = body.position
    radius = circle_data["radius"]
    num_segments = circle_data["num_segments"]

    center = (int(pos[0]), int(WORLD_HEIGHT - pos[1]))  # Convert Box2D to pygame coordinates

    # Calculate the angle step for each segment
    angle_step = 360 / num_segments
    rotation_angle = math.degrees(body.angle)  # Get body's current rotation in degrees

    # Draw each segment of the circle with its respective color
    for i, color in enumerate(circle_data["colors"]):
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step
        draw_segment(screen, pygame.Color(color), center, radius, start_angle, end_angle, rotation_angle)

    # Draw the outline of the circle
    pygame.draw.circle(screen, pygame.Color("white"), center, radius, 1)

# Function to apply repulsive force when circles get close to the boundaries
def apply_repulsive_force(circle):
    body = circle['body']
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

# Function to apply color-based repulsion force between circles
def apply_color_repulsion(circles):
    for i, circle1 in enumerate(circles):
        for j, circle2 in enumerate(circles):
            if i != j:  # Only compare different circles
                pos1 = circle1['body'].position
                pos2 = circle2['body'].position
                distance = np.linalg.norm([pos1.x - pos2.x, pos1.y - pos2.y])

                if distance < COLOR_REPULSION_DISTANCE:
                    # Check if any of the colors match between circle1 and circle2
                    for color1 in circle1['colors']:
                        if color1 in circle2['colors']:
                            # Apply repulsive force proportional to the distance between the circles
                            direction = vec2(pos1.x - pos2.x, pos1.y - pos2.y)
                            direction.Normalize()
                            force = direction * COLOR_REPULSION_STRENGTH * (COLOR_REPULSION_DISTANCE - distance) / COLOR_REPULSION_DISTANCE
                            circle1['body'].ApplyForce(force, circle1['body'].worldCenter, wake=True)
                            circle2['body'].ApplyForce(-force, circle2['body'].worldCenter, wake=True)

# Function to apply color-based attraction between specific color pairs
def apply_color_attraction(circles):
    for i, circle1 in enumerate(circles):
        for j, circle2 in enumerate(circles):
            if i != j:  # Only compare different circles
                pos1 = circle1['body'].position
                pos2 = circle2['body'].position
                distance = np.linalg.norm([pos1.x - pos2.x, pos1.y - pos2.y])

                if distance < COLOR_ATTRACTION_DISTANCE:
                    # Check for attraction between specific color pairs
                    for color1 in circle1['colors']:
                        for color2 in circle2['colors']:
                            if ATTRACTING_COLOR_PAIRS.get(color1) == color2:
                                # Apply attractive force inversely proportional to the distance
                                direction = vec2(pos2.x - pos1.x, pos2.y - pos1.y)
                                direction.Normalize()
                                force = direction * COLOR_ATTRACTION_STRENGTH * (COLOR_ATTRACTION_DISTANCE - distance) / COLOR_ATTRACTION_DISTANCE
                                circle1['body'].ApplyForce(force, circle1['body'].worldCenter, wake=True)
                                circle2['body'].ApplyForce(-force, circle2['body'].worldCenter, wake=True)

# Simulation loop
running = True
time_step = 1.0

while running:
    screen.fill((0, 0, 0))  # Clear screen with black background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Apply edge repulsive force to each circle when it gets close to the edges
    for circle in circles:
        apply_repulsive_force(circle)

    # Apply color-based repulsion between circles
    apply_color_repulsion(circles)

    # Apply color-based attraction between circles
    apply_color_attraction(circles)

    # Draw each circle
    for circle in circles:
        draw_circle(circle)

    # Update physics
    world.Step(time_step, 6, 2)

    pygame.display.flip()  # Refresh the screen
    # clock.tick(FPS)

pygame.quit()
