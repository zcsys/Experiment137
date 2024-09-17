import pygame
import random
import math

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Function to create a circle
def create_circle(radius, position):
    shape = pygame.Rect(position[0] - radius, position[1] - radius, radius * 2, radius * 2)  # Pygame rect for movement
    return shape, radius

# Create random circles that move towards the bottom-right corner
def create_random_circles(num_circles, screen_width, screen_height):
    circles = []
    for _ in range(num_circles):
        radius = random.randint(10, 20)
        x = random.randint(radius, screen_width - radius)
        y = random.randint(radius, screen_height - radius)
        circle, radius = create_circle(radius, (x, y))
        circles.append((circle, radius))  # Circles will move step by step manually
    return circles

# Check if two circles overlap
def circles_overlap(circle1, radius1, circle2, radius2):
    dist_x = circle1.centerx - circle2.centerx
    dist_y = circle1.centery - circle2.centery
    distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
    return distance < (radius1 + radius2)

# Check if a new position causes overlap with any other circle
def check_prospective_collision(new_x, new_y, radius, circles, circle_idx):
    for i, (other_circle, other_radius) in enumerate(circles):
        if i != circle_idx:
            dist_x = (new_x + radius) - (other_circle.centerx)
            dist_y = (new_y + radius) - (other_circle.centery)
            distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
            if distance < (radius + other_radius):
                return True
    return False

# Set up the simulation
screen_width, screen_height = 800, 600
circles = create_random_circles(100, screen_width, screen_height)

# Pygame loop
running = True
step_size = 2  # Step size for position update towards bottom-right
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Manually update circle positions step by step
    for i, (circle, radius) in enumerate(circles):
        # Tentative move calculation towards bottom-right corner
        new_x = circle.x + step_size
        new_y = circle.y + step_size

        # Check for wall collisions
        if new_x <= 0 or new_x + circle.width >= screen_width:
            new_x = circle.x  # Prevent horizontal movement out of bounds
        if new_y <= 0 or new_y + circle.height >= screen_height:
            new_y = circle.y  # Prevent vertical movement out of bounds

        # Check for collisions with other circles before moving
        if not check_prospective_collision(new_x, new_y, radius, circles, i):
            # If no collision, update the circle's position
            circle.x = new_x
            circle.y = new_y

        # Update the circle in the list
        circles[i] = (circle, radius)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw circles
    for circle, radius in circles:
        pygame.draw.circle(screen, (255, 0, 0), (circle.centerx, circle.centery), radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
