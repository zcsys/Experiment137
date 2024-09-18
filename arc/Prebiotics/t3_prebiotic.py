import pygame
import numpy as np
import math
import time

# Initialize Pygame
pygame.init()

# Constants
width, height = 800, 600  # Size of the window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Sine Wave Heatmap")

# Define colors
def get_heat_color(value):
    """Convert a value (-1 to 1) to a color (gradient from blue to red)."""
    r = (255 * (value + 1) / 2).astype(np.uint8)  # Scale from (-1, 1) to (0, 255)
    g = np.zeros_like(r, dtype=np.uint8)  # Green is fixed to 0
    b = 255 - r  # Blue is the inverse of red
    return np.stack((r, g, b), axis=-1)  # Stack to create (height, width, 3) array

# Generate Sine Waves using NumPy
def sine_wave_horizontal(x, t, frequency, phase):
    """Calculate horizontal sine wave based on x, time (t), frequency, and phase."""
    return np.sin(frequency * (x / width * 2 * np.pi) + phase + t)

def sine_wave_vertical(y, t, frequency, phase):
    """Calculate vertical sine wave based on y, time (t), frequency, and phase."""
    return np.sin(frequency * (y / height * 2 * np.pi) + phase + t)

def combined_wave(x, y, t):
    """Superimpose a horizontal and vertical sine wave."""
    wave1 = sine_wave_horizontal(x, t, 5, 0)  # Horizontal wave
    wave2 = sine_wave_vertical(y, t, 5, np.pi / 4)  # Vertical wave with phase shift
    return (wave1 + wave2) / 2

# Create heatmap using NumPy
def draw_heatmap(t):
    """Draw a sine wave heatmap on the screen using NumPy."""
    x = np.linspace(0, width - 1, width)  # Generate x coordinates
    y = np.linspace(0, height - 1, height)  # Generate y coordinates

    X, Y = np.meshgrid(x, y)  # Create a 2D grid of coordinates

    # Calculate the combined wave at each point
    wave_value = combined_wave(X, Y, t)

    # Convert the wave value to colors
    color_array = get_heat_color(wave_value)

    # Transpose the array to match (width, height, 3) format
    color_array = np.transpose(color_array, (1, 0, 2))

    # Convert NumPy array to a Pygame surface
    pygame.surfarray.blit_array(screen, color_array)

if __name__ == "__main__":
    # Main loop
    running = True
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate time elapsed to propagate the wave
        t = time.time() - start_time

        # Draw heatmap with time-based propagation
        draw_heatmap(t)

        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
