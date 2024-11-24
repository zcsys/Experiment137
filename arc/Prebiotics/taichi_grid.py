import taichi as ti
import numpy as np
import time

ti.init()

# Full HD resolution
WIDTH = 1920
HEIGHT = 1080

# Fixed cell size
CELL_SIZE = 60

# Calculate grid dimensions based on cell size
GRID_WIDTH = WIDTH // CELL_SIZE  # 16 cells (1920/120 = 16)
GRID_HEIGHT = HEIGHT // CELL_SIZE  # 9 cells (1080/120 = 9)

# Field to store the heatmap values
field = ti.field(dtype=float, shape=(GRID_WIDTH, GRID_HEIGHT))
pixels = ti.Vector.field(4, dtype=ti.u8, shape=(WIDTH, HEIGHT))

@ti.kernel
def render():
    for i, j in pixels:
        # Convert pixel coordinates to grid coordinates
        grid_x = i // CELL_SIZE
        grid_y = j // CELL_SIZE

        # Get the field value (0 to 1)
        value = field[grid_x, grid_y]

        # Convert value to yellow intensity (R,G,B,A)
        r = ti.u8(255 * value)
        g = ti.u8(255 * value)
        b = ti.u8(0)
        a = ti.u8(255)  # Full opacity

        pixels[i, j] = ti.Vector([r, g, b, a])

# Create window
window = ti.ui.Window("2D Heatmap", (WIDTH, HEIGHT))
canvas = window.get_canvas()

start_time = time.time()

# Main loop
while window.running:
    current_time = time.time() - start_time

    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            field[i, j] = 0.5 + 0.5 * np.sin(current_time * 2.0 + (i + j) * 0.5)

    render()
    canvas.set_image(pixels)
    window.show()
