import time
import taichi as ti
import numpy as np
from scipy.spatial import cKDTree
from base_vars import *

ti.init(arch = ti.gpu)

@ti.data_oriented
class Grid:
    def __init__(self, cell_size=10, feature_dim=3, diffusion_rate=0.001, num_circles=25):
        self.cell_size = cell_size
        self.feature_dim = feature_dim
        self.grid_x = SIMUL_WIDTH // cell_size
        self.grid_y = SIMUL_HEIGHT // cell_size
        self.grid = ti.Vector.field(feature_dim, dtype=ti.f32, shape=(self.grid_x, self.grid_y))
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(SIMUL_WIDTH, SIMUL_HEIGHT))
        self.diffusion_rate = diffusion_rate
        self.window = ti.ui.Window("Abundance 2", (SIMUL_WIDTH, SIMUL_HEIGHT))
        self.canvas = self.window.get_canvas()

        self.num_circles = num_circles
        self.circle_positions = ti.Vector.field(2, dtype=ti.f32, shape=num_circles)
        self.circle_radius = 50
        self.min_distance = 2 * self.circle_radius
        self.boxsize = np.array([SIMUL_WIDTH, SIMUL_HEIGHT], dtype=np.float32)

        self.init_random_points()
        self.init_circles()

    def init_circles(self):
        positions = []
        for _ in range(self.num_circles):
            while True:
                pos = np.array([
                    np.random.random() * SIMUL_WIDTH,
                    np.random.random() * SIMUL_HEIGHT
                ], dtype=np.float32)

                if not positions:
                    positions.append(pos)
                    break

                tree = cKDTree(positions, boxsize=self.boxsize)
                if tree.query(pos)[0] >= self.min_distance:
                    positions.append(pos)
                    break

        positions = np.array(positions, dtype=np.float32)
        self.circle_positions.from_numpy(positions)

    def move_circles(self):
        positions = self.circle_positions.to_numpy()
        for i in range(self.num_circles):
            original_pos = positions[i]
            proposed_move = np.random.uniform(-1, 1, 2).astype(np.float32)
            proposed_pos = original_pos + proposed_move
            proposed_pos = proposed_pos % self.boxsize

            temp_positions = np.delete(positions, i, axis=0)
            if temp_positions.size > 0:
                tree = cKDTree(temp_positions, boxsize=self.boxsize)
                if tree.query(proposed_pos)[0] >= self.min_distance:
                    positions[i] = proposed_pos

        self.circle_positions.from_numpy(positions)

    @ti.func
    def get_grid_value(self, channel, x, y):
        return self.grid[x % self.grid_x, y % self.grid_y][channel]

    @ti.kernel
    def init_random_points(self):
        for channel, i, j in ti.ndrange(self.feature_dim, self.grid_x, self.grid_y):
            if ti.random() < 0.3:
                self.grid[i, j][channel] = 1.0

    @ti.kernel
    def diffuse(self):
        for channel, i, j in ti.ndrange(self.feature_dim, self.grid_x, self.grid_y):
            center = self.get_grid_value(channel, i, j)
            left = self.get_grid_value(channel, i-1, j)
            right = self.get_grid_value(channel, i+1, j)
            up = self.get_grid_value(channel, i, j-1)
            down = self.get_grid_value(channel, i, j+1)
            laplacian = left + right + up + down - 4.0 * center
            self.grid[i, j][channel] += self.diffusion_rate * laplacian

    @ti.kernel
    def draw(self):
        for i, j in self.pixels:
            self.pixels[i, j] = self.grid[i // self.cell_size, j // self.cell_size]

        for circle_idx in range(self.num_circles):
            pos = self.circle_positions[circle_idx]
            for i, j in ti.ndrange((-self.circle_radius, self.circle_radius + 1),
                                 (-self.circle_radius, self.circle_radius + 1)):
                x = int(pos[0] + i)
                y = int(pos[1] + j)
                print(x, y)
                if (x >= 0 and x < SIMUL_WIDTH and y >= 0 and y < SIMUL_HEIGHT and
                    i * i + j * j <= self.circle_radius * self.circle_radius):
                    self.pixels[x, y] = ti.Vector([1.0, 1.0, 1.0])

    def display(self):
        self.move_circles()
        self.draw()
        self.canvas.set_image(self.pixels)
        self.window.show()

def main():
    grid = Grid()
    last_time = time.time()
    while grid.window.running:
        if grid.window.is_pressed(ti.ui.ESCAPE):
            break
        start_time = time.time()
        grid.diffuse()
        grid.display()
        last_time = time.time()
        total_duration = int((last_time - start_time) * 2400)
        grid.window.GUI.begin("Stats", 0.02, 0.02, 0.2, 0.1)
        grid.window.GUI.text(f"Est. cycle: {total_duration}s")
        grid.window.GUI.end()

if __name__ == "__main__":
    main()
