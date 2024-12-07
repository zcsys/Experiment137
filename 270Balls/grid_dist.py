import json
import seaborn as sns
from diffusion import Grid
from matplotlib import pyplot as plt

def plot_grid_distribution(grid, bins=50):
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

   colors = ['red', 'green', 'blue']

   for i in range(grid.feature_dim):
       sns.histplot(grid.grid[0,i].flatten(), bins=bins, ax=ax1,
                   color=colors[i], alpha=0.5, label=f'Channel {i}')
       sns.kdeplot(grid.grid[0,i].flatten(), ax=ax2,
                   color=colors[i], label=f'Channel {i}')

   ax1.set_title('Value Distribution per Channel')
   ax2.set_title('Density Distribution per Channel')
   ax1.legend()
   ax2.legend()

   plt.tight_layout()
   plt.show()

load_file = "simulation_20241206_233339.json"
with open(load_file, 'r') as f:
    saved_data = json.load(f)
    grid = Grid(saved_state = saved_data["grid"])

if __name__ == '__main__':
    plot_grid_distribution(grid)
