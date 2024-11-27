import torch
import numpy as np
from scipy.spatial import cKDTree
import time

# Simulation parameters
SIMUL_WIDTH = 1920.0
SIMUL_HEIGHT = 1080.0
N_PARTICLES = 1000  # Number of particles
RADIUS = 60.0        # Search radius

def toroidal_vicinity(positions, radius):
    tree = cKDTree(positions.numpy(), boxsize=(SIMUL_WIDTH, SIMUL_HEIGHT))
    distances = tree.sparse_distance_matrix(tree, radius, p=2.0)
    rows, cols = distances.nonzero()
    return np.stack([rows, cols]), distances, positions[cols] - positions[rows]

def generate_random_positions():
    # Generate random positions using torch
    positions = torch.rand(N_PARTICLES, 2)
    # Scale positions to simulation dimensions
    positions[:, 0] *= SIMUL_WIDTH
    positions[:, 1] *= SIMUL_HEIGHT
    return positions

def run_performance_test(n_trials=5):
    total_time = 0

    print(f"Running {n_trials} trials with {N_PARTICLES} particles...")

    for i in range(n_trials):
        # Generate new random positions
        positions = generate_random_positions()

        # Time the vicinity calculation
        start_time = time.time()
        indices, distances, relative_positions = toroidal_vicinity(positions, RADIUS)
        end_time = time.time()

        trial_time = end_time - start_time
        total_time += trial_time

        print(f"Trial {i+1}: {2400*trial_time:.2f} seconds for 2400 such steps")
        print(f"Found {len(indices[0])} neighbor pairs")

    avg_time = total_time / n_trials
    print(f"\nAverage execution time: {avg_time:.4f} seconds")

    return avg_time

if __name__ == "__main__":
    run_performance_test()
