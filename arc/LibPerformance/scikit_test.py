# main.py
import numpy as np
import time
import torch
from sklearn.neighbors import NearestNeighbors

def main():
    # Generate random particle coordinates
    np.random.seed(42)
    num_particles = 10000  # Adjust this number for testing
    particles = np.random.rand(num_particles, 2) * 100  # 10,000 particles in a 100x100 area
    max_distance = 10

    # Approach 1: NearestNeighbors
    nbrs = NearestNeighbors(radius=max_distance).fit(particles)

    def find_neighbors_nn(point):
        indices = nbrs.radius_neighbors([point], return_distance=False)
        return indices[0]

    # Timing NearestNeighbors
    start_time_nn = time.time()
    neighbors_nn = find_neighbors_nn(particles[500])  # Example point
    end_time_nn = time.time()

    print(f"Neighbors (NearestNeighbors): {neighbors_nn}")
    print(f"Time taken (NearestNeighbors): {end_time_nn - start_time_nn:.6f} seconds")

    # Approach 2: Pairwise distance with PyTorch
    particles_tensor = torch.tensor(particles, dtype=torch.float32)

    def find_neighbors_torch(point, max_distance):
        point_tensor = torch.tensor(point, dtype=torch.float32)
        distances = torch.norm(particles_tensor - point_tensor, dim=1)
        return (distances <= max_distance).nonzero(as_tuple=True)[0]

    # Timing PyTorch
    start_time_torch = time.time()
    neighbors_torch = find_neighbors_torch(particles_tensor[500], max_distance)  # Example point
    end_time_torch = time.time()

    print(f"Neighbors (PyTorch): {neighbors_torch}")
    print(f"Time taken (PyTorch): {end_time_torch - start_time_torch:.6f} seconds")

if __name__ == "__main__":
    main()
