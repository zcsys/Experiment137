import torch
import time

# Setup
N = 100000  # Number of positions
D = 20      # Dimensions (e.g., x, y coordinates)

# Initialize positions randomly
positions = torch.rand((N, D), dtype=torch.float32)
updates = torch.rand((N, D), dtype=torch.float32)

# ----- Approach 1: For loop with matrix addition -----
print("Testing for loop approach...")
positions_loop = positions.clone()

start_loop = time.time()
for i in range(N):
    positions_loop[i] += updates[i]
end_loop = time.time()

loop_time = end_loop - start_loop
print(f"For loop approach time: {loop_time:.6f} seconds")


# ----- Approach 2: Vectorized matrix addition (no loop) -----
print("\nTesting vectorized approach (no loop)...")
positions_vec = positions.clone()

start_vec = time.time()
positions_vec += updates  # Vectorized addition
end_vec = time.time()

vec_time = end_vec - start_vec
print(f"Vectorized approach time: {vec_time:.6f} seconds")
