import numpy as np
import time
import torch
import pybullet as p

# Parameters
N = 1000  # Number of particles
steps = 1000  # Number of position updates

# Create initial random positions and movements for PyTorch
positions = np.random.rand(N, 3).astype(np.float32)  # Random initial positions in 3D
movements = (np.random.rand(N, 3) - 0.5).astype(np.float32)  # Random movement vectors

# PyTorch approach (MPS backend if available)
positions_torch = torch.tensor(positions, device='mps', dtype=torch.float32)
movements_torch = torch.tensor(movements, device='mps', dtype=torch.float32)

# PyTorch timing
start_time = time.time()
for _ in range(steps):
    positions_torch += movements_torch  # Update positions
torch_time = time.time() - start_time
print(f"PyTorch time: {torch_time:.6f} seconds")

# PyBullet approach (CPU, as GPU is limited in PyBullet unless specific settings)
# Initialize PyBullet
p.connect(p.DIRECT)  # No GUI
p.resetSimulation()

# Create particles in PyBullet
particle_ids = []
for i in range(N):
    particle_id = p.createMultiBody(baseMass=1,
                                    basePosition=positions[i],
                                    baseCollisionShapeIndex=-1)  # No collision
    particle_ids.append(particle_id)

# PyBullet timing
start_time = time.time()
for _ in range(steps):
    for i in range(N):
        current_pos, _ = p.getBasePositionAndOrientation(particle_ids[i])
        new_pos = np.add(current_pos, movements[i])  # Update with random movements
        p.resetBasePositionAndOrientation(particle_ids[i], new_pos, [0, 0, 0, 1])
bullet_time = time.time() - start_time
print(f"PyBullet time: {bullet_time:.6f} seconds")

# Disconnect PyBullet
p.disconnect()
