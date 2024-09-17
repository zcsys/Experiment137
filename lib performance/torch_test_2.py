import torch
import time

# Initialize position tensor with random values for 1000 particles in 2D
num_particles = 1000
positions = torch.rand((num_particles, 2))  # Shape: (1000, 2)

# A sample movement function: Here we simulate movement by adding a random vector
def calculate_movement(index):
    # The movement is simulated as a random vector
    return torch.rand(2)

# Scenario 1: Apply the movement function and update positions one by one
def scenario_one(positions):
    start_time = time.time()

    for i in range(num_particles):
        movement = calculate_movement(i)
        positions[i] += movement

    end_time = time.time()
    return positions, end_time - start_time

# Scenario 2: Compute a compound movement tensor and add to positions at once
def scenario_two(positions):
    start_time = time.time()

    # Initialize the movement tensor
    movement_tensor = torch.zeros((num_particles, 2))

    # Calculate the movement for each particle and store it in the movement tensor
    for i in range(num_particles):
        movement = calculate_movement(i)
        movement_tensor[i] = movement

    # Apply the compound movement to the position tensor at once
    positions += movement_tensor

    end_time = time.time()
    return positions, end_time - start_time

# Clone the position tensor for fair comparison
positions_scenario_one = positions.clone()
positions_scenario_two = positions.clone()

# Run scenario 1
_, time_scenario_one = scenario_one(positions_scenario_one)

# Run scenario 2
_, time_scenario_two = scenario_two(positions_scenario_two)

# Print the times for comparison
print(f"Time taken for Scenario 1 (one-by-one updates): {time_scenario_one:.6f} seconds")
print(f"Time taken for Scenario 2 (compound movement tensor): {time_scenario_two:.6f} seconds")
