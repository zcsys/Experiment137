import torch
import time

# Matrix dimensions
MATRIX_SIZE = 2500

# Ensure MPS device is available, else use CPU
DEVICE_MPS = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Generate random matrices for both CPU and MPS
A_cpu = torch.rand((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32)
B_cpu = torch.rand((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32)

# Copy matrices to MPS device (if available)
A_mps = A_cpu.to(DEVICE_MPS)
B_mps = B_cpu.to(DEVICE_MPS)

# CPU matrix multiplication timing
start_cpu = time.time()
C_cpu = torch.matmul(A_cpu, B_cpu)
end_cpu = time.time()

# MPS matrix multiplication timing (if available)
start_mps = time.time()
C_mps = torch.matmul(A_mps, B_mps)
end_mps = time.time()

# Calculate time taken for both operations
cpu_time = end_cpu - start_cpu
mps_time = end_mps - start_mps

# Display results
print(f"CPU matrix multiplication time: {cpu_time:.6f} seconds")
print(f"MPS matrix multiplication time: {mps_time:.6f} seconds")
