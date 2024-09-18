import numpy as np
import time
from numba import njit
import torch

# Size of the matrices
N = 1000

# Create two random NxN matrices as float32
matrix1 = np.random.rand(N, N).astype(np.float32)
matrix2 = np.random.rand(N, N).astype(np.float32)

# Numba version with nested loops
@njit
def matmul_numba(A, B):
    result = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i, j] += A[i, k] * B[k, j]
    return result

# Measure time for Numba version
start_time = time.time()
result_numba = matmul_numba(matrix1, matrix2)
numba_time = time.time() - start_time
print(f"Numba time: {numba_time:.5f} seconds")

# PyTorch version using M1 chip (float32)
matrix1_torch = torch.tensor(matrix1, device='mps', dtype=torch.float32)
matrix2_torch = torch.tensor(matrix2, device='mps', dtype=torch.float32)

# Measure time for PyTorch version
start_time = time.time()
result_torch = torch.matmul(matrix1_torch, matrix2_torch)
torch_time = time.time() - start_time
print(f"PyTorch MPS time: {torch_time:.5f} seconds")

# Move the PyTorch result back to CPU and convert to NumPy array
result_torch_np = result_torch.cpu().numpy()

# Verify that the results are close within float32 precision
are_equal = np.allclose(result_numba, result_torch_np, rtol=1e-5, atol=1e-8)
print(f"Results are equal within float32 tolerance: {are_equal}")
