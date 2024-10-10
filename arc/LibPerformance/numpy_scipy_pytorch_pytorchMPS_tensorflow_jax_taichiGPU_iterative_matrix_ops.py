import numpy as np
import torch
import time
import jax.numpy as jnp
from jax import random
import tensorflow as tf
from scipy import linalg
import taichi as ti

# Matrix and vector dimensions
VECTOR_SIZE = 10000
NUM_ITERATIONS = 10

# Function to calculate pairwise differences within the same vector
def pairwise_diff_matrix_with_self(vec):
    return np.subtract.outer(vec, vec)

# ----- NumPy (A + B) * C -----
print("\nTesting NumPy (A + B) * C")
start_numpy = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences
    A_numpy_1D = np.random.rand(VECTOR_SIZE)
    C_numpy_1D = np.random.rand(VECTOR_SIZE)

    # Generate pairwise difference matrices A and C with themselves
    A_numpy = pairwise_diff_matrix_with_self(A_numpy_1D)
    C_numpy = pairwise_diff_matrix_with_self(C_numpy_1D)

    # Generate random B matrix
    B_numpy = np.random.rand(VECTOR_SIZE, VECTOR_SIZE)

    # Perform proper matrix multiplication (A + B) * C
    result_numpy = np.matmul(A_numpy + B_numpy, C_numpy)

end_numpy = time.time()
numpy_time = end_numpy - start_numpy
print(f"NumPy total time for {NUM_ITERATIONS} iterations: {numpy_time:.6f} seconds")


# ----- SciPy (A + B) * C -----
print("\nTesting SciPy (A + B) * C")
start_scipy = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences
    A_scipy_1D = np.random.rand(VECTOR_SIZE)
    C_scipy_1D = np.random.rand(VECTOR_SIZE)

    # Generate pairwise difference matrices A and C with themselves
    A_scipy = pairwise_diff_matrix_with_self(A_scipy_1D)
    C_scipy = pairwise_diff_matrix_with_self(C_scipy_1D)

    # Generate random B matrix
    B_scipy = np.random.rand(VECTOR_SIZE, VECTOR_SIZE)

    # Perform proper matrix multiplication (A + B) * C using SciPy
    result_scipy = linalg.blas.dgemm(1.0, A_scipy + B_scipy, C_scipy)

end_scipy = time.time()
scipy_time = end_scipy - start_scipy
print(f"SciPy total time for {NUM_ITERATIONS} iterations: {scipy_time:.6f} seconds")


# ----- PyTorch (A + B) * C -----
print("\nTesting PyTorch (A + B) * C")
device = torch.device("cpu")  # Use Metal Performance Shaders (MPS) on M1
start_torch = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences on the M1 device
    A_torch_1D = torch.rand(VECTOR_SIZE, dtype=torch.float32, device=device)
    C_torch_1D = torch.rand(VECTOR_SIZE, dtype=torch.float32, device=device)

    # Generate pairwise difference matrices A and C with themselves
    A_torch = A_torch_1D.unsqueeze(1) - A_torch_1D.unsqueeze(0)
    C_torch = C_torch_1D.unsqueeze(1) - C_torch_1D.unsqueeze(0)

    # Generate random B matrix
    B_torch = torch.rand((VECTOR_SIZE, VECTOR_SIZE), dtype=torch.float32, device=device)

    # Perform proper matrix multiplication (A + B) * C
    result_torch = torch.matmul(A_torch + B_torch, C_torch)

end_torch = time.time()
torch_cpu_time = end_torch - start_torch
print(f"PyTorch total time for {NUM_ITERATIONS} iterations: {torch_cpu_time:.6f} seconds")


# ----- PyTorch MPS (A + B) * C -----
print("\nTesting PyTorch (A + B) * C with M1 acceleration")
device = torch.device("mps")  # Use Metal Performance Shaders (MPS) on M1
start_torch = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences on the M1 device
    A_torch_1D = torch.rand(VECTOR_SIZE, dtype=torch.float32, device=device)
    C_torch_1D = torch.rand(VECTOR_SIZE, dtype=torch.float32, device=device)

    # Generate pairwise difference matrices A and C with themselves
    A_torch = A_torch_1D.unsqueeze(1) - A_torch_1D.unsqueeze(0)
    C_torch = C_torch_1D.unsqueeze(1) - C_torch_1D.unsqueeze(0)

    # Generate random B matrix
    B_torch = torch.rand((VECTOR_SIZE, VECTOR_SIZE), dtype=torch.float32, device=device)

    # Perform proper matrix multiplication (A + B) * C
    result_torch = torch.matmul(A_torch + B_torch, C_torch)

end_torch = time.time()
torch_cpu_time = end_torch - start_torch
print(f"PyTorch (MPS) total time for {NUM_ITERATIONS} iterations: {torch_cpu_time:.6f} seconds")


# ----- TensorFlow (A + B) * C -----
print("\nTesting TensorFlow (A + B) * C")
start_tf = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences
    A_tf_1D = tf.random.normal((VECTOR_SIZE,))
    C_tf_1D = tf.random.normal((VECTOR_SIZE,))

    # Generate pairwise difference matrices A and C with themselves
    A_tf = tf.expand_dims(A_tf_1D, 1) - tf.expand_dims(A_tf_1D, 0)
    C_tf = tf.expand_dims(C_tf_1D, 1) - tf.expand_dims(C_tf_1D, 0)

    # Generate random B matrix
    B_tf = tf.random.normal((VECTOR_SIZE, VECTOR_SIZE))

    # Perform proper matrix multiplication (A + B) * C
    result_tf = tf.linalg.matmul(A_tf + B_tf, C_tf)

end_tf = time.time()
tensorflow_time = end_tf - start_tf
print(f"TensorFlow total time for {NUM_ITERATIONS} iterations: {tensorflow_time:.6f} seconds")


# ----- JAX (A + B) * C -----
print("\nTesting JAX (A + B) * C")
key = random.PRNGKey(0)
start_jax = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences
    A_jax_1D = random.normal(key, (VECTOR_SIZE,))
    C_jax_1D = random.normal(key, (VECTOR_SIZE,))

    # Generate pairwise difference matrices A and C with themselves
    A_jax = jnp.expand_dims(A_jax_1D, 1) - jnp.expand_dims(A_jax_1D, 0)
    C_jax = jnp.expand_dims(C_jax_1D, 1) - jnp.expand_dims(C_jax_1D, 0)

    # Generate random B matrix
    B_jax = random.normal(key, (VECTOR_SIZE, VECTOR_SIZE))

    # Perform proper matrix multiplication (A + B) * C
    result_jax = jnp.matmul(A_jax + B_jax, C_jax)

end_jax = time.time()
jax_time = end_jax - start_jax
print(f"JAX total time for {NUM_ITERATIONS} iterations: {jax_time:.6f} seconds\n")


# ----- Taichi GPU (A + B) * C -----
# Initialize Taichi
ti.init(arch=ti.gpu)

# Define Taichi kernels
@ti.kernel
def pairwise_diff_matrix_with_self(vec: ti.types.ndarray(), result: ti.types.ndarray()):
    for i, j in result:
        result[i, j] = vec[i] - vec[j]

@ti.kernel
def matrix_add(A: ti.types.ndarray(), B: ti.types.ndarray(), result: ti.types.ndarray()):
    for i, j in result:
        result[i, j] = A[i, j] + B[i, j]

@ti.kernel
def matrix_multiply(A: ti.types.ndarray(), B: ti.types.ndarray(), result: ti.types.ndarray()):
    for i, j in result:
        for k in range(A.shape[1]):
            result[i, j] += A[i, k] * B[k, j]

print("\nTesting Taichi GPU (A + B) * C")
start_taichi = time.time()
for _ in range(NUM_ITERATIONS):
    # Generate random 1D vectors for pairwise differences
    A_taichi_1D = np.random.rand(VECTOR_SIZE).astype(np.float32)
    C_taichi_1D = np.random.rand(VECTOR_SIZE).astype(np.float32)

    # Generate pairwise difference matrices A and C with themselves
    A_taichi = np.zeros((VECTOR_SIZE, VECTOR_SIZE), dtype=np.float32)
    C_taichi = np.zeros((VECTOR_SIZE, VECTOR_SIZE), dtype=np.float32)
    pairwise_diff_matrix_with_self(A_taichi_1D, A_taichi)
    pairwise_diff_matrix_with_self(C_taichi_1D, C_taichi)

    # Generate random B matrix
    B_taichi = np.random.rand(VECTOR_SIZE, VECTOR_SIZE).astype(np.float32)

    # Perform matrix addition (A + B)
    AB_taichi = np.zeros((VECTOR_SIZE, VECTOR_SIZE), dtype=np.float32)
    matrix_add(A_taichi, B_taichi, AB_taichi)

    # Perform matrix multiplication (A + B) * C
    result_taichi = np.zeros((VECTOR_SIZE, VECTOR_SIZE), dtype=np.float32)
    matrix_multiply(AB_taichi, C_taichi, result_taichi)

end_taichi = time.time()
taichi_time = end_taichi - start_taichi
print(f"Taichi (GPU) total time for {NUM_ITERATIONS} iterations: {taichi_time:.6f} seconds")
