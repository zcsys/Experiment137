import numpy as np
import torch
import time
import jax.numpy as jnp
from jax import random
import tensorflow as tf

# Matrix dimensions
MATRIX_SIZE = 1000

# ----- NumPy matrix multiplication (Baseline) -----
print("Testing NumPy matrix multiplication")
A_numpy = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
B_numpy = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)

start_numpy = time.time()
C_numpy = np.dot(A_numpy, B_numpy)
end_numpy = time.time()

numpy_time = end_numpy - start_numpy
print(f"NumPy matrix multiplication time: {numpy_time:.6f} seconds")


# ----- PyTorch matrix multiplication (CPU) -----
print("\nTesting PyTorch CPU matrix multiplication")
A_torch = torch.rand((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32)
B_torch = torch.rand((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32)

start_torch = time.time()
C_torch = torch.matmul(A_torch, B_torch)
end_torch = time.time()

torch_cpu_time = end_torch - start_torch
print(f"PyTorch CPU matrix multiplication time: {torch_cpu_time:.6f} seconds")

# ----- TensorFlow matrix multiplication (CPU or GPU) -----
print("\nTesting TensorFlow matrix multiplication")
A_tf = tf.random.normal((MATRIX_SIZE, MATRIX_SIZE))
B_tf = tf.random.normal((MATRIX_SIZE, MATRIX_SIZE))

start_tf = time.time()
C_tf = tf.linalg.matmul(A_tf, B_tf)
end_tf = time.time()

tensorflow_time = end_tf - start_tf
print(f"TensorFlow matrix multiplication time: {tensorflow_time:.6f} seconds")


# ----- JAX matrix multiplication -----
print("\nTesting JAX matrix multiplication")
key = random.PRNGKey(0)
A_jax = random.normal(key, (MATRIX_SIZE, MATRIX_SIZE))
B_jax = random.normal(key, (MATRIX_SIZE, MATRIX_SIZE))

start_jax = time.time()
C_jax = jnp.matmul(A_jax, B_jax)
end_jax = time.time()

jax_time = end_jax - start_jax
print(f"JAX matrix multiplication time: {jax_time:.6f} seconds")
