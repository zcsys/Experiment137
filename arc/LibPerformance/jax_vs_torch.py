import jax
import jax.numpy as jnp
import torch
import time
import numpy as np

def jax_operation():
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (1000, 1000))
    B = jax.random.normal(key, (1000, 1000))
    C = jax.random.normal(key, (1000, 1000))

    @jax.jit
    def step(x):
        return jnp.matmul(A, B) + C

    start_time = time.time()
    for _ in range(1000):
        x = step(None)
    jax.block_until_ready(x)
    end_time = time.time()

    return end_time - start_time

def pytorch_operation():
    A = torch.randn(1000, 1000)
    B = torch.randn(1000, 1000)
    C = torch.randn(1000, 1000)

    start_time = time.time()
    for _ in range(1000):
        x = torch.matmul(A, B) + C
    end_time = time.time()

    return end_time - start_time

# Run the comparisons
jax_time = jax_operation()
pytorch_time = pytorch_operation()

print(f"JAX time: {jax_time:.4f} seconds")
print(f"PyTorch time: {pytorch_time:.4f} seconds")
