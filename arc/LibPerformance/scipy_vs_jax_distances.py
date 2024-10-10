import numpy as np
import scipy.spatial.distance
import jax
import jax.numpy as jnp
import time

def time_function(func, *args, n_runs=100):
    start_time = time.time()
    for _ in range(n_runs):
        result = func(*args)
    end_time = time.time()
    return result, (end_time - start_time) / n_runs

# JAX implementation
@jax.jit
def jax_pdist(points):
    diff = jnp.expand_dims(points, 1) - jnp.expand_dims(points, 0)
    squared_dist = jnp.sum(diff**2, axis=-1)
    triu_indices = jnp.triu_indices(points.shape[0], k=1)
    return jnp.sqrt(squared_dist[triu_indices])

# Test for different sizes
sizes = [10, 100, 1000, 10000]

for size in sizes:
    print(f"\nTesting with {size} points:")

    # Generate random points
    np.random.seed(0)
    points_np = np.random.rand(size, 3)
    points_jax = jnp.array(points_np)

    # Time SciPy implementation
    scipy_result, scipy_time = time_function(scipy.spatial.distance.pdist, points_np)

    # Time JAX implementation
    jax_result, jax_time = time_function(jax_pdist, points_jax)

    # Compile JAX function (this is usually done automatically, but we're doing it explicitly for fair timing)
    _ = jax_pdist(points_jax).block_until_ready()

    # Time compiled JAX implementation
    jax_compiled_result, jax_compiled_time = time_function(lambda x: jax_pdist(x).block_until_ready(), points_jax)

    print(f"SciPy time: {scipy_time:.6f} seconds")
    print(f"JAX time (including compilation): {jax_time:.6f} seconds")
    print(f"JAX time (compiled): {jax_compiled_time:.6f} seconds")

    # Check results
    max_diff = np.max(np.abs(scipy_result - np.array(jax_compiled_result)))
    print(f"Max absolute difference: {max_diff:.6e}")
