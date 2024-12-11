import torch
import random
import math
import struct
import base64
import numpy as np
from base_vars import *
from scipy.spatial import KDTree

def unique(x):
    """Gets a list and returns its unique values as a list in same order"""
    seen = set()
    result = []
    for item in x:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def add_positions(sizes,
                  existing_sizes = torch.empty(0),
                  existing_positions = torch.empty((0, 2)),
                  width = SIMUL_WIDTH,
                  height = SIMUL_HEIGHT):
    existing_N = len(existing_positions)
    total_N = existing_N + len(sizes)

    positions = existing_positions
    sizes = torch.cat((existing_sizes, sizes), dim = 0)

    i = existing_N
    while i < total_N:
        new_position = torch.tensor(
            [
                random.randint(int(sizes[i]), int(width - sizes[i])),
                random.randint(int(sizes[i]), int(height - sizes[i]))
            ],
            dtype = torch.float32
        ).unsqueeze(0)

        distances = torch.norm(new_position - positions, dim = 1)
        if (distances < sizes[i] + sizes[:i]).any():
            continue

        positions = torch.cat((positions, new_position), dim = 0)
        i += 1

    return sizes, positions

def remove_element(tensor, i):
    return torch.cat((tensor[:i], tensor[i + 1:]), dim = 0)

def get_color_by_genome(genome, scale = 1., base_color = colors["rgb"]):
    n = len(genome) // 3
    return (
        base_color[0] + torch.clamp(genome[:n].sum(), 0, 255).int().item(),
        base_color[1] + torch.clamp(genome[n:2 * n].sum(), 0, 255).int().item(),
        base_color[2] + torch.clamp(genome[2 * n:].sum(), 0, 255).int().item()
    )

def flattened_identity_matrix(N, x = None):
    lt = x if x else N
    return [1 if i == j and i < lt else 0 for j in range(N) for i in range(N)]

def create_initial_genomes(num_monads, input, output):
    if input > 2 * output:
        return torch.tensor(
            flattened_identity_matrix(input) +
            [0 for _ in range(3 * input ** 2)] +
            [0 for _ in range(4 * input)] +
            flattened_identity_matrix(4 * input)[:4 * input ** 2] +
            [0 for _ in range(input)] +
            [0 for _ in range((input + 1) * output)],
            dtype = torch.float32
        ).repeat(num_monads, 1)
    else:
        return torch.tensor(
            flattened_identity_matrix(input) +
            [0 for _ in range(3 * input ** 2)] +
            [0 for _ in range(4 * input)] +
            flattened_identity_matrix(4 * input)[:8 * input * output] +
            [0 for _ in range(2 * output)] +
            [0 for _ in range((2 * output + 1) * output)],
            dtype = torch.float32
        ).repeat(num_monads, 1)

def vicinity(source_positions, radius = SIGHT, target_positions = None):
    source_tree = KDTree(source_positions.numpy())
    if target_positions:
        target_tree = KDTree(target_positions.numpy())
    else:
        target_tree, target_positions = source_tree, source_positions

    distances = source_tree.sparse_distance_matrix(target_tree, radius, p = 2.0)
    rows, cols = distances.nonzero()

    vector_diff = torch.zeros(
        (len(source_positions), len(target_positions), 2),
        dtype = torch.float32
    )
    vector_diff[rows, cols] = target_positions[cols] - source_positions[rows]

    return (
        torch.from_numpy(np.stack([rows, cols])),
        torch.tensor(distances.toarray(), dtype = torch.float32),
        vector_diff
    )
