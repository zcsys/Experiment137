import torch
import random
import math
import struct
import base64
import numpy as np
from base_vars import *
from scipy.spatial import cKDTree

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

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def get_color_by_genome(genome, scale = 100., base_color = (160, 160, 160)):
    n = len(genome) // 3
    return (
        max(min(base_color[0] + int(scale * genome[:n].sum().item()),
            255), 64),
        max(min(base_color[1] + int(scale * genome[n:2 * n].sum().item()),
            255), 64),
        max(min(base_color[2] + int(scale * genome[2 * n:3 * n].sum().item()),
            255), 64)
    )

def reverse_color(color):
    r, g, b = color
    return 255 - r, 255 - g, 255 - b

def float_msg_to_str(float_msg):
    packed_bytes = struct.pack('>ff', np.float32(float_msg[0]),
                               np.float32(float_msg[1]))
    return base64.b64encode(packed_bytes).decode('ascii')[:4]

def get_box(positions):
    return (positions[:, 0] // 120 + positions[:, 1] // 120 * 16).int()

def flattened_identity_matrix(N, x = None):
    lt = x if x else N
    return [1 if i == j and i < lt else 0 for j in range(N) for i in range(N)]

def create_initial_genomes(num_monads, num_input, num_output):
    return torch.tensor(
        flattened_identity_matrix(num_input) +
        [0 for _ in range(3 * num_input ** 2)] +
        [0 for _ in range(4 * num_input)] +
        flattened_identity_matrix(4 * num_input)[:4 * num_input ** 2] +
        [0 for _ in range(num_input)] +
        [0 for _ in range((num_input + 1) * num_output)],
        dtype = torch.float32
    ).repeat(num_monads, 1)

def circular_pad(x, pad):
    padded = torch.cat(
        [x[..., -pad:], x, x[..., :pad]], dim = -1
    ).transpose(-1, -2)
    return torch.cat(
        [padded[..., -pad:], padded, padded[..., :pad]], dim = -1
    ).transpose(-1, -2)

def toroidal_vicinity(positions, radius):
    tree = cKDTree(positions.numpy(), boxsize = (SIMUL_WIDTH, SIMUL_HEIGHT))
    distances = tree.sparse_distance_matrix(tree, radius, p = 2.0)
    rows, cols = distances.nonzero()
    return np.stack([rows, cols]), distances, positions[cols] - positions[rows]
