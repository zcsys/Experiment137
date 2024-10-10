import torch
import random
import math
import struct
import base64
import numpy as np
from base_vars import *

identity = lambda x: x

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

def generate_wave(frequency, duration, waveform = "sine", sample_rate = 44100,
                  amplitude = 400):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if waveform == "sine":
        wave = np.sin(2 * np.pi * frequency * t) * amplitude
    elif waveform == "square":
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "sawtooth":
        wave = 2 * amplitude * (t * frequency - np.floor(1/2 + t * frequency))
    elif waveform == "noise":
        wave = np.random.uniform(-1, 1, size=t.shape) * amplitude
    else:
        raise ValueError("Invalid waveform type. Choose 'sine', 'square'," +
                         "'sawtooth', or 'noise'.")
    wave = wave.astype(np.int16)
    sound_array = np.array([wave, wave]).T
    sound_array = np.ascontiguousarray(sound_array)
    return pygame.sndarray.make_sound(sound_array)

def remove_element(tensor, i):
    return torch.cat((tensor[:i], tensor[i + 1:]), dim = 0)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def get_color_by_genome(genome, scale = 10., base_color = (160, 160, 160)):
    n = len(genome) // 6
    result = (
        max(min(base_color[0] + int(scale * genome[:n].sum().item()),
            255), 64),
        max(min(base_color[1] + int(scale * genome[n:2*n].sum().item()),
            255), 64),
        max(min(base_color[2] + int(scale * genome[2*n:3*n].sum().item()),
            255), 64)
    )
    print(result)
    return result

def reverse_color(color):
    r, g, b = color
    return 255 - r, 255 - g, 255 - b

def float_msg_to_str(float_msg):
    packed_bytes = struct.pack('>f', np.float32(float_msg))
    return base64.b64encode(packed_bytes)[:4].decode('ascii')

def get_box(positions):
    return (positions[:, 0] // 120 + positions[:, 1] // 120 * 16).int()
