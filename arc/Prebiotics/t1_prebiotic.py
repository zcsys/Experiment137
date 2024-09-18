import pygame
import torch
import numpy as np
import random

# Initialize Pygame and its mixer for sound
pygame.init()
pygame.mixer.init()

# Set sound parameters
SAMPLE_RATE = 44100  # Standard audio sample rate
TONE_DURATION = 0.5  # Duration of each tone in seconds
AMPLITUDE = 32767  # Maximum amplitude for 16-bit sound

# Initialize display
screen = pygame.display.set_mode((1920, 1080))
clock = pygame.time.Clock()

# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

# Define a light source, move to GPU
light_source = {
    'position': torch.tensor([960, 540], device=device),  # Center of screen
    'intensity': torch.tensor(1.0, device=device),  # Full intensity
    'color': torch.tensor([1.0, 0.5, 0.2], device=device)  # Orange light
}

# Define the space (Full HD) as a 3D tensor (height, width, RGB), move to GPU
space = torch.zeros(1080, 1920, 3, device=device)

def propagate_light(space, light_source):
    height, width, _ = space.shape
    x, y = light_source['position']
    intensity = light_source['intensity']
    color = light_source['color']

    # Generate a grid of coordinates for the entire space using GPU
    y_coords, x_coords = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')

    # Calculate the distance from the light source to each pixel on the GPU
    distances = torch.sqrt((x_coords - x)**2 + (y_coords - y)**2)

    # Calculate light falloff (inverse square law) on GPU
    light_effect = intensity / (distances + 1e-6)

    # Apply light effect to the entire space tensor on the GPU
    space += (light_effect.unsqueeze(-1) * color).clamp(0, 1)

    return space, light_effect

def render_space(space):
    # Convert PyTorch tensor to numpy on CPU for Pygame
    space_np = (space.cpu().numpy() * 255).astype(np.uint8)  # Convert to 0-255 range for Pygame
    pygame_surface = pygame.surfarray.make_surface(space_np.transpose(1, 0, 2))  # Correct shape for Pygame
    return pygame_surface

def generate_tone(frequencies, duration, sample_rate=SAMPLE_RATE):
    """Generate a complex tone with multiple frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate multiple sine waves for different frequencies and sum them
    waveform = sum(np.sin(2 * np.pi * freq * t) for freq in frequencies) * (AMPLITUDE / len(frequencies))

    # Add slight random noise for a playful effect
    waveform += np.random.uniform(-AMPLITUDE * 0.05, AMPLITUDE * 0.05, waveform.shape)

    return waveform.astype(np.int16)

def play_sound(frequencies):
    """Create and play a sine wave sound dynamically based on multiple frequencies."""
    tone = generate_tone(frequencies, TONE_DURATION)

    # Convert the tone to a Pygame Sound object and play it
    sound_array = np.stack([tone, tone], axis=-1)  # Stereo sound
    sound = pygame.sndarray.make_sound(sound_array)
    sound.play()

def modulate_sound_based_on_light(light_effect):
    """Modulate the sound based on the average light intensity."""
    avg_intensity = light_effect.mean().item()  # Ensure avg_intensity is a float

    # Playful frequency variation based on intensity
    base_frequency = 220 + avg_intensity * 880  # Base frequency range (220 Hz to 1100 Hz)

    # Add harmonic and random variations to make it more playful
    frequencies = [
        base_frequency,
        base_frequency * 1.5,  # Harmonic
        base_frequency * random.uniform(0.9, 1.2),  # Slightly random frequency for variation
    ]

    play_sound(frequencies)

if __name__ == "__main__":
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Propagate light in the space (on GPU)
        space, light_effect = propagate_light(space, light_source)

        # Render the space (on CPU, as Pygame doesn't support GPU rendering directly)
        surface = render_space(space)

        # Display the surface
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Modulate sound based on the visual propagation
        modulate_sound_based_on_light(light_effect)

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
