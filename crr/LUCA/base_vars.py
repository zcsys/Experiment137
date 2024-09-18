import random
import torch

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MENU_WIDTH = 180

SPEED_CONSTANT = 1.
INITIAL_ENERGY = 1000.

SQRT2 = 2 ** 0.5

class color:
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    CYAN = (0, 255, 255)
    RED = (255, 0, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

def default_action_function():
    if random.random() < 0.5:
        return torch.tensor(random.uniform(0, 2 * torch.pi))
    return torch.tensor(float('nan'))

def move_diagonally():
    return torch.tensor(torch.pi / 4)

def just_dont():
    return torch.tensor(float('nan'))

def get_controlled_action(controlled_direction):
    if controlled_direction.norm().item() > 0:
        angle = torch.atan2(
            controlled_direction[1],
            controlled_direction[0]
        )
        return angle
    return torch.tensor(float('nan'))

THING_TYPES = {
    "controlled_cell": {
        "color": color.BLUE,
        "size": 20,
        "action_function": get_controlled_action,
        "overlap": False
    },
    "cell": {
        "color": color.GREEN,
        "size": 20,
        "action_function": move_diagonally,
        "overlap": False
    },
    "sugar": {
        "color": color.YELLOW,
        "size": 1,
        "action_function": default_action_function,
        "overlap": True
    }
}
