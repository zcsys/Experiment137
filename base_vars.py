import random
import torch

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MENU_WIDTH = 180

SPEED_CONSTANT = 1.

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

THING_TYPES = {
    "controlled_cell": {
        "color": color.BLUE,
        "size": 50,
        "action_function": None
    },
    "cell": {
        "color": color.GREEN,
        "size": 50,
        "action_function": default_action_function
    },
    "sugar": {
        "color": color.YELLOW,
        "size": 10,
        "action_function": default_action_function
    }
}
