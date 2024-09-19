from base_vars import *
from action_functions import *

THING_TYPES = {
    "controlled_cell": {
        "color": BLUE,
        "size": 20,
        "initial_energy": 10000.,
        "action_function": controlled_action,
        "overlap": False,
        "draw_as": "circle"
    },
    "cell": {
        "color": GREEN,
        "size": 20,
        "initial_energy": 1000.,
        "action_function": move_random,
        "overlap": False,
        "draw_as": "circle"
    },
    "sugar": {
        "color": YELLOW,
        "size": 1,
        "initial_energy": 0.,
        "action_function": move_random,
        "overlap": True,
        "draw_as": "line"
    }
}
