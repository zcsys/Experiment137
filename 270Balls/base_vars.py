SIMUL_WIDTH = 1920
SIMUL_HEIGHT = 1080
MENU_WIDTH = 180
SCREEN_WIDTH = SIMUL_WIDTH + MENU_WIDTH
SCREEN_HEIGHT = SIMUL_HEIGHT

SIGHT = 60
N_TARGET = 500
AUTO_FISSION_THRESHOLD = 10000
METABOLIC_ACTIVITY_CONSTANT = 0.1
SUGAR_ENERGY = 1000
SYSTEM_HEAT = 11

colors = {
    "0": (0, 0, 0),
    "A": (0, 0, 127),
    "B": (0, 0, 254),
    "C": (0, 127, 0),
    "D": (0, 127, 127),
    "E": (0, 127, 254),
    "F": (0, 254, 0),
    "G": (0, 254, 127),
    "H": (0, 254, 254),
    "I": (127, 0 ,0),
    "J": (127, 0, 127),
    "K": (127, 0, 254),
    "L": (127, 127, 0),
    "M": (127, 127, 127),
    "N": (127, 127, 254),
    "O": (127, 254, 0),
    "P": (127, 254, 127),
    "Q": (127, 254, 254),
    "R": (254, 0, 0),
    "S": (254, 0, 127),
    "T": (254, 0, 254),
    "U": (254, 127, 0),
    "V": (254, 127, 127),
    "W": (254, 127, 254),
    "X": (254, 254, 0),
    "Y": (254, 254, 127),
    "Z": (254, 254, 254),
}

THING_TYPES = {
    "monad": {
        "color": colors["A"],
        "size": 5,
        "initial_energy": 1000.,
        "overlap": False
    },
    "sugar": {
        "color": colors["F"],
        "size": 1,
        "initial_energy": 0.,
        "overlap": True
    }
}
