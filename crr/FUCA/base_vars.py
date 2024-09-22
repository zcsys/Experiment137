SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MENU_WIDTH = 180
SIMUL_WIDTH = SCREEN_WIDTH - MENU_WIDTH
SIMUL_HEIGHT = SCREEN_HEIGHT

SIGHT = 120
N_TARGET = 500
AUTO_FISSION_THRESHOLD = 10000
METABOLIC_ACTIVITY_CONSTANT = 0.01

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

THING_TYPES = {
    "controlled_cell": {
        "color": BLUE,
        "size": 20,
        "initial_energy": 10000.,
        "overlap": False
    },
    "cell": {
        "color": GREEN,
        "size": 20,
        "initial_energy": 1000.,
        "overlap": False
    },
    "sugar": {
        "color": YELLOW,
        "size": 1,
        "initial_energy": 0.,
        "overlap": True
    }
}
