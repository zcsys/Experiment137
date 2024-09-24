SIMUL_WIDTH = 1920
SIMUL_HEIGHT = 1080
MENU_WIDTH = 180
SCREEN_WIDTH = SIMUL_WIDTH + MENU_WIDTH
SCREEN_HEIGHT = SIMUL_HEIGHT

SIGHT = 120
N_TARGET = 500
AUTO_FISSION_THRESHOLD = 10000
METABOLIC_ACTIVITY_CONSTANT = 0.1

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
        "size": 10,
        "initial_energy": 10000.,
        "overlap": False
    },
    "cell": {
        "color": GREEN,
        "size": 10,
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

GENOME211_11 = [
    0.0,
    0.0,
    0.0,
    0.6830495595932007,
    0.1516951322555542,
    0.0,
    -0.42980408668518066,
    -0.26049137115478516,
    -0.9150177240371704,
    0.019066452980041504,
    0.3572014570236206,
    0.0,
    0.14348268508911133,
    -0.7865908145904541,
    -0.2719993591308594,
    0.0,
    0.0,
    0.8695175647735596,
    0.978837251663208,
    -1.5926415920257568,
    -0.7710322141647339,
    0.0,
    1.7021629810333252,
    -0.29253268241882324,
    0.8500052690505981,
    -2.354734420776367,
    -0.5327942371368408,
    0.287997841835022,
    0.0,
    0.0,
    0.2556288242340088,
    0.0,
    0.0677412748336792,
    -0.051197171211242676
]
