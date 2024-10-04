SIMUL_WIDTH = 1920
SIMUL_HEIGHT = 1080
MENU_WIDTH = 180
SCREEN_WIDTH = SIMUL_WIDTH + MENU_WIDTH
SCREEN_HEIGHT = SIMUL_HEIGHT

SIGHT = 120
N_TARGET = 500
AUTO_FISSION_THRESHOLD = 10000
METABOLIC_ACTIVITY_CONSTANT = 0.1
SUGAR_ENERGY = 1000

colors = {
    "∅": (0, 0, 0),
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
        "color": GREEN,
        "size": 20,
        "nucleus_size": 5,
        "initial_energy": 1000.,
        "overlap": False
    },
    "sugar": {
        "color": YELLOW,
        "size": 1,
        "nucleus_size": "N/A",
        "initial_energy": 0.,
        "overlap": True
    }
}

REGULATORY_GENOME_429_0 = [
    # 1 sense organ active, 4 not: [[0, 0], [1, 1], [0], [0], [0, 0, 0]]
    # Activation function for layer 1 is the identity function.
    [[0, 0, 1, 1, 0, 0, 0, 0, 0] for i in range(2)] + # layer 1 active only for
    [[0 for _ in range(9)] for i in range(6)],        # the active sense and
                                                      # action organs
    [[0 for _ in range(8)] for i in range(8)], # hidden state 1 not active
    [[1], [1], [0], [0], [0], [0], [0], [0]], # bias search for layer 1 active
                                              # only for the active action
                                              # organs

    # W_1_2 is 8x8 identity martrix.
    # Activation function for layer 2 is the identity function.
    [[0 for _ in range(8)] for i in range(8)], # layer 2 not active
    [[0 for _ in range(8)] for i in range(8)], # hidden state 2 not active
    [[0] for i in range(8)], # bias search for layer 2 not active

    # W_2_o = [[1 if j == i and i in range(2) else 0 for j in range(8)]
    #          for i in range(4)]
    # Activation function for output layer is tanh.
    [[0 for _ in range(8)] for i in range(4)], # weight and bias search for
    [[0] for _ in range(4)]                    # output layer not active till
                                               # all hidden layers are activated

]
