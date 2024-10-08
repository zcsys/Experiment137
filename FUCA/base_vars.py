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
        "color": colors["F"],
        "size": 20,
        "nucleus_size": 5,
        "initial_energy": 1000.,
        "overlap": False
    },
    "sugar": {
        "color": colors["X"],
        "size": 1,
        "nucleus_size": "N/A",
        "initial_energy": 0.,
        "overlap": True
    }
}

GENOME429_0 = [
    # CODING PART

    # Sense organs
    0, 0,                   # compass to layer 1 neuron 1
    0, 0,                   # hunger to layer 1 neuron 1
    0,                      # energy to layer 1 neuron 1
    0,                      # LMWS neuron to layer 1 neuron 1
    0, 0, 0,                # ear to layer 1 neuron 1
    0, 0,                   # compass to layer 1 neuron 2
    0, 0,                   # hunger to layer 1 neuron 2
    0,                      # energy to layer 1 neuron 2
    0,                      # LMWS neuron to layer 1 neuron 2
    0, 0, 0,                # ear to layer 1 neuron 2
    0, 0,                   # compass to layer 1 neuron 3
    0, 0,                   # hunger to layer 1 neuron 3
    0,                      # energy to layer 1 neuron 3
    0,                      # LMWS neuron to layer 1 neuron 3
    0, 0, 0,                # ear to layer 1 neuron 3
    0, 0,                   # compass to layer 1 neuron 4
    0, 0,                   # hunger to layer 1 neuron 4
    0,                      # energy to layer 1 neuron 4
    0,                      # LMWS neuron to layer 1 neuron 4
    0, 0, 0,                # ear to layer 1 neuron 4
    0, 0,                   # compass to layer 1 neuron 5
    0, 0,                   # hunger to layer 1 neuron 5
    0,                      # energy to layer 1 neuron 5
    0,                      # LMWS neuron to layer 1 neuron 5
    0, 0, 0,                # ear to layer 1 neuron 5
    0, 0,                   # compass to layer 1 neuron 6
    0, 0,                   # hunger to layer 1 neuron 6
    0,                      # energy to layer 1 neuron 6
    0,                      # LMWS neuron to layer 1 neuron 6
    0, 0, 0,                # ear to layer 1 neuron 6
    0, 0,                   # compass to layer 1 neuron 7
    0, 0,                   # hunger to layer 1 neuron 7
    0,                      # energy to layer 1 neuron 7
    0,                      # LMWS neuron to layer 1 neuron 7
    0, 0, 0,                # ear to layer 1 neuron 7
    0, 0,                   # compass to layer 1 neuron 8
    0, 0,                   # hunger to layer 1 neuron 8
    0,                      # energy to layer 1 neuron 8
    0,                      # LMWS neuron to layer 1 neuron 8
    0, 0, 0,                # ear to layer 1 neuron 8

    # Hidden state and biases for layer 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 2
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 3
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 4
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 5
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 6
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 7
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 8
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 biases

    # Layer 2
    1, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 1
    0, 1, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 2
    0, 0, 1, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 3
    0, 0, 0, 1, 0, 0, 0, 0, # layer 1 to layer 2 neuron 4
    0, 0, 0, 0, 1, 0, 0, 0, # layer 1 to layer 2 neuron 5
    0, 0, 0, 0, 0, 1, 0, 0, # layer 1 to layer 2 neuron 6
    0, 0, 0, 0, 0, 0, 1, 0, # layer 1 to layer 2 neuron 7
    0, 0, 0, 0, 0, 0, 0, 1, # layer 1 to layer 2 neuron 8
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 2
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 3
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 4
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 5
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 6
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 7
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 8
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 biases

    # Action organs
    # Flagellum
    1, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 1
    0, 1, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 2

    # Divisor
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 3

    # Messager
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 4

    0, 0, 0, 0, 0, 0, 0, 0,  # output layer biases

    # REGULATORY PART

    # Sense organs
    0, 0,                   # compass to layer 1 neuron 1
    1, 1,                   # hunger to layer 1 neuron 1
    0,                      # energy to layer 1 neuron 1
    0,                      # LMWS neuron to layer 1 neuron 1
    0, 0, 0,                # ear to layer 1 neuron 1
    0, 0,                   # compass to layer 1 neuron 2
    1, 1,                   # hunger to layer 1 neuron 2
    0,                      # energy to layer 1 neuron 2
    0,                      # LMWS neuron to layer 1 neuron 2
    0, 0, 0,                # ear to layer 1 neuron 2
    0, 0,                   # compass to layer 1 neuron 3
    0, 0,                   # hunger to layer 1 neuron 3
    0,                      # energy to layer 1 neuron 3
    0,                      # LMWS neuron to layer 1 neuron 3
    0, 0, 0,                # ear to layer 1 neuron 3
    0, 0,                   # compass to layer 1 neuron 4
    0, 0,                   # hunger to layer 1 neuron 4
    0,                      # energy to layer 1 neuron 4
    0,                      # LMWS neuron to layer 1 neuron 4
    0, 0, 0,                # ear to layer 1 neuron 4
    0, 0,                   # compass to layer 1 neuron 5
    0, 0,                   # hunger to layer 1 neuron 5
    0,                      # energy to layer 1 neuron 5
    0,                      # LMWS neuron to layer 1 neuron 5
    0, 0, 0,                # ear to layer 1 neuron 5
    0, 0,                   # compass to layer 1 neuron 6
    0, 0,                   # hunger to layer 1 neuron 6
    0,                      # energy to layer 1 neuron 6
    0,                      # LMWS neuron to layer 1 neuron 6
    0, 0, 0,                # ear to layer 1 neuron 6
    0, 0,                   # compass to layer 1 neuron 7
    0, 0,                   # hunger to layer 1 neuron 7
    0,                      # energy to layer 1 neuron 7
    0,                      # LMWS neuron to layer 1 neuron 7
    0, 0, 0,                # ear to layer 1 neuron 7
    0, 0,                   # compass to layer 1 neuron 8
    0, 0,                   # hunger to layer 1 neuron 8
    0,                      # energy to layer 1 neuron 8
    0,                      # LMWS neuron to layer 1 neuron 8
    0, 0, 0,                # ear to layer 1 neuron 8

    # Hidden state and biases for layer 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 2
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 3
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 4
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 5
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 6
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 7
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 1 weight 8
    1, 1, 0, 0, 0, 0, 0, 0, # layer 1 biases

    # Layer 2
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 1
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 2
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 3
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 4
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 5
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 6
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 7
    0, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 8
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 1
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 2
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 3
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 4
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 5
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 6
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 7
    0, 0, 0, 0, 0, 0, 0, 0, # hidden state 2 weight 8
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 biases

    # Action organs
    # Flagellum
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 1
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 2

    # Divisor
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 3

    # Messager
    0, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 4

    0, 0, 0, 0, 0, 0, 0, 0  # output layer biases
]
