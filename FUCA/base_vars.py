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
        "color": colors["L"],
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
    2, 0, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 1
    0, 2, 0, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 2
    0, 0, 2, 0, 0, 0, 0, 0, # layer 1 to layer 2 neuron 3
    0, 0, 0, 2, 0, 0, 0, 0, # layer 1 to layer 2 neuron 4
    0, 0, 0, 0, 2, 0, 0, 0, # layer 1 to layer 2 neuron 5
    0, 0, 0, 0, 0, 2, 0, 0, # layer 1 to layer 2 neuron 6
    0, 0, 0, 0, 0, 0, 2, 0, # layer 1 to layer 2 neuron 7
    0, 0, 0, 0, 0, 0, 0, 2, # layer 1 to layer 2 neuron 8
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
    2, 0, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 1
    0, 2, 0, 0, 0, 0, 0, 0, # layer 2 to output layer neuron 2

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

neighbors = {
    1: [1, 2, 17, 18],
    2: [1, 2, 3, 17, 18, 19],
    3: [2, 3, 4, 18, 19, 20],
    4: [3, 4, 5, 19, 20, 21],
    5: [4, 5, 6, 20, 21, 22],
    6: [5, 6, 7, 21, 22, 23],
    7: [6, 7, 8, 22, 23, 24],
    8: [7, 8, 9, 23, 24, 25],
    9: [8, 9, 10, 24, 25, 26],
    10: [9, 10, 11, 25, 26, 27],
    11: [10, 11, 12, 26, 27, 28],
    12: [11, 12, 13, 27, 28, 29],
    13: [12, 13, 14, 28, 29, 30],
    14: [13, 14, 15, 29, 30, 31],
    15: [14, 15, 16, 30, 31, 32],
    16: [15, 16, 31, 32],
    17: [1, 2, 17, 18, 33, 34],
    18: [1, 2, 3, 17, 18, 19, 33, 34, 35],
    19: [2, 3, 4, 18, 19, 20, 34, 35, 36],
    20: [3, 4, 5, 19, 20, 21, 35, 36, 37],
    21: [4, 5, 6, 20, 21, 22, 36, 37, 38],
    22: [5, 6, 7, 21, 22, 23, 37, 38, 39],
    23: [6, 7, 8, 22, 23, 24, 38, 39, 40],
    24: [7, 8, 9, 23, 24, 25, 39, 40, 41],
    25: [8, 9, 10, 24, 25, 26, 40, 41, 42],
    26: [9, 10, 11, 25, 26, 27, 41, 42, 43],
    27: [10, 11, 12, 26, 27, 28, 42, 43, 44],
    28: [11, 12, 13, 27, 28, 29, 43, 44, 45],
    29: [12, 13, 14, 28, 29, 30, 44, 45, 46],
    30: [13, 14, 15, 29, 30, 31, 45, 46, 47],
    31: [14, 15, 16, 30, 31, 32, 46, 47, 48],
    32: [15, 16, 31, 32, 47, 48],
    33: [17, 18, 33, 34, 49, 50],
    34: [17, 18, 19, 33, 34, 35, 49, 50, 51],
    35: [18, 19, 20, 34, 35, 36, 50, 51, 52],
    36: [19, 20, 21, 35, 36, 37, 51, 52, 53],
    37: [20, 21, 22, 36, 37, 38, 52, 53, 54],
    38: [21, 22, 23, 37, 38, 39, 53, 54, 55],
    39: [22, 23, 24, 38, 39, 40, 54, 55, 56],
    40: [23, 24, 25, 39, 40, 41, 55, 56, 57],
    41: [24, 25, 26, 40, 41, 42, 56, 57, 58],
    42: [25, 26, 27, 41, 42, 43, 57, 58, 59],
    43: [26, 27, 28, 42, 43, 44, 58, 59, 60],
    44: [27, 28, 29, 43, 44, 45, 59, 60, 61],
    45: [28, 29, 30, 44, 45, 46, 60, 61, 62],
    46: [29, 30, 31, 45, 46, 47, 61, 62, 63],
    47: [30, 31, 32, 46, 47, 48, 62, 63, 64],
    48: [31, 32, 47, 48, 63, 64],
    49: [33, 34, 49, 50, 65, 66],
    50: [33, 34, 35, 49, 50, 51, 65, 66, 67],
    51: [34, 35, 36, 50, 51, 52, 66, 67, 68],
    52: [35, 36, 37, 51, 52, 53, 67, 68, 69],
    53: [36, 37, 38, 52, 53, 54, 68, 69, 70],
    54: [37, 38, 39, 53, 54, 55, 69, 70, 71],
    55: [38, 39, 40, 54, 55, 56, 70, 71, 72],
    56: [39, 40, 41, 55, 56, 57, 71, 72, 73],
    57: [40, 41, 42, 56, 57, 58, 72, 73, 74],
    58: [41, 42, 43, 57, 58, 59, 73, 74, 75],
    59: [42, 43, 44, 58, 59, 60, 74, 75, 76],
    60: [43, 44, 45, 59, 60, 61, 75, 76, 77],
    61: [44, 45, 46, 60, 61, 62, 76, 77, 78],
    62: [45, 46, 47, 61, 62, 63, 77, 78, 79],
    63: [46, 47, 48, 62, 63, 64, 78, 79, 80],
    64: [47, 48, 63, 64, 79, 80],
    65: [49, 50, 65, 66, 81, 82],
    66: [49, 50, 51, 65, 66, 67, 81, 82, 83],
    67: [50, 51, 52, 66, 67, 68, 82, 83, 84],
    68: [51, 52, 53, 67, 68, 69, 83, 84, 85],
    69: [52, 53, 54, 68, 69, 70, 84, 85, 86],
    70: [53, 54, 55, 69, 70, 71, 85, 86, 87],
    71: [54, 55, 56, 70, 71, 72, 86, 87, 88],
    72: [55, 56, 57, 71, 72, 73, 87, 88, 89],
    73: [56, 57, 58, 72, 73, 74, 88, 89, 90],
    74: [57, 58, 59, 73, 74, 75, 89, 90, 91],
    75: [58, 59, 60, 74, 75, 76, 90, 91, 92],
    76: [59, 60, 61, 75, 76, 77, 91, 92, 93],
    77: [60, 61, 62, 76, 77, 78, 92, 93, 94],
    78: [61, 62, 63, 77, 78, 79, 93, 94, 95],
    79: [62, 63, 64, 78, 79, 80, 94, 95, 96],
    80: [63, 64, 79, 80, 95, 96],
    81: [65, 66, 81, 82, 97, 98],
    82: [65, 66, 67, 81, 82, 83, 97, 98, 99],
    83: [66, 67, 68, 82, 83, 84, 98, 99, 100],
    84: [67, 68, 69, 83, 84, 85, 99, 100, 101],
    85: [68, 69, 70, 84, 85, 86, 100, 101, 102],
    86: [69, 70, 71, 85, 86, 87, 101, 102, 103],
    87: [70, 71, 72, 86, 87, 88, 102, 103, 104],
    88: [71, 72, 73, 87, 88, 89, 103, 104, 105],
    89: [72, 73, 74, 88, 89, 90, 104, 105, 106],
    90: [73, 74, 75, 89, 90, 91, 105, 106, 107],
    91: [74, 75, 76, 90, 91, 92, 106, 107, 108],
    92: [75, 76, 77, 91, 92, 93, 107, 108, 109],
    93: [76, 77, 78, 92, 93, 94, 108, 109, 110],
    94: [77, 78, 79, 93, 94, 95, 109, 110, 111],
    95: [78, 79, 80, 94, 95, 96, 110, 111, 112],
    96: [79, 80, 95, 96, 111, 112],
    97: [81, 82, 97, 98, 113, 114],
    98: [81, 82, 83, 97, 98, 99, 113, 114, 115],
    99: [82, 83, 84, 98, 99, 100, 114, 115, 116],
    100: [83, 84, 85, 99, 100, 101, 115, 116, 117],
    101: [84, 85, 86, 100, 101, 102, 116, 117, 118],
    102: [85, 86, 87, 101, 102, 103, 117, 118, 119],
    103: [86, 87, 88, 102, 103, 104, 118, 119, 120],
    104: [87, 88, 89, 103, 104, 105, 119, 120, 121],
    105: [88, 89, 90, 104, 105, 106, 120, 121, 122],
    106: [89, 90, 91, 105, 106, 107, 121, 122, 123],
    107: [90, 91, 92, 106, 107, 108, 122, 123, 124],
    108: [91, 92, 93, 107, 108, 109, 123, 124, 125],
    109: [92, 93, 94, 108, 109, 110, 124, 125, 126],
    110: [93, 94, 95, 109, 110, 111, 125, 126, 127],
    111: [94, 95, 96, 110, 111, 112, 126, 127, 128],
    112: [95, 96, 111, 112, 127, 128],
    113: [97, 98, 113, 114, 129, 130],
    114: [97, 98, 99, 113, 114, 115, 129, 130, 131],
    115: [98, 99, 100, 114, 115, 116, 130, 131, 132],
    116: [99, 100, 101, 115, 116, 117, 131, 132, 133],
    117: [100, 101, 102, 116, 117, 118, 132, 133, 134],
    118: [101, 102, 103, 117, 118, 119, 133, 134, 135],
    119: [102, 103, 104, 118, 119, 120, 134, 135, 136],
    120: [103, 104, 105, 119, 120, 121, 135, 136, 137],
    121: [104, 105, 106, 120, 121, 122, 136, 137, 138],
    122: [105, 106, 107, 121, 122, 123, 137, 138, 139],
    123: [106, 107, 108, 122, 123, 124, 138, 139, 140],
    124: [107, 108, 109, 123, 124, 125, 139, 140, 141],
    125: [108, 109, 110, 124, 125, 126, 140, 141, 142],
    126: [109, 110, 111, 125, 126, 127, 141, 142, 143],
    127: [110, 111, 112, 126, 127, 128, 142, 143, 144],
    128: [111, 112, 127, 128, 143, 144],
    129: [113, 114, 129, 130],
    130: [113, 114, 115, 129, 130, 131],
    131: [114, 115, 116, 130, 131, 132],
    132: [115, 116, 117, 131, 132, 133],
    133: [116, 117, 118, 132, 133, 134],
    134: [117, 118, 119, 133, 134, 135],
    135: [118, 119, 120, 134, 135, 136],
    136: [119, 120, 121, 135, 136, 137],
    137: [120, 121, 122, 136, 137, 138],
    138: [121, 122, 123, 137, 138, 139],
    139: [122, 123, 124, 138, 139, 140],
    140: [123, 124, 125, 139, 140, 141],
    141: [124, 125, 126, 140, 141, 142],
    142: [125, 126, 127, 141, 142, 143],
    143: [126, 127, 128, 142, 143, 144],
    144: [127, 128, 143, 144]
}
