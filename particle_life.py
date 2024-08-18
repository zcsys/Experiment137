import pygame
import math
import random

WINDOW_SIZE = [1920, 1080]
ATOMS = []

def randomxy(size):
    x = round(random.random() * WINDOW_SIZE[0])
    x = max(x, size)
    x = min(x, WINDOW_SIZE[0] - size)
    y = round(random.random() * WINDOW_SIZE[1])
    y = max(y, size)
    y = min(y, WINDOW_SIZE[1] - size)
    return (x, y)

def create(N, C, S, E = 100, v = None):
    for i in range(N):
        x, y = randomxy(S)
        if v == None:
            v = [0, 0]
        ATOMS.append(
            {
                "x": x,
                "y": y,
                "color": C,
                "size": S,
                "energy": E,
                "velocity": v[:]
            }
        )

def distance(i, j):
    if i == j:
        return (0, 0, 0)

    i_x = ATOMS[i]["x"]
    i_y = ATOMS[i]["y"]
    j_x = ATOMS[j]["x"]
    j_y = ATOMS[j]["y"]

    d = ((j_x - i_x) ** 2 + (j_y - i_y) ** 2) ** 0.5

    if d > 0:
        cos = (j_x - i_x) / d
        sin = (j_y - i_y) / d
    else:
        return (0, 0, 0)

    return (d, sin, cos)

def Rules(n):
    # Apply velocities and prevent overflow
    if n == 0:
        edge_friction = 0.5
        for this_atom in ATOMS:
            this_atom["x"] += this_atom["velocity"][0]
            this_atom["y"] += this_atom["velocity"][1]

            if this_atom["x"] <= this_atom["size"]:
                this_atom["x"] = this_atom["size"]
                this_atom["velocity"][0] *= -1 + edge_friction
                this_atom["velocity"][1] *= 1 - edge_friction
            elif this_atom["x"] >= WINDOW_SIZE[0] - this_atom["size"]:
                this_atom["x"] = WINDOW_SIZE[0] - this_atom["size"]
                this_atom["velocity"][0] *= -1 + edge_friction
                this_atom["velocity"][1] *= 1 - edge_friction

            if this_atom["y"] <= this_atom["size"]:
                this_atom["y"] = this_atom["size"]
                this_atom["velocity"][0] *= 1 - edge_friction
                this_atom["velocity"][1] *= -1 + edge_friction
            elif this_atom["y"] >= WINDOW_SIZE[1] - this_atom["size"]:
                this_atom["y"] = WINDOW_SIZE[1] - this_atom["size"]
                this_atom["velocity"][0] *= 1 - edge_friction
                this_atom["velocity"][1] *= -1 + edge_friction

    # Rules based on colors
    if n == 1:
        k_bb = -30
        k_by = 10
        k_yy = -1
        for i in range(len(ATOMS)):
            ATOMS[i]["force"] = [0., 0.]
            for j in range(len(ATOMS)):
                d, sin, cos = distance(i, j)
                i_col = ATOMS[i]["color"]
                j_col = ATOMS[j]["color"]
                i_size = ATOMS[i]["size"]
                j_size = ATOMS[j]["size"]

                if d > i_size + j_size:
                    F = ATOMS[j]["size"] / d ** 2
                else:
                    F = ATOMS[j]["size"] * d / (i_size + j_size) ** 3

                if i_col == (0, 0, 255) and j_col == (0, 0, 255):
                        F *= k_bb
                if (i_col == (0, 0, 255) and j_col == (255, 255, 0) or
                    i_col == (255, 255, 0) and j_col == (0, 0, 255)):
                        F *= k_by
                if i_col == (255, 255, 0) and j_col == (255, 255, 0):
                        F *= k_yy

                ATOMS[i]["force"][0] += F * cos
                ATOMS[i]["force"][1] += F * sin

            # Apply forces to velocities
            ATOMS[i]["velocity"][0] += ATOMS[i]["force"][0]
            ATOMS[i]["velocity"][1] += ATOMS[i]["force"][1]


if __name__ == '__main__':
    create(4, (0, 0, 255), 100)  # Blue atom
    create(100, (255, 255, 0), 3)  # Yellow atom

    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)

    run = True
    clock = pygame.time.Clock()

    while run:
        window.fill((0, 0, 0))

        for this_atom in ATOMS:
            pygame.draw.circle(
                window,
                this_atom["color"],
                (int(this_atom["x"]), int(this_atom["y"])),
                this_atom["size"]
            )
        pygame.display.flip()

        Rules(0)
        Rules(1)

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    pygame.quit()
