from base_vars import *
from base_vars import (METABOLIC_ACTIVITY_CONSTANT, AUTO_FISSION_THRESHOLD,
                       N_TARGET)
import torch

def Rules(simul, n):
    global METABOLIC_ACTIVITY_CONSTANT, AUTO_FISSION_THRESHOLD, N_TARGET

    # Accumulated system energy pops out sugars
    if 0 in n:
        if simul.things.E > 1000:
            number_of_new, simul.things.E = divmod(simul.things.E, 1000)
            simul.things.add_sugars(int(number_of_new))

    # Living conditions get harder as population goes high
    if 4 in n:
        if simul.things.Pop <= 20:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 20 < simul.things.Pop <= 30:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 * (simul.things.Pop - 20)
        elif 30 < simul.things.Pop:
            METABOLIC_ACTIVITY_CONSTANT = 1. * (simul.things.Pop - 30)

    # Different rules for the initial ("incubation") epochs
    if 5 in n:
        if simul.epochs > 6:
            simul.things.heat = 3
            N_TARGET = 100
            AUTO_FISSION_THRESHOLD = 20000
        elif simul.epochs == 6:
            simul.things.heat = 3
            N_TARGET = 150
            AUTO_FISSION_THRESHOLD = 15000
        elif simul.epochs == 5:
            simul.things.heat = 3
            N_TARGET = 200
            AUTO_FISSION_THRESHOLD = 15000
        elif 0 < simul.epochs < 5:
            simul.things.heat = 11
            N_TARGET = 350

    # Populate universe with sugars until N_TARGET is reached
    if 1 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_sugars(N_TARGET - simul.things.N)

    # Auto fission
    if 2 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if mask:
                simul.things.monad_division(i)

    # Aging and death
    if 3 in n:
        simul.things.energies[simul.things.monad_mask] -= (
            METABOLIC_ACTIVITY_CONSTANT
        )
        to_remove = torch.nonzero(
            simul.things.energies[simul.things.monad_mask] <= 0
        )
        if len(to_remove) > 0:
            simul.things.monad_death(to_remove.squeeze(1).tolist())
