from base_vars import *
from base_vars import (METABOLIC_ACTIVITY_CONSTANT, AUTO_FISSION_THRESHOLD,
                       N_TARGET)
import torch

def Rules(simul, n):
    global N_TARGET

    # More resources in earlier epochs
    if 0 in n:
        if 20 <= simul.epochs:
            N_TARGET = 200
        elif 15 <= simul.epochs < 20:
            N_TARGET = 300
        elif 10 <= simul.epochs < 15:
            N_TARGET = 400
        elif 5 <= simul.epochs < 10:
            N_TARGET = 500

    # Auto fission
    if 1 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if mask:
                simul.things.monad_division(i)

    # Aging and death
    if 2 in n:
        simul.things.energies[simul.things.monad_mask] -= (
            METABOLIC_ACTIVITY_CONSTANT
        )
        to_remove = torch.nonzero(
            simul.things.energies[simul.things.monad_mask] <= 0
        )
        if len(to_remove) > 0:
            # Autogenetic breeding
            if simul.things.Pop <= 5:
                for idx in range(simul.things.Pop):
                    simul.things.monad_autogenesis_v1(idx)
            simul.things.monad_death(to_remove.squeeze(1).tolist())

    # Populate universe with sugars until N_TARGET is reached
    if 3 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_sugars(N_TARGET - simul.things.N)
