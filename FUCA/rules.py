from base_vars import *
from base_vars import METABOLIC_ACTIVITY_CONSTANT
import torch

def Rules(simul, n):
    global METABOLIC_ACTIVITY_CONSTANT

    # Accumulated system energy pops out sugars
    if 0 in n:
        if simul.things.E > 1000:
            number_of_new, simul.things.E = divmod(simul.things.E, 1000)
            simul.things.add_sugars(int(number_of_new))

    # Populate universe with sugars until N_TARGET is reached
    if 1 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_sugars(N_TARGET - simul.things.N)

    # Auto fission
    if 2 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if simul.things.thing_types[i] == "controlled_cell":
                continue
            if mask:
                simul.things.cell_division(i)

    # Aging and death
    if 3 in n:
        simul.things.energies[simul.things.cell_mask] -= (
            METABOLIC_ACTIVITY_CONSTANT
        )
        to_remove = torch.nonzero(
            simul.things.energies[simul.things.cell_mask] <= 0
        )
        if len(to_remove) > 0:
            simul.things.cell_death(to_remove.squeeze(1).tolist())

    # Living conditions get harder as population goes high and as epochs pass
    if 4 in n:
        if simul.things.Pop <= 20:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif simul.things.Pop > 20 and simul.things.Pop <= 30:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 * (simul.things.Pop - 20)
        elif simul.things.Pop > 30:
            METABOLIC_ACTIVITY_CONSTANT = 1. * (simul.things.Pop - 30)

        if simul.epochs != 0:
            simul.things.heat = 0
