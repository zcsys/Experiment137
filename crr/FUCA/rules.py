from base_vars import *
import torch

def Rules(simul, n):
    # Accumulated system energy pops out sugars
    if 0 in n:
        if simul.things.E > 1000:
            number_of_new, simul.things.E = divmod(simul.things.E, 1000)
            simul.things.add_sugars(number_of_new)

    # Populate universe with sugars until N_TARGET is reached
    if 1 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_sugars(N_TARGET - simul.things.N)

    # Auto fission
    if 2 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if i == 0:
                continue
            if mask:
                if simul.things.cell_division(i):
                    print("cell division at:", i)

    # Aging and death
    if 3 in n:
        energies = simul.things.energies[simul.things.cell_mask]
        energies -= METABOLIC_ACTIVITY_CONSTANT
        simul.things.remove_things(
            torch.nonzero(energies <= 0, as_tuple = False).squeeze(1).tolist()
        )
        simul.things.energies[simul.things.cell_mask] = energies
