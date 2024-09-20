from base_vars import *

def Rules(simul, n):
    # Accumulated system energy pops out sugars
    if 0 in n:
        if simul.things.E > 1000:
            number_of_new, simul.things.E = divmod(simul.things.E, 1000)
            simul.things.add_things(["sugar"] * int(number_of_new))
            simul.things.calc_state()

    # Populate universe with sugars until N_TARGET is reached
    if 1 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_things(["sugar"])
            simul.things.calc_state()

    # Auto fission
    if 2 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if i == 0:
                continue
            if mask:
                simul.things.cell_division(i)
