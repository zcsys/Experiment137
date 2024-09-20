from base_vars import *

def Rules(simul, n):
    # Accumulated system energy pops out sugars
    if n == 0:
        if simul.things.E > 1000:
            number_of_new, simul.things.E = divmod(simul.things.E, 1000)
            simul.things.add_things(["sugar"] * int(number_of_new))
            simul.things.diffs = (simul.things.positions.unsqueeze(1) -
                                  simul.things.positions.unsqueeze(0))
