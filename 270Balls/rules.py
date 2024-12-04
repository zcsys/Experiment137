from base_vars import *
from base_vars import N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT
import torch

def Rules(simul, n):
    global N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT

    # Birth and death
    if 0 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if mask:
                simul.things.monad_division(i)
        simul.things.energies -= METABOLIC_ACTIVITY_CONSTANT
        to_remove = torch.nonzero(simul.things.energies <= 0)
        if len(to_remove) > 0:
            simul.things.monad_death(to_remove.squeeze(1).tolist())
        simul.things.E = simul.things.energies.sum().item() // 1000

    # Hyperparameter adjustment for the ages of incubation
    if 1 in n:
        if simul.epoch > 0 or simul.age >= 20:
            N_TARGET = 100
            SYSTEM_HEAT = 3
        elif simul.age >= 15:
            N_TARGET = 200
            SYSTEM_HEAT = 5
        elif simul.age >= 10:
            N_TARGET = 300
            SYSTEM_HEAT = 7
        elif simul.age >= 5:
            N_TARGET = 400
            SYSTEM_HEAT = 9

        if simul.things.N < N_TARGET:
            simul.things.add_sugars(N_TARGET - simul.things.N)

    # Population control
    if 2 in n:
        if simul.things.E <= 100:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 100 < simul.things.E <= 200:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 + 0.009 * (simul.things.E - 100)
        elif 200 < simul.things.E:
            METABOLIC_ACTIVITY_CONSTANT = 1. + 0.09 * (simul.things.E - 200)
