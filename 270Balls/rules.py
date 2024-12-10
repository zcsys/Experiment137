from base_vars import *
from base_vars import (N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT,
                       AUTO_FISSION_THRESHOLD)
import torch

def Rules(simul, n):
    global N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT, \
           AUTO_FISSION_THRESHOLD

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

    # Population control
    if 1 in n:
        if simul.period > 0 or simul.epoch >= 20:
            pass
        elif simul.epoch >= 15:
            update_system_heat(5)
            AUTO_FISSION_THRESHOLD = 60000
        elif simul.epoch >= 10:
            N_TARGET = 500
            update_system_heat(7)
            AUTO_FISSION_THRESHOLD = 30000
            if simul.age % 5 == 0 and simul.step == 0:
                simul.things.add_structuralUnits(1)
        elif simul.epoch >= 5:
            N_TARGET = 500
            update_system_heat(9)
            AUTO_FISSION_THRESHOLD = 15000
            if simul.age % 5 == 0 and simul.step == 0:
                simul.things.add_structuralUnits(1)
        else:
            N_TARGET = 500
            update_system_heat(11)
            AUTO_FISSION_THRESHOLD = 10000
            if simul.epoch == 0 and simul.age == 0 and simul.step == 1:
                simul.things.add_structuralUnits()

        if simul.things.N < N_TARGET:
            simul.things.add_energyUnits(N_TARGET - simul.things.N)

        if simul.things.E <= 100:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 100 < simul.things.E <= 200:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 + 0.009 * (simul.things.E - 100)
        elif 200 < simul.things.E:
            METABOLIC_ACTIVITY_CONSTANT = 1. + 0.09 * (simul.things.E - 200)

    # Resource management
    if 2 in n:
        numberOf_energyUnits_to_create, simul.excess = divmod(simul.excess, 10)
        if numberOf_energyUnits_to_create > 0:
            simul.things.add_energyUnits(int(numberOf_energyUnits_to_create))
