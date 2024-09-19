from simulation import Simulation
from things import Things
import argparse

POP_0 = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Experiment 137.03: LUCA"
    )
    parser.add_argument('--load', type = str,
                        help = "Load simulation state from a JSON file")
    args = parser.parse_args()

    things_instance = Things(
        ["controlled_cell"] +
        ["cell"] * POP_0 +
        ["sugar"] * POP_0
    )

    simulation = Simulation(things_instance, load_file = args.load)
    simulation.run()
