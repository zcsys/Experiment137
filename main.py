from simulation import Simulation
from things import Things, default_action_function
import argparse

POP_0 = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Experiment 137.03: Monadology"
    )
    parser.add_argument('--load', type = str,
                        help = "Load simulation state from a JSON file")
    args = parser.parse_args()

    things_instance = Things([default_action_function] * POP_0)
    simulation = Simulation(things_instance, load_file = args.load)
    simulation.run()
