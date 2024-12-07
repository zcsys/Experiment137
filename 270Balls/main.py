from simulation import Simulation
from things import Things
import argparse

POP_0 = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Eon 1 Era 0: 270Balls"
    )
    parser.add_argument('--load', type = str)
    args = parser.parse_args()

    if args.load:
        things_instance = Things(state_file = args.load)
    else:
        things_instance = Things(["monad" for _ in range(2 * POP_0)])

    simulation = Simulation(things_instance, load_file = args.load)
    simulation.run()
