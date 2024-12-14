import sys
from particle_life import *
from time import time

GENOMES = []

def load_genomes_with_scores(folder_path = "genomes/0"):
    GENOMES.clear()
    for filename in os.listdir(folder_path):
        if filename[0] != ".":
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                genome = []
                with open(file_path, 'rb') as file:
                    for line in file:
                        genes = [gene.decode() for gene in line.strip().split()]
                        genome.append(genes)
                genome[1][0] = int(genome[1][0])
                GENOMES.append(genome)

def get_scores():
    return sorted([float(x[0][0]) for x in GENOMES])

def mutate_population(path = "genomes/mutated"):
    percentile = 0.1

    scores = get_scores()
    print("The base population's average score:", sum(scores) / len(scores))

    N = int(len(GENOMES) * percentile)
    print(f"The top {N}'s average:", sum(scores[:N]) / N)

    genomes_to_reproduce = sorted(GENOMES, key = lambda x: float(x[0][0]))[:N]

    if not os.path.exists(path):
        os.makedirs(path)

        GENOMES.clear()

        for i, this_genome in enumerate(genomes_to_reproduce):
            GENOMES.append(this_genome)
            dump_genome_to_file(this_genome[1], folder = path)

            for j in range(int(1 / percentile - 1)):
                mutated_genome = mutate(this_genome)
                GENOMES.append(mutated_genome)
                dump_genome_to_file(mutated_genome[1], folder = path)

def ground_zero():
    initiate_world([random_genome(), random_genome()])

    while True:
        STEP[0] += 1

        Rules(0)
        Rules(1)
        Rules(2)
        Rules(3)
        Rules(4)
        Rules(6)

        if EPOCH[0] == 50:
            break

def simulate_from_pool(epoch, source, target, slice):
    EPOCH[0] = epoch
    load_genomes_with_scores(source)

    start = (slice - 1) * 10
    finish = start + 9
    for i in range(start, finish, 2):
        initiate_world([GENOMES[i][1], GENOMES[i + 1][1]])

        while True:
            STEP[0] += 1

            Rules(0)
            Rules(1)
            Rules(2)
            Rules(3)
            Rules(4)

            if STEP[0] == 2400:
                ATOMS[0]["score"] = 0
                ATOMS[1]["score"] = 0

                for i, this_atom in enumerate(ATOMS):
                    if this_atom["color"] != COL_RED:
                        continue

                    ATOMS[0]["score"] += distance (0, i)[0]
                    ATOMS[1]["score"] += distance (1, i)[0]

                Organism_1 = ATOMS[0]["Organism"]
                Organism_2 = ATOMS[1]["Organism"]

                dump_genome_to_file(
                    ATOMS[0]["Organism"].genome,
                    ATOMS[0]["score"],
                    target
                )
                dump_genome_to_file(
                    ATOMS[1]["Organism"].genome,
                    ATOMS[1]["score"],
                    target
                )

                # print("Score 0:", ATOMS[0]["score"])
                # print("Score 1:", ATOMS[1]["score"])

                break


if __name__ == '__main__':
    # ground_zero()

    simulate_from_pool(
        int(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4])
    )
