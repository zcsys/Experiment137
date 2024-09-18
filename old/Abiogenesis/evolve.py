from evolve_slice import *
import subprocess

def evolve_epoch(epoch, source = "0", destination = "1", noscore = "1_noscore"):
    load_genomes_with_scores("genomes/" + source)
    mutate_population("genomes/" + noscore)

    processes = []

    for slice in range(1, 11):
        process = subprocess.Popen([
            "python3",
            "evolve_slice.py",
            str(epoch),
            "genomes/" + noscore,
            "genomes/" + destination,
            str(slice)
        ])
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()

def evolve(last_epoch, target_epoch):
    for epoch in range(last_epoch, target_epoch):
        current_epoch = epoch + 1
        print(f"Stepped in the epoch {current_epoch}.")
        evolve_epoch(
            current_epoch,
            f"{epoch}",
            f"{current_epoch}",
            f"{current_epoch}_noscore"
        )


if __name__ == '__main__':
    start_time = time()

    evolve(16, 1000)

    elapsed_time = time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Elapsed Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
