import pygame
import pickle
import numpy as np
from sklearn.cluster import KMeans
from monadology import Thing, Monad016

HEATMAP_WIDTH = 1920
HEATMAP_HEIGHT = 1080
CELL_SIZE = 20
MARGIN = 50
CLUSTER_GAP = 30
SCROLL_SPEED = 20

THINGS = []

def collect_genomes(filename):
    global THINGS
    with open(filename, 'rb') as f:
        game_state = pickle.load(f)
    THINGS = game_state["THINGS"]
    return [thing.genome[:76] for thing in THINGS
        if thing.color != (170, 170, 0)]

def genome_statistics(genomes):
    genomes = np.array(genomes)
    return {
        "mean": np.mean(genomes, axis = 0),
        "min": np.min(genomes, axis = 0),
        "max": np.max(genomes, axis = 0),
        "std": np.std(genomes, axis = 0)
    }

def genome_correlation(genomes):
    genome_array = np.array(genomes)
    return np.corrcoef(genome_array, rowvar = False)

def genome_clustering(genomes, how_many = 4):
    genome_array = np.array(genomes)
    kmeans = KMeans(n_clusters=how_many, random_state = 0).fit(genome_array)
    cluster_labels = kmeans.labels_

    cluster_averages = []
    for cluster in range(how_many):
        cluster_genomes = genome_array[cluster_labels == cluster]
        cluster_avg = np.mean(cluster_genomes, axis = 0)
        cluster_averages.append(cluster_avg)

    return cluster_labels, cluster_averages

def create_heatmap_window():
    heatmap_window = pygame.display.set_mode((HEATMAP_WIDTH, HEATMAP_HEIGHT))
    pygame.display.set_caption("Genome Heatmap")
    return heatmap_window

def map_value_to_color(value, min_value, max_value):
    if max_value == min_value:
        ratio = 0
    else:
        ratio = (value - min_value) / (max_value - min_value)
    return (50, int(255 * ratio), 50)

def draw_heatmap(heatmap_window, genomes, cluster_labels, cluster_averages,
    scroll_offset):
    heatmap_window.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 24)

    if len(genomes) == 0:
        return

    genome_array = np.array(genomes)
    num_genomes, num_genes = genome_array.shape

    min_col0 = np.min(genome_array[:, 0])
    max_col0 = np.max(genome_array[:, 0])
    min_other_cols = np.min(genome_array[:, 1:])
    max_other_cols = np.max(genome_array[:, 1:])

    sorted_indices = np.argsort(cluster_labels)
    sorted_genomes = genome_array[sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]

    y_offset = MARGIN - scroll_offset
    current_cluster = sorted_labels[0]

    for genome_idx, (genome, label) in enumerate(zip(sorted_genomes,
        sorted_labels)):
        if label != current_cluster:
            y_offset += CLUSTER_GAP
            current_cluster = label

        if genome_idx == 0 or sorted_labels[genome_idx - 1] != label:
            avg_formatted = ', '.join([f'{avg:.2f}'
                for avg in cluster_averages[label]])
            average_text = f"Cluster {label} Averages: [{avg_formatted}]"
            avg_surface = font.render(average_text, True, (0, 0, 0))
            heatmap_window.blit(avg_surface, (MARGIN, y_offset))
            y_offset += CELL_SIZE * 2

        row_number_surface = font.render(str(sorted_indices[genome_idx]), True,
            (0, 0, 0))
        heatmap_window.blit(row_number_surface, (MARGIN - 30,
            y_offset + CELL_SIZE // 4))

        for gene_idx, gene_value in enumerate(genome):
            if gene_idx == 0:
                color = map_value_to_color(
                    gene_value,
                    min_col0,
                    max_col0
                )
            else:
                color = map_value_to_color(
                    gene_value,
                    min_other_cols,
                    max_other_cols
                )

            x = MARGIN + gene_idx * CELL_SIZE
            y = y_offset

            pygame.draw.rect(heatmap_window, color, (x, y, CELL_SIZE,
                CELL_SIZE))
            pygame.draw.rect(heatmap_window, (255, 255, 255), (x, y, CELL_SIZE,
                CELL_SIZE), 1)

        y_offset += CELL_SIZE

    pygame.display.update()

def run_heatmap_visualization(genomes):
    pygame.init()
    heatmap_window = create_heatmap_window()

    cluster_labels, cluster_averages = genome_clustering(genomes, how_many = 4)

    scroll_offset = 0
    running = True

    while running:
        draw_heatmap(
            heatmap_window,
            genomes,
            cluster_labels,
            cluster_averages,
            scroll_offset)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    scroll_offset += SCROLL_SPEED
                elif event.key == pygame.K_UP:
                    scroll_offset -= SCROLL_SPEED
                scroll_offset = max(0, scroll_offset)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    genomes = collect_genomes("SavedState20240912_165126.pkl")
    run_heatmap_visualization(genomes)
