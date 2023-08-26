"""Plots archives with pyribs."""
import matplotlib.pyplot as plt
import numpy as np
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

objectives = np.load("objectives.npy")
measures = np.load("measures.npy")
solutions = np.load("solutions.npy")
occupied = np.load("occupied.npy")

archive = GridArchive(solution_dim=solutions.shape[-1],
                      dims=[20, 20],
                      ranges=[(-1, 1), (-1, 1)])
archive.add(
    solutions[occupied],
    objectives[occupied],
    measures[occupied],
)

grid_archive_heatmap(archive)
plt.savefig("archive.png")
