#! /home/noyk/projects/ga/venv/bin/python3

import math

import matplotlib.pyplot as plt
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import PandasLogger, StdOutLogger

plt.style.use('ggplot')

# Declare the objective function


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    A = 10
    (_, n) = x.shape
    return A * n + torch.sum((x**2) - A * torch.cos(2 * math.pi * x), 1)


# Declare the problem
problem = Problem(
    "min",
    rastrigin,
    initial_bounds=(-5.12, 5.12),
    solution_length=100,
    vectorized=True,
    device="cuda:0", # enable this line if you wish to use GPU
    vectorised=True,
)

# Initialize the SNES algorithm to solve the problem
searcher = SNES(problem, popsize=1000, stdev_init=10.0)

# Initialize a standard output logger, and a pandas logger
_ = StdOutLogger(searcher, interval=100)
pandas_logger = PandasLogger(searcher)

# Run SNES for the specified amount of generations
searcher.run(2000)

# Get the progress of the evolution into a DataFrame with the
# help of the PandasLogger, and then plot the progress.
pandas_frame = pandas_logger.to_dataframe()
pandas_frame.plot(subplots=False)
plt.show()
