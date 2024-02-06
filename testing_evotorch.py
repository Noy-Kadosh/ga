#! /home/noyk/projects/ga/venv/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import torch
from evotorch import Problem, Solution
from evotorch.algorithms import SNES
from evotorch.logging import PandasLogger, StdOutLogger
from PIL import Image

from utils import draw_individual, get_individual

plt.style.use('ggplot')
SIZE = (255, 255)
# GEN_NUM = 0
TARGET = torch.Tensor(np.array(
    Image.open("color_box.png")
    .resize(size=SIZE),
    dtype=float,
))

# Declare a custom Problemd


class MyProblem(Problem):

    def __init__(
        self,
        target: torch.Tensor,
        device: str,
        **kwargs,
    ):
        self.target = torch.tensor(target, device=device)

        super().__init__(device=device, **kwargs)

    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        # Declare the objective function
        canvas = torch.zeros_like(self.target, device=x.device) + 255
        canvas = draw_individual(get_individual(x), canvas)
        return torch.abs(canvas - self.target).sum()

    def _evaluate(self, solution: Solution):
        # This is where we declare the procedure of evaluating a solution

        # Get the decision values of the solution as a PyTorch tensor
        x = solution.values

        # Compute the fitness
        # fitness = torch.sum(x ** (2 * self.q))
        fitness = self.fitness(x)

        # Register the fitness into the Solution object
        solution.set_evaluation(fitness)


# Declare the problem
problem = MyProblem(
    target=TARGET,
    objective_sense='min',
    initial_bounds=(0, 255),
    solution_length=28,
    device='cuda:0',
    # num_actors=2,
)

# Initialize the SNES algorithm to solve the problem
searcher = SNES(problem, popsize=1000, stdev_init=2.5)

# Initialize a standard output logger, and a pandas logger
_ = StdOutLogger(searcher, interval=5)
pandas_logger = PandasLogger(searcher)

# Run SNES for the specified amount of generations
searcher.run(250)
# best solution
best_individual = searcher._population.take_best(1)
best_score = best_individual._evdata.detach().cpu().item()
best_solution = best_individual._data.detach().cpu().view(-1, )
ind_dict = get_individual(best_solution)

CANVAS = torch.zeros_like(problem.target.detach().cpu()) + 255
CANVAS = draw_individual(ind_dict, CANVAS).numpy()

print(ind_dict)

plt.imshow(CANVAS.astype(np.uint8))
plt.title(f'{best_score:.3f}')
plt.axis('off')
plt.savefig('./best_evotorch.png')


# Get the progress of the evolution into a DataFrame with the
# help of the PandasLogger, and then plot the progress.
pandas_frame = pandas_logger.to_dataframe()
pandas_frame.plot(subplots=False)
plt.show()
