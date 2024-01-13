#! /home/noyk/projects/ga/venv/bin/python3
'''Image reconstracting using GA'''
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
import pygad
from clearml import Task
from PIL import Image

from utils import draw_individual, score_batch, min_func

task = Task.init(task_name='PyGAD',
                 project_name='Simple Image Reconstraction', reuse_last_task_id=False)
logger = task.get_logger()


SIZE = (500, 500)
GEN_NUM = 0
TARGET = np.array(
    Image.open("./draw1.png")
    .resize(size=SIZE),
    dtype=int,
)
logger.report_image(title="Original Image",
                    series="Source", image=TARGET)


def fitness_func(ga_instance, solution, solution_idx):
    fitness = 1.0 / score_batch(solution, TARGET)
    return fitness


def on_generation(ga_instance):
    global GEN_NUM
    solution, solution_fitness, _ = ga_instance.best_solution(
        ga_instance.last_generation_fitness)
    # print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    # print(f"Index of the best solution : {solution_idx}")
    canvas = np.zeros_like(TARGET) + 255
    canvas = draw_individual(solution, canvas)
    # save_solution_as_image(sol_canvas=canvas, gen_num=GEN_NUM,
    #                        title=f'MAE: {int(1 / solution_fitness)} as gen. {GEN_NUM}')
    GEN_NUM += 1
    logger.report_scalar(title="Best Fitness", series="Fitness over Gens.",
                         value=solution_fitness, iteration=GEN_NUM)
    logger.report_image(title=f'MAE: {int(1 / solution_fitness)} at gen. {GEN_NUM}',
                        series="Samples", iteration=GEN_NUM, image=canvas)


def main() -> None:
    n_patches = 8
    ga_instance = pygad.GA(
        num_generations=10_000,
        num_parents_mating=50,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=7 * n_patches,
        gene_type=int,
        init_range_low=0,
        init_range_high=499,
        parent_selection_type='sss',
        keep_elitism=5,
        crossover_type='single_point',
        mutation_type='random',
        mutation_num_genes=2,
        mutation_by_replacement=True,
        fitness_batch_size=10,
        # parallel_processing=3,
        gene_space=[*repeat({'low': 0, 'high': 500, 'step': 1}, times=2),
                    *repeat({'low': 0, 'high': 256, 'step': 1}, times=5)] * n_patches,
        on_generation=on_generation,
        stop_criteria='saturate_10',
        # random_seed=np.random.randint(0, 100000),
    )

    logger.report_text(ga_instance.summary())
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    canvas = np.zeros_like(TARGET) + 255
    canvas = draw_individual(solution, canvas)

    # plt.imshow(canvas)
    # plt.title(f'{min_func(solution[0], TARGET)}')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('/mnt/c/Users/noyoy/Downloads/out/final_sol.png')


if __name__ == "__main__":
    main()
