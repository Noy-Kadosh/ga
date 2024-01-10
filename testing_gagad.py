
#! /home/noyk/projects/ga/venv/bin/python3
'''Image reconstracting using GA'''
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
import pygad
from clearml import Task
from PIL import Image

task = Task.init(task_name='PyGAD', project_name='Simple Image Reconstraction', reuse_last_task_id=False)
logger = task.get_logger()


SIZE = (500, 500)
GEN_NUM = 0
TARGET = np.array(
    Image.open("draw1.png")
    .resize(size=SIZE),
    dtype=int,
)
logger.report_image(title="Original Image",
                        series="Source", image=TARGET)



def read_image(path: str, size: tuple) -> np.array:
    return np.array(
        Image.open(path)
        .resize(size=size),
        dtype=int,
    )


def get_individual(x: np.array) -> dict:
    return {
        'c0': x[:2],  # 0-1
        'c1': x[2:4],  # 2-3
        'c2': x[4:6],  # 4-5
        'c3': x[6:8],  # 6-7
        'r0': x[8],
        'r1': x[9],
        'r2': x[10],
        'r3': x[11],
        'color0': x[12:15],  # 12-14
        'color1': x[15:18],
        'color2': x[18:21],
        'color3': x[21:24],
    }


def create_circle(center, radius, color, canvas):
    canvas_shape = canvas.shape
    x, y = np.meshgrid(np.arange(canvas_shape[1]), np.arange(canvas_shape[0]))
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2
    canvas[circle_mask] = np.clip(color, 0, 255)

    return canvas


def draw_individual(individual, canvas):
    for i in range(len(individual) // 3):
        center = individual[f'c{i}']
        radius = individual[f'r{i}']
        color = individual[f'color{i}']
        canvas = create_circle(center, radius, color, canvas)
    return canvas


def score(individual: dict, target: np.array) -> float:

    canvas = np.zeros_like(target) + 255
    canvas = draw_individual(individual, canvas)

    if canvas.shape != target.shape:
        raise ValueError('shape do not match!')

    return np.abs(canvas - target).sum()


def min_func(x: np.array, target: np.array) -> float:
    canvas = np.zeros_like(target) + 255
    canvas = draw_individual(get_individual(x), canvas)

    if canvas.shape != target.shape:
        raise ValueError('shape do not match!')

    return np.abs(canvas - target).sum()


def random_individual() -> None:
    x = np.concatenate((
        np.random.randint(0, 499, size=(4 * 3)),
        np.random.randint(0, 499, size=(4 * 3)),
    ))

    # individual = get_individual(x)
    # draw_individual(individual, canvas)
    print('test', min_func(x, TARGET))


def fitness_func(ga_instance, solution, solution_idx):
    fitness = 1.0 / min_func(solution, TARGET)
    return fitness


def on_generation(ga_instance):
    global GEN_NUM
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        ga_instance.last_generation_fitness)
    # print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    # print(f"Index of the best solution : {solution_idx}")
    canvas = np.zeros_like(TARGET) + 255
    canvas = draw_individual(get_individual(solution), canvas)
    # save_solution_as_image(sol_canvas=canvas, gen_num=GEN_NUM,
    #                        title=f'MAE: {int(1 / solution_fitness)} as gen. {GEN_NUM}')
    GEN_NUM += 1
    logger.report_scalar(title="Best Fitness", series="Fitness over Gens.",
                         value=solution_fitness, iteration=GEN_NUM)
    logger.report_image(title=f'MAE: {int(1 / solution_fitness)} at gen. {GEN_NUM}',
                        series="Samples", iteration=GEN_NUM, image=canvas)


def save_solution_as_image(sol_canvas: np.array, gen_num: int, title: str) -> None:
    plt.imshow(sol_canvas)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/mnt/c/Users/noyoy/Downloads/out/gagad_gen{gen_num}.png')


def main() -> None:
    ga_instance = pygad.GA(
        num_generations=10_000,
        num_parents_mating=50,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=24,
        gene_type=int,
        init_range_low=0,
        init_range_high=499,
        parent_selection_type='sss',
        keep_elitism=10,
        crossover_type='uniform',
        mutation_type='random',
        mutation_num_genes=1,
        parallel_processing=3,
        gene_space=list(repeat(range(500), times=12)) +
        list(repeat(range(256), times=12)),
        on_generation=on_generation,
        stop_criteria='saturate_10',
        random_seed=np.random.randint(0, 100000),
    )

    task.connect_configuration(vars(ga_instance), name="GA GAD configurations")
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    canvas = np.zeros_like(TARGET) + 255
    canvas = draw_individual(get_individual(solution), canvas)

    plt.imshow(canvas)
    plt.title(f'{min_func(solution, TARGET)}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/noyoy/Downloads/out/final_sol.png')


if __name__ == "__main__":
    main()
