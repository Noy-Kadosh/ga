#! /home/noyk/projects/ga/venv/bin/python3
'''Image reconstracting using GA'''
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor


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
    canvas[circle_mask] = color

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


def random_individual(canvas) -> dict:
    x = np.concatenate((
        np.random.randint(0, 499, size=(4 * 3)),
        np.random.randint(0, 499, size=(4 * 3)),
    ))

    individual = get_individual(x)
    draw_individual(individual, canvas)


def main() -> None:
    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    RES = differential_evolution(
        min_func,
        bounds=list(repeat((0, 499), 12)) + list(repeat((0, 255), 12)),
        args=(TARGET,),
        disp=True,
        popsize=150,
        mutation=(0.001, 0.01),
        recombination=0.4,
        maxiter=100,
        # workers=3,
    )
    print(RES)
    X = RES.x
    print(min_func(X, TARGET))

    CANVAS = np.zeros_like(TARGET) + 255
    CANVAS = draw_individual(get_individual(X), CANVAS)

    plt.imshow(CANVAS)
    plt.title(f'{RES.fun}')
    plt.axis('off')
    plt.savefig('/mnt/c/Users/noyoy/Downloads/out/a.png')


def test():

    from time import time

    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    CANVAS = np.zeros_like(TARGET) + 255
    n = 100
    stats = np.empty(n)

    for i in range(n):
        tic = time()
        random_individual(CANVAS)
        stats[i] = time() - tic

    M = stats.mean()
    STD = stats.std()
    print(f'{n} loops: mean {M:.3} std {STD:.3} total {stats.sum():.3f}')


if __name__ == "__main__":
    pass
