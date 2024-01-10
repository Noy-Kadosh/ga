#! /home/noyk/projects/ga/venv/bin/python3
'''Image reconstracting using GA'''
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import differential_evolution

from utils import draw_individual, get_individual, min_func, read_image


def random_individual(canvas) -> dict:
    # TODO: move to test.py
    x = np.concatenate((
        np.random.randint(0, 499, size=4 * 3),
        np.random.randint(0, 256, size=4 * 4),
    ))

    individual = get_individual(x)
    draw_individual(individual, canvas)


def main() -> None:
    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    RES = differential_evolution(
        min_func,
        bounds=list(repeat((0, 499), 12)) + list(repeat((0, 255), 16)),
        args=(TARGET,),
        disp=True,
        popsize=15,
        mutation=(0.001, 0.01),
        recombination=0.4,
        maxiter=10,
        workers=3,
    )
    print(RES)
    X = RES.x
    print(min_func(X, TARGET))

    CANVAS = np.zeros_like(TARGET) + 255
    CANVAS = draw_individual(get_individual(X), CANVAS)

    plt.imshow(CANVAS.astype(np.uint8))
    plt.title(f'{RES.fun}')
    plt.axis('off')
    plt.savefig('/mnt/c/Users/noyoy/Downloads/out/a.png')


def test():

    from time import time

    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    CANVAS = np.zeros_like(TARGET) + 255
    n = 10
    stats = np.empty(n)

    for i in range(n):
        tic = time()
        random_individual(CANVAS)
        stats[i] = time() - tic

    M = stats.mean()
    STD = stats.std()
    print(f'{n} loops: mean {M:.3} std {STD:.3} total {stats.sum():.3f}')


if __name__ == "__main__":
    main()
