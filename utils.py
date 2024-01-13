#! /home/noyk/projects/ga/venv/bin/python3
'''Utilites'''

import matplotlib.pylab as plt
import numpy as np
from numba import jit
from PIL import Image


def read_image(path: str, size: tuple) -> np.array:
    return np.array(
        Image.open(path)
        .resize(size=size),
        dtype=int,
    )


def get_individual(x: np.array) -> dict:
    '''
    Deprecated!
    c - center (x, y) (2,)
    r - radius (r) (1,)
    color - r, g, b (3,)
    alpha - (1,)
    return: [dict.from_keys({'c', 'r', 'color', 'alpha'})]
    '''
    patches = []
    for patch in x.reshape(-1, 7):
        patches.append(
            {
                'c': patch[:2],
                'r': patch[2],
                'color': patch[3:6],
                'alpha': patch[6],
            }
        )
    return patches


@jit(cache=True, nopython=True)
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for i in range(x.size):
            xx[i, j] = i
            yy[i, j] = j
    return xx, yy


@jit(cache=True, nopython=True)
def create_circle(center, radius, color, alpha, canvas):
    alpha /= 255
    canvas_shape = canvas.shape
    x, y = meshgrid(np.arange(canvas_shape[1]), np.arange(canvas_shape[0]))
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2

    # color = np.array(color)
    for i in range(circle_mask.shape[0]):
        for j in range(circle_mask.shape[1]):
            if circle_mask[i, j] is False:
                continue
            canvas[i, j] = alpha * canvas[i, j] + \
                (1 - alpha) * np.clip(color, 0, 255)

    return canvas


@jit(nopython=True)
def draw_individual(individual, canvas):
    for (x, y, r, R, G, B, a) in individual.reshape(-1, 7):
        canvas = create_circle(center=(x, y), radius=r,
                               color=np.array([R, G, B]), alpha=a, canvas=canvas)
    return canvas


@jit(nopython=True)
def score_batch(individuals, target):
    n = len(individuals)
    scores = np.empty(n)
    for i in range(n):
        scores[i] = min_func(individuals[i], target)

    return scores


@jit(nopython=True)
def min_func(x: np.array, target: np.array):
    canvas = np.zeros_like(target) + 255
    canvas = draw_individual(x, canvas)

    return np.abs(canvas - target).sum()


def random_individual(canvas) -> dict:
    i = 20  # num of patches
    x = np.random.randint(0, 256, size=7 * i)
    draw_individual(x, canvas)


def save_solution_as_image(sol_canvas: np.array, gen_num: int, title: str) -> None:
    plt.imshow(sol_canvas)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'/mnt/c/Users/noyoy/Downloads/out/gagad_gen{gen_num}.png')


def test():

    from time import time

    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    CANVAS = np.zeros_like(TARGET) + 255
    n = 10
    stats = np.empty(n)

    for i in range(n):
        canvas = CANVAS.copy()
        tic = time()
        random_individual(canvas)
        stats[i] = time() - tic

    M = stats.mean()
    STD = stats.std()
    print(f'{n} loops: mean {M:.3} std {STD:.3} total {stats.sum():.3f}')

    save_solution_as_image(canvas, gen_num=-1, title='test')


def test_batch():

    from time import time

    SIZE = (500, 500)
    TARGET = read_image(path="color_box.png", size=SIZE)
    CANVAS = np.zeros_like(TARGET) + 255
    n = 10
    stats = np.empty(n)

    for i in range(n):
        tic = time()
        x = np.random.randint(0, 256, size=(4, 7 * 20))
        score_batch(x, TARGET)
        stats[i] = time() - tic

    M = stats.mean()
    STD = stats.std()
    print(f'{n} loops: mean {M:.3} std {STD:.3} total {stats.sum():.3f}')

    canvas = draw_individual(x[0], CANVAS)
    save_solution_as_image(canvas, gen_num=-1, title='test')


if __name__ == "__main__":
    test_batch()
