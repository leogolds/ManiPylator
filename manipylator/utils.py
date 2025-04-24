import numpy as np


def parametric_heart_1(t):
    x = np.sqrt(2) * np.sin(t) ** 3
    y = -np.cos(t) ** 3 - np.cos(t) ** 2 + 2 * np.cos(t)

    return np.stack((x, y, np.zeros(t.shape)), axis=1)


def parametric_heart_2(t):
    x = (16 * np.sin(t) ** 3) / 11
    y = (
        13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - 1 * np.cos(4 * t) - 5
    ) / 11

    return np.stack((x, y, np.zeros(t.shape)), axis=1)
