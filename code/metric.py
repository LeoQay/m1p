import numpy as np


def special_func(G, coefs, pred):
    return np.dot(((G - pred[:, None]) ** 2), coefs).mean()
