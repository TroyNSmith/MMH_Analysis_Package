import numpy as np
from numpy.typing import NDArray

def inter_combinations(n_atoms: int)->NDArray:
    """
    Return all unique pairs (i < j) for n atoms.

    :param n_atoms: Number of atoms in the Universe.
    :return combinatorials: All unique pairs of atoms.
    """
    return np.array([
        (i, j)
        for i in range(n_atoms)
        for j in range(i + 1, n_atoms)
    ])