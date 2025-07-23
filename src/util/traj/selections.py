import mdtraj as md
from numpy.typing import ArrayLike


def select_pairs(Universe: md.Trajectory, Group1: str, Group2: str) -> ArrayLike:
    """
    Select indices for each unique pair of atoms using mdtraj.

    :param Universe: Trajectory/topology as mdtraj object.
    :param Group1: Selection criteria string for first group of atoms.
    :param Group2: Selection criteria string for second group of atoms.
    :return Pairs: ArrayLike object containing the indices of two atoms within each row.
    """
    return Universe.topology.select_pairs(Group1, Group2)