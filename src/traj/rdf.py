import mdtraj as md
from numpy.typing import ArrayLike, NDArray


def rdf(Universe: md.Trajectory, Pairs: ArrayLike) -> tuple[NDArray, NDArray]:
    """
    Compute RDF for pairs in every frame of trajectory using mdtraj.
    """
    return md.compute_rdf(Universe, Pairs)
