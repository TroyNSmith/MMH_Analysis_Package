import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray


class q_Analysis:
    """
    Class for calculating q value from RDF and performing related analysis.
    """

    @staticmethod
    def rdf(Universe: md.Trajectory, Pairs: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Compute RDF for pairs in every frame of trajectory using mdtraj.
        """
        return md.compute_rdf(Universe, Pairs)

    def scattering_vector(R: NDArray, G_R: NDArray) -> float:
        """
        Identify the scattering vector (q) for a radial distribution.

        :param R: Array of radial bins.
        :param G_R: Radial distribution results associated with R.
        :return q: Scattering vector corresponding to absolute maximum in radial distribution.
        """
        max_G_R_index = np.argmax(G_R)

        return R[max_G_R_index]

    def incoherent_scattering(start_indices: np.ndarray, frame_indices: np.ndarray, q: float) -> np.ndarray:
        """
        
        """
        vec = start_indices - frame_indices
        distance = np.linalg.norm(vec, axis=1)
        print(distance)