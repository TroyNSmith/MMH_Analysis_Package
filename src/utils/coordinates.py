import numpy as np
import sys

from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import mdevaluate as mde


def centers_of_masses(
    mde_coords: mde.coordinates.CoordinateFrame, res_name: str
) -> mde.coordinates.CoordinatesMap:
    @mde.coordinates.map_coordinates
    def center_of_masses(coordinates, atom_idxs):
        res_ids = coordinates.residue_ids[atom_idxs]
        masses = coordinates.masses[atom_idxs]
        coords = coordinates.whole[atom_idxs]
        positions = np.array(
            [
                np.bincount(res_ids, weights=c * masses)[1:]
                / np.bincount(res_ids, weights=masses)[1:]
                for c in coords.T
            ]
        ).T[np.bincount(res_ids)[1:] != 0]
        return np.array(positions)

    return center_of_masses(
        mde_coords,
        atom_idxs=mde_coords.subset(residue_name=res_name).atom_subset.indices,
    ).nojump


def multi_radial_selector(atoms, bins: np.ndarray) -> list:
    """
    Selects all of the atoms within a radial bin.

    Args:
        atoms: Atoms to select from.
        bins: NumPy array of the radial bins.

    Returns:
        list: A list of the atoms meeting bin criteria.
    """

    indices = []
    for i in range(len(bins) - 1):
        index = mde.coordinates.selector_radial_cylindrical(
            atoms, r_min=bins[i], r_max=bins[i + 1]
        )
        indices.append(index)
    return indices


def vectorize_residue(
    mde_coords: mde.coordinates.CoordinateFrame,
    res_name: str,
    atoms: list[str],
) -> mde.coordinates.CoordinatesMap:
    """
    Process and save mdevaluate indices/vectors.

    Args:
        mde_trajectory: Trajectory object initialized by mdevaluate.
        residue: Name of the residue of interest.
        atom1: First reference atom for center of mass.
        atom2: Second reference atom for center of mass.

    Returns:
        NDArray: NumPy array containing the center of mass coordinates for selected residues.
    """
    atom_1_indices = mde_coords.subset(
        atom_name=atoms[0], residue_name=res_name
    ).atom_subset.indices
    atom_2_indices = mde_coords.subset(
        atom_name=atoms[1], residue_name=res_name
    ).atom_subset.indices
    vectorized_coords = mde.coordinates.vectors(
        mde_coords,
        atom_indices_a=atom_1_indices,
        atom_indices_b=atom_2_indices,
        normed=True,
    )

    return vectorized_coords
