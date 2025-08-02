import numpy as np
import pandas as pd

from functools import partial

from .. import mdevaluate as mde


def radial_distribution_function(
    coords_1: mde.coordinates.CoordinatesMap,
    coords_2: mde.coordinates.CoordinatesMap = None,
    num_segments: int = 1000,
    mode: str = "total",
    column_label: str = "g(r)",
) -> pd.DataFrame:
    """
    Computes a radial distribution function (RDF) for the centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: Resulting RDF.
        float: Value for q constant.
        float: Value for g max.
    """
    if coords_2 is None:
        coords_2 = coords_1

    column_labels = ["r / nm", column_label]

    bins = np.arange(0, 2.2, 0.01)

    rdf = mde.distribution.time_average(
        partial(mde.distribution.rdf, bins=bins, mode=mode),
        coords_1,
        coords_2,
        segments=num_segments,
        skip=0.01,
    )

    df = pd.DataFrame(np.column_stack([bins[:-1], rdf]), columns=column_labels)

    return df


def spatial_density_function(
    coords: mde.coordinates.CoordinatesMap,
    res_atom_pairs: dict[str, list[str]],
    pore_diameter: float,
) -> pd.DataFrame:
    """
    Calculates a radial spatial density function (rSDF) for an atom on a residue.

    Args:
        mde_trajectory: Universe object initialized by mdevaluate.
        residue: Topology name for the residue of interest.
        atom1: Topology name for an atom on the residue.
        diameter: Diameter of the pore.

    Returns:
        NDArray: Resulting rSDF.
    """
    radius_buffer = 0.1
    radius = pore_diameter / 2 + radius_buffer

    bins = np.arange(0.0, radius, 0.025)

    out = np.array(
        [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    )

    column_labels = ["t / ps"]

    for residue, atoms in res_atom_pairs.items():
        for atom in atoms:
            column_labels.append(f"{residue}:{atom}")

            result = mde.distribution.time_average(
                partial(mde.distribution.radial_density, bins=bins),
                coords.subset(atom_name=atom, residue_name=residue),
                segments=1000,
                skip=0.01,
            )
            out = np.column_stack([out, result])

    df = pd.DataFrame(out, columns=column_labels)
    cols_to_drop = (df == 0).all(axis=0)
    df = df.loc[:, ~cols_to_drop]

    return df
