# Imports #
# ------- #
from functools import partial
from MDAnalysis.analysis.rdf import InterRDF
import mdevaluate as mde
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Any
# ------- #

def radial_density(mde_trajectory: Any, 
                   residue: str, 
                   atom: str, 
                   diameter: float,
                   )-> NDArray:
    '''
    Calculates a radial spatial density function (rSDF) for an atom on a residue.

    Args:
        mde_trajectory: Universe object initialized by mdevaluate.
        residue: Topology name for the  residue of interest.
        atom1: Topology name for an atom on the residue.
        diameter: Diameter of the pore.

    Returns:
        NDArray: Resulting rSDF.
    '''
    bins = np.arange(0.0, diameter/2 + 0.1, 0.025)
    pos = mde.distribution.time_average(partial(mde.distribution.radial_density, bins=bins), mde_trajectory.subset(atom_name=atom, residue_name=residue), segments=1000, skip=0.01)

    return np.column_stack([bins[1:], pos])

def rdf_com(com: NDArray, 
            segments: int = 1000,
            )-> tuple[NDArray, float, float]: 
    '''
    Computes a radial distribution function (RDF) for the centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: Resulting RDF.
        float: Value for q constant.
        float: Value for g max.
    '''
    bins = np.arange(0, 2.2, 0.01)
    result_rdf = mde.distribution.time_average(partial(mde.distribution.rdf, bins=bins), 
                                                com, 
                                                segments=segments, 
                                                skip=0.01)

    return np.column_stack([bins[:-1], result_rdf]), 2*np.pi*(1/result_rdf.max()), result_rdf.max()

def rdf_inter(mdaUniverse: Any, 
              residue: str, 
              atom1: str, 
              atom2: str, 
              start: int = 0, 
              stop: int = 1000
              )-> NDArray:
    '''
    Computes a radial distribution function (RDF) for intermolecular atom-wise distances.

    Args:
        mdaUniverse: Universe object initialized by MDAnalysis.
        residue: Topology name for the  residue of interest.
        atom1: Topology name for an atom on the residue.
        atom2: Topology name for the corresponding atom on the residue.
        start: First frame to analyze.
        stop: Final frame to analyze.

    Returns:
        NDArray: Resulting (averaged) intermolecular RDF and time series.
    '''
    residue_atoms = mdaUniverse.select_atoms(f'resname {residue}')

    rdf_values_list_1 = []  # For RDF between atom1 and non-residue atom1
    rdf_values_list_2 = []  # For RDF between atom1 and non-residue atom2
    rdf_values_list_3 = []  # For RDF between atom2 and non-residue atom1
    rdf_values_list_4 = []  # For RDF between atom2 and non-residue atom2
    
    with tqdm(total=4 * len(set(residue_atoms.resids)), desc="Inter RDF Progress", unit="Resid") as pbar:
        for resid in sorted(set(residue_atoms.resids)):
            # Atom 1 vs non-residue atom1
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom1}')
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_1.append(a1_a2_rdf.results.rdf)
            pbar.update(1)
            
            # Atom 1 vs non-residue atom2
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom2}')
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_2.append(a1_a2_rdf.results.rdf)
            pbar.update(1)
            
            # Atom 2 vs non-residue atom1
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom2}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom1}')
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_3.append(a1_a2_rdf.results.rdf)
            pbar.update(1)
            
            # Atom 2 vs non-residue atom2
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom2}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom2}')
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_4.append(a1_a2_rdf.results.rdf)
            pbar.update(1)

    # Convert lists of RDF values into NumPy arrays
    rdf_array_1 = np.array(rdf_values_list_1)
    rdf_array_2 = np.array(rdf_values_list_2)
    rdf_array_3 = np.array(rdf_values_list_3)
    rdf_array_4 = np.array(rdf_values_list_4)

    # Compute the mean for each group (atom pair) separately
    mean_rdf_1 = np.mean(rdf_array_1, axis=0)
    mean_rdf_2 = np.mean(rdf_array_2, axis=0)
    mean_rdf_3 = np.mean(rdf_array_3, axis=0)
    mean_rdf_4 = np.mean(rdf_array_4, axis=0)

    # Get the distance bins (from the last RDF calculation)
    distance_bins = a1_a2_rdf.results.bins / 10

    # Stack the results together
    result = np.column_stack([distance_bins, mean_rdf_1, mean_rdf_2, mean_rdf_3, mean_rdf_4])

    return result

def rdf_intra(mdaUniverse: Any, 
              residue: str, 
              atom1: str, 
              atom2: str, 
              start: int = 0, 
              stop: int = 1000
              )-> NDArray:
    '''
    Computes a radial distribution function (RDF) for intramolecular atom-wise distances.

    Args:
        mdaUniverse: Universe object initialized by MDAnalysis.
        residue: Topology name for the  residue of interest.
        atom1: Topology name for an atom on the residue.
        atom2: Topology name for the corresponding atom on the residue.
        start: First frame to analyze.
        stop: Final frame to analyze.

    Returns:
        NDArray: Resulting (averaged) intramolecular RDF and time series.
    '''
    residue_atoms = mdaUniverse.select_atoms(f'resname {residue}')
    rdf_values_list = []
    
    with tqdm(total=len(set(residue_atoms.resids)), desc="Intra RDF Progress", unit="Resid") as pbar:
        for resid in sorted(set(residue_atoms.resids)):
            a1_a2_rdf = InterRDF(g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'), 
                                g2=mdaUniverse.select_atoms(f'resid {resid} and name {atom2}'), )
            a1_a2_rdf.run(start=start, stop=stop)

            rdf_values_list.append(a1_a2_rdf.results.rdf)
            pbar.update(1)

    rdf_array = np.array(rdf_values_list)

    return np.column_stack([a1_a2_rdf.results.bins / 10, np.mean(rdf_array, axis=0) / np.max(np.mean(rdf_array, axis=0))])

def z_align(vectors: NDArray, 
            segments: int
            )-> NDArray:
    """
    Calculates Z-axis alignment based on centers of masses.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: A NumPy array containing the Z-axis alignment orientations.
    """
    def z_angle_orientation(vectors, segments, skip=0.1):
        z_vector=[0,0,1]
        bins = np.linspace(0,180,361)
        x = bins[1:] - (bins[1]-bins[0])/2

        def angles(start, end, z_vector, bins):
            angle = np.arccos((start * z_vector).sum(axis=-1))
            angle = angle[(angle>=0)*(angle<=np.pi)]
            hist, _ = np.histogram(angle *360/(2*np.pi), bins)
            return 1 / len(start) * hist
        
        t, S = mde.correlation.shifted_correlation(partial(angles, z_vector=z_vector, bins=bins), vectors, segments=segments, skip=skip)  
        time = np.array([t_i for t_i in t for entry in x])
        angle = np.array([entry for t_i in t for entry in x])
        result = S.flatten()

        return np.column_stack([time, angle, result])
    return z_angle_orientation(vectors, segments)

def z_histogram(vectors: NDArray, 
                segments: int, 
                skip: float = 0.1
                )-> NDArray:
    """
    Calculates Z-axis radial positions based on centers of masses.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        segments: Number of segments to divide the trajectory into.
        skip: ***

    Returns:
        NDArray: A NumPy array containing the Z-axis alignment orientations.
    """
    bins = np.linspace(-1,1,201)
    x = bins[1:] - (bins[1]-bins[0])/2

    def z_comp(start, end, bins):
        norm_vectors = np.linalg.norm(start, axis=1)
        z_comp = start[:,2]/norm_vectors
        hist, _ = np.histogram(z_comp, bins)
        return 1 / len(start) * hist

    t, S = mde.correlation.shifted_correlation(partial(z_comp, bins=bins), vectors, segments=segments, skip=skip)  
    time = np.array([t_i for t_i in t for entry in x])
    angle = np.array([entry for t_i in t for entry in x])
    result = S.flatten()

    return np.column_stack([time, angle, result])