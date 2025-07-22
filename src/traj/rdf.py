import numpy as np
import mdtraj as md

import util.traj.combinatorials as combinatorials

def inter_rdf(Universe:md.Trajectory, selection:str="all", r_max:float=15.0, n_bins:int=75)->tuple:
    """
    
    """

    indices = Universe.topology.select(selection)
    positions = Universe.atom_slice(indices).xyz
    n_frames, n_atoms = positions.shape[:2]

    volume = np.prod(Universe.unitcell_lengths[0])
    density = n_atoms / volume

    dr = r_max / n_bins

    rdf_hist = np.zeros(n_bins)
    radii = np.linspace(0.0, r_max, n_bins)

    for frame in positions:
        dists = md.compute_distances(md.Trajectory([frame], Universe.atom_slice(indices).topology),
                                    combinations = combinatorials.inter_combinations(n_atoms), periodic=True)

        dists = dists[0]

        hist, _ = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
        rdf_hist += hist

    rdf_hist /= n_frames
    shell_volumes = 4/3 * np.pi * ((radii + dr)**3 - radii**3)
    norm = density * n_atoms * shell_volumes
    g_r = rdf_hist / norm
    return radii, g_r