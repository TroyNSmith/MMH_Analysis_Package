# ----- Last updated: 07/02/2025 -----
# ----- By Troy N. Smith :-) -----

# Imports #
# ------- #
import helpers
from itertools import chain
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any
# ------- #

def hbonds(mdaUniverse: Any, start: int = 0, stop: int = 5000, residue: str = None, pore: bool = False)-> NDArray:
    """
    Gathers hydrogen bond data for a given residue as both an acceptor and a donor through MDAnalysis.

    Args:
        mdaUniverse : Universe object initialized by MDAnalysis.
        start       : Starting frame of analysis.
        stop        : Ending frame of analysis.

    Returns:
        NDArray     : NumPy array containing hydrogen bond information.
    """
    hbonds_data = hba(universe=mdaUniverse,
                      donors_sel='(type O or type N) and not resname PORE',
                      hydrogens_sel='type H and not resname PORE',
                      acceptors_sel='(type O or type N) and not resname PORE',
                      d_a_cutoff=3.5,               # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
                      d_h_a_angle_cutoff=150,       # Same as GMX default parameters
                      update_selections=False       # No dynamic selections
                      ).run(start=start, stop=stop, verbose=True)
    
    hbonds_not_pore = np.column_stack([
        hbonds_data.results.hbonds[:, 0].astype(int),                                                       # 0: Frame
        hbonds_data.results.hbonds[:, 1].astype(int),                                                       # 1: Donor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids,                             # 2: Donor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resnames,                           # 3: Donor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,                              # 4: Donor atom name
        hbonds_data.results.hbonds[:, 3].astype(int),                                                       # 5: Acceptor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids,                             # 6: Acceptor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resnames,                           # 7: Acceptor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,                              # 8: Acceptor atom name
        hbonds_data.results.hbonds[:, 4],                                                                   # 9: Hydrogen bond distance
        hbonds_data.results.hbonds[:, 5]                                                                    # 10: Hydrogen bond angle
    ])

    ids = hbonds_data.count_by_ids()

    not_pore_by_ids = np.column_stack([
        ids[:,0],                                                                                           # 0: Donor index
        mdaUniverse.atoms[ids[:,0]-1].resids,                                                               # 1: Donor resid
        mdaUniverse.atoms[ids[:,0]-1].resnames,                                                             # 2: Donor resname
        mdaUniverse.atoms[ids[:,0]-1].names,                                                                # 3: Donor atom name
        ids[:,1],                                                                                           # 4: Hydrogen index
        ids[:,2],                                                                                           # 5: Acceptor index
        mdaUniverse.atoms[ids[:,2]-1].resids,                                                               # 6: Acceptor resid
        mdaUniverse.atoms[ids[:,2]-1].resnames,                                                             # 7: Acceptor resname
        mdaUniverse.atoms[ids[:,2]-1].names,                                                                # 8: Acceptor atom name
        ids[:,3]                                                                                            # 9: Number of occcurrences
    ])

    if pore == True:
        hbonds_data = hba(universe=mdaUniverse,
                        donors_sel=f'(type O or type N) and resname {residue}',
                        hydrogens_sel=f'type H and resname {residue}',
                        acceptors_sel='type O and resname PORE',
                        d_a_cutoff=3.5,               # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
                        d_h_a_angle_cutoff=150,       # Same as GMX default parameters
                        update_selections=False       # No dynamic selections
                        ).run(start=start, stop=stop, verbose=True)
        
        hbonds_donor = np.column_stack([
            hbonds_data.results.hbonds[:, 0].astype(int),                                                       # 0: Frame
            hbonds_data.results.hbonds[:, 1].astype(int),                                                       # 1: Donor index
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids,                             # 2: Donor resid
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resnames,                           # 3: Donor resname
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,                              # 4: Donor atom name
            hbonds_data.results.hbonds[:, 3].astype(int),                                                       # 5: Acceptor index
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids,                             # 6: Acceptor resid
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resnames,                           # 7: Acceptor resname
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,                              # 8: Acceptor atom name
            hbonds_data.results.hbonds[:, 4],                                                                   # 9: Hydrogen bond distance
            hbonds_data.results.hbonds[:, 5]                                                                    # 10: Hydrogen bond angle
        ])

        ids = hbonds_data.count_by_ids()

        donor_by_ids = np.column_stack([
            ids[:,0],                                                                                           # 0: Donor index
            mdaUniverse.atoms[ids[:,0]-1].resids,                                                               # 1: Donor resid
            mdaUniverse.atoms[ids[:,0]-1].resnames,                                                             # 2: Donor resname
            mdaUniverse.atoms[ids[:,0]-1].names,                                                                # 3: Donor atom name
            ids[:,1],                                                                                           # 4: Hydrogen index
            ids[:,2],                                                                                           # 5: Acceptor index
            mdaUniverse.atoms[ids[:,2]-1].resids,                                                               # 6: Acceptor resid
            mdaUniverse.atoms[ids[:,2]-1].resnames,                                                             # 7: Acceptor resname
            mdaUniverse.atoms[ids[:,2]-1].names,                                                                # 8: Acceptor atom name
            ids[:,3]                                                                                            # 9: Number of occcurrences
        ])

        hbonds_data = hba(universe=mdaUniverse,
                        donors_sel='type O and resname PORE',
                        hydrogens_sel='type H and resname PORE',
                        acceptors_sel=f'(type O or type N) and resname {residue}',
                        d_a_cutoff=3.5,               # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
                        d_h_a_angle_cutoff=150,       # Same as GMX default parameters
                        update_selections=False       # No dynamic selections
                        ).run(start=start, stop=stop, verbose=True)
        
        hbonds_acceptor = np.column_stack([
            hbonds_data.results.hbonds[:, 0].astype(int),                                                       # 0: Frame
            hbonds_data.results.hbonds[:, 1].astype(int),                                                       # 1: Donor index
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids,                             # 2: Donor resid
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resnames,                           # 3: Donor resname
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,                              # 4: Donor atom name
            hbonds_data.results.hbonds[:, 3].astype(int),                                                       # 5: Acceptor index
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids,                             # 6: Acceptor resid
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resnames,                           # 7: Acceptor resname
            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,                              # 8: Acceptor atom name
            hbonds_data.results.hbonds[:, 4],                                                                   # 9: Hydrogen bond distance
            hbonds_data.results.hbonds[:, 5]                                                                    # 10: Hydrogen bond angle
        ])

        ids = hbonds_data.count_by_ids()

        acceptor_by_ids = np.column_stack([
            ids[:,0],                                                                                           # 0: Donor index
            mdaUniverse.atoms[ids[:,0]-1].resids,                                                               # 1: Donor resid
            mdaUniverse.atoms[ids[:,0]-1].resnames,                                                             # 2: Donor resname
            mdaUniverse.atoms[ids[:,0]-1].names,                                                                # 3: Donor atom name
            ids[:,1],                                                                                           # 4: Hydrogen index
            ids[:,2],                                                                                           # 5: Acceptor index
            mdaUniverse.atoms[ids[:,2]-1].resids,                                                               # 6: Acceptor resid
            mdaUniverse.atoms[ids[:,2]-1].resnames,                                                             # 7: Acceptor resname
            mdaUniverse.atoms[ids[:,2]-1].names,                                                                # 8: Acceptor atom name
            ids[:,3]                                                                                            # 9: Number of occcurrences
        ])

        hbonds_pore_all = np.vstack([hbonds_donor, hbonds_acceptor, hbonds_not_pore])
        ids_pore_all = np.vstack([donor_by_ids, acceptor_by_ids, not_pore_by_ids])

        return  hbonds_pore_all[hbonds_pore_all[:,0].argsort()], ids_pore_all                                   # Unique ID pairs and unique typing pairs
    
    else:
        return  hbonds_not_pore, not_pore_by_ids

def hbond_counts(hbonds: NDArray, residue: str)-> dict:
    """
    Counts the inter- and intramolecular hydrogen bonds present in a system, as well as categorizes them by the donor-acceptor pair (e.g., O01 to O0E).

    Args:
        hbonds       : NumPy array containing the hydrogen bonds.
        reside       : The residue of interest.

    Returns:
        dict         : Dictionary containing the counts of hydrogen bond types and pairs.
    """
    counts = {'Intra H-bonds': 0,
              'Inter H-bonds': 0,
              'Total H-bonds': hbonds.shape[0],
              'Average H-bonds per frame': round(hbonds.shape[0] / np.max(hbonds[:,0]), 2)}

    for row in range(len(hbonds)):
        if hbonds[row,2] == hbonds[row,6] == residue:
            counts['Intra H-bonds'] += 1
        else:
            counts['Inter H-bonds'] += 1

        if f'{hbonds[row,3]}_{hbonds[row,4]} to {hbonds[row,7]}_{hbonds[row,8]} H-bonds' in counts:
            counts[f'{hbonds[row,3]}_{hbonds[row,4]} to {hbonds[row,7]}_{hbonds[row,8]} H-bonds'] += 1
        else:
            counts[f'{hbonds[row,3]}_{hbonds[row,4]} to {hbonds[row,7]}_{hbonds[row,8]} H-bonds'] = 1

    return counts

def hbonds_heatmap(mda_universe: Any, residue: str, atom1: str, atom2: str, pore_diameter: float, workdir: str, pore: bool = False)->None:
    
    hyd_bonds, _ = hbonds(mdaUniverse=mda_universe, residue=residue, pore=pore)

    # Create a structured NumPy array to store the center coordinates for each frame and bond
    # Define the dtype for the structured array
    dtype = [('frame', 'i4'), ('center_x', 'f4'), ('center_y', 'f4')]

    # Initialize an empty list to store the rows (each row will be a tuple)
    data = []

    for ts in mda_universe.trajectory:
        frame = ts.frame
        hyd_bonds_frame = hyd_bonds[hyd_bonds[:, 0] == frame]

        for bond in hyd_bonds_frame:
            donor = mda_universe.atoms[bond[1]].name
            acceptor = mda_universe.atoms[bond[5]].name
            if donor or acceptor in [atom1, atom2, 'NL', 'OEE', 'NV', 'OVE', 'OVH']:
                # Get the atom indices from the bond (columns 1 and 4)
                donor_index = int(bond[1])  # Donor atom index
                acceptor_index = int(bond[5])  # Acceptor atom index
                
                # Access the positions of the donor and acceptor atoms
                donor_position = mda_universe.atoms[donor_index].position  # x, y, z of donor
                acceptor_position = mda_universe.atoms[acceptor_index].position  # x, y, z of acceptor
                
                # Extract the x and y coordinates of the donor and acceptor
                donor_x, donor_y = donor_position[0], donor_position[1]
                acceptor_x, acceptor_y = acceptor_position[0], acceptor_position[1]
                
                # Calculate the center coordinate in nm between the donor and acceptor (only x, y)
                center_x = (donor_x + acceptor_x) / 20.0  # Convert from Å to nm (divide by 10)
                center_y = (donor_y + acceptor_y) / 20.0  # Convert from Å to nm (divide by 10)
                
                # Append the new row to the data list
                data.append((frame, center_x, center_y))
    
    # Convert the data list to a structured NumPy array
    structured_array = np.array(data, dtype=dtype)

    # Extract center_x and center_y values from the structured array
    center_x = structured_array['center_x']
    center_y = structured_array['center_y']

    # Define the number of bins for the x and y axes (you can adjust this depending on your data)
    x_bins = round(pore_diameter*100)  # Number of bins for center_x
    y_bins = round(pore_diameter*100)  # Number of bins for center_y

    # Create a 2D histogram (count how many points fall into each bin)
    heatmap, xedges, yedges = np.histogram2d(center_x, center_y, bins=[x_bins, y_bins])

    # Create a meshgrid for the x, y coordinates of the heatmap
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Create a plot for the current atom type
    plt.figure(figsize=(6, 6))
    im = plt.pcolormesh(X, Y, heatmap.T, shading='auto', norm=LogNorm())
    plt.xlabel("X (nm)")
    plt.ylabel("Y (nm)")
    plt.colorbar(im)

    plt.savefig(f'{workdir}/analysis/graphs/HBonds/HBonds_heatmap_{residue}_{atom1}_{atom2}.png')
    plt.close()

def hbonds_pore(mdaUniverse: Any, residue: str, start: int = 0, stop: int = 5000)-> NDArray:
    """
    Gathers hydrogen bond data for a given residue as both an acceptor and a donor through MDAnalysis.

    Args:
        mdaUniverse : Universe object initialized by MDAnalysis.
        start       : Starting frame of analysis.
        stop        : Ending frame of analysis.

    Returns:
        NDArray     : NumPy array containing hydrogen bond information.
    """
    hbonds_data = hba(universe=mdaUniverse,
                      donors_sel=f'(type O or type N) and resname {residue}',
                      hydrogens_sel=f'type H and resname {residue}',
                      acceptors_sel='type O and resname PORE',
                      d_a_cutoff=3.5,               # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
                      d_h_a_angle_cutoff=150,       # Same as GMX default parameters
                      update_selections=False       # No dynamic selections
                      ).run(start=start, stop=stop, verbose=True)
    
    hbonds_donor = np.column_stack([
        hbonds_data.results.hbonds[:, 0].astype(int),                                                       # 0: Frame
        hbonds_data.results.hbonds[:, 1].astype(int),                                                       # 1: Donor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids,                             # 2: Donor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resnames,                           # 3: Donor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,                              # 4: Donor atom name
        hbonds_data.results.hbonds[:, 3].astype(int),                                                       # 5: Acceptor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids,                             # 6: Acceptor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resnames,                           # 7: Acceptor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,                              # 8: Acceptor atom name
        hbonds_data.results.hbonds[:, 4],                                                                   # 9: Hydrogen bond distance
        hbonds_data.results.hbonds[:, 5]                                                                    # 10: Hydrogen bond angle
    ])

    ids = hbonds_data.count_by_ids()

    donor_by_ids = np.column_stack([
        ids[:,0],                                                                                           # 0: Donor index
        mdaUniverse.atoms[ids[:,0]-1].resids,                                                               # 1: Donor resid
        mdaUniverse.atoms[ids[:,0]-1].resnames,                                                             # 2: Donor resname
        mdaUniverse.atoms[ids[:,0]-1].names,                                                                # 3: Donor atom name
        ids[:,1],                                                                                           # 4: Hydrogen index
        ids[:,2],                                                                                           # 5: Acceptor index
        mdaUniverse.atoms[ids[:,2]-1].resids,                                                               # 6: Acceptor resid
        mdaUniverse.atoms[ids[:,2]-1].resnames,                                                             # 7: Acceptor resname
        mdaUniverse.atoms[ids[:,2]-1].names,                                                                # 8: Acceptor atom name
        ids[:,3]                                                                                            # 9: Number of occcurrences
    ])

    hbonds_data = hba(universe=mdaUniverse,
                      donors_sel='type O and resname PORE',
                      hydrogens_sel='type H and resname PORE',
                      acceptors_sel=f'(type O or type N) and resname {residue}',
                      d_a_cutoff=3.5,               # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
                      d_h_a_angle_cutoff=150,       # Same as GMX default parameters
                      update_selections=False       # No dynamic selections
                      ).run(start=start, stop=stop, verbose=True)
    
    hbonds_acceptor = np.column_stack([
        hbonds_data.results.hbonds[:, 0].astype(int),                                                       # 0: Frame
        hbonds_data.results.hbonds[:, 1].astype(int),                                                       # 1: Donor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids,                             # 2: Donor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resnames,                           # 3: Donor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,                              # 4: Donor atom name
        hbonds_data.results.hbonds[:, 3].astype(int),                                                       # 5: Acceptor index
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids,                             # 6: Acceptor resid
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resnames,                           # 7: Acceptor resname
        mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,                              # 8: Acceptor atom name
        hbonds_data.results.hbonds[:, 4],                                                                   # 9: Hydrogen bond distance
        hbonds_data.results.hbonds[:, 5]                                                                    # 10: Hydrogen bond angle
    ])

    ids = hbonds_data.count_by_ids()

    acceptor_by_ids = np.column_stack([
        ids[:,0],                                                                                           # 0: Donor index
        mdaUniverse.atoms[ids[:,0]-1].resids,                                                               # 1: Donor resid
        mdaUniverse.atoms[ids[:,0]-1].resnames,                                                             # 2: Donor resname
        mdaUniverse.atoms[ids[:,0]-1].names,                                                                # 3: Donor atom name
        ids[:,1],                                                                                           # 4: Hydrogen index
        ids[:,2],                                                                                           # 5: Acceptor index
        mdaUniverse.atoms[ids[:,2]-1].resids,                                                               # 6: Acceptor resid
        mdaUniverse.atoms[ids[:,2]-1].resnames,                                                             # 7: Acceptor resname
        mdaUniverse.atoms[ids[:,2]-1].names,                                                                # 8: Acceptor atom name
        ids[:,3]                                                                                            # 9: Number of occcurrences
    ])

    hbonds_pore_all = np.vstack([hbonds_donor, hbonds_acceptor])
    ids_pore_all = np.vstack([donor_by_ids, acceptor_by_ids])

    return  hbonds_pore_all[hbonds_pore_all[:,0].argsort()], ids_pore_all[ids_pore_all[:,0].argsort()], hbonds_data.count_by_type               # Unique ID pairs and unique typing pairs


def find_clusters(hbonds: NDArray,
                  filename: str = None,
                  statistics: bool = True
                  )-> tuple[NDArray, str]:
    """
    Find hydrogen bonds that are related to one another through donor-acceptor and/or residue-residue chains.

    Args:
        hbonds    : NumPy array containing hydrogen bond information.
        filename  : Filename for exported document (optional).
        statistics: Whether to perform statistical analysis on the cluster data.

    Returns:
        NDArray: Array of clusters, each containing a frame number and the list of residue IDs in the cluster.
    """
    # Initialize the result structure for the clusters
    all_clusters = []
    stats = None

    def __get_neighbors__(index, np_frame, visited):
        """
        Find hydrogen bonds that are related to the current bond
        """
        donors = np_frame[index, 1]
        acceptors = np_frame[index, 5]
        donor_resid = np_frame[index, 2]
        acceptor_resid = np_frame[index, 6]

        neighbors = []

        for i in range(len(np_frame)):
            if not visited[i]:  # Only consider non-visited bonds
                # Donor acts as an acceptor or acceptor acts as a donor
                if np_frame[i, 1] == acceptors or np_frame[i, 5] == donors:
                    neighbors.append(i)

                # Check for atoms in the same residue being donor/acceptor
                if np_frame[i, 2] == donor_resid or np_frame[i, 6] == acceptor_resid:
                    neighbors.append(i)

        return neighbors

    def __explore_cluster__(start_index, np_frame, visited):
        """
        Recursively explore and find all hydrogen bonds that form a cluster.
        """
        cluster_resids = set()  # Store unique residue IDs for the cluster
        to_visit = [start_index]

        while to_visit:
            current = to_visit.pop()
            if visited[current]:
                continue
            visited[current] = True

            # Get the donor and acceptor residue IDs and add them to the cluster
            donor_resid = np_frame[current, 2]
            acceptor_resid = np_frame[current, 6]
            cluster_resids.add(donor_resid)
            cluster_resids.add(acceptor_resid)

            neighbors = __get_neighbors__(current, np_frame, visited)
            to_visit.extend(neighbors)

        return cluster_resids
    
    def __export_to_txt__(clusters, filename):
        """
        Custom export function for the clusters data.
        """
        with open(filename, 'w') as txtfile:
            txtfile.write('Frame: Residues\n')
            for cluster in clusters:
                frame = cluster[0]
                residues = ', '.join(map(str, cluster[1]))
                txtfile.write(f'{frame}: {residues}\n')
                
    def __statistics__(all_clusters):
        """
        Get mean and standard deviation for the number (per frame) and (overall) size of clusters.
        """
        # Extract the unique frames (set of frames that appear in the data)
        unique_frames = {frame for frame, _ in all_clusters}

        # Determine the full range of frames (you can set this based on your context)
        min_frame = min(unique_frames)
        max_frame = max(unique_frames)

        # Count the occurrences of each frame (initialize skipped frames with 0 count)
        frame_counts = {frame: 0 for frame in range(min_frame, max_frame + 1)}
        res_size = []
        for frame, residue in all_clusters:
            frame_counts[frame] += 1
            res_size.append(len(residue))

        # Cluster frequency statistics (per frame)        
        freq = list(frame_counts.values())
        tot_freq = len(all_clusters)
        mean_freq = round(np.mean(freq), 2)
        stdev_freq = round(np.std(freq), 2)

        # Cluster size statistics (overall)
        mean_size = round(np.mean(res_size), 2)
        stdev_size = round(np.std(res_size), 2)
        
        return str(f'Total clusters: {tot_freq} in {frame} frames\n' \
                   f'Clusters per frame: {mean_freq} +/- {stdev_freq}\n' \
                   f'Cluster size: {mean_size} +/- {stdev_size}')
        

    # Go through all hydrogen bonds and find clusters for each frame
    for frame in np.unique(hbonds[:, 0]):                  # Loop over unique frames
        np_frame = hbonds[(hbonds[:, 0] == frame) & (hbonds[:, 2] != "PORE") & (hbonds[:, 7] != "PORE")]
        visited = np.zeros(len(np_frame), dtype=bool)      # Initialize visited array for this frame

        for i in range(len(np_frame)):                     # Loop over bonds in the current frame
            if not visited[i]:                             # If bond hasn't been visited, start a cluster search
                cluster_resids = __explore_cluster__(i, np_frame, visited)
                if len(cluster_resids) > 2:                # Only consider clusters with more than 1 bond
                    all_clusters.append([frame, list(cluster_resids)])

    if not filename == None:
        __export_to_txt__(np.array(all_clusters, dtype=object), filename)
    
    if statistics:
        stats = __statistics__(all_clusters)
    
    return np.array(all_clusters, dtype=object), stats