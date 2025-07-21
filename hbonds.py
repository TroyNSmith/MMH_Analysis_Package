# Imports #
# ------- #
import helpers
from itertools import chain
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba
import numpy as np
from numpy.typing import NDArray
from typing import Any
# ------- #

def hbonds(mdaUniverse: Any,
           donors_sel: str,
           hydrogens_sel: str = 'type H',
           acceptors_sel: str = 'type O or type N',
           start: int = 0,
           stop: int = 5000,
           )-> NDArray:
    """
    Gathers hydrogen bond data for a given residue as both an acceptor and a donor through MDAnalysis.

    Args:
        mdaUniverse : Universe object initialized by MDAnalysis.
        residue     : Residue to analyze.
        start       : Starting frame of analysis.
        stop        : Ending frame of analysis.

    Returns:
        NDArray     : NumPy array containing hydrogen bond information.
    """
    hbonds_data = hba(universe=mdaUniverse,
                      donors_sel=donors_sel,
                      hydrogens_sel=hydrogens_sel,
                      acceptors_sel=acceptors_sel,
                      d_a_cutoff=3.0,
                      d_h_a_angle_cutoff=150,
                      update_selections=False
                      ).run(start=start, stop=stop)
    
    return np.column_stack([hbonds_data.results.hbonds[:, 0].astype(int),                           # 0: Frame
                            hbonds_data.results.hbonds[:, 1].astype(int),                           # 1: Donor index
                            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].resids, # 2: Donor resid
                            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 1].astype(int)].names,  # 3: Donor atom name
                            hbonds_data.results.hbonds[:, 3].astype(int),                           # 4: Acceptor index
                            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].resids, # 5: Acceptor resid
                            mdaUniverse.atoms[hbonds_data.results.hbonds[:, 3].astype(int)].names,  # 6: Acceptor atom name
                            hbonds_data.results.hbonds[:, 4],                                       # 7: Hydrogen bond distance
                            hbonds_data.results.hbonds[:, 5]                                        # 8: Hydrogen bond angle
                            ])                       

def hbond_counts(hbonds: NDArray,
                 )-> dict:
    """
    Counts the inter- and intramolecular hydrogen bonds present in a system, as well as categorizes them by the donor-acceptor pair (e.g., O01 to O0E).

    Args:
        hbonds       : NumPy array containing the hydrogen bonds.

    Returns:
        dict         : Dictionary containing the counts of hydrogen bond types and pairs.
    """
    counts = {'Intra H-bonds': 0,
              'Inter H-bonds': 0,
              'Total H-bonds': hbonds.shape[0],
              'Average H-bonds per frame': round(hbonds.shape[0] / np.max(hbonds[:,0]), 2)}

    for row in range(len(hbonds)):
        if hbonds[row,2] == hbonds[row,5]:
            counts['Intra H-bonds'] += 1
        else:
            counts['Inter H-bonds'] += 1

        if f'{hbonds[row,3]} to {hbonds[row,6]} H-bonds' in counts:
            counts[f'{hbonds[row,3]} to {hbonds[row,6]} H-bonds'] += 1
        else:
            counts[f'{hbonds[row,3]} to {hbonds[row,6]} H-bonds'] = 1

    return counts

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
        acceptors = np_frame[index, 4]
        donor_resid = np_frame[index, 2]
        acceptor_resid = np_frame[index, 5]

        neighbors = []

        for i in range(len(np_frame)):
            if not visited[i]:  # Only consider non-visited bonds
                # Donor acts as an acceptor or acceptor acts as a donor
                if np_frame[i, 1] == acceptors or np_frame[i, 4] == donors:
                    neighbors.append(i)

                # Check for atoms in the same residue being donor/acceptor
                if np_frame[i, 2] == donor_resid or np_frame[i, 5] == acceptor_resid:
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
            acceptor_resid = np_frame[current, 5]
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
        np_frame = hbonds[hbonds[:, 0] == frame]           # Get all bonds in the current frame
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