# Imports #
# ------- #
import pexpect
from functools import partial
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rdf import InterRDF
import mdevaluate as mde
import numpy as np
from numpy.typing import NDArray
import os, re, subprocess
import plotting
from tqdm import tqdm
from typing import Any
# ------- #

try:
    plt.style.use('MPL_Styles/ForPapers.mplstyle')
except FileNotFoundError:
    pass

def dihedrals(workdir: str):
    try:
        subprocess.run(
            [
                'gmx', 'mk_angndx',
                '-s', f'{workdir}/run.tpr',
                '-n', f'{workdir}/angle.ndx',
                '-type', 'ryckaert-bellemans'
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create angle index file: {e}")
        return

    handles = []
    with open(os.path.join(workdir, 'angle.ndx')) as f:
        for line in f:
            if "[" in line:
                handle = line.strip().strip('[]').strip()
                handles.append(handle)

    _, ax = plt.subplots()
    x_values = None  # All x-values will be the same

    for group_number, handle in enumerate(handles):
        try:
            xvg_file = os.path.join(workdir, 'analysis', 'data_files', 'Dihedrals', f'dihedrals_{handle}.xvg')

            process = subprocess.Popen(
                [
                    'gmx', 'angle',
                    '-n', os.path.join(workdir, 'angle.ndx'),
                    '-f', os.path.join(workdir, 'out', 'traj.xtc'),
                    '-type', 'ryckaert-bellemans',
                    '-od', xvg_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            _, stderr = process.communicate(input=f"{group_number}\n")

            if process.returncode != 0:
                print(f"[ERROR] gmx angle failed at group {group_number}:\n{stderr}")
                continue

            x, y = np.loadtxt(xvg_file, comments=["#", "@"], unpack=True)
            if x_values is None:
                x_values = x  # Save common x-axis
            ax.plot(x_values, y, label=handle)

        except Exception as e:
            print(f"[ERROR] Exception at group {group_number}: {e}")
            continue

    ax.set_xlabel("Time / ps")
    ax.set_ylabel("Angle / degrees")
    ax.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, 'analysis', 'graphs', 'Dihedrals', 'dihedrals.png'))

    angle_ndx_path = os.path.join(workdir, 'angle.ndx')
    if os.path.exists(angle_ndx_path):
        os.remove(angle_ndx_path)

def end_to_end(mda_universe: Any,
               residue: str,
               workdir: str,
               ):
    selected_atoms = mda_universe.select_atoms(f'resname {residue} and (type O or type C)')
    
    upper_name = selected_atoms.names[0]
    lower_name = selected_atoms.names[-1]

    upper_atoms = mda_universe.select_atoms(f'name {upper_name}')
    lower_atoms = mda_universe.select_atoms(f'name {lower_name}')

    end_to_end_distances = []

    for dt in mda_universe.trajectory:
        upper_positions = upper_atoms.positions
        lower_positions = lower_atoms.positions

        if len(upper_positions) != len(lower_positions):
            raise ValueError(f"Number of upper atoms ({len(upper_positions)}) doesn't match the number of lower atoms ({len(lower_positions)})")

        for up_pos, low_pos in zip(upper_positions, lower_positions):
            difference = up_pos - low_pos
            distance = np.linalg.norm(difference)

            if distance < 20:
                end_to_end_distances.append(distance / 10)

    mean_distance = np.mean(end_to_end_distances)
    std_distance = np.std(end_to_end_distances)

    title_text = f'Mean: {mean_distance:.2f} +/- {std_distance:.2f}'

    hist, bin_edges = np.histogram(end_to_end_distances, bins=100)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plotting.plot_line(f'{workdir}/analysis/graphs/End_to_End/End_to_end_{residue}.png', bin_centers, hist, title=title_text, xlabel='Distance (nm)', ylabel='Frequency')
    np.savetxt(f'{workdir}/analysis/data_files/End_to_End/End_to_end_{residue}.csv', np.column_stack([bin_centers, hist]), delimiter=',', header='Distance / nm, Frequency')

def gyration(workdir: str):
    summary_lines = []
    summary_file = os.path.join(workdir, 'analysis', 'graphs', 'Gyration', 'gyration_summary.txt')

    # Prepare the command to get group info
    command = f"gmx gyrate -f {workdir}/out/traj.xtc -s {workdir}/run.tpr -o dummy.xvg"
    child = pexpect.spawn(command, encoding='utf-8')

    try:
        child.expect(r"[Ss]elect a group", timeout=30)
        group_output = child.before
        child.close(force=True)  # force close since we're not selecting anything yet

        # Parse group info from output
        group_pattern = re.compile(r"Group\s+(\d+)\s+\(\s*(\S+)\s*\)")
        excluded = {"System", "Other"}

        selected = []
        for line in group_output.splitlines():
            match = group_pattern.search(line)
            if match:
                group_num = match.group(1)
                group_name = match.group(2)
                if group_name not in excluded:
                    selected.append((group_num, group_name))

        if not selected:
            print("[WARNING] No valid groups to analyze.")
            return

        print(f"[INFO] Will analyze groups: {selected}")

        # Run gmx gyrate for each selected group
        for group_num, group_name in selected:
            print(f"[INFO] Running gyrate for group '{group_name}' ({group_num})")
            xvg_file = os.path.join(workdir, 'analysis', 'data_files', 'Gyration', f'gyration_{group_name}.xvg')
            run_cmd = f"gmx gyrate -f {workdir}/out/traj.xtc -s {workdir}/run.tpr -n {workdir}/gyration.ndx -o {xvg_file}"

            child = pexpect.spawn(run_cmd, encoding='utf-8')
            child.expect(r"[Ss]elect a group", timeout=20)
            child.sendline(group_num)
            child.expect(pexpect.EOF)
            child.close()

            try:
                t, rg_all, rg_x, rg_y, rg_z = np.loadtxt(xvg_file, comments=["#", "@"], unpack=True)

                mean_all, std_all = np.mean(rg_all), np.std(rg_all)
                mean_x, std_x = np.mean(rg_x), np.std(rg_x)
                mean_y, std_y = np.mean(rg_y), np.std(rg_y)
                mean_z, std_z = np.mean(rg_z), np.std(rg_z)

                summary_lines.append(
                    f"{group_name:10s} | All: {mean_all:.3f} ± {std_all:.3f} nm | "
                    f"X: {mean_x:.3f} ± {std_x:.3f} nm | "
                    f"Y: {mean_y:.3f} ± {std_y:.3f} nm | "
                    f"Z: {mean_z:.3f} ± {std_z:.3f} nm"
                )

            except Exception as e:
                print(f"[ERROR] Failed to process {xvg_file}: {e}")
                continue

        with open(summary_file, 'w') as f:
            f.write("Radius of Gyration Summary (mean ± std, in nm)\n")
            f.write("-" * 80 + "\n")
            for line in summary_lines:
                f.write(line + "\n")

        print(f"[DONE] Summary written to: {summary_file}")

    except pexpect.exceptions.ExceptionPexpect as e:
        print(f"[ERROR] Failed to read or run gmx gyrate: {e}")
      
def radial_density(mde_trajectory: Any, residue: str, atom: str, diameter: float)-> NDArray:
    '''
    Calculates a radial spatial density function (rSDF) for an atom on a residue.

    Args:
        mde_trajectory: Universe object initialized by mdevaluate.
        residue: Topology name for the residue of interest.
        atom1: Topology name for an atom on the residue.
        diameter: Diameter of the pore.

    Returns:
        NDArray: Resulting rSDF.
    '''
    bins = np.arange(0.0, diameter/2 + 0.1, 0.025)
    pos = mde.distribution.time_average(partial(mde.distribution.radial_density, bins=bins), mde_trajectory.subset(atom_name=atom, residue_name=residue), segments=1000, skip=0.01)

    return np.column_stack([bins[1:], pos])

def radial_distances(mda_universe: Any, workdir: str, pore_diameter: float, exclusions: str = '')->None:
    '''
    Calculates and stores radial distances for N, O, and F atoms in the x,y-plane.
    Prints the transpose of the positions, with trajectory time as the first column.

    Args:
        mda_universe: Universe object initialized by MDAnalysis.
        exclusions  : Exclusion groups from calculations (e.g. 'resname PORE').
                      If an exclusion is provided, the selection will exclude those atoms.
        save_dir    : Directory to save the generated heatmap images (default is './heatmaps/')
    '''
    
    # Start building the selection string
    selection_string = "type N or type O or type F"
    
    # If exclusions are provided, append to the selection string
    if exclusions:
        selection_string += f" and not ({exclusions})"
    
    # Select atoms based on the dynamically constructed selection string
    selected_atoms = mda_universe.select_atoms(selection_string)

    # Get unique atom names to avoid duplicate processing
    unique_atoms = set(selected_atoms.names)

    # Iterate through unique atom names and compute their radial distributions
    for atom in unique_atoms:
        # Select atoms of the current type
        atoms = mda_universe.select_atoms(f'name {atom}')
        
        # List to store the positions of atoms
        data = []

        # Iterate over the trajectory
        for dt in mda_universe.trajectory:
            # Get the positions of atoms in the x,y-plane (first two coordinates)
            positions = atoms.positions[:, :2]
            data.extend(positions)  # Append the positions to data

        # Convert data to numpy array
        data = np.array(data)

        # Create a 2D histogram to generate the heatmap
        x_bins = round(pore_diameter*100)  # Number of bins for center_x
        y_bins = round(pore_diameter*100)  # Number of bins for center_y
        heatmap, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=[x_bins, y_bins], density=True)

        # Create a meshgrid for the x, y coordinates of the heatmap
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        # Create a plot for the current atom type
        plt.figure(figsize=(6, 6))
        im = plt.pcolormesh(X, Y, heatmap.T, shading='auto', norm=LogNorm())
        plt.title(f"Radial Distribution Heatmap for Atom: {atom}")
        plt.xlabel("X (Å)")
        plt.ylabel("Y (Å)")
        plt.colorbar(im)

        plt.savefig(f'{workdir}/analysis/graphs/HBonds/Positions_heatmap_{atom}.png')
        plt.close()

def rdf_com(com: NDArray, segments: int = 1000) -> tuple[NDArray, float, float]:
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
    bins = np.arange(0, 2.2, 0.01)
    rdf = mde.distribution.time_average(
        partial(mde.distribution.rdf, bins=bins), com, segments=segments, skip=0.01
    )
    out = np.column_stack([bins[:-1], rdf])

    YMaxIdx = np.argmax(out[:, 1])
    XAtMax = out[YMaxIdx, 0]
    YAtMax = out[YMaxIdx, 1]
    MagScattVector = 2 * np.pi / XAtMax

    return out, MagScattVector, YAtMax

def rdf_inter(mdaUniverse: Any, residue: str, atom1: str, atom2: str, start: int = 0, stop: int = 5000)-> NDArray:
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
    rdf_values_list_3 = []  # For RDF between atom2 and non-residue atom2
    
    with tqdm(total=3 * len(set(residue_atoms.resids)), desc="Inter RDF Progress", unit="Resid") as pbar:
        for resid in sorted(set(residue_atoms.resids)):
            # Atom 1 vs non-residue atom1
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom1}'),
                nbins=250
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_1.append(a1_a2_rdf.results.rdf)
            pbar.update(1)
            
            # Atom 1 vs non-residue atom2
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom2}'),
                nbins=250
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_2.append(a1_a2_rdf.results.rdf)
            pbar.update(1)
            
            # Atom 2 vs non-residue atom2
            a1_a2_rdf = InterRDF(
                g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom2}'),
                g2=mdaUniverse.select_atoms(f'not resid {resid} and name {atom2}'),
                nbins=250
            )
            a1_a2_rdf.run(start=start, stop=stop)
            rdf_values_list_3.append(a1_a2_rdf.results.rdf)
            pbar.update(1)

    # Convert lists of RDF values into NumPy arrays
    rdf_array_1 = np.array(rdf_values_list_1)
    rdf_array_2 = np.array(rdf_values_list_2)
    rdf_array_3 = np.array(rdf_values_list_3)

    # Compute the mean for each group (atom pair) separately
    mean_rdf_1 = np.mean(rdf_array_1, axis=0)
    mean_rdf_2 = np.mean(rdf_array_2, axis=0)
    mean_rdf_3 = np.mean(rdf_array_3, axis=0)

    # Get the distance bins (from the last RDF calculation)
    distance_bins = a1_a2_rdf.results.bins / 10

    # Stack the results together
    result = np.column_stack([distance_bins, mean_rdf_1, mean_rdf_2, mean_rdf_3])

    return result

def rdf_intra(mdaUniverse: Any, residue: str, atom1: str, atom2: str, start: int = 0, stop: int = 5000)-> NDArray:
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
            a1_a2_rdf = InterRDF(g1=mdaUniverse.select_atoms(f'resid {resid} and name {atom1}'), g2=mdaUniverse.select_atoms(f'resid {resid} and name {atom2}'), nbins=250)
            a1_a2_rdf.run(start=start, stop=stop)

            rdf_values_list.append(a1_a2_rdf.results.rdf)
            pbar.update(1)

    rdf_array = np.array(rdf_values_list)

    return np.column_stack([a1_a2_rdf.results.bins / 10, np.mean(rdf_array, axis=0) / np.max(np.mean(rdf_array, axis=0))])

def z_align(vectors: NDArray, segments: int)-> NDArray:
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

def z_histogram(vectors: NDArray, segments: int, skip: float = 0.1)-> NDArray:
    """
    Calculates Z-axis radial positions based on centers of masses.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        segments: Number of segments to divide the trajectory into.
        skip: Fraction of the beginning of the trajectory to skip over.

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