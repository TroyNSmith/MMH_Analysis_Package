import click
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdevaluate as mde
import os
import pandas as pd

from functools import reduce
from pathlib import Path

from .functions import dynamics
from .functions.coordinates import centers_of_masses, vectorize_residue
from .functions.plotting import plot_line
from .functions.utils import log_analysis_yaml


@click.group()
def cli():
    """Main CLI entry point."""
    pass


@click.command()
@click.option(
    "-i",
    "--input",
    "sim_dir",
    help="Path to simulation directory containing trajectory and topology files. This directory will also be used to dump output files unless otherwise specified.",
)
@click.option(
    "-tr",
    "--trajectory",
    "trajectory",
    help="Name of the trajectory file located within -d / --directory.",
)
@click.option(
    "-tp",
    "--topology",
    "topology",
    help="Name of the topology file located within -d / --directory.",
)
@click.option(
    "-r",
    "--resname",
    "res_name",
    help="Name of the residue to be analyzed.",
    default=None,
)
@click.option(
    "-a",
    "--atoms",
    "atoms",
    nargs=2,
    help="Two atoms for analysis located on the residue specified in -res / --resname.",
    default=None,
)
@click.option(
    "-s",
    "--segments",
    "num_segments",
    help="Number of starting points to use with correlation functions.",
    type=click.IntRange(10, 1000, clamp=True),
    default=500,
)
@click.option(
    "-d",
    "--diameter",
    "pore_diameter",
    help="Diameter of the pore in nm.",
    type=click.FloatRange(0, 1000, clamp=False),
    default=None,
)
@click.option(
    "-q",
    "--q",
    "q_magnitude",
    type=float,
    help="Magnitude of the scattering vector, q.",
    default=None,
)
@click.option(
    "-o",
    "--out",
    "output_dir",
    help="Designated path for analysis output files.",
    default=None,
)
@click.option(
    "-ov",
    "--override",
    "override",
    help="Force re-run of calculations.",
    is_flag=True,
)
@click.option(
    "-po",
    "--plot_only",
    "plot_only",
    help="Skip calculations and regenerate plots of existing data.",
    is_flag=True,
)
def Run(
    sim_dir: Path,
    trajectory: str,
    topology: str,
    res_name: str,
    atoms: list[str],
    num_segments: int,
    pore_diameter: float,
    q_magnitude: float,
    override: bool,
    plot_only: bool,
):
    # ===== Step 1: Initialize =====
    mde_coords = mde.open(
        directory=sim_dir, topology=topology, trajectory=trajectory, nojump=True
    )
    mde_vectors = vectorize_residue(
        mde_coords == mde_coords, residue=res_name, atoms=atoms
    )
    mde_com = centers_of_masses(mde_coords=mde_coords, residue=res_name)

    mda_coords = mda.Universe(
        os.path.join(sim_dir, topology), os.path.join(sim_dir, trajectory)
    )

    parameters={
        "trajectory": trajectory,
        "topology": topology,
        "res_name": res_name,
        "atoms": atoms,
        "num_segments": num_segments,
        "pore_diameter": pore_diameter,
        "q_magnitude": q_magnitude,
    }

    dir_out = os.path.join(sim_dir, "analysis")

    yaml_out = os.path.join(dir_out, "log.yaml")

    # ===== Step 2: Mean square displacement =====
    csv_out_all = os.path.join(dir_out, "MSD", "MSD_all.csv")
    csv_out_z = os.path.join(dir_out, "MSD", "MSD_z.csv")
    csv_out_xy = os.path.join(dir_out, "MSD", "MSD_xy.csv")
    png_out = os.path.join(dir_out, "MSD", "MSD.png")

    if not os.path.exists(csv_out_all) or override:
        msd_all = dynamics.mean_square_displacement(
            coords=mde_com, num_segments=num_segments
        )
        msd_all.to_csv(csv_out_all)
        log_analysis_yaml(log_path=yaml_out,
                          analysis_name='MSD (z-axis)',
                          parameters=parameters)

    if not os.path.exists(csv_out_z) or override:
        msd_z = dynamics.mean_square_displacement(
            coords=mde_com, axis="z", num_segments=num_segments
        )
        msd_z.to_csv(csv_out_z)

    if not os.path.exists(csv_out_xy) or override:
        msd_xy = dynamics.mean_square_displacement(
            coords=mde_com, axis="xy", num_segments=num_segments
        )
        msd_xy.to_csv(csv_out_xy)

    if not os.path.exists(png_out) or plot_only:
        msd_all = pd.read_csv(csv_out_all)
        msd_z = pd.read_csv(csv_out_z)
        msd_xy = pd.read_csv(csv_out_xy)

        msd_merged = reduce(
            lambda left, right: pd.merge(left, right, on="time / ps"),
            [msd_all, msd_z, msd_xy],
        )

        plot_line(
            output_path=png_out,
            x_data=msd_merged[:, 0],
            y_data=msd_merged[:, 1:],
            x_axis_scale="log",
            x_axis_label=r"$\mathbf{\mathit{t}}$ / ps",
            y_axis_scale="log",
            y_axis_label=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
        )


"""
# ===== Step 3: Calculate the resolved mean square displacement =====
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD/Resolved_MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Resolved MSD")
            msd, r = dynamics.resolved_msd(com=com, diameter=pore_inf['D'], segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD/Resolved_MSD_{residue}.png', msd[:,0], msd[:,1:], xlabel=r"$\mathbf{\mathit{t}}$ / ps", ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log', yscale='log', legend=True, handles=[f'{bin:.2f}' for bin in r], ncols=2)
            np.savetxt(f'{workdir}/analysis/data_files/MSD/Resolved_MSD_{residue}.csv', msd, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in r]))
        pbar.update(1)

# ===== Step 4: Calculate the RDF / q constant =====
        if q_val == 0 or overwrite:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com, segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/RDF_com_{residue}_{atom1}_{atom2}.png', rdf[:,0], rdf[:,1], xlabel='r / nm', ylabel='g(r)')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/RDF_com_{residue}_{atom1}_{atom2}.csv', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

# ===== Step 5: Calculate the incoherent scattering function =====
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.incoherent_scattering_function(coords=com, q_val=q_val, num_segments=segments, pore_diameter=pore_inf['D']/2)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png', isf[:,0], isf[:,1:], xlabel=r"$t$ / ps", ylabel="ISF", xscale='log', legend=True, handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{workdir}/analysis/data_files/ISF/ISF_{residue}_{atom1}_{atom2}.csv', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

# ===== Step 6: Calculate the resolved incoherent scattering function =====
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF/Resolved_ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Resolved ISF")
            isf, r = dynamics.resolved_isf(com=com, q_val=q_val, diameter=pore_inf['D'], segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF/Resolved_ISF_{residue}.png', isf[:,0], isf[:,1:], xlabel=r"$t$ / ps", ylabel="ISF", xscale='log',
                               legend=True, handles=[f'{bin:.2f}' for bin in r], ncols=2)
            np.savetxt(f'{workdir}/analysis/data_files/ISF/Resolved_ISF_{residue}.csv', isf, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in r]))
        pbar.update(1)

# ===== Step 7: Calculate rotational correlation coefficients =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational Correlation")
            rot_corr = dynamics.rotational_corr(vectors=mde_vectors, num_segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.png', rot_corr[:,0], rot_corr[:,1:], xlabel=r"$t$ / ps",
                               ylabel="F(t)", xscale='log', legend=True, handles=[r'$F_1(t)$', r'$F_2(t)$'], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.csv', rot_corr, delimiter=',', header='Time / ps, F1, F2')
        pbar.update(1)

# ===== Step 8: Calculate the non-Gaussian displacement statistics =====
        if not os.path.exists(f'{workdir}/analysis/graphs/nonGauss/nonGauss_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Non-Gaussian Displacement")
            non_gauss = dynamics.non_Gauss(com=com, segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/nonGauss/nonGauss_{residue}_{atom1}_{atom2}.png', non_gauss[:,0], non_gauss[:,1:],
                               xlabel=r"$t$ / ps", ylabel="Non-Gaussian Displacement", xscale='log')
            np.savetxt(f'{workdir}/analysis/data_files/nonGauss/nonGauss_{residue}_{atom1}_{atom2}.csv', non_gauss, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)

# ===== Step 9: Calculate the translational van Hove correlations =====
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove/vanHove_transl_center_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Translational van Hove")
            plt.style.use('MPL_Styles/3D_Plot.mplstyle')            # Use custom styling
            vH_wall, vH_center, bins = dynamics.van_hove_translation(coords=com, pore_diameter=pore_inf['D'], num_segments=segments, pore=True)
            complete_matrix = np.column_stack([bins, vH_wall.T])

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            for i in range(2, complete_matrix.shape[1]):
                bins = complete_matrix[1:, 0]
                vH_all = complete_matrix[1:, i]
                t = complete_matrix[0, i]
                times = np.full_like(bins, t)

                ax.plot(bins, times, vH_all, alpha=0.8)

            ax.set_xlabel("r / nm")
            ax.set_ylabel("t / ps")
            ax.set_zlabel("S(r, t)", labelpad=10)
            ax.grid(False)
            ax.view_init(elev=30, azim=120)

            plt.savefig(f'{workdir}/analysis/graphs/vanHove/vanHove_transl_wall_{residue}_{atom1}_{atom2}.png')
            np.savetxt(f'{workdir}/analysis/data_files/vanHove/vanHove_transl_wall_{residue}_{atom1}_{atom2}.csv', vH_wall, delimiter=',')
            plt.clf()

            complete_matrix = np.column_stack([np.insert(bins,0,0), vH_center.T])

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            for i in range(2, complete_matrix.shape[1]):
                bins = complete_matrix[1:, 0]
                vH_all = complete_matrix[1:, i]
                t = complete_matrix[0, i]
                times = np.full_like(bins, t)

                ax.plot(bins, times, vH_all, alpha=0.8)

            ax.set_xlabel("r / nm")
            ax.set_ylabel("t / ps")
            ax.set_zlabel("S(r, t)", labelpad=10)
            ax.grid(False)
            ax.view_init(elev=30, azim=120)

            plt.savefig(f'{workdir}/analysis/graphs/vanHove/vanHove_transl_center_{residue}_{atom1}_{atom2}.png')
            np.savetxt(f'{workdir}/analysis/data_files/vanHove/vanHove_transl_center_{residue}_{atom1}_{atom2}.csv', vH_wall, delimiter=',')
            plt.clf()
            plt.style.use('MPL_Styles/ForPapers.mplstyle')      # Go back to normal styling

        pbar.update(1)

# ===== Step 10: Calculate the rotational van Hove correlations =====
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational van Hove")
            vH_rot =  dynamics.van_hove_rotation(vectors=mde_vectors, num_segments=segments)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.png', vH_rot[vH_rot[:,0] == 0, 1], 
                                  np.column_stack([vH_rot[vH_rot[:,0] == t, 2] for t in np.unique(vH_rot[:,0])[5::10]]),
                                  xlabel=r'$\varphi$', ylabel=r'$S(\varphi)$', legend=True,
                                  handles=[f'{t} ps' for t in np.unique(vH_rot[:,0])[5::10]], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.csv', vH_rot, delimiter=',')
        pbar.update(1)

# ===== Step 11: Calculate the fourth-order susceptibility =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Susceptibility/Susceptibility_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating 4th Order Susceptibility")
            sus = dynamics.chi_4_susceptibility(coords=com, q_val=q_val)
            plotting.plot_line(f'{workdir}/analysis/graphs/Susceptibility/Susceptibility_{residue}_{atom1}_{atom2}.png', sus[:,0], sus[:,1:], xlabel=r"$t$ / ps", 
                               ylabel=r'$\chi_4 \cdot 10^{-5}$', xscale='log', legend=True, handles=[r'$\chi_4$', r'$\chi_4$ Smoothed'])
            np.savetxt(f'{workdir}/analysis/data_files/Susceptibility/Susceptibility_{residue}.csv', sus, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)
        
# ===== Step 12: Calculate the Z-axis radial alignments =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Z_Axis/Z_align_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Z-Axis Alignment")
            Z_align =  structural.z_align(vectors=mde_vectors, segments=segments)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/Z_Axis/Z_align_{residue}_{atom1}_{atom2}.png', Z_align[Z_align[:,0] == 0, 1], 
                                  np.column_stack([Z_align[Z_align[:,0] == t, 2] for t in np.unique(Z_align[:,0])[::20]]),
                                  xlabel=r'$\varphi$', ylabel=r'$S(\varphi)$', legend=True,
                                  handles=[f'{t} ps' for t in np.unique(Z_align[:,0])[::20]], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/Z_Axis/Z_align_{residue}_{atom1}_{atom2}.csv', Z_align, delimiter=',', header=r'Time / ps, $\varphi$, S($\varphi$)')
        pbar.update(1)
        
# ===== Step 13: Calculate the Z-axis radial positions =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Z_Axis/Z_histogram_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Z-Axis Radial Positions")
            Z_histo =  structural.z_histogram(vectors=mde_vectors, segments=segments)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/Z_Axis/Z_histogram_{residue}_{atom1}_{atom2}.png',  Z_histo[Z_histo[:,0] == 0, 1], 
                                  np.column_stack([Z_histo[Z_histo[:,0] == t, 2] for t in np.unique(Z_histo[:,0])[::20]]),
                                  xlabel=r'$\varphi$', ylabel=r'$S(\varphi)$', legend=True,
                                  handles=[f'{t} ps' for t in np.unique(Z_histo[:,0])[::20]], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/Z_Axis/Z_histogram_{residue}_{atom1}_{atom2}.csv', Z_histo, delimiter=',', header=r'Time / ps, $\varphi$, dist($\varphi$)')
        pbar.update(1)
        
# ===== Step 14: Calculate the intramolecular atom-wise RDF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF/RDF_Intra_{residue}_{atom1}_to_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Intra Atom-Wise RDFs")
            rdf_intra = structural.rdf_intra(mdaUniverse=mda_universe, residue=residue, atom1=atom1, atom2=atom2) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/RDF_Intra_{residue}_{atom1}_to_{atom2}.png', rdf_intra[:,0], rdf_intra[:,1], xlabel='r / nm', ylabel=r'$g_{intra}(r)$')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/RDF_Intra_{residue}_{atom1}_to_{atom2}.csv', rdf_intra, delimiter=',', header='Distance / nm, Mean RDF')          
        pbar.update(1)
        
# ===== Step 15: Calculate the intermolecular atom-wise RDF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF/RDF_Inter_{residue}_{atom1}_to_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Inter Atom-Wise RDFs")
            rdf_inter = structural.rdf_inter(mdaUniverse=mda_universe, residue=residue, atom1=atom1, atom2=atom2) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/RDF_Inter_{residue}_{atom1}_to_{atom2}.png', rdf_inter[:,0], rdf_inter[:,1:], xlabel='r / nm', ylabel=r'$g_{inter}(r)$', legend=True,
                               handles=[f'{residue}:{atom1} to {residue}:{atom1}',
                                        f'{residue}:{atom1} to {residue}:{atom2}',
                                        f'{residue}:{atom2} to {residue}:{atom2}'])
            np.savetxt(f'{workdir}/analysis/data_files/RDF/RDF_Intra_{residue}_{atom1}_to_{atom2}.csv', rdf_inter, delimiter=',', header='Distance / nm, Mean RDF')               
        pbar.update(1)
        
# ===== Step 16: Calculate the radial spatial density function(s) =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Spatial_Density/Spatial_Density_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Radial Spatial Density Function")

            results = []
            residues = [residue, residue, 'LNK', 'ETH', 'VAN', 'VAN', 'VAN']
            names = [atom1, atom2, 'NL', 'OEE', 'NV', 'OVE', 'OVH']

            with tqdm(total=len(residues), desc="Spatial Density Progress", unit="Pair") as bar:
                for res, atom in zip(residues, names):
                    if res in [f'{residue.resname}' for residue in mda_universe.residues]:
                        rSDF = structural.radial_density(mde_trajectory=mde_trajectory, residue=res, atom=atom, diameter=pore_inf['D'])
                        results.append(rSDF[:,1])
                    bar.update(1)

            results = [rSDF[:,0]] + results
            rSDF = np.column_stack(results)
            plotting.plot_line(f'{workdir}/analysis/graphs/Spatial_Density/Spatial_Density_{residue}_{atom1}_{atom2}.png', rSDF[:,0], rSDF[:,1:],
                               xlabel="r / nm", ylabel=r"Number Density / nm$^{-3}$", legend=True,
                               handles=[f'{res}:{atom}' for res, atom in zip(residues, names)])
            np.savetxt(f'{workdir}/analysis/data_files/Spatial_Density/Spatial_Density_{residue}_{atom1}_{atom2}.csv', rSDF, delimiter=',', header='r / nm, ' + ''.join([f'{res}:{atom}, ' for res, atom in zip(residues, names)])) 
        pbar.update(1)

# ===== Step 17: Initialize the hydrogen bond information =====
        if not os.path.exists(f'{workdir}/analysis/data_files/HBonds/All_HBonds_NEW.csv') or overwrite:
            pbar.set_postfix(step="Initializing Hydrogen Bond Information")
            hyd_bonds, unique_pairs = hbonds.hbonds(mdaUniverse=mda_universe, residue=residue, pore=True)
            np.savetxt(f'{workdir}/analysis/data_files/HBonds/All_HBonds_NEW.csv', hyd_bonds, delimiter=',', fmt='%d,%d,%d,%s,%s,%d,%d,%s,%s,%.3f,%.2f', header='Frame,Donor_Index,Donor_ResID,Donor_ResName,Donor_Atom_Name,Acceptor_Index,Acceptor_ResID,Acceptor_ResName,Acceptor_Atom_Name,Distance,Angle')
            try:
                np.savetxt(f'{workdir}/analysis/data_files/HBonds/Unique_HBonds_IDs.csv', unique_pairs, delimiter=',', fmt='%d,%d,%s,%s,%d,%d,%d,%s,%s,%d', header='Donor_Index,Donor_ResID,Donor_ResName,Donor_Atom_Name,Hydrogen_Index,Acceptor_Index,Acceptor_ResName,Acceptor_ResID,Acceptor_Atom_Name,Number_of_Occurrences')
            except ValueError:
                pass
            counts = hbonds.hbond_counts(hbonds=hyd_bonds, residue=residue)

            for pair, count in counts.items():
                helpers.log_info(workdir, f'{pair}: {count}')

            print("Finding Clusters...")
            _, stats = hbonds.find_clusters(hbonds=hyd_bonds, filename=f'{workdir}/analysis/data_files/HBonds/Clusters_{residue}.txt')

            if not stats == None:
                helpers.log_info(workdir, stats)
                with open(f'{workdir}/analysis/data_files/HBonds/Clusters_Summary_{residue}.txt', 'w') as f:
                    f.write(stats)
                    f.close()

        pbar.update(1)

# ===== Step 18: Generate hydrogen bonds heatmap =====
        #if not os.path.exists(f'{workdir}/analysis/graphs/HBonds/HBonds_heatmap_{residue}_{atom1}_{atom2}.png') or overwrite:
        if 1 == 1:
            pbar.set_postfix(step="Calculating hydrogen bonds heatmap")
            hbonds.hbonds_heatmap(mda_universe=mda_universe, residue=residue, atom1=atom1, atom2=atom2, pore_diameter=pore_inf['D'], workdir=workdir, pore=True)
        pbar.update(1)

# ===== Step 19: Generate positional heatmap for any hydrogen bonding species =====
        #if not os.path.exists(f'{workdir}/analysis/graphs/HBonds_heatmap_{atom1}.png') or overwrite:
        if 1 == 1:
            pbar.set_postfix(step="Calculating positional heatmaps")
            structural.radial_distances(mda_universe=mda_universe, workdir = workdir, pore_diameter=pore_inf['D'], exclusions='resname PORE')
        pbar.update(1)

# ===== Step 20: Generate end to end distance =====
        if not os.path.exists(f'{workdir}/analysis/graphs/End_to_End/End_to_end_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating end-to-end distances")
            structural.end_to_end(mda_universe=mda_universe, residue=residue, workdir=workdir)
        pbar.update(1)

# ===== Step 21: Calculate dihedral angles =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Dihedrals/dihedrals.png') or overwrite:
            pbar.set_postfix(step="Gathering Dihedral Angles")
            structural.dihedrals(workdir=workdir)
        pbar.update(1)

# ===== Step 22: Calculate radii of gyration =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Gyration/gyration_summary.txt') or overwrite:
            pbar.set_postfix(step="Gathering Radii of Gyration")
            structural.gyration(workdir=workdir)
        pbar.update(1)

# ===== Complete! =====
        pbar.set_postfix(step="Completed Full Analysis!")
        pbar.update(1)

def minimal(args_dict: dict)-> None:
    '''
    Performs minimal analysis on a pore simulation (including MSD, RDF, and ISF).

    Args:
        args_dict: A dictionary containing all of the imported arguments from main.py
        ver_dir  : Name of the verification folder.
    '''
    workdir = args_dict['workdir']
    segments = args_dict['segments']
    residue = args_dict['residue']
    atom1 = args_dict['atoms'][0]
    atom2 = args_dict['atoms'][1]
    q_val = args_dict['q_value'] if 'q_value' in args_dict.keys() else 0
    overwrite = args_dict['overwrite']
        
    with tqdm(total=4, desc="Minimal Analysis Progress", unit="step") as pbar:
# ===== Step 1: Initialize =====
        pbar.set_postfix(step="Initializing")           # Sets the progress bar step seen on the right hand side of the terminal interface
        # Loop through possible topology and trajectory names (add more if necessary; avoids issues with naming inconsistencies).
        topologies = ["md.tpr", "run.tpr"]
        trajectories = ["out/traj.xtc", "out/out.xtc"]

        mde_trajectory = None
    
        for tpr in topologies:
            for xtc in trajectories:
                print(workdir, mde_trajectory)
                try:
                    mde_trajectory = mde.open(directory=workdir, topology=tpr, trajectory=xtc, nojump=True)
                    mde_vectors = handle.handle_vectors(mde_trajectory=mde_trajectory, residue=residue, atom1=atom1, atom2=atom2)
                    com = helpers.center_of_masses(trajectory=mde_trajectory, residue=residue)
                    print(f"Opened with: topology={tpr}, trajectory={xtc}")
                    break       # Break if successful
                except FileNotFoundError:
                    continue    # Continue if unsuccessful
            if mde_trajectory is not None:
                break           # Break if successful
        if mde_trajectory is None:
            raise FileNotFoundError("No valid combination of topology and trajectory files found.\nPlease ensure that the files exist and/or the file name is listed in the possible topology/trajectory names (bulk.py Step 1)")
        
        # Loop through possible configuration and trajectory names (add more if necessary)
        configurations = ["out/out.gro", "out/md.gro"]
        trajectories = ["out/traj.xtc", "out/out.xtc"]

        mda_universe = None

        for gro in configurations:
            for xtc in trajectories:
                print(f'{workdir}/{gro}')
                try:
                    mda_universe = mda.Universe(f'{workdir}/{gro}', f'{workdir}/{xtc}')
                    print(f"Opened with: configuration={gro}, trajectory={xtc}")
                    break       # Break if successful
                except FileNotFoundError:
                    continue    # Continue if unsuccessful
            if mda_universe is not None:
                break           # Break if successful
        if mda_universe is None:
            raise FileNotFoundError("No valid combination of topology and trajectory files found.\nPlease ensure that the files exist and/or the file name is listed in the possible topology/trajectory names (bulk.py Step 1)")
        
        pore_inf = {}
        pore_dir = next((dir for dir in workdir.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir): # re.findall will search a string for a specific pattern and return all of the matches as a list.
                pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])
        helpers.log_info(workdir, pore_inf, f'\n@ ANALYSIS RESULTS', )  # Log extracted informati      

        pbar.update(1)          # Update the progress bar after initializing

# ===== Step 2: Calculate the mean square displacement =====
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(coords=com, num_segments=segments, pore=True)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png', msd[:,0], msd[:,1], msd[:,2], xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$", xscale='log', legend=True, handles=['Average', 'Z-Direction'])
            np.savetxt(f'{workdir}/analysis/data_files/MSD/MSD_{residue}.csv', msd, delimiter=',', header='Time / ps, Average MSD, Average Z-MSD')
        pbar.update(1)

# ===== Step 3: Calculate the RDF / q constant =====
        if q_val == 0 or overwrite:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com, segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/rdf_{residue}_{atom1}_{atom2}.png', rdf[:,0], rdf[:,1], xlabel='r / nm', ylabel='g(r)')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/rdf_{residue}_{atom1}_{atom2}.csv', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

# ===== Step 4: Calculate the ISF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png') or overwrite or q_val > 0:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.incoherent_scattering_function(coords=com, q_val=q_val, num_segments=segments, pore_diameter=pore_inf['D']/2)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png', isf[:,0], isf[:,1], isf[:,2], isf[:,3], xlabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$", xscale='log', legend=True, handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{workdir}/analysis/data_files/ISF/ISF_{residue}_{atom1}_{atom2}.csv', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

# ===== Complete! =====
        pbar.set_postfix(step="Completed Minimal Analysis!")
        pbar.update(1)
"""

cli.add_command(Run)

if __name__ == "__main__":
    cli()
