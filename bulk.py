# ----- Last updated: 07/02/2025 -----
# ----- By Troy N. Smith :-) -----

# Imports #
# ------- #
import dynamics, handle, hbonds, helpers, plotting, structural
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdevaluate as mde
import numpy as np
import os, sys
sys.stdout.reconfigure(line_buffering=True)
from tqdm import tqdm
# ------- #

def full(args_dict: dict)-> None:
    '''
    Performs full analysis on a pore simulation (including MSD, RDF, and ISF).

    Args:
        args_dict: A dictionary containing all of the imported arguments from main.py
    '''
    workdir = args_dict['workdir']
    segments = args_dict['segments']
    residue = args_dict['residue']
    atom1 = args_dict['atoms'][0]
    atom2 = args_dict['atoms'][1]
    q_val = args_dict['q_value'] if 'q_value' in args_dict.keys() else 0
    overwrite = args_dict['overwrite']

    # The tqdm package allows the developer to implement progress bars in the terminal (for those of us who are impatient :-)
    with tqdm(total=13, desc="Analysis Progress", unit="step", file=sys.stdout) as pbar:
# ===== Step 1: Initialize =====
        pbar.set_postfix(step="Initializing")           # Sets the progress bar step seen on the right hand side of the terminal interface
        # Loop through possible topology and trajectory names (add more if necessary)
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
        
        pbar.update(1)          # Update the progress bar after initializing
        
# ===== Read the box dimensions =====
        box_dims = mda_universe.dimensions
        box_len = box_dims[0] / 10

# ===== Step 2: Calculate the mean square displacement =====
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(com=com, segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png', msd[:,0], msd[:,1], xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$", xscale='log', yscale='log', legend=True, handles=['Average'])
            np.savetxt(f'{workdir}/analysis/data_files/MSD/MSD_{residue}.csv', msd, delimiter=',', header='Time / ps, Average MSD')
        pbar.update(1)

# ===== Step 3: Calculate the RDF / q constant =====
        if q_val == 0 or overwrite:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com, segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/rdf_{residue}_{atom1}_{atom2}.png', rdf[:,0], rdf[:,1], xlabel='r / nm', ylabel='g(r)')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/rdf_{residue}_{atom1}_{atom2}.csv', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

# ===== Step 4: Calculate the incoherent scattering function =====
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF/ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.isf(com=com, q_val=q_val, segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF/ISF_{residue}.png', isf[:,0], isf[:,1], xlabel=r"$t$ / ps",
                               ylabel="ISF", xscale='log', legend=True, handles=['All'])
            np.savetxt(f'{workdir}/analysis/data_files/ISF/ISF_{residue}.csv', isf, delimiter=',', header='Time / ps, All')
        pbar.update(1)
    
# ===== Step 5: Calculate rotational correlation coefficients =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational Correlation")
            rot_corr = dynamics.rotational_corr(vectors=mde_vectors, segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.png', rot_corr[:,0], rot_corr[:,1:], xlabel=r"$t$ / ps",
                               ylabel="F(t)", xscale='log', legend=True, handles=[r'$F_1(t)$', r'$F_2(t)$'], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/Rotation/Rotational_Corr_{residue}_{atom1}_{atom2}.csv', rot_corr, delimiter=',', header='Time / ps, F1, Fit1, F2, Fit2')
        pbar.update(1)

# ===== Step 6: Calculate the non-Gaussian displacement statistics =====
        if not os.path.exists(f'{workdir}/analysis/graphs/nonGauss/nonGauss_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Non-Gaussian Displacement")
            non_gauss = dynamics.non_Gauss(com=com, segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/nonGauss/nonGauss_{residue}.png', non_gauss[:,0], non_gauss[:,1:],
                               xlabel=r"$t$ / ps", ylabel="Non-Gaussian Displacement", xscale='log')
            np.savetxt(f'{workdir}/analysis/data_files/nonGauss/nonGauss_{residue}.csv', non_gauss, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)

# ===== Step 7: Calculate the translational van Hove correlations =====
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove/vanHove_transl_{residue}.png') or overwrite:
            plt.style.use('MPL_Styles/3D_Plot.mplstyle')            # Use custom styling
            pbar.set_postfix(step="Calculating Translational van Hove")
            vH_all, bins = dynamics.vanHove_translation(com=com, diameter=box_len, segments=segments, pore=False)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            complete_matrix = np.column_stack([bins, vH_all.T])

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

            plt.savefig(f'{workdir}/analysis/graphs/vanHove/vanHove_transl_{residue}.png', bbox_inches=None)
            np.savetxt(f'{workdir}/analysis/data_files/vanHove/vanHove_transl_{residue}.csv', complete_matrix, delimiter=',')
            plt.style.use('MPL_Styles/ForPapers.mplstyle')      # Go back to normal styling
        pbar.update(1)
        
# ===== Step 8: Calculate the rotational van Hove correlations =====
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational van Hove")
            vH_rot =  dynamics.vanHove_rotation(vectors=mde_vectors, segments=segments)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.png', vH_rot[vH_rot[:,0] == 0, 1], np.column_stack([vH_rot[vH_rot[:,0] == time, 2] for time in np.unique(vH_rot[:,0])[5::10]]),
                                  xlabel=r'$\varphi$', ylabel=r'$S(\varphi)$', legend=True, handles=[f'{t} ps' for t in np.unique(vH_rot[:,0])[5::10]], ncols=1)
            np.savetxt(f'{workdir}/analysis/data_files/vanHove/vanHove_rot_{residue}_{atom1}_{atom2}.csv', vH_rot, delimiter=',')
        pbar.update(1)

# ===== Step 9: Calculate the fourth-order susceptibility =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Susceptibility/Susceptibility_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating 4th Order Susceptibility")
            sus = dynamics.susceptibility(com=com, q_val=q_val)
            plotting.plot_line(f'{workdir}/analysis/graphs/Susceptibility/Susceptibility_{residue}.png', sus[:,0], sus[:,1:], xlabel=r"$t$ / ps", ylabel=r'$\chi_4 \cdot 10^{-5}$',
                               xscale='log', legend=True, handles=[r'$\chi_4$', r'$\chi_4$ Smoothed'])
            np.savetxt(f'{workdir}/analysis/data_files/Susceptibility/Susceptibility_{residue}.csv', sus, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)
        
# ===== Step 10: Calculate the intramolecular atom-wise RDF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF/MDAnal_RDF_Intra_{residue}_{atom1}_to_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Intra Atom-Wise RDFs")
            rdf_intra = structural.rdf_intra(mdaUniverse=mda_universe, residue=residue, atom1=atom1, atom2=atom2, start= 2500, stop=3000) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/MDAnal_RDF_Intra_{residue}_{atom1}_to_{atom2}.png', rdf_intra[:,0], rdf_intra[:,1], xlabel='r / nm', ylabel=r'$g_{intra}(r)$')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/MDAnal_RDF_Intra_{residue}_{atom1}_to_{atom2}.csv', rdf_intra, delimiter=',', header='Distance / nm, Mean RDF')          
        pbar.update(1)

# ===== Step 11: Calculate the intermolecular atom-wise RDF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF/RDF_Inter_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Inter Atom-Wise RDFs")
            rdf_inter = structural.rdf_inter(mdaUniverse=mda_universe, residue=residue, atom1=atom1, atom2=atom2, start=2500, stop=3000) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/RDF_Inter_{residue}_{atom1}_{atom2}.png', rdf_inter[:,0], rdf_inter[:,1:], xlabel='r / nm', ylabel=r'$g_{inter}(r)$',
                               legend=True, handles=[f'{residue}:{atom1} to {residue}:{atom1}', f'{residue}:{atom1} to {residue}:{atom2}', f'{residue}:{atom2} to {residue}:{atom1}', f'{residue}:{atom2} to {residue}:{atom2}'])
            np.savetxt(f'{workdir}/analysis/data_files/RDF/RDF_Intra_{residue}_{atom1}_{atom2}.csv', rdf_inter, delimiter=',', header='Distance / nm, Mean RDF')               
        pbar.update(1)

# ===== Step 12: Initialize the hydrogen bond information =====
        if not os.path.exists(f'{workdir}/analysis/data_files/HBonds/All_HBonds.csv') or overwrite:
            pbar.set_postfix(step="Initializing Hydrogen Bond Information")
            hyd_bonds, unique_pairs, unique_types = hbonds.hbonds(mdaUniverse=mda_universe)
            np.savetxt(f'{workdir}/analysis/data_files/HBonds/All_HBonds.csv', hyd_bonds, delimiter=',', fmt='%d,%d,%d,%s,%d,%d,%s,%.3f,%.2f', header='Frame,Donor_Index,Donor_ResID,Donor_Atom_Name,Acceptor_Index,Acceptor_ResID,Acceptor_Atom_Name,Distance,Angle')
            try:
                np.savetxt(f'{workdir}/analysis/data_files/HBonds/Unique_HBonds_IDs.csv', unique_pairs, delimiter=',', header='Donor_Index,Hydrogen_Index,Acceptor_Index,Num_Occurrences')
            except ValueError:
                pass
            try:
                np.savetxt(f'{workdir}/analysis/data_files/HBonds/Unique_HBonds_Types.csv', unique_types, delimiter=',', header='Frame,Donor_Index,Donor_ResID,Donor_Atom_Name,Acceptor_Index,Acceptor_ResID,Acceptor_Atom_Name,Distance,Angle')
            except ValueError:
                pass
            counts = hbonds.hbond_counts(hbonds=hyd_bonds, residue=residue)

            print("Finding Clusters...")
            for pair, count in counts.items():
                helpers.log_info(workdir, f'{pair}: {count}')
            _, stats = hbonds.find_clusters(hbonds=hyd_bonds, filename=f'{workdir}/analysis/data_files/HBonds/Clusters_{residue}.txt')

            if not stats == None:
                helpers.log_info(workdir, stats)
                with open(f'{workdir}/analysis/data_files/HBonds/Clusters_Summary_{residue}.txt', 'w') as f:
                    f.write(stats)
                    f.close()
            
        pbar.update(1)

# ===== Step 13: Generate end to end distance =====
        if not os.path.exists(f'{workdir}/analysis/graphs/End_to_end/End_to_end_{residue}') or overwrite:
            pbar.set_postfix(step="Calculating end-to-end distances")
            structural.end_to_end(mda_universe=mda_universe, residue=residue, workdir=workdir)
        pbar.update(1)
        
# ===== Complete! =====
        pbar.set_postfix(step="Completed Full Analysis!")
        pbar.update(1)