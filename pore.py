# Imports #
# ------- #
import dynamics, handle, hbonds, helpers, plotting, structural, pore_additional
import MDAnalysis as mda
import mdevaluate as mde
import numpy as np
import os, re
from tqdm import tqdm
import warnings
# ------- #

# Suppress specific warnings:
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def minimal(args_dict: dict,
            ver_dir: str = None,
            )-> None:
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
    overwrite = True if 'overwrite' in args_dict.keys() else False

    if ver_dir is not None:
        workdir = ver_dir
        
    with tqdm(total=8, desc="Analysis Progress", unit="step") as pbar:
        
        # Step 1: Change working directory and open trajectory
        pbar.set_postfix(step="Initializing")
        mde_trajectory = mde.open(directory=workdir,
                                  topology='run.tpr',
                                  trajectory='out/traj.xtc',
                                  nojump=True)
        pbar.update(1)  # Update the progress bar after initializing
        
        # Step 2: Handle vectors
        pbar.set_postfix(step="Handling Vectors")
        mde_vectors = handle.handle_vectors(mde_trajectory=mde_trajectory,
                                            residue=residue,
                                            atom1=atom1,
                                            atom2=atom2)
        pbar.update(1)
        
        # Step 3: Extract pore simulation details from directory name
        pbar.set_postfix(step="Logging Pore Information")
        pore_inf = {}
        pore_dir = next((dir for dir in workdir.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir):
                pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])
        helpers.log_info(workdir, pore_inf, f'\n@ ANALYSIS RESULTS', )  # Log extracted information
        pbar.update(1)
        
        # Step 4: Gather centers of masses
        pbar.set_postfix(step="Gathering Centers of Masses")
        com = helpers.center_of_masses(trajectory=mde_trajectory,
                                       residue=residue)
        pbar.update(1)

        # Step 5: Calculate the mean square displacement
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(com=com,
                                       segments=100)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD_{residue}.png', 
                               msd[:,0], 
                               msd[:,1],
                               msd[:,2],
                               xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log', 
                               legend=True,
                               handles=['Average', 'Z-Direction'])
            np.savetxt(f'{workdir}/analysis/xvg_files/MSD_{residue}.xvg', msd, delimiter=',', header='Time / ps, Average MSD, Average Z-MSD')
        pbar.update(1)

        # Step 6: Calculate the RDF / q constant
        if q_val == 0:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com,
                                                   segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/rdf_{residue}_{atom1}_{atom2}.png', 
                               rdf[:,0], 
                               rdf[:,1],
                               xlabel='r / nm',
                               ylabel='g(r)'
                               )
            np.savetxt(f'{workdir}/analysis/xvg_files/rdf_{residue}_{atom1}_{atom2}.xvg', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

        # Step 7: Calculate the ISF
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.isf(com=com,
                               q_val=q_val,
                               segments=segments,
                               radius=pore_inf['D']/2)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF_{residue}.png', 
                               isf[:,0], 
                               isf[:,1],
                               isf[:,2],
                               isf[:,3],
                               xlabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log',
                               legend=True,
                               handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{workdir}/analysis/xvg_files/ISF_{residue}.xvg', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

        # Complete!
        pbar.set_postfix(step="Completed Minimal Analysis!")
        pbar.update(1)

def full(args_dict: dict,
         ver_dir: str = None,
         )-> None:
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

    if ver_dir is not None:
        workdir = ver_dir

    with tqdm(total=20, desc="Analysis Progress", unit="step") as pbar: # The tqdm package allows the developer to implement progress bars
        
        print(overwrite)
        # Step 1: Change working directory and open trajectory / universe
        pbar.set_postfix(step="Initializing") # Sets the progress bar step seen on the right hand side of the terminal interface
        mde_trajectory = mde.open(directory=workdir,
                                  topology='run.tpr',
                                  trajectory='out/traj.xtc',
                                  nojump=True)
        mda_universe = mda.Universe(f'{workdir}/out/out.gro', 
                                    f'{workdir}/out/traj.xtc')
        pbar.update(1)  # Update the progress bar after initializing
        
        # Step 2: Handle vectors
        pbar.set_postfix(step="Handling Vectors")
        mde_vectors = handle.handle_vectors(mde_trajectory=mde_trajectory,
                                            residue=residue,
                                            atom1=atom1,
                                            atom2=atom2)
        pbar.update(1)
        
        # Step 3: Extract pore simulation details from directory name
        pbar.set_postfix(step="Logging Pore Information")
        pore_inf = {}
        pore_dir = next((dir for dir in workdir.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir):
                pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])
        helpers.log_info(workdir, pore_inf, f'\n@ ANALYSIS RESULTS', )  # Log extracted information
        pbar.update(1)
        
        # Step 4: Gather centers of masses
        pbar.set_postfix(step="Gathering Centers of Masses")
        com = helpers.center_of_masses(trajectory=mde_trajectory,
                                       residue=residue)
        pbar.update(1)

        # Step 5: Calculate the mean square displacement
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(com=com,
                                       segments=100)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD_{residue}.png', 
                               msd[:,0], 
                               msd[:,1],
                               msd[:,2],
                               xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log',
                               yscale='log', 
                               legend=True,
                               handles=['Average', 'Z-Direction'])
            np.savetxt(f'{workdir}/analysis/xvg_files/MSD_{residue}.xvg', msd, delimiter=',', header='Time / ps, Average MSD, Average Z-MSD')
        pbar.update(1)

        # Step 6: Calculate the resolved mean square displacement
        if not os.path.exists(f'{workdir}/analysis/graphs/Resolved_MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Resolved MSD")
            msd, r = dynamics.resolved_msd(com=com,
                                           diameter=pore_inf['D'],
                                           segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/Resolved_MSD_{residue}.png', 
                               msd[:,0], 
                               msd[:,1:],
                               xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log',
                               yscale='log', 
                               legend=True,
                               handles=[f'{bin:.2f}' for bin in r],
                               ncols=2)
            np.savetxt(f'{workdir}/analysis/xvg_files/Resolved_MSD_{residue}.xvg', msd, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in r]))
        pbar.update(1)

        # Step 7: Calculate the RDF / q constant
        if q_val == 0:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com,
                                                   segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/rdf_{residue}_{atom1}_{atom2}.png', 
                               rdf[:,0], 
                               rdf[:,1],
                               xlabel='r / nm',
                               ylabel='g(r)'
                               )
            np.savetxt(f'{workdir}/analysis/xvg_files/rdf_{residue}_{atom1}_{atom2}.xvg', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

        # Step 8: Calculate the incoherent scattering function
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.isf(com=com,
                               q_val=q_val,
                               segments=segments,
                               radius=pore_inf['D']/2)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF_{residue}.png', 
                               isf[:,0], 
                               isf[:,1:],
                               xlabel=r"$t$ / ps",
                               ylabel="ISF",
                               xscale='log',
                               legend=True,
                               handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{workdir}/analysis/xvg_files/ISF_{residue}.xvg', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

        # Step 9: Calculate the resolved incoherent scattering function
        if not os.path.exists(f'{workdir}/analysis/graphs/Resolved_ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Resolved ISF")
            isf, r = dynamics.resolved_isf(com=com,
                                           q_val=q_val,
                                           diameter=pore_inf['D'],
                                           segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/Resolved_ISF_{residue}.png', 
                               isf[:,0], 
                               isf[:,1:],
                               xlabel=r"$t$ / ps",
                               ylabel="ISF",
                               xscale='log',
                               legend=True,
                               handles=[f'{bin:.2f}' for bin in r],
                               ncols=2)
            np.savetxt(f'{workdir}/analysis/xvg_files/Resolved_ISF_{residue}.xvg', isf, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in r]))
        pbar.update(1)

        # Step 10: Calculate rotational correlation coefficients
        if not os.path.exists(f'{workdir}/analysis/graphs/Rotational_Corr_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational Correlation")
            rot_corr = dynamics.rotational_corr(vectors = mde_vectors, 
                                                segments = segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/Rotational_Corr_{residue}_{atom1}_{atom2}.png', 
                               rot_corr[:,0], 
                               rot_corr[:,1:],
                               xlabel=r"$t$ / ps",
                               ylabel="F(t)",
                               xscale='log',
                               legend=True,
                               handles=[r'$F_1(t)$', r'$Fit_1(t)$', r'$F_2(t)$', r'$Fit_2(t)$'],
                               ncols=1)
            np.savetxt(f'{workdir}/analysis/xvg_files/Rotational_Corr_{residue}_{atom1}_{atom2}.xvg', rot_corr, delimiter=',', header='Time / ps, F1, Fit1, F2, Fit2')
        pbar.update(1)

        # Step 11: Calculate the non-Gaussian displacement statistics
        if not os.path.exists(f'{workdir}/analysis/graphs/nonGauss_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Non-Gaussian Displacement")
            non_gauss = dynamics.non_Gauss(com=com,
                                           segments=10)
            plotting.plot_line(f'{workdir}/analysis/graphs/nonGauss_{residue}.png', 
                               non_gauss[:,0], 
                               non_gauss[:,1:],
                               xlabel=r"$t$ / ps",
                               ylabel="Non-Gaussian Displacement",
                               xscale='log')
            np.savetxt(f'{workdir}/analysis/xvg_files/nonGauss_{residue}.xvg', non_gauss, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)

        # Step 12: Calculate the translational van Hove correlations
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove_transl_center_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Translational van Hove")
            vH_wall, vH_center, bins = dynamics.vanHove_translation(com=com,
                                                                    diameter=pore_inf['D'],
                                                                    segments=segments)
            plotting.plot_line(f'{workdir}/analysis/graphs/vanHove_transl_wall_{residue}.png', 
                               vH_wall[:,0], 
                               vH_wall[:,2:],
                               xlabel="t / ps",
                               ylabel="Dist(t)",
                               legend=True,
                               handles=[f'{bin:.2f}' for bin in bins[1:]],
                               ncols=2)
            np.savetxt(f'{workdir}/analysis/xvg_files/vanHove_transl_wall_{residue}.xvg', vH_wall, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in bins]))
            plotting.plot_line(f'{workdir}/analysis/graphs/vanHove_transl_center_{residue}.png', 
                               vH_center[:,0], 
                               vH_center[:,2:],
                               xlabel=r"t / $10^{-6} \cdot$ ps",
                               ylabel="Dist(t)",
                               legend=True,
                               handles=[f'{bin:.2f}' for bin in bins[1:]],
                               ncols=2)
            np.savetxt(f'{workdir}/analysis/xvg_files/vanHove_transl_center_{residue}.xvg', vH_center, delimiter=',', header='Time / ps,'.join([f'{bin:.2f},' for bin in bins]))
        pbar.update(1)

        # Step 13: Calculate the rotational van Hove correlations
        if not os.path.exists(f'{workdir}/analysis/graphs/vanHove_rot_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Rotational van Hove")
            vH_rot =  dynamics.vanHove_rotation(vectors=mde_vectors,
                                                segments=segments)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/vanHove_rot_{residue}_{atom1}_{atom2}.png', 
                                  vH_rot[vH_rot[:,0] == 0, 1], 
                                  np.column_stack([vH_rot[vH_rot[:,0] == time, 2] for time in np.unique(vH_rot[:,0])[5::10]]),
                                  xlabel=r'$\varphi$',
                                  ylabel=r'$S(\varphi)$',
                                  legend=True,
                                  handles=[f'{time} ps' for time in np.unique(vH_rot[:,0])[5::10]],
                                  ncols=1)
            np.savetxt(f'{workdir}/analysis/xvg_files/vanHove_rot_{residue}_{atom1}_{atom2}.xvg', vH_rot, delimiter=',')
        pbar.update(1)

        # Step 14: Calculate the fourth-order susceptibility
        if not os.path.exists(f'{workdir}/analysis/graphs/Susceptibility_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating 4th Order Susceptibility")
            sus = dynamics.susceptibility(com=com,
                                          q_val=q_val)
            plotting.plot_line(f'{workdir}/analysis/graphs/Susceptibility_{residue}.png', 
                               sus[:,0], 
                               sus[:,1:],
                               xlabel=r"$t$ / ps",
                               ylabel=r'$\chi_4 \cdot 10^{-5}$',
                               xscale='log',
                               legend=True,
                               handles=[r'$\chi_4$', r'$\chi_4$ Smoothed'])
            np.savetxt(f'{workdir}/analysis/xvg_files/Susceptibility_{residue}.xvg', sus, delimiter=',', header='Time / ps, Displacement')
        pbar.update(1)
        
        # Step 15: Calculate the Z-axis radial alignments
        if not os.path.exists(f'{workdir}/analysis/graphs/Z_align_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Z-Axis Alignment")
            Z_align =  structural.z_align(vectors=mde_vectors,
                                          segments=10)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/Z_align_{residue}_{atom1}_{atom2}.png', 
                                  Z_align[Z_align[:,0] == 0, 1], 
                                  np.column_stack([Z_align[Z_align[:,0] == time, 2] for time in np.unique(Z_align[:,0])[::20]]),
                                  xlabel=r'$\varphi$',
                                  ylabel=r'$S(\varphi)$',
                                  legend=True,
                                  handles=[f'{time} ps' for time in np.unique(Z_align[:,0])[::20]],
                                  ncols=1)
            np.savetxt(f'{workdir}/analysis/xvg_files/Z_align_{residue}_{atom1}_{atom2}.xvg', Z_align, delimiter=',', header=r'Time / ps, $\varphi$, S($\varphi$)')
        pbar.update(1)
        
        # Step 16: Calculate the Z-axis radial positions
        if not os.path.exists(f'{workdir}/analysis/graphs/Z_histogram_{residue}_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Z-Axis Radial Positions")
            Z_histo =  structural.z_histogram(vectors=mde_vectors,
                                              segments=10)
            plotting.plot_scatter(f'{workdir}/analysis/graphs/Z_histogram_{residue}_{atom1}_{atom2}.png', 
                                  Z_histo[Z_histo[:,0] == 0, 1], 
                                  np.column_stack([Z_histo[Z_histo[:,0] == time, 2] for time in np.unique(Z_histo[:,0])[::20]]),
                                  xlabel=r'$\varphi$',
                                  ylabel=r'$S(\varphi)$',
                                  legend=True,
                                  handles=[f'{time} ps' for time in np.unique(Z_histo[:,0])[::20]],
                                  ncols=1)
            np.savetxt(f'{workdir}/analysis/xvg_files/Z_histogram_{residue}_{atom1}_{atom2}.xvg', Z_histo, delimiter=',', header=r'Time / ps, $\varphi$, dist($\varphi$)')
        pbar.update(1)
        
        # Step 17: Calculate the intramolecular atom-wise RDF
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF_Intra_{atom1}_to_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Intra Atom-Wise RDFs")
            rdf_intra = structural.rdf_intra(mdaUniverse=mda_universe,
                                             residue=residue,
                                             atom1=atom1,
                                             atom2=atom2,
                                             stop=5000) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF_Intra_{atom1}_to_{atom2}.png', 
                               rdf_intra[:,0], 
                               rdf_intra[:,1],
                               xlabel='r / nm',
                               ylabel=r'$g_{intra}(r)$')
            np.savetxt(f'{workdir}/analysis/xvg_files/RDF_Intra_{atom1}_to_{atom2}.xvg', rdf_intra, delimiter=',', header='Distance / nm, Mean RDF')          
        pbar.update(1)
        
        # Step 18: Calculate the intermolecular atom-wise RDF
        if not os.path.exists(f'{workdir}/analysis/graphs/RDF_Inter_{atom1}_{atom2}.png') or overwrite:
            pbar.set_postfix(step="Calculating Inter Atom-Wise RDFs")
            rdf_inter = structural.rdf_inter(mdaUniverse=mda_universe,
                                             residue=residue,
                                             atom1=atom1,
                                             atom2=atom2,
                                             stop=5000) 
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF_Inter_{atom1}_{atom2}.png', 
                               rdf_inter[:,0], 
                               rdf_inter[:,1:],
                               xlabel='r / nm',
                               ylabel=r'$g_{inter}(r)$',
                               legend=True,
                               handles=[f'{residue}:{atom1} to {residue}:{atom1}',
                                        f'{residue}:{atom1} to {residue}:{atom2}',
                                        f'{residue}:{atom2} to {residue}:{atom1}',
                                        f'{residue}:{atom2} to {residue}:{atom2}'])
            np.savetxt(f'{workdir}/analysis/xvg_files/RDF_Intra_{atom1}_{atom2}.xvg', rdf_inter, delimiter=',', header='Distance / nm, Mean RDF')               
        pbar.update(1)
        
        # Step 19: Calculate the radial spatial density function(s)
        if not os.path.exists(f'{workdir}/analysis/graphs/Spatial_Density_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating Radial Spatial Density Function")
            results = []
            residues = [residue, residue, 'LNK', 'ETH', 'VAN', 'VAN', 'VAN']
            names = [atom1, atom2, 'NL', 'OEE', 'NV', 'OVE', 'OVH']
            with tqdm(total=len(residues), desc="Spatial Density Progress", unit="Pair") as bar:
                for res, atom in zip(residues, names):
                    if residue in [f'{residue.resname}' for residue in mda_universe.residues]:
                        rSDF = structural.radial_density(mde_trajectory=mde_trajectory,
                                                         residue=res,
                                                         atom=atom,
                                                         diameter=pore_inf['D'])
                        results.append(rSDF[:,1])
                    bar.update(1)
            results = [rSDF[:,0]] + results
            rSDF = np.column_stack(results)
            plotting.plot_line(f'{workdir}/analysis/graphs/Spatial_Density_{residue}.png', 
                               rSDF[:,0], 
                               rSDF[:,1:],
                               xlabel="r / nm",
                               ylabel=r"Number Density / nm$^{-3}$",
                               legend=True,
                               handles=[f'{res}:{atom}' for res, atom in zip(residues, names)])
            np.savetxt(f'{workdir}/analysis/xvg_files/Spatial_Density_{residue}.xvg', rSDF, delimiter=',', header='r / nm, ' + ''.join([f'{res}:{atom}, ' for res, atom in zip(residues, names)])) 
        pbar.update(1)

        # Step 20: Initialize the hydrogen bond information
        if not os.path.exists(f'debug.png') or overwrite:
            pbar.set_postfix(step="Initializing Hydrogen Bond Information")
            hyd_bonds = hbonds.hbonds(mdaUniverse=mda_universe,
                                      donors_sel=f'type O and resname {residue}',
                                      hydrogens_sel=f'type H and resname {residue}',
                                      acceptors_sel=f'type O and resname {residue}',
                                      start=0,
                                      stop=5000)
            counts = hbonds.hbond_counts(hbonds=hyd_bonds)
            for pair, count in counts.items():
                helpers.log_info(workdir,
                                 f'{pair}: {count}')
            
            clusters, stats = hbonds.find_clusters(hbonds=hyd_bonds,
                                                   filename=f'{workdir}/analysis/xvg_files/Clusters_{residue}.txt')

            if not stats == None:
                helpers.log_info(workdir,
                                 stats)

        # Complete!
        pbar.set_postfix(step="Completed Full Analysis!")
        pbar.update(1)

        additional = False
        if additional:
            print('Beginning additional analysis!')
            pore_additional.additional(args_dict=args_dict)