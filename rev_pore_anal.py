# Metadata
__author__ = 'Hoffmann, M.M.'
__rev_by__ = 'Smith, T.N.'
__last_rev__ = '20 Dec 2024'

# Imports
import mdevaluate as mde
import MDAnalysis as mda
import argparse
import os
import re
import datetime
from functools import partial
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib import cm
from scipy.optimize import curve_fit
from math import gamma
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba
from itertools import product

# Suppress annoying messages
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import logging
logging.basicConfig(level=logging.WARNING)
mda.logger.setLevel(logging.WARNING)
mde.logger.setLevel(logging.WARNING)

#plt.style.use('/home/mmh/BrockportMDFiles/mpl_styles/ForPapers.mplstyle')

# Benefits of using a class function involve 1) __init__ does not have to be called when using the function, so processes are automatically performed
#                                            2) A self object  is introduced which is passed to all functions within the class; avoids repetition
#                                            3) functions can be called out of order

class Global:
    def __init__(self):
        """
        Initialize the Global class with command-line argument parsing,
        simulation data setup, and mode determination (interactive/non-interactive).
        """
        # Argument parser setup
        parser = argparse.ArgumentParser(description="Standard analysis of pore simulations")
        parser.add_argument('-w', '--workdir', type=str, help="Specific simulation working directory", required=False)
        parser.add_argument('-m', '--minimal', help="Select True if minimal analysis only, else False", required=False, action='store_true')
        parser.add_argument('-s', '--segments', type=int, help="Number of segments", required=False)
        parser.add_argument('-r', '--residue', type=str, help="Index of reference residue (must match topology)", required=False)
        parser.add_argument('-a', '--atoms', nargs=2, type=str, help="Indices of reference atoms (two required, must match topology)", required=False)
        parser.add_argument('-q', '--q_value', type=float, help='q-value as determined by previous RDF calculations', required=False)
        parser.add_argument('-ow', '--overwrite', type=bool, help="your responsibilty to change filenames so they don't get overwritten", required=False)
        args = parser.parse_args()

        print('Initializing analysis')

        # Initialize working directory and simulation data. os.getcwd() extracts the current working directory
        if args.workdir:
            self.cwd = args.workdir
        else:
            self.cwd = os.getcwd()

        # Build a string with the name of the parent work directory (for access to files in 5 or 6 later on)
        self.pwd = '/'.join(self.cwd.split('/')[:-1])

        self.mda_universe = mda.Universe(f'{self.cwd}/out/out.gro', f'{self.cwd}/out/traj.xtc')
        self.mde_trajectory = mde.open(directory=self.cwd,
                                       topology='run.tpr',
                                       trajectory='out/traj.xtc',
                                       nojump=True)

        # Collect residue information. Coding info "key=lambda some variable name:" is a construct that allows for selecting which columns is used to sort by
        self.residues = sorted(
            {residue.resname for residue in self.mda_universe.residues},
            key=lambda name: name[0]
        )

        # Initialize choices and pore information, both are of type dictionary
        self.choices, self.pore_inf = {}, {}

        # Extract pore simulation details from directory name, here the diameter, length and wall thickness of the pore, and the decorations which are all part of the pore file name
        # Coding info: the next function keeps going in this through each "/" of directory path until requirement, here "pore_D" is met or found 
        pore_dir = next((dir for dir in self.cwd.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir):
                self.pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])

        self.folder = self.cwd.split('/')[-1]

        # Determine mode (non-interactive or interactive)
        if all([args.workdir, args.segments, args.residue, args.atoms]):
            self.handle_args(args)
        else:
            self.handle_interactive()

        os.system(f'mkdir -p {self.cwd}/analysis/graphs \
                  mkdir -p {self.cwd}/analysis/xvg_files')

        if '5_nvt' in self.folder:
            self.minimal_procedure()

            if not self.choices['Minimal Analysis']:
                self.full_procedure()
        
        if '6_nvt' in self.folder:
            self.additional_procedure()

        if '7_nvt' in self.folder:
            self.hbond_procedure()

    def handle_args(self, args):
        """
        Handle non-interactive mode by processing command-line arguments.
        """
        # Store method choices
        self.choices['Minimal Analysis'] = args.minimal
        print(self.choices['Minimal Analysis'])
        self.choices['Segments'] = args.segments

        # Map residue index to residue name
        self.choices['Residue'] = args.residue

        self.choices['Overwrite'] = True if args.overwrite else False

        self.q_const = args.q_value if args.q_value else 0

        # Process atom references
        self.handle_atoms(args)

    def handle_interactive(self):
        """
        Handle interactive mode, prompting the user for input.
        """
        print("Interactive mode activated.")
        while True:
            try:
                # Select method
                print("Please select a method:")
                for i, choice in enumerate(['Minimal', 'Full'], start=1):
                    print(f'{i}. {choice}')
                user_input = int(input('Enter the method: '))
                self.choices['Minimal Analysis'] = True if user_input == 1 else False
# Coding info: user_input in [1,2] checks if user inputted 1 or 2 as the option, returns true or false which is stored in the dictionaly "self.choices" under each key 'Minimal Analysis' and 'Average Analysis'
                # Enter the number of segments
                self.choices['Segments'] = int(input('Enter the number of segments, typical value is 100: '))

                # Choose residue
                print("Please choose a residue from the following options:")
                for i, choice in enumerate(self.residues, start=1):
                    print(f'{i}. {choice}')
                residue_input = int(input('Enter the reference residue: '))
                self.choices['Residue'] = self.residues[residue_input - 1]
                self.choices['Overwrite'] = True if int(input('Overwrite? Enter 1 for yes, 2 for no: ')) == 1 else False

                # Process atom references
                self.handle_atoms()

                break
            except ValueError:
                print('Invalid input. Please try again.')

    def handle_atoms(self, *args):
        """
        Process and save reference atom choices.
        """
        print('Importing atom information')

        if args and hasattr(args[0], 'atoms'):
            # Non-interactive mode: Extract atoms from arguments
            self.mde_indices = {}
            atoms = args[0].atoms
            for i, atom in enumerate(atoms, start=1):
                print(i,atom)
                self.choices[f'Atom_{i}'] = atom
                # Another dictionary to store the two atoms for defining a vector between these two atoms later
                self.mde_indices[f'Atom_{i}'] = self.mde_trajectory.subset(atom_name=atom, residue_name=self.choices['Residue']).atom_subset.indices[0]
            self.mde_vectors = mde.coordinates.vectors(self.mde_trajectory,
                                                        atom_indices_a=self.mde_indices['Atom_1'],
                                                        atom_indices_b=self.mde_indices['Atom_2'],
                                                        normed=True
                                                        )
        else:
            # Interactive mode: Prompt user for atom choices
            self.atoms = sorted(
                {atom.name for atom in self.mda_universe.select_atoms(f"resname {self.choices['Residue']}")},
                key=lambda name: name[-1]
            )

            for i, atom_name in enumerate(self.atoms, start=1):
                print(f'{i}. {atom_name}')

            for i in range(2):
                choice = int(input(f'Enter reference atom {i+1}: '))
                self.choices[f'Atom_{i+1}'] = self.atoms[choice - 1]

            self.mde_indices = {
                'Atom_1': self.mde_trajectory.subset(atom_name=self.choices['Atom_1'], residue_name=self.choices['Residue']).atom_subset.indices[0],
                'Atom_2': self.mde_trajectory.subset(atom_name=self.choices['Atom_2'], residue_name=self.choices['Residue']).atom_subset.indices[0],
            }
            self.mde_vectors = mde.coordinates.vectors(self.mde_trajectory,
                                                       atom_indices_a=self.mde_indices['Atom_1'],
                                                       atom_indices_b=self.mde_indices['Atom_2'],
                                                       normed=True
                                                       )

        # Save choices to a summary file
        os.makedirs(f'{self.cwd}/analysis', exist_ok=True)
        with open(f'{self.cwd}/analysis/summary.txt', 'a+') as summary_file:
            summary_file.write('@ REFERENCE INFORMATION\n')
            summary_file.write(f'{self.cwd}\n')
            now = datetime.datetime.now()
            summary_file.write(f'Date/Time: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
            for key, value in {**self.choices, **self.pore_inf}.items():
                summary_file.write(f'{key}: {value}\n')

            print(f'Exported initialization information to: {summary_file.name}')  

    def minimal_procedure(self):
            
        with open(f'{self.cwd}/analysis/summary.txt', 'a') as summary_file:
            summary_file.write('\n @ ANALYTICAL RESULTS\n')

        print("Gathering centers of masses")
        self.com = self.Analysis.center_of_masses(trajectory=self.mde_trajectory,
                                                resname=self.choices['Residue'])
        
        if not os.path.exists(f'{self.cwd}/analysis/graphs/MSD_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating average mean square displacement")
            self.Analysis.msd_average(workdir=self.cwd,
                                          com=self.com,
                                          resname=self.choices['Residue'],
                                          segments=self.choices['Segments'])
        if self.q_const == 0:     
            print("Calculating RDF")
            self.q_const, self.g_max = self.Analysis.rdf_com(self=self)

        with open(f'{self.cwd}/analysis/summary.txt', 'a') as summary_file:
            summary_file.write(f'q constant : {self.q_const}\n')
            try:
                summary_file.write(f'g max : {self.g_max}\n')
            except AttributeError:
                pass
        
        if not os.path.exists(f'{self.cwd}/analysis/graphs/ISF_com_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating ISF")
            self.Analysis.isf(self=self)

    def full_procedure(self):
# msd resolve, isf radial dissolved, F1 and F2, non-Gauss, van Hoove trans and rotation, susceptibility, z-align, z-histogram, atoms radial binning 
        if not os.path.exists(f'{self.cwd}/analysis/graphs/MSD_resolved_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating radially resolved mean square displacement")
            self.Analysis.msd_resolved(workdir=self.cwd,
                                          com=self.com,
                                          resname=self.choices['Residue'],
                                          diameter=self.pore_inf['D'],
                                          segments=self.choices['Segments'])

        if not os.path.exists(f'{self.cwd}/analysis/graphs/ISF_resolved_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating radially resolved intermediate incoherent scattering function")
            self.Analysis.ISF_radial(workdir=self.cwd, 
                                        com=self.com, 
                                        q_const=self.q_const,
                                        diameter=self.pore_inf['D'], 
                                        resname=self.choices['Residue'], 
                                        segments=self.choices['Segments'])

        if not os.path.exists(f'{self.cwd}/analysis/graphs/F1_F2-fit{self.choices['Residue']}_{self.choices['Atom_1']}_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating rotational correlation function time constants F1 and F2")
            self.Analysis.F1_F2(workdir=self.cwd, 
                                        vector=self.mde_vectors, 
                                        resname=self.choices['Residue'],
                                        atom1=self.choices['Atom_1'],
                                        atom2=self.choices['Atom_2'], 
                                        segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/nonGauss_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Inspecting non-Gaussian displacement statistics")
            self.Analysis.non_Gauss(workdir=self.cwd,
                                    com=self.com,
                                    resname=self.choices['Residue'],
                                    segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/vanHove_transl_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating van Hove correlation for translational motion")
            self.Analysis.vanHove_translation(workdir=self.cwd,
                                              com=self.com,
                                              diameter=self.pore_inf['D'],
                                              resname=self.choices['Residue'],
                                              segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/vanHove_rotation_{self.choices['Residue']}_{self.choices['Atom_1']}_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating van Hove correlation for rotational motion")
            self.Analysis.vanHove_rotation(workdir=self.cwd,
                                           trajectory=self.mde_trajectory,
                                           vectors=self.mde_vectors,
                                           atom1=self.choices['Atom_1'],
                                           atom2=self.choices['Atom_2'],
                                           resname=self.choices['Residue'],
                                           segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/Susceptibility_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating susceptibility")
            self.Analysis.susceptibility(workdir=self.cwd,
                                         com=self.com,
                                         q_const=self.q_const,
                                         resname=self.choices['Residue'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/z_align_{self.choices['Residue']}_{self.choices['Atom_1']}_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating Z alignment")
            self.Analysis.z_align(workdir=self.cwd,
                                  trajectory=self.mde_trajectory,
                                  vectors=self.mde_vectors,
                                  atom1=self.choices['Atom_1'],
                                  atom2=self.choices['Atom_2'],
                                           resname=self.choices['Residue'],
                                           segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/z_histogram_{self.choices['Residue']}_{self.choices['Atom_1']}_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating Z histogram")
            self.Analysis.z_histogram(workdir=self.cwd,
                                  trajectory=self.mde_trajectory,
                                  vectors=self.mde_vectors,
                                  atom1=self.choices['Atom_1'],
                                  atom2=self.choices['Atom_2'],
                                  resname=self.choices['Residue'],
                                  segments=self.choices['Segments'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/RDF_Intra_{self.choices['Atom_1']}_to_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating atom-atom intra RDF")            
            self.Analysis.rdf_a2a_intra(workdir=self.cwd,
                                    mdaUniverse=self.mda_universe,
                                    residue=self.choices['Residue'],
                                    atom1=self.choices['Atom_1'],
                                    atom2=self.choices['Atom_2'])
            

        if not os.path.exists(f'{self.cwd}/analysis/graphs/RDF_Inter_{self.choices['Atom_1']}_to_{self.choices['Atom_2']}.png') or self.choices['Overwrite']:
            print("Calculating atom-atom inter RDF")            
            self.Analysis.rdf_a2a_inter(workdir=self.cwd,
                                    mdaUniverse=self.mda_universe,
                                    residue=self.choices['Residue'],
                                    atom1=self.choices['Atom_1'],
                                    atom2=self.choices['Atom_2'])
            
        if not os.path.exists(f'{self.cwd}/analysis/graphs/Spatial_Density_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating radial density distribution")
            plt.clf()
            data = pd.DataFrame({})
            residues = [self.choices['Residue'], 'LNK', 'ETH', 'VAN', 'VAN', 'VAN']
            names = [self.choices['Atom_1'], 'NL', 'OEE', 'NV', 'OVE', 'OVH']
            for residue, name in zip(residues, names):
                if residue in self.residues:
                        bins, pos = self.Analysis.radial_density(workdir=self.cwd,
                                                    trajectory=self.mde_trajectory,
                                                    atom1=name,
                                                    residue=residue,
                                                    diameter=self.pore_inf['D'],
                                                    length=self.pore_inf['L'],
                                                    segments=self.choices['Segments'])
                        
                        plt.plot(bins, pos, label=f'{residue} {name}')
                        data[f'{residue} {name}'] = pos
            
            plt.xlabel(r"$r$ / nm")
            plt.ylabel(r"Number Density / nm$^{-3}$")

            plt.tight_layout()
            plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))

            plt.savefig(f'{self.cwd}/analysis/graphs/Spatial_Density_{self.choices['Residue']}.png')
            data[f'Bins'] = bins
            data.to_csv(f'{self.cwd}/analysis/xvg_files/Spatial_Density_{self.choices['Residue']}.xvg', sep=' ', index=False)
    
    def additional_procedure(self):
        """
        For analysis of the additional (step 6) production run.
        1. ISF
        2. Hydrogen bond counts
        3. Hydrogen bond clusters
        4. Hydrogen bond lifetimes
        5. 
        """

        print("Gathering centers of masses")
        self.com = self.Analysis.center_of_masses(trajectory=self.mde_trajectory,
                                                resname=self.choices['Residue'])
        
        if self.q_const == 0:     
            print("Calculating RDF")
            self.q_const, self.g_max = self.Analysis.rdf_com(self=self)

        with open(f'{self.cwd}/analysis/summary.txt', 'a') as summary_file:
            summary_file.write(f'q constant : {self.q_const}\n')
            try:
                summary_file.write(f'g max : {self.g_max}\n')
            except AttributeError:
                pass

        if not os.path.exists(f'{self.cwd}/analysis/graphs/ISF_com_{self.choices['Residue']}.png') or self.choices['Overwrite']:
            print("Calculating ISF")
            self.Analysis.isf(self=self)


            try:
                ISF_5 = pd.read_csv(f'{self.pwd}/5_nvt_prod_system/analysis/xvg_files/isf_{self.choices['Residue']}.xvg', sep=' ', index_col=False)
                ISF_6 = pd.read_csv(f'{self.cwd}/analysis/xvg_files/isf_{self.choices['Residue']}.xvg', sep=' ', index_col=False)

                ISF_6_max = ISF_6.iloc[-1,0]

                ISF_5_filtered = ISF_5[ISF_5.iloc[:,0] > ISF_6_max]

                ISF_merged = pd.concat([ISF_6,ISF_5_filtered], ignore_index=True)

                plt.clf()

                plt.plot(ISF_merged.iloc[1:,0], ISF_merged.iloc[1:,1], label='All')
                plt.plot(ISF_merged.iloc[1:,0], ISF_merged.iloc[1:,2], label='Wall')
                plt.plot(ISF_merged.iloc[1:,0], ISF_merged.iloc[1:,3], label='Center')
                plt.xscale('log')
                plt.xlabel(r"$t$ / ps")
                plt.ylabel(r"ISF")
                plt.axvspan(0, ISF_6_max, color='gray', alpha=0.2)
                plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))

                plt.savefig(self.cwd + "/analysis/graphs/ISF_merged_" + self.choices['Residue'])

            except FileNotFoundError:
                print("File does not exist!")
                pass
    
    def hbond_procedure(self):
        print('Calculating hydrogen bond counts')
        Global.Analysis.hbond_counts(workdir=self.cwd,
                                     mdaUniverse=self.mda_universe,
                                     resname=self.choices['Residue'],
                                     atom=self.choices['Atom_1'])
        
        Global.Analysis.hbond_counts(workdir=self.cwd,
                                     mdaUniverse=self.mda_universe,
                                     resname=self.choices['Residue'],
                                     atom=self.choices['Atom_2'])

    class Analysis:
        def center_of_masses(trajectory, resname):
            @mde.coordinates.map_coordinates
            def center_of_masses(coordinates, atoms, shear: bool = False):
                res_ids = coordinates.residue_ids[atoms]
                masses = coordinates.masses[atoms]
                coords = coordinates.whole[atoms]
                mask = np.bincount(res_ids)[1:] != 0
                positions = np.array(
                    [
                        np.bincount(res_ids, weights=c * masses)[1:]
                        / np.bincount(res_ids, weights=masses)[1:]
                        for c in coords.T
                    ]
                ).T[mask]
                return np.array(positions)
            return center_of_masses(trajectory, atoms=trajectory.subset(residue_name=resname).atom_subset.indices).nojump

        def multi_radial_selector(atoms, bins):
            indices = []
            for i in range(len(bins) - 1):
                index = mde.coordinates.selector_radial_cylindrical(
                    atoms, r_min=bins[i], r_max=bins[i + 1]
                )
                indices.append(index)
            return indices
        
        def isf(self):
            time, result_all = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=self.q_const), self.com, segments=self.choices['Segments'], skip=0.1)
            time, result_wall = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=self.q_const), self.com,selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=1.0, r_max=1.5),segments=self.choices['Segments'], skip=0.1)
            time, result_center = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=self.q_const), self.com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=0.0, r_max=0.5), segments=self.choices['Segments'], skip=0.1)

            mask = time > 3e-1
            fitted = False
            try:
                fit_com, cov_com = curve_fit(mde.functions.kww, time[mask], result_all[mask])
                #tau = md.functions.kww_1e(*fit)
                tau_com = fit_com[1]
                beta_com = fit_com[2]
                with open(f'{self.cwd}/analysis/summary.txt', 'a') as summary_file:
                    summary_file.write(f'Fit parameters (tau, beta) : {tau_com}, {beta_com}\n Fit covariance : {cov_com}\n')
                    
                fitted = True
            except RuntimeError:
                print("ISF function could not be fitted successfully")
                pass

            plt.figure()
            plt.plot(time, result_all, label="all")
            plt.plot(time, result_wall, label="wall")
            plt.plot(time, result_center, label="center")
            if fitted:
                plt.plot(time, mde.functions.kww(time, *fit_com), '-', label=r'KWW, $\tau$={:.2f}, $\beta$={:.2f}ps'.format(tau_com, beta_com))
            plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xscale("log")
            plt.xlabel(r"$t$ / ps")
            plt.ylabel(r"ISF")
            plt.savefig(self.cwd + "/analysis/graphs/ISF_com_" + self.choices['Residue'])

            df = pd.DataFrame({'# t in ps': time, 'Qall in nm-1': result_all, 'Qwall in nm-1': result_wall, 'Qcent in nm-1': result_center})
            df.to_csv(f'{self.cwd}/analysis/xvg_files/isf_{self.choices['Residue']}.xvg', sep=' ', index=False)

        def ISF_radial(workdir, com, q_const, diameter, resname, segments): 
            bins = np.arange(0.0, diameter/2, 0.1)
            r = (bins[:-1] + bins[1:]) / 2
        
            time, results = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=q_const), 
                com, 
                selector=partial(Global.Analysis.multi_radial_selector, bins=bins), 
                segments=segments, 
                skip=0.1)
        
            df = pd.DataFrame({'time': time})
            c = [cm.plasma(i) for i in np.linspace(0, 1, len(r))]
            plt.figure()
            for i, result in enumerate(results):
                header = str(r[i])
                df[header]=results[i]
                plt.plot(time, result, "-", c=c[i], label=round(r[i], 2))
            plt.legend(title=r"$r$ / nm", ncols=2, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xscale("log")
            plt.xlabel(r"$t$ / ps")
            plt.ylabel(r"ISF")
            plt.savefig(workdir + "/analysis/graphs/ISF_resolved_" + resname)
            df.to_csv(workdir + "/analysis/xvg_files/ISF_resolved" + resname +".xvg", sep=' ', index=False)

        def msd_average(workdir, com, resname, segments):
            time, msd_com  = mde.correlation.shifted_correlation(mde.correlation.msd, com, segments=segments)
            time, msd_com_z  = mde.correlation.shifted_correlation(partial(mde.correlation.msd, axis = "z"), com, segments= segments)

            df = pd.DataFrame({'time / ps': time, 'msd_com / nm': msd_com, 'msd_com_z / nm': msd_com_z,})
            df.to_csv(f'{workdir}/analysis/xvg_files/MSD_{resname}.xvg', sep=' ', index=False)

            plt.plot(time, msd_com, label='MSD')
            plt.plot(time, msd_com_z, label='MSD (Z-direction)')

            plt.xscale("log")
            plt.yscale("log")
            
            plt.legend()
            plt.xlabel(r"$\mathbf{\mathit{t}}$ / ps")
            plt.ylabel(r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$")

            plt.tight_layout()

            plt.savefig(f'{workdir}/analysis/graphs/MSD_{resname}.png')
    
        def msd_resolved(workdir, com, resname, diameter, segments):
            bins = np.arange(0.0, diameter/2+0.1, 0.1)
            r = (bins[:-1] + bins[1:]) / 2
        
            time, results = mde.correlation.shifted_correlation(
                partial(mde.correlation.msd, axis="z"),
                com,
                selector=partial(Global.Analysis.multi_radial_selector, bins=bins),
                segments=segments,
                skip=0.1,
                average=True,
            )
        #    print(len(results))
            df = pd.DataFrame({'time': time})
            c = [cm.plasma(i) for i in np.linspace(0, 0.8, len(r))]
            plt.figure()
            for i, result in enumerate(results):
                header = str(r[i])
                df[header]=results[i]
        #        print(df) 
                plt.plot(time, result, c=c[i], label=round(r[i], 2))
            plt.legend(title=r"$r$", ncols=2, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel(r"$t$ / ps")
            plt.ylabel(r"⟨z$^2$⟩ / nm$^2$")
        #    plt.show()        
            plt.savefig(workdir + "/analysis/graphs/MSD_resolved_" + resname)
            df.to_csv(workdir + "/analysis/xvg_files/MSD_resolved" + resname +".xvg", sep=' ', index=False)

        def rdf_com(self, segments: int = 1000): 
            bins = np.arange(0, 2.2, 0.01)
            result_rdf = mde.distribution.time_average(partial(mde.distribution.rdf, bins=bins), 
                                                       self.com, 
                                                       segments=segments, 
                                                       skip=0.01)
            
            r = (bins[1:] + bins[:-1])/2
            q = 2*np.pi*np.linspace(1/r[result_rdf>0].max(), 5, 100)
            n = len(self.com[0])/np.prod(self.com[0].box.diagonal())

            plt.plot(bins[:-1], result_rdf, label='RDF (COM)')

            plt.xlabel(r"$r $ / nm")
            plt.ylabel(r"$g(r)$")
            plt.legend(loc='upper right')
            plt.savefig(f'{self.cwd}/analysis/graphs/RDF_COM_{self.choices['Residue']}')

            plt.clf()

            plt.plot(q, self.Analysis.Sq_from_gr(r, result_rdf, q, n), label=f'{self.choices['Residue']} (COM)')

            plt.xlabel(r"q / nm$^{-1}$")
            plt.ylabel(r"S(q)")
            plt.legend(loc='upper right')
            plt.savefig(f'{self.cwd}/analysis/graphs/SQ_COM_{self.choices['Residue']}')

            df = pd.DataFrame({'# r in nm': r, 'rdf in nm': result_rdf})
            df.to_csv(f'{self.cwd}//analysis/xvg_files/rdf_com_{self.choices['Residue']}.xvg', sep=' ', index=False)

            q_const = 2*np.pi*(1/result_rdf.max())

            return q_const, result_rdf.max()

        def Sq_from_gr(r, gr, q, n: float):
            ydata = ((gr - 1) * r).reshape(-1, 1) * np.sin(r.reshape(-1, 1) * q.reshape(1, -1))
            return np.trapz(x=r, y=ydata, axis=0) * (4 * np.pi * n / q) + 1

        def F1_F2(workdir, vector, resname, atom1, atom2, segments):
            plt.clf()
            f1s = {}
            f2s = {}
            time1, result_f1 = mde.correlation.shifted_correlation(partial(mde.correlation.rotational_autocorrelation, order=1), vector,segments=segments, skip=0.1, average=True)
            time2, result_f2 = mde.correlation.shifted_correlation(partial(mde.correlation.rotational_autocorrelation, order=2), vector, segments=segments, skip=0.1, average=True)
            f1s = (time1, result_f1)
            f2s = (time2, result_f2)
        #    print(result_f1)
            plt.plot(time1, result_f1, ".", label="$F_1$")
        #    plt.plot(time1[fit_start_index:], md.functions.kww(time1[fit_start_index:], *popt1), "--", label="fit")
        #   if fit_start is not None:
        #      plt.axvline(fit_start, linestyle="--", label="fit boundary")
            plt.xlabel(r"$t$ in ps")
            plt.ylabel(r"$F_1(t)$")
            plt.xscale("log")
        # if title is not None:
        #    plt.title(title)
            plt.legend()
        #plt.show()
#            plt.savefig(workdir + "/analysis/graphs/F1" + atom1 + "_" + atom2)
            df = pd.DataFrame({'# t in ps': time1, 'F1': result_f1})
            df.to_csv(workdir + "/analysis/xvg_files/F1" + resname + "_" + atom1 + "_" + atom2 +".xvg", sep=' ', index=False)

            plt.plot(time2, result_f2, ".", label="$F_2$")
        #  plt.plot(time2[fit_start_index:], md.functions.kww(time2[fit_start_index:], *popt2), "--", label="fit")
        #    if fit_start is not None:
        #       plt.axvline(fit_start, linestyle="--", label="fit boundary")
            plt.xlabel(r"$t$ in ps")
            plt.ylabel(r"$F_2(t)$")
            plt.xscale("log")
        # if title is not None:
        #    plt.title(title)
            plt.legend()
        #plt.show()
#            plt.savefig(workdir + "/analysis/graphs/F2" + atom1 + "_" + atom2)
            df = pd.DataFrame({'# t in ps': time1, 'F2': result_f2})
            df.to_csv(workdir + "/analysis/xvg_files/F2" + resname + "_" + atom1 + "_" + atom2 +".xvg", sep=' ', index=False)

        #analysis, fitting to exponential decay
            def exp_decay(t, tau):
                return np.exp(-t / tau)

            exp_decay.log = lambda tlog, tau: exp_decay(np.exp(tlog), tau)

            def plot_f1(title=None, fit_start=None):
                time1, result_f1 = f1s
                result_f1 = result_f1[time1 > 0]
                time1 = time1[time1 > 0]
                fit_start_index = 0
                if fit_start is not None:
                    fit_start_index = np.argmax(time1 > fit_start)

                popt1, pcov1 = curve_fit(
                    mde.functions.kww, time1[fit_start_index:], result_f1[fit_start_index:]
                )
                perr1 = np.sqrt(np.diag(pcov1))
                print (popt1[1])
                print (popt1[2])

                mean_τ_F1 = (popt1[1] / popt1[2]) * gamma(1 / popt1[2])
                print("Mean_tau_F1 (⟨τ_F1⟩):", mean_τ_F1)
                #print (pcov)
                #print (perr)
                #plt.plot(time1, result_f1, ".", label="$F_1$_" + str(popt1[1]) + "_" + str(popt1[2]))
                plt.plot(time1[fit_start_index:], mde.functions.kww(time1[fit_start_index:], *popt1), "-", label="fit")
                if fit_start is not None:
                    plt.axvline(fit_start, linestyle="--", label="fit boundary")
                plt.xlabel(r"$t$ in ps")
                plt.ylabel(r"$F_1(t)$")
                plt.xscale("log")
                if title is not None:
                    plt.title(title)
                plt.legend()
#                plt.savefig(workdir + "/analysis/graphs/F1-fit" + atom1 + "_" + atom2)

            plot_f1()
        # analyse F_2
        #def exp_decay(t, tau):
        #    return np.exp(-t / tau)

        #exp_decay.log = lambda tlog, tau: exp_decay(np.exp(tlog), tau)

            def plot_f2(title=None, fit_start=None):
                time2, result_f2 = f2s
                result_f2 = result_f2[time2 > 0]
                time2 = time2[time2 > 0]
                fit_start_index = 0
                if fit_start is not None:
                    fit_start_index = np.argmax(time2 > fit_start)

                popt2, pcov2 = curve_fit(
                    mde.functions.kww, time2[fit_start_index:], result_f2[fit_start_index:]
                )
                perr2 = np.sqrt(np.diag(pcov2))
                print (popt2[1])
                print (popt2[2])

                mean_τ_F2 = (popt2[1] / popt2[2]) * gamma(1 / popt2[2])
                print("Mean_tau_F2 (⟨τ_F2⟩):", mean_τ_F2)
            #print (pcov)
            #print (perr)
                #plt.plot(time2, result_f2, ".", label="$F_2$_" + str(popt2[1]) + "_" + str(popt2[2]))
                plt.plot(time2[fit_start_index:], mde.functions.kww(time2[fit_start_index:], *popt2), "-", label="fit")
                if fit_start is not None:
                    plt.axvline(fit_start, linestyle="--", label="fit boundary")
                plt.xlabel(r"$t$ in ps")
                plt.ylabel(r"$F(t)$")
                plt.xscale("log")
                if title is not None:
                    plt.title(title)
                plt.legend(loc='upper right')
                plt.savefig(workdir + "/analysis/graphs/F1_F2-fit" + resname + "_" + atom1 + "_" + atom2)

            plot_f2()

        def non_Gauss(workdir,com, resname, segments):
            time, result = mde.correlation.shifted_correlation(mde.correlation.non_gaussian_parameter
                                                    ,com, segments= segments)
            fig, ax = plt.subplots(1, 1)
            ax.plot(time, result,label=resname)

            ax.set_xscale("log")  
            ax.legend()
            ax.set_xlabel(r"$t$ / ps")
            ax.set_ylabel(r"Non-Gaussian Displacement")
            fig.savefig(workdir + "/analysis/graphs/nonGauss_" + resname)
            df = pd.DataFrame({'# t in ps': time, 'nonGauss': result})
            df.to_csv(workdir + "/analysis/xvg_files/nonGauss_" + resname +".xvg", sep=' ', index=False)

        def vanHove_translation(workdir, com, diameter, resname, segments):
            bins = np.arange(0, diameter/2, 0.1)
            time1, vH_wall = mde.correlation.shifted_correlation(partial(mde.correlation.van_hove_self, bins=bins), 
                                                                 com, 
                                                                 selector=partial(mde.coordinates.selector_radial_cylindrical, 
                                                                                  r_min=diameter/2-1, 
                                                                                  r_max=diameter/2), 
                                                                 segments=segments,
                                                                 skip=0.1)

            time2, vH_center = mde.correlation.shifted_correlation(partial(mde.correlation.van_hove_self, bins=bins), com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=0.0, r_max=0.5), segments=segments, skip=0.1)
        #    print(len(time1))
        #    print(len(bins))
            fig, ax = plt.subplots(1, 2, figsize = (20, 8))
            ax[0].set_title(f"{resname} van Hove_translation_wall")
            ax[1].set_title(f"{resname} van Hove_translation_center")
            df = pd.DataFrame({'radius/nm': bins[:-1]})
            for i, (t, S) in enumerate(zip(time1, vH_wall)):
                header = str(round(t, 1))
                df[header]= S
        #        print(i,t)
        #        print(len(S))
                if i%10 != 1:
                    continue
                ax[0].plot(bins[:-1], S, "-", label=round(t, 2))
            df.to_csv(workdir + "/analysis/xvg_files/vanHove_wall" + resname +".xvg", sep=' ', index=False)

            df = pd.DataFrame({'radius/nm': bins[:-1]})
            for i, (t, S) in enumerate(zip(time2, vH_center)):
                header = str(round(t, 1))
                df[header]= S
                if i%10 != 1:
                    continue
                ax[1].plot(bins[:-1], S, "--", label=round(t, 2))
            df.to_csv(workdir + "/analysis/xvg_files/vanHove_center" + resname +".xvg", sep=' ', index=False)

            ax[0].legend ()#   (loc = 'lower left',    bbox_to_anchor = (1.05,0,1,1),  fontsize = 20 )
            ax[0].set_xlabel(r"$r $ / nm")
            ax[0].set_ylabel(r"$distr(t)$", fontsize=18)
            ax[1].legend ()#loc = 'lower left',    bbox_to_anchor = (1.05,0,1,1),  fontsize = 20 )
            ax[1].set_xlabel(r"$r $ / nm")
            ax[1].set_ylabel(r"$distr(t)$", fontsize=18)

            fig.savefig(workdir + "/analysis/graphs/vanHove_transl_" + resname)

        def vanHove_rotation(workdir, trajectory, vectors, atom1, atom2, resname, segments):
            def van_hove_angle(trajectory, segments, vectors, skip=0.1):
                traj = trajectory.nojump
                bins = np.linspace(0,180,361)
                x = bins[1:] - (bins[1]-bins[0])/2
                def van_hove_angle_dist(start, end, bins):
                #  print("start",start)
                    scalar_prod = (start * end).sum(axis=-1)
            # print("scalar_prod",scalar_prod)
                    angle = np.arccos(scalar_prod)
                    angle = angle[(angle>=0)*(angle<=np.pi)]
                    hist, edges = np.histogram(angle *360/(2*np.pi), bins)
                    return 1 / len(start) * hist
        
                t, S = mde.correlation.shifted_correlation(partial(van_hove_angle_dist, bins=bins)
                                                    , vectors, segments=segments, skip=skip)  
                time = np.array([t_i for t_i in t for entry in x])
                angle = np.array([entry for t_i in t for entry in x])
                result = S.flatten()
                
                return pd.DataFrame({"t":time, "angle":angle, "vector":result})
            
            data = van_hove_angle(trajectory,segments,vectors)
            data.to_csv(workdir + "/analysis/xvg_files/vanHove_rot" + resname + "_" + atom1 + "_" + atom2 + ".xvg", sep=' ', index=False)

            c = [cm.plasma(x) for x in np.linspace(0, 1, len(data.groupby('t')))]
            plt.figure(figsize=(7,5))
            i = 0
            times = []
            for t, dt in data.groupby('t'):
                d = np.array(dt[['angle','vector']])
                y_data = d[:,1] # * (np.sum(d[:,1])*0.5*np.sum(np.sin(d[:,0]/180*np.pi)))
                if i% 20 == 1:
                    plt.plot(d[:,0], y_data, '.', c=c[i], label=str(round(t ,2)) + ' ps')
                i += 1
            #plt.ylim(0,0.5)
            plt.xlabel(r'$\varphi$',fontsize=20)
            plt.ylabel(r'$S(\varphi)$',fontsize=20)
            plt.legend(loc='upper right')
            plt.savefig(workdir + "/analysis/graphs/vanHove_rotation_" + resname + "_" + atom1 + "_" + atom2)

        def susceptibility(workdir, com, q_const, resname):
            time, chi_com = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=q_const), com, average=False, segments=50, window=0.02)

            chi4_com = len(com[0])*chi_com.var(axis=0)*1E5
            plt.clf()
            plt.figure()
            plt.plot(time[2:-2], chi4_com[2:-2], 'o', markerfacecolor='none', label=r'$\chi_4$', color='k')
            plt.plot(time[2:-2], mde.utils.moving_average(chi4_com, 5), label='smoothed')

            plt.xscale('log')
            plt.xlabel('$t$ / ps')
            plt.ylabel('$\\chi_4 \cdot 10^{-5}$')
            plt.legend(loc='upper left')
            plt.savefig(workdir + "/analysis/graphs/Susceptibility_" + resname)
            df = pd.DataFrame({'# time in ps': time[2:-2], 'susz': mde.utils.moving_average(chi4_com, 5)})
            df.to_csv(workdir + "/analysis/xvg_files/Susceptibility_" + resname +".xvg", sep=' ', index=False)

        def z_align(workdir, trajectory, vectors, atom1, atom2, resname, segments):

            def z_angle_orientation(trajectory, vectors, segments, skip=0.1):

                traj = trajectory.nojump
                z_vector=[0,0,1]
                bins = np.linspace(0,180,361)
                x = bins[1:] - (bins[1]-bins[0])/2
                def angles(start, end, z_vector, bins):
                    scalar_prod = (start * z_vector).sum(axis=-1)
                    angle = np.arccos(scalar_prod)
                    angle = angle[(angle>=0)*(angle<=np.pi)]
                    hist, edges = np.histogram(angle *360/(2*np.pi), bins)
                    return 1 / len(start) * hist
        #
                t, S = mde.correlation.shifted_correlation(partial(angles, z_vector=z_vector, bins=bins)
                                                    , vectors, segments=segments, skip=skip)  
        #   
                                    
        #        t, S = md.distribution.time_average(partial(angles, z_vector=z_vector, bins=bins), vectors, segments=segments, skip=skip)
        # above command instead of shifted_correlation gives me unpack problem, so I let go for now even though averaging through entire trajectory would be preferrable  
                time = np.array([t_i for t_i in t for entry in x])
                angle = np.array([entry for t_i in t for entry in x])
                result = S.flatten()
            
                return pd.DataFrame({"t":time, "angle":angle, "vector":result})

            data = z_angle_orientation(trajectory, vectors, segments)
            data.to_csv(workdir + "/analysis/xvg_files/z_align" + resname + "_" + atom1 + "_" + atom2 + ".xvg", sep=' ', index=False)

            c = [cm.plasma(x) for x in np.linspace(0, 1, len(data.groupby('t')))]
            plt.figure(figsize=(7,5))
            i = 0
            times = []
            for t, dt in data.groupby('t'):
                d = np.array(dt[['angle','vector']])
                y_data = d[:,1] # * (np.sum(d[:,1])*0.5*np.sum(np.sin(d[:,0]/180*np.pi)))
                if i% 10 == 1:
                    plt.plot(d[:,0], y_data, '.', c=c[i], label=str(round(t ,2)) + ' ps')
                i += 1
        #plt.ylim(0,0.5)
            plt.xlabel(r'$\varphi$',fontsize=20)
            plt.ylabel(r'$dist(\varphi)$',fontsize=20)
            plt.legend()
            plt.savefig(workdir + "/analysis/graphs/z_align_" + resname + "_" + atom1 + "_" + atom2)

        def z_histogram(workdir, trajectory, vectors, atom1, atom2, resname, segments): 
            """
            This could be plotted as a histogram
            """
            def z_componentent_histogram(trajectory, vectors, segments, skip=0.1):
                traj = trajectory.nojump
                bins = np.linspace(-1,1,201)
                x = bins[1:] - (bins[1]-bins[0])/2
                def z_comp(start, end, bins):
                    norm_vectors = np.linalg.norm(start, axis=1)
                    z_comp = start[:,2]/norm_vectors
                    hist, edges = np.histogram(z_comp, bins)
                    return 1 / len(start) * hist

                t, S = mde.correlation.shifted_correlation(partial(z_comp, bins=bins)
                                                    , vectors, segments=segments, skip=skip)  
        #   
                                    
        
        #        t, S = md.distribution.time_average(partial(z_comp, bins=bins), vectors, segments=segments, skip=skip)  
        # above command instead of shifted_correlation gives me unpack problem, so I let go for now even though averaging through entire trajectory would be preferrable  
                time = np.array([t_i for t_i in t for entry in x])
                angle = np.array([entry for t_i in t for entry in x])
                result = S.flatten()
            
                return pd.DataFrame({"t":time, "angle":angle, "vector":result})
            data = z_componentent_histogram(trajectory, vectors, segments)
            data.to_csv(workdir + "/analysis/xvg_files/z_histogram" + resname + "_" + atom1 + "_" + atom2 + ".xvg", sep=' ', index=False)
            c = [cm.plasma(x) for x in np.linspace(0, 1, len(data.groupby('t')))]
            plt.figure(figsize=(7,5))
            i = 0
            times = []
            for t, dt in data.groupby('t'):
                d = np.array(dt[['angle','vector']])
                y_data = d[:,1] # * (np.sum(d[:,1])*0.5*np.sum(np.sin(d[:,0]/180*np.pi)))
                if i% 20 == 1:
                    plt.plot(d[:,0], y_data, '.', c=c[i], label=str(round(t ,2)) + ' ps')
                i += 1
            plt.xlabel(r'$\varphi$',fontsize=20)
            plt.ylabel(r'$S(\varphi)$',fontsize=20)
            plt.legend()
            plt.savefig(workdir + "/analysis/graphs/z_histogram_" + resname + "_" + atom1 + "_" + atom2)

        def rdf_a2a_intra(workdir, mdaUniverse, residue,  atom1, atom2):
            residue = mdaUniverse.select_atoms(f'resname {residue}')
            df = pd.DataFrame()
            for i, resid in enumerate(set(residue.resids)):
                a1 = mdaUniverse.select_atoms(f'resid {resid} and name {atom1}')
                a2 = mdaUniverse.select_atoms(f'resid {resid} and name {atom2}')

                a1_a2_rdf = rdf.InterRDF(g1=a1,
                                        g2=a2,
                                        norm='density',
                                        )

                a1_a2_rdf.run(start=0, stop=500)

                df[resid] = a1_a2_rdf.results.rdf

            df2 = pd.DataFrame({'Distance': a1_a2_rdf.results.bins, 'Mean RDF': df.mean(axis=1)})

            plt.clf()
            plt.plot(df2['Distance']/10, df2['Mean RDF'])
            plt.xlabel(r'$r$ / nm')
            plt.ylabel(r'g(r)')
            plt.savefig(f'{workdir}/analysis/graphs/RDF_Intra_{atom1}_to_{atom2}')
            
        def rdf_a2a_inter(workdir, mdaUniverse, residue,  atom1, atom2):
            """
            test
            """
            residue = mdaUniverse.select_atoms(f'resname {residue}')
            df = pd.DataFrame()
            for i, resid in enumerate(set(residue.resids)):
                a1 = mdaUniverse.select_atoms(f'resid {resid} and name {atom1}')
                a2 = mdaUniverse.select_atoms(f'not resid {resid} and name {atom2}')

                a1_a2_rdf = rdf.InterRDF(g1=a1,
                                        g2=a2,
                                        norm='density',
                                        )

                a1_a2_rdf.run(start=0, stop=500)

                df[resid] = a1_a2_rdf.results.rdf

            df2 = pd.DataFrame({'Distance': a1_a2_rdf.results.bins, 'Mean RDF': df.mean(axis=1)})

            plt.clf()
            plt.plot(df2['Distance']/10, df2['Mean RDF'])
            plt.xlabel(r'$r$ / nm')
            plt.ylabel(r'g(r)')
            plt.savefig(f'{workdir}/analysis/graphs/RDF_Inter_{atom1}_to_{atom2}')

        def radial_density(workdir, trajectory, atom1, residue, diameter, length, segments):
            bins = np.arange(0.0, diameter/2 + 0.1, 0.025)
            pos = mde.distribution.time_average(partial(mde.distribution.radial_density, bins=bins), trajectory.subset(atom_name=atom1, residue_name=residue), segments=1000, skip=0.01)

            return bins[1:], pos

        def hbond_counts(workdir, mdaUniverse, resname, atom):
            def hbonds(hbonds_analysis, mdaUniverse, label):
                hbonds_analysis.run()
                pd.DataFrame({
                    'Frame': hbonds_analysis.results.hbonds[:,0].astype(int),
                    'Donor': hbonds_analysis.results.hbonds[:,1].astype(int),
                    'Acceptor': hbonds_analysis.results.hbonds[:,3].astype(int),
                    'Cluster': 0,
                    'Distance': hbonds_analysis.results.hbonds[:,4],
                    'Angle': hbonds_analysis.results.hbonds[:,5]
                }).to_csv(f'{workdir}/analysis/xvg_files/HBonds_{label}.xvg', sep=' ', index=False)

                tau_frames, hbond_lifetimes = hbonds_analysis.lifetime(tau_max=500)
                tau_times = tau_frames * mdaUniverse.trajectory.dt
                np_lifetimes = np.stack((tau_times, hbond_lifetimes), axis=-1)
                if not np.any(np.isnan(hbond_lifetimes)):
                    np.savetxt(f'{workdir}/analysis/xvg_files/HBond_Lifetimes_{label}.xvg', np_lifetimes, delimiter=' ')
                    if len(tau_frames) > 0:
                        plt.plot(tau_times, hbond_lifetimes, label=label)

            residues = ['LNK', 'ETH', 'VAN', 'VAN', 'VAN']
            names = ['NL', 'OEE', 'NV', 'OVE', 'OVH']
            to_analyze = [[res1, res2] for res1, res2 in product(zip(residues,names), [[resname, atom]])] # Generate all possible unique combinatorials between residue types
            
            print(to_analyze)

            plt.clf()
            for set in to_analyze:
                donor = set[0]
                acceptor = set[1]

                hbond_analysis = hba(universe=mdaUniverse,
                                        donors_sel=f'resname {donor[0]} and name {donor[1]} ',
                                        hydrogens_sel=f'type H and resname {donor[0]}',
                                        acceptors_sel=f'resname {acceptor[0]} and name {acceptor[1]}',
                                        d_a_cutoff=3.0,
                                        d_h_a_angle_cutoff=150,
                                        update_selections=False)
                hbonds(hbonds_analysis=hbond_analysis, 
                    mdaUniverse=mdaUniverse,
                    label=f'{donor[0]}_{donor[1]}_to_{acceptor[0]}_{acceptor[1]}')
                
                hbond_analysis = hba(universe=mdaUniverse,
                                        donors_sel=f'resname {acceptor[0]} and name {acceptor[1]} ',
                                        hydrogens_sel=f'type H and resname {acceptor[0]}',
                                        acceptors_sel=f'resname {donor[0]} and name {donor[1]}',
                                        d_a_cutoff=3.0,
                                        d_h_a_angle_cutoff=150,
                                        update_selections=False)
                hbonds(hbonds_analysis=hbond_analysis, 
                    mdaUniverse=mdaUniverse,
                    label=f'{acceptor[0]}_{acceptor[1]}_to_{donor[0]}_{donor[1]}')
            #plt.yscale('log')
            plt.xlabel(r"$t$ / ps")
            plt.ylabel(r"$\tau$")
            plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(f'{workdir}/analysis/graphs/tau_lifetimes_{resname}_{atom}')

if __name__ == "__main__":
    Global()
