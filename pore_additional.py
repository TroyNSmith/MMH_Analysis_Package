# Imports #
# ------- #
import dynamics, handle, helpers, plotting, structural
import MDAnalysis as mda
import mdevaluate as mde
import numpy as np
import os, re
from tqdm import tqdm
import warnings
# ------- #

def additional(args_dict: dict
               )-> None:
    """
    Executes additional analysis simulation (if necessary) and performs the analysis.

    Args:
        args_dict: A dictionary containing all of the imported arguments from main.py
    """
    workdir = args_dict['workdir']
    segments = args_dict['segments']
    residue = args_dict['residue']
    atom1 = args_dict['atoms'][0]
    atom2 = args_dict['atoms'][1]
    q_val = args_dict['q_value'] if 'q_value' in args_dict.keys() else 0
    overwrite = True if 'overwrite' in args_dict.keys() else False

    addtnldir = os.path.join(os.path.dirname(workdir), '6_nvt_addtnl')

    if not os.path.exists(f'{addtnldir}/out/out.gro'):
        run_6(addtnldir=addtnldir,
              workdir=workdir)
        
    os.system(f'mkdir -p {addtnldir}/analysis/graphs \
                mkdir -p {addtnldir}/analysis/xvg_files')
        
    with tqdm(total=7, desc="Additional Analysis Progress", unit="step") as pbar:          
        # Step 1: Change working directory and open trajectory
        pbar.set_postfix(step="Initializing")
        os.system(f'cd {addtnldir}')
        mde_trajectory = mde.open(directory=addtnldir,
                                  topology='run.tpr',
                                  trajectory='out/traj.xtc',
                                  nojump=True)
        pbar.update(1)  # Update the progress bar after initializing
        
        # Step 3: Extract pore simulation details from directory name
        pbar.set_postfix(step="Logging Pore Information")
        pore_inf = {}
        pore_dir = next((dir for dir in addtnldir.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir):
                pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])
        helpers.log_info(addtnldir, pore_inf, f'\n@ ANALYSIS RESULTS', )  # Log extracted information
        pbar.update(1)
        
        # Step 4: Gather centers of masses
        pbar.set_postfix(step="Gathering Centers of Masses")
        com = helpers.center_of_masses(trajectory=mde_trajectory,
                                       residue=residue)
        pbar.update(1)

        # Step 5: Calculate the mean square displacement
        if not os.path.exists(f'{addtnldir}/analysis/graphs/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(com=com,
                                       segments=100)
            plotting.plot_line(f'{addtnldir}/analysis/graphs/MSD_{residue}.png', 
                               msd[:,0], 
                               msd[:,1],
                               msd[:,2],
                               xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log',
                               yscale='log', 
                               legend=True,
                               handles=['Average', 'Z-Direction'])
            np.savetxt(f'{addtnldir}/analysis/xvg_files/MSD_{residue}.xvg', msd, delimiter=',', header='Time / ps, Average MSD, Average Z-MSD')
        pbar.update(1)

        # Step 6: Calculate the RDF / q constant
        if q_val == 0:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com,
                                                   segments=1000)
            helpers.log_info(addtnldir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{addtnldir}/analysis/graphs/rdf_{residue}_{atom1}_{atom2}.png', 
                               rdf[:,0], 
                               rdf[:,1],
                               xlabel='r / nm',
                               ylabel='g(r)'
                               )
            np.savetxt(f'{addtnldir}/analysis/xvg_files/rdf_{residue}_{atom1}_{atom2}.xvg', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

        # Step 7: Calculate the ISF
        if not os.path.exists(f'{addtnldir}/analysis/graphs/ISF_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.isf(com=com,
                               q_val=q_val,
                               segments=segments,
                               radius=pore_inf['D']/2)
            plotting.plot_line(f'{addtnldir}/analysis/graphs/ISF_{residue}.png', 
                               isf[:,0], 
                               isf[:,1],
                               isf[:,2],
                               isf[:,3],
                               xlabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               xscale='log',
                               yscale='log', 
                               legend=True,
                               handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{addtnldir}/analysis/xvg_files/ISF_{residue}.xvg', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

        # Complete!
        pbar.set_postfix(step="Completed Additional Analysis!")
        pbar.update(1)

def run_6(addtnldir:str,
          workdir: str
          )-> None:
    """
    Runs the additional simulation (if necessary).

    Args:
        addtnldir: Additional simulation directory.
        workdir: Production simulation directory.
    """
    os.system(f'mkdir -p {addtnldir}/out')
    os.system(f'cp {workdir}/*.gro {workdir}/*.tpr {workdir}/*.cpt {workdir}/*.ndx {workdir}/*.top {addtnldir}/')
    with open(f'{addtnldir}/mdp_nvt_addtnl_system.mdp', 'w') as f:
        f.write('; very basics of the simulation\n' 
                'integrator               = md              ; solve newton\'s equation of motion\n' 
                'dt                       = 0.002           ; integration time step / ps\n' 
                'nsteps                   = 1000000         ; number of steps\n' 
                '\n' 
                '; remove drifts of the center of mass\n' 
                'comm-mode                = None            ; do not remove COM translation due to pore absolute position restraint\n' 
                '\n' 
                '; control frequency of output\n' 
                'nstvout                  = 0               ; write velocities to trajectory file every number of steps\n' 
                'nstfout                  = 0               ; write forces to trajectory file every number of steps\n' 
                'nstlog                   = 500000          ; update log file every number of steps\n' 
                'nstcalcenergy            = 5               ; calculate energies/pressures every nstenergy steps\n' 
                'nstenergy                = 5               ; write energies to energy file every number of steps\n' 
                'nstxout-compressed       = 2000            ; write positions using compression (saves memory, worse quality)\n' 
                'compressed-x-precision   = 200             ; precision to write compressed trajectory\n' 
                '\n' 
                '; next neighbor search and periodic boundary conditions\n' 
                'nstlist                  = 20              ; freq. to update neighbor list & long range forces (>= 20 w/ GPUs)\n' 
                'rlist                    = 1.0             ; short-range neighbor list cutoff (nm)\n' 
                'cutoff-scheme            = Verlet          ; atom-based neighbor search with an implicit buffer region\n' 
                'pbc                      = xyz             ; periodicity in x, y, and z\n' 
                'periodic-molecules       = yes             ; silica pore is infinitely bonded to itself through pbc\n' 
                '\n' 
                '; coulomb interaction\n' 
                'coulombtype              = PME             ; particle-mesh ewald summation for long range (>rcoulomb)\n' 
                'rcoulomb                 = 1.0             ; short-range electrostatic cutoff (in nm) (with PME, rcoulomb >= rvdw)\n' 
                'coulomb-modifier         = Potential-shift ; shifts potential by constant so potential is 0 at cut-off\n' 
                'fourierspacing           = 0.12            ; spacing of FFT in reciprocal space in PME long range treatment (in nm)\n' 
                'pme-order                = 4               ; cubic PME interpolation order\n' 
                '\n' 
                '; lennard-jones potential handling\n' 
                'vdwtype                  = PME             ; particle-mesh ewald summation for long range (>rvdw)\n' 
                'rvdw                     = 1.0             ; short-range vdw cutoff (in nm)\n' 
                'vdw-modifier             = Potential-shift ; shifts potential by constant so potential is 0 at cut-off\n' 
                '\n' 
                '; temperature coupling\n' 
                'tcoupl                   = v-rescale       ; the algorithm to use, v-rescale generates correct canonical ensemble\n' 
                'tc-grps                  = pore_graft solvent ; groups to couple to temperature bath\n' 
                'tau-t                    = 1.0 1.0         ; time constant (in ps), meaning varies by algorithm\n' 
                'ref-t                    = 328.0 328.0     ; temperature for coupling (K)\n' 
                'nsttcouple               = 1               ; frequency to couple temperature\n' 
                '\n' 
                '; velocity generation\n' 
                'gen-vel                  = no              ; generate velocities according to Maxwell distr. (no for a continuation run)\n' 
                '\n' 
                '; pressure coupling\n' 
                'pcoupl                   = no              ; the algorithm to use (no for NVT)\n' 
                '\n' 
                '; constraints\n' 
                'constraints              = h-bonds         ; constrains bonds only involving hydrogen\n' 
                'constraint_algorithm     = lincs           ; algorithm to use, lincs should NOT be used for angle constraining\n' 
                'lincs-order              = 4               ; highest order in constraint coupling matrix expansion\n' 
                'lincs-iter               = 1               ; accuracy of lincs algorithm\n' 
                'continuation             = yes             ; no for applying constraints at start of run\n')
        f.close()

    os.system(f'gmx grompp \
             -f {addtnldir}/mdp_nvt_addtnl_system.mdp \
             -c {addtnldir}/gro_nvt_prod_system.gro \
             -r {addtnldir}/gro_nvt_prod_system.gro \
             -t {addtnldir}/prev_sim.cpt \
             -n {addtnldir}/tc_grps.ndx \
             -p {addtnldir}/top_nvt_prod_system.top \
             -o {addtnldir}/run.tpr \
             -po {addtnldir}/out/mdout.mdp \
             -quiet no')
    
    os.system(f'gmx mdrun \
             -s {addtnldir}/run.tpr \
             -o {addtnldir}/out/traj.trr \
             -x {addtnldir}/out/traj.xtc \
             -c {addtnldir}/out/out.gro \
             -e {addtnldir}/out/energy.edr \
             -g {addtnldir}/out/log.log \
             -cpo {addtnldir}/out/state.cpt \
             -cpi {addtnldir}/out/state.cpt \
             -cpt 5 \
             -quiet no -v')