# ----- Last updated: 06/05/2025 -----
# ----- By Troy N. Smith :-) -----

# Imports
import argparse
import datetime
import os
import MDAnalysis as mda
import pkg.bulk as bulk, handle, helpers, pkg.pore as pore

# Argument parser setup
parser = argparse.ArgumentParser(description="Standard analysis of pore simulations")
parser.add_argument('-w', '--workdir', type=str, help="Simulation working directory", default=os.getcwd())
parser.add_argument('-m', '--minimal', action='store_true', help="Perform minimal analysis only")
parser.add_argument('-ad', '--additional', action='store_true', help="Perform additional analysis")
parser.add_argument('-sys', '--system', help="Currently supported: Pore, Bulk")
parser.add_argument('-s', '--segments', type=int, help="Number of segments", default=1000)
parser.add_argument('-r', '--residue', type=str, help="Index of reference residue (matches topology)")
parser.add_argument('-a', '--atoms', nargs=2, type=str, help="Indices of reference atoms (two required, matches topology)")
parser.add_argument('-q', '--q_value', type=float, help="q-value from RDF calculations")
parser.add_argument('-ow', '--overwrite', action='store_true', help="Overwrite previously generated files")
parser.add_argument('-auto', '--auto_select', action='store_true', help="Automatically select end-to-end on all relevant residues for analysis.")
args = parser.parse_args()

# Initialization
workdir = args.workdir
try:
    MDA_universe = mda.Universe(f'{workdir}/out/out.gro', f'{workdir}/out/traj.xtc')
except FileNotFoundError:
    MDA_universe = mda.Universe(f'{workdir}/out/md.gro', f'{workdir}/out/traj.xtc')

# Collect unique residues, sorted by name
residues = sorted({residue.resname for residue in MDA_universe.residues})

# Determine working directory name
pardir = workdir.split('/')[-1]

# Determine mode (interactive or non-interactive)
if (args.atoms and args.residue) or args.auto_select:
    args_dict = handle.handle_args(args)
else:
    args_dict = handle.handle_interactive(MDA_universe, residue=args.residue)
    args_dict['workdir'] = workdir

# Log reference information
helpers.log_info(workdir, '\n@ REFERENCE INFORMATION', f'Date/Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}', args_dict)

if args_dict['auto_select'] == True:

    residues = {residue.resname for residue in MDA_universe.residues} # Automatically select end-to-end atoms for all but the grafting/pore residues (useful for PEG mixtures)
    for residue in residues:
         if residue not in ['LNK', 'VAN', 'PORE']:
            args_dict['residue'] = residue
            atoms = MDA_universe.select_atoms(f'resname {residue}').names

            seen = set()
            retain = []

            for item in atoms:
                if item.startswith('H'):
                    continue
                if item not in seen:
                    seen.add(item)
                    retain.append(item)
            
            args_dict['atoms'] = [retain[0], retain[-1]]

    if args_dict['system'] == 'Pore':
        if args_dict['minimal']:
            pore.minimal(args_dict=args_dict)
        else:
            pore.full(args_dict=args_dict)

    elif args_dict['system'] == 'Bulk':
            bulk.full(args_dict=args_dict)

    else:
        print(f'Unrecognized system ({args_dict["system"]}) or method ({args_dict["minimal"]})\nPlease check spelling (case-sensitive) or use -h for help.')
        raise SystemError

# Perform Pore analysis
if args_dict['system'].lower() == 'pore':
    if args_dict['minimal']:
        pore.minimal(args_dict=args_dict)
    else:
        pore.full(args_dict=args_dict)

# Perform Bulk analysis
elif args_dict['system'].lower() == 'bulk':
        bulk.full(args_dict=args_dict)

else:
    print(f'Unrecognized system ({args_dict["system"]}) or method ({args_dict["minimal"]})\nPlease check spelling or use -h for help.')
    raise SystemError