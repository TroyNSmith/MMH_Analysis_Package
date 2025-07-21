# Imports
import argparse
import datetime
import os
import MDAnalysis as mda
import handle, helpers, pore
import mdevaluate as mde

# Argument parser setup
parser = argparse.ArgumentParser(description="Standard analysis of pore simulations")
parser.add_argument('-w', '--workdir', type=str, help="Simulation working directory", default=os.getcwd())
parser.add_argument('-m', '--minimal', action='store_true', help="Perform minimal analysis only")
parser.add_argument('-ad', '--additional', action='store_true', help="Perform additional analysis")
parser.add_argument('-sys', '--system', help="Currently supported: Pore")
parser.add_argument('-s', '--segments', type=int, help="Number of segments")
parser.add_argument('-r', '--residue', type=str, help="Index of reference residue (matches topology)")
parser.add_argument('-a', '--atoms', nargs=2, type=str, help="Indices of reference atoms (two required, matches topology)")
parser.add_argument('-q', '--q_value', type=float, help="q-value from RDF calculations")
parser.add_argument('-ow', '--overwrite', action='store_true', help="Overwrite previously generated files")
args = parser.parse_args()

# Initialization
print('Initializing analysis')
workdir = args.workdir
MDA_universe = mda.Universe(f'{workdir}/out/out.gro', f'{workdir}/out/traj.xtc')

# Collect unique residues, sorted by name
residues = sorted({residue.resname for residue in MDA_universe.residues})

# Determine working directory name
pardir = workdir.split('/')[-1]

# Determine mode (interactive or non-interactive)
if all([args.workdir, args.segments, args.residue, args.atoms]):
    args_dict = handle.handle_args(args)
else:
    args_dict = handle.handle_interactive(MDA_universe, residue=args.residue)
    args_dict['workdir'] = workdir

# Create necessary directories
os.makedirs(f'{workdir}/analysis/graphs', exist_ok=True)
os.makedirs(f'{workdir}/analysis/xvg_files', exist_ok=True)

# Log reference information
helpers.log_info(workdir, '\n@ REFERENCE INFORMATION', f'Date/Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}', args_dict)

# Perform Pore analysis
if args_dict['system'] == 'Pore':
    if args_dict['minimal']:
        pore.minimal(args_dict=args_dict)
    else:
        pore.full(args_dict=args_dict)

else:
    print(f'Unrecognized system ({args_dict["system"]}) or method ({args_dict["minimal"]})\nPlease check spelling (case-sensitive) or use -h for help.')
    raise SystemError
