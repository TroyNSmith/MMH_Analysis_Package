# Imports
import argparse
import datetime
import os
import random
import MDAnalysis as mda
import handle, helpers, pore
import mdevaluate as mde

# Argument parser setup
parser = argparse.ArgumentParser(description="Standard analysis of pore simulations")
parser.add_argument('-w', '--workdir', type=str, required=True, help="Simulation working directory")
parser.add_argument('-nr', '--number_repetitions', type=int, required=True, help="Number of repetitions for data verification")
parser.add_argument('-nf', '--number_frames', type=int, required=True, help="Frames to analyze in each verification")
parser.add_argument('-sf', '--start_frame', type=int, help="Starting frame for analysis")
parser.add_argument('-ef', '--end_frame', type=int, help="Ending frame for analysis")
parser.add_argument('-m', '--minimal', action='store_true', help="Minimal analysis only")
parser.add_argument('-ad', '--additional', action='store_true', help="Additional analysis only")
parser.add_argument('-sys', '--system', required=True, help="Analysis to perform (currently supported: Pore)")
parser.add_argument('-s', '--segments', type=int, required=True, help="Number of segments")
parser.add_argument('-r', '--residue', required=True, help="Index of reference residue (matches topology)")
parser.add_argument('-a', '--atoms', nargs=2, required=True, type=str, help="Indices of reference atoms (two required, matches topology)")
parser.add_argument('-q', '--q_value', type=float, help="q-value from RDF calculations")
parser.add_argument('-ow', '--overwrite', type=bool, help="Overwrite previously generated files", default=False)
args = parser.parse_args()

# Initialization
print('Initializing analysis')
workdir = args.workdir
mda_Universe = mda.Universe(f'{workdir}/out/out.gro', f'{workdir}/out/traj.xtc')
total_frames = mda_Universe.trajectory.n_frames

# Initialize arguments and frame checks
args_dict = handle.handle_args(args)
args_dict.setdefault('start_frame', 0)
args_dict.setdefault('end_frame', total_frames)

# Adjust frame ranges if needed
if total_frames < args_dict['number_frames']:
    print(f'\n### WARNING ###\nTotal frames ({total_frames}) < desired frames ({args_dict["number_frames"]})\nUsing {total_frames//2} instead.')
    args_dict['number_frames'] = total_frames // 2

elif args_dict['start_frame'] + args_dict['number_frames'] > args_dict['end_frame']:
    print(f'\n### WARNING ###\nNumber of frames ({args_dict["number_frames"]}) exceeds the domain ({args_dict["start_frame"]} to {args_dict["end_frame"]})\nAdjusting to {(args_dict["end_frame"] - args_dict["start_frame"]) // 2}.')
    args_dict['number_frames'] = (args_dict['end_frame'] - args_dict['start_frame']) // 2

if total_frames < args_dict['end_frame']:
    print(f'\n### WARNING ###\nTotal frames ({total_frames}) < ending frame ({args_dict["end_frame"]})\nAdjusting to {total_frames}.')
    args_dict['end_frame'] = total_frames

if args_dict['start_frame'] < 0 or args_dict['start_frame'] >= total_frames:
    print(f'\n### WARNING ###\nInvalid start frame ({args_dict["start_frame"]})\nStarting from frame 0 instead.')
    args_dict['start_frame'] = 0
    args_dict['end_frame'] = total_frames

# Verification loop
for ver_it in range(args_dict['number_repetitions']):
    start_frame = random.randint(args_dict['start_frame'], args_dict['end_frame'] - args_dict['number_frames'])
    end_frame = start_frame + args_dict['number_frames']
    ver_dir = f'{workdir}/Verification/s{start_frame}_e{end_frame}'

    print(f"Processing frames: {start_frame} to {end_frame}")

    # Setup verification directories
    os.makedirs(f'{ver_dir}/out', exist_ok=True)
    os.makedirs(f'{ver_dir}/analysis/graphs', exist_ok=True)
    os.makedirs(f'{ver_dir}/analysis/xvg_files', exist_ok=True)

    # Copy necessary files
    os.system(f'cp {workdir}/out/*.gro {ver_dir}/out')
    os.system(f'cp {workdir}/*.tpr {ver_dir}')

    # Trjconv for frame selection
    os.system(f'gmx trjconv -f {workdir}/out/traj.xtc -o {ver_dir}/out/traj.xtc -b {start_frame * mda_Universe.trajectory.dt} -e {end_frame * mda_Universe.trajectory.dt}')

    # Logging info
    helpers.log_info(ver_dir, '\n@ REFERENCE INFORMATION', f'Date/Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}', args_dict)

    # Perform analysis based on system type
    if args_dict['system'] == 'Pore':
        if args_dict['minimal']:
            pore.minimal(args_dict=args_dict, ver_dir=ver_dir)
        else:
            pore.full(args_dict=args_dict, ver_dir=ver_dir)
    else:
        print(f'Unrecognized system ({args_dict["system"]}) or method ({args_dict["minimal"]})\nPlease check spelling (case-sensitive) or use -h for help.')
        raise SystemError