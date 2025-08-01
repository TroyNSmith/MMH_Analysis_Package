# Imports #
# ------- #
import MDAnalysis as mda
import mdevaluate as mde
from numpy.typing import NDArray
from typing import Any
# ------- #

def handle_args(args)-> dict:
    """
    This will create a dictionary of all provided keyword arguments for easier passing.
    """
    args_dict = {}
    for arg in vars(args):
        if getattr(args, arg) is not None:
            args_dict[arg] = getattr(args, arg)
            
    return args_dict

def handle_interactive(MDA_universe: Any, residue: str)-> dict:
    """
    Handle interactive mode, prompting the user for various inputs.
    """
    print("Interactive mode activated.")
    args_dict = {}
    while True:
        try:
            print("Please select a method:")
            for i, choice in enumerate(['Minimal', 'Full'], start=1):
                print(f'{i}. {choice}')
            args_dict['minimal'] = True if user_input == 1 else False

            supported_analysis = [
                'Pore'
            ]

            print('Please choose the type of analysis: ')
            for i, choice in enumerate(supported_analysis, start=1):
                print(f'{i}. {choice}')
            user_input = int(input('Enter the analysis: '))
            args_dict['system'] = supported_analysis[user_input - 1]

            args_dict['segments'] = int(input('Enter the number of segments (typically 100): '))

            print("Please choose a residue from the following options:")
            for i, choice in enumerate(residue, start=1):
                print(f'{i}. {choice}')
            residue_input = int(input('Enter the reference residue: '))
            args_dict['residue'] = residue[residue_input - 1]
            
            atoms = sorted(
                {atom.name for atom in MDA_universe.select_atoms(f"resname {args_dict['residue']}")},
                key=lambda name: name[-1]
                )

            for i, atom_name in enumerate(atoms, start=1):
                print(f'{i}. {atom_name}')

            for i in range(2):
                choice = int(input(f'Enter reference atom {i+1}: '))
                args_dict[f'atom{i+1}'] = atoms[choice - 1]

            print('Overwrite existing analysis?')
            for i, choice in enumerate(['Yes', 'No'], start=1):
                print(f'{i}. {choice}')
            args_dict['overwrite'] = True if int(input('Enter selection: ')) == 1 else False

            return args_dict
        
        except ValueError:
            print('Invalid input. Please try again.')

def handle_vectors(mde_trajectory,
                   residue:str,
                   atom1:str,
                   atom2:str,
                   )-> NDArray:
    """
    Process and save mdevaluate indices/vectors.

    Args:
        mde_trajectory: Trajectory object initialized by mdevaluate.
        residue: Name of the residue of interest.
        atom1: First reference atom for center of mass.
        atom2: Second reference atom for center of mass.

    Returns:
        NDArray: NumPy array containing the center of mass coordinates for selected residues.
    """
    mde_indices_1 = mde_trajectory.subset(atom_name=atom1, residue_name=residue).atom_subset.indices
    mde_indices_2 = mde_trajectory.subset(atom_name=atom2, residue_name=residue).atom_subset.indices
    mde_vectors = mde.coordinates.vectors(mde_trajectory,
                                          atom_indices_a=mde_indices_1,
                                          atom_indices_b=mde_indices_2,
                                          normed=True)

    return mde_vectors