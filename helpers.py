# Imports #
# ------- #
import logging
import mdevaluate as mde
import numpy as np
from numpy.typing import NDArray
from typing import Any, Iterable
import warnings
# ------- #

# Suppress specific warnings:
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def center_of_masses(trajectory: Any, 
                     residue: str
                     )-> NDArray:

    @mde.coordinates.map_coordinates
    def center_of_masses(coordinates, atoms, shear: bool = False):
        res_ids = coordinates.residue_ids[atoms]
        masses = coordinates.masses[atoms]
        coords = coordinates.whole[atoms]
        positions = np.array(
            [
                np.bincount(res_ids, weights=c * masses)[1:]
                / np.bincount(res_ids, weights=masses)[1:]
                for c in coords.T
            ]
        ).T[np.bincount(res_ids)[1:] != 0]
        return np.array(positions)
    return center_of_masses(trajectory, atoms=trajectory.subset(residue_name=residue).atom_subset.indices).nojump

def exponential_model(t: NDArray,
                      A: float,
                      k: float, 
                      C: float
                      )-> NDArray:
    """
    Returns the model associated with a set of exponential fit variables.

    Args:
        t: NumPy array containing the time series.
        A: Exponential fit coefficient.
        k: Exponential fit k value.
        C: Exponential fit translation.

    Returns:
        NDArray: NumPy array containing the fitted y data series.
    """
    return A * np.exp(t * k) + C

def fit_exp_linear(t: NDArray, 
                   y: NDArray, 
                   C: int = 0, 
                   order: int = 1,
                   max_attempts: int = 100,
                   increment: float = 1e-3,
                   )-> tuple[float, float, float]:
    """
    Performs an exponential fit by linearizing the time and y data series then converting the fit coefficient back to linear.

    Args:
        t: NumPy array containing the time series.
        y: NumPy array containing the y data series.
        C: Arithmetic translation of y data series for first attempt.
        order: Order of the fit.
        max_attempts: How many times to try applying the incremental increase to C in order to linearize the y data series.
        increment: Magnitude of the incremental increases to C.

    Returns:
        float: Exponential fit coefficient, A.
        float: Exponential fit k value.
        float: Exponential fit translation, C.
    """
    t_filtered = t[t > 0]
    y_filtered = y[t > 0]

    # Try incrementing C until the values for A and k are valid
    attempts = 0
    while attempts < max_attempts:
        # Perform the log transformation of y
        try:
            y_transformed = np.log(y_filtered + C)
            k, A_log = np.polyfit(t_filtered, y_transformed, order)
            A = np.exp(A_log)
            
            # Check if the result is not NaN
            if not np.any(np.isnan(A)) and not np.any(np.isnan(k)):
                return A, k, C  # Return valid result when found
        except Exception as e:
            # If the fit fails due to any other reason, catch the exception and continue
            pass
        
        # Increment C for the next attempt
        C += increment
        attempts += 1

    print("Unable to find valid fit after {} attempts with incremental C.".format(max_attempts))

def log_info(workdir: str,
             *args: Any
             )-> None:
    """
    Log information to a summary file in the work directory.

    Args:
        workdir: Current work directory.

    *Args:
        arg: Object to be written to the summary file.
    """
    with open(f'{workdir}/summary.txt', 'a+') as f:
        for arg in args:
            if isinstance(arg, (list, tuple, set)):
                # Handle list/tuple/set: iterate over its items and log them
                for item in arg:
                    f.write(str(item) + '\n')  # Write each item on a new line
            elif isinstance(arg, dict):
                # Handle dictionary: log keys and values
                for key, value in arg.items():
                    f.write(f'{key}: {value}\n')
            else:
                # Handle simple types (str, int, etc.)
                f.write(str(arg) + '\n')

def multi_radial_selector(atoms, 
                          bins: NDArray
                          )-> list:
    """
    Selects all of the atoms within a radial bin.

    Args:
        atoms: Atoms to select from.
        bins: NumPy array of the radial bins.
    
    Returns:
        list: A list of the atoms meeting bin criteria.
    """

    indices = []
    for i in range(len(bins) - 1):
        index = mde.coordinates.selector_radial_cylindrical(
            atoms, r_min=bins[i], r_max=bins[i + 1]
        )
        indices.append(index)
    return indices