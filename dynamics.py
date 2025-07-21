# Imports #
# ------- #
from functools import partial
import helpers
import matplotlib.pyplot as plt
import mdevaluate as mde
import numpy as np
from numpy.typing import NDArray
import warnings
# ------- #

def average_msd(com: NDArray,
                segments:int = 1000,
                pore: bool = False,
                )-> NDArray:
    """
    Calculated mean square displacement (MSD) based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        segments: Number of segments to divide the trajectory into.
        pore: Whether or not the system is a  pore.

    Returns:
        NDArray: NumPy array containing the average (and z-component) MSD.
    """
    if pore:
        time, msd_com  = mde.correlation.shifted_correlation(partial(mde.correlation.msd, axis = 'all'), com, segments=segments)
        time, msd_com_z  = mde.correlation.shifted_correlation(partial(mde.correlation.msd, axis = "z"), com, segments= segments)
        return np.column_stack([time, msd_com, msd_com_z])

    else:
        time, msd_com  = mde.correlation.shifted_correlation(partial(mde.correlation.msd, axis = 'all'), com, segments=segments)

        return np.column_stack([time, msd_com])
    
def resolved_msd(com: NDArray,
                 diameter: float,
                 segments: int=100,
                 )-> tuple[NDArray, NDArray]:
    """
    Calculates radially resolved mean square displacement (MSD) based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        diameter: Diameter of the pore.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: A NumPy array containing the radial MSD results.
        NDArray: A NumPy array containing the bins used.
    """
    bins = np.arange(0.0, diameter/2+0.1, 0.1)
    r = (bins[:-1] + bins[1:]) / 2

    output, results = mde.correlation.shifted_correlation(
        partial(mde.correlation.msd, axis="z"),
        com,
        selector=partial(helpers.multi_radial_selector, bins=bins),
        segments=segments,
        skip=0.1,
        average=True,
    )

    for result in results:
        output = np.column_stack([output, result])

    return output, r

def isf(com: NDArray,
        q_val: float,
        segments: int = 1,
        radius: float = 0,
        )-> NDArray:
    """
    Calculates intermediate / incoherent scattering functions (ISF) based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        q_val: Value of the q constant as calculated from the RDF.
        segments: Number of segments to divide the trajectory into.
        radius: Radius of the pore.

    Returns:
        NDArray: NumPy array containing the ISF for all residues, residues in the center, and residues at the wall.
    """
    times, result_all = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=q_val), com, segments=segments)

    if not radius == 0:
        times, result_wall = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=q_val), com,selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=2*radius/3, r_max=radius),segments=segments, skip=0.0)
        times, result_center = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=q_val), com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=0.0, r_max=radius/3), segments=segments, skip=0.0)
        return np.column_stack([times, result_all, result_wall, result_center])

    return np.column_stack([times, result_all.T])

def non_Gauss(com: NDArray, 
              segments: int,
              )-> NDArray:
    """
    Calculates non-Gaussian displacements based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: NumPy array containing the non-Gaussian displacements and corresponding time series.
    """
    times, results = mde.correlation.shifted_correlation(mde.correlation.non_gaussian_parameter, com, segments=segments)

    return np.column_stack([times, results])

def resolved_isf(com:NDArray, 
                 q_val:float, 
                 diameter:float, 
                 segments:int = 100
                 )-> tuple[NDArray, NDArray]: 
    """
    Calculates radially resolved  intermediate / incoherent scattering functions (ISF) based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        q_val: Value of the q constant as calculated from the RDF.
        diameter: Diameter of the pore.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: A NumPy array containing the radial ISF results.
        NDArray: A NumPy array containing the bins used.
    """
    bins = np.arange(0.0, diameter/2+0.1, 0.1)
    r = (bins[:-1] + bins[1:]) / 2

    output, results = mde.correlation.shifted_correlation(
        partial(mde.correlation.isf, q=q_val), 
        com, 
        selector=partial(helpers.multi_radial_selector, bins=bins), 
        segments=segments, 
        skip=0.1)

    for result in results:
        output = np.column_stack([output, result])

    return output, r

def rotational_corr(vectors: NDArray, 
                    segments: int = 100
                    )-> NDArray:
    """
    Calculates the first and second order rotational correlations for atom-to-atom vectors.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: NumPy array containing the times and correlation results.
    """
    time, results_f1 = mde.correlation.shifted_correlation(partial(mde.correlation.rotational_autocorrelation, order=1), vectors,segments=segments, skip=0.1, average=True)
    time, results_f2 = mde.correlation.shifted_correlation(partial(mde.correlation.rotational_autocorrelation, order=2), vectors, segments=segments, skip=0.1, average=True)
    warnings.warn("\nFitting is not currently being performed for rotational correlation.\nPlease either manually perform fitting or add a fitting procedure.")
    return np.column_stack([time, results_f1, results_f2])

def susceptibility(com: NDArray,
                   q_val: float
                   )-> NDArray:
    """
    Calculates fourth-order susceptibility of the residues based on provided centers of masses.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        q_val: Value of the q constant.

    Returns:
        NDArray: NumPy array containing the susceptibility and time series.
    """
    time, chi_com = mde.correlation.shifted_correlation(partial(mde.correlation.isf, q=q_val), com, average=False, segments=50)

    chi4_results = len(com[0])*chi_com.var(axis=0)*1E5
    chi4_smooth = mde.utils.moving_average(chi4_results, 5)

    return np.column_stack([time[2:-2], chi4_results[2:-2], chi4_smooth])

def vanHove_rotation(vectors: NDArray, 
                     segments: int
                     )-> NDArray:
    """
    Calculates van Hove rotational dynamics based on centers of masses.

    Args:
        vectors: NumPy array containing the atom-to-atom vectors.
        segments: Number of segments to divide the trajectory into.

    Returns:
        NDArray: A NumPy array containing the van Hove rotational dynamics.
    """
    def van_hove_angle(segments, vectors, skip=0.1):
        bins = np.linspace(0,180,361)
        x = bins[1:] - (bins[1]-bins[0])/2

        def van_hove_angle_dist(start, end, bins):
            scalar_prod = (start * end).sum(axis=-1)
            angle = np.arccos(scalar_prod)
            angle = angle[(angle>=0)*(angle<=np.pi)]
            hist, _ = np.histogram(angle *360/(2*np.pi), bins)
            return 1 / len(start) * hist

        t, S = mde.correlation.shifted_correlation(partial(van_hove_angle_dist, bins=bins), vectors, segments=segments, skip=skip)  
        time = np.array([t_i for t_i in t for entry in x])
        angle = np.array([entry for t_i in t for entry in x])
        result = S.flatten()
        
        return np.column_stack([time, angle, result])
    
    return van_hove_angle(segments,vectors)


def vanHove_translation(com:NDArray, 
                        segments: int,
                        diameter: float = 0,
                        pore: bool = True,
                        )-> tuple[NDArray, NDArray, NDArray]:
    """
    Calculates van Hove translational dynamics based on centers of masses.

    Args:
        com: NumPy array containing the center of mass coordinates.
        diameter: Diameter of the pore.
        segments: Number of segments to divide the trajectory into.
        pore: Whether or not the system is a pore.

    Returns:
        * NDArray: A NumPy array containing the van Hove translational dynamics at the wall of the pore.
        * NDArray: A NumPy array containing the van Hove translational dynamics at the center of the pore.
        NDArray: A NumPy array containing the bins used.

        * If the system is a pore, otherwise only the overall van Hove translational array is returned.
    """
    if pore:
        bins = np.arange(0, diameter/2+0.1, 0.1)
        time1, vH_wall = mde.correlation.shifted_correlation(partial(mde.correlation.van_hove_self, bins=bins), com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=diameter/2-1, r_max=diameter/2), segments=segments, skip=0.1)
        time2, vH_center = mde.correlation.shifted_correlation(partial(mde.correlation.van_hove_self, bins=bins), com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=0.0, r_max=0.5), segments=segments, skip=0.1)

        return np.column_stack([time1 * 10e-6, vH_wall]), np.column_stack([time2 * 10e-6, vH_center]), bins
    else:
        bins = np.arange(0, diameter/2, 0.05)
        time, vH_all = mde.correlation.shifted_correlation(partial(mde.correlation.van_hove_self, bins=bins), com, selector=partial(mde.coordinates.selector_radial_cylindrical, r_min=diameter/2-1, r_max=diameter/2), segments=segments, skip=0.1)

        return np.column_stack([time * 10e-6, vH_all]), bins