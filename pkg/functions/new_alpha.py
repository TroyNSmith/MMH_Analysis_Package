__author__ = 'Markus Hoffmann'

from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import mdevaluate as md

workdir = "/home/tee/pore_files_FINAL/simulations/noreservoir/pore_D3_L6_W2_S5.0_E0.0_A0.2_V0.2_no_reservoir_N1/OCT/328K/5_nvt_prod_system"
resname = "OCT"
q_const = 11.64
segments = 10

print(workdir)
print(resname)

def get_coordinates(workdir): 
    trajectory = md.open(f"{workdir}", trajectory="out/traj.xtc", topology="run.tpr", nojump=True)
    return trajectory

def center_of_masses(trajectory, resname):
    @md.coordinates.map_coordinates
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
    atoms = trajectory.subset(residue_name=resname).atom_subset.indices
#    print(atoms[0])
    com = center_of_masses(trajectory, atoms=atoms).nojump
    return com
"""
not needed after all, as Robin's Chi4 calculation was wrong"

def isf_mean_var(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    q: float = 22.7,
    trajectory: Coordinates = None,
    axis: str = "all",
) -> float:

 #   Incoherent intermediate scattering function. To specify q, use
 #   water_isf = functools.partial(isf, q=22.77) # q has the value 22.77 nm^-1

     if trajectory is None:
        displacements = start_frame - end_frame
    else:
        displacements = displacements_without_drift(start_frame, end_frame, trajectory)
    if axis == "all":
        distance = (displacements**2).sum(axis=1) ** 0.5
        return np.sinc(distance * q / np.pi).mean(), np.sinc(distance * q / np.pi).var()
    elif axis == "xy" or axis == "yx":
        distance = (displacements[:, [0, 1]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean(), np.real(jn(0, distance * q)).var()
    elif axis == "xz" or axis == "zx":
        distance = (displacements[:, [0, 2]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean(), np.real(jn(0, distance * q)).var()
    elif axis == "yz" or axis == "zy":
        distance = (displacements[:, [1, 2]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean(), np.real(jn(0, distance * q)).var()
    elif axis == "x":
        distance = np.abs(displacements[:, 0])
        return np.mean(np.cos(np.abs(q * distance)))
    elif axis == "y":
        distance = np.abs(displacements[:, 1])
        return np.mean(np.cos(np.abs(q * distance)))
    elif axis == "z":
        distance = np.abs(displacements[:, 2])
        return np.mean(np.cos(np.abs(q * distance)))
    else:
        raise ValueError('Parameter axis has to be ether "all", "x", "y", or "z"!')

"""

#   return (np.mean(r**2) / ((1 + 2 / dimensions) * (np.mean(r) ** 2))) - 1

def non_Gauss(workdir, trajectory, com, resname, segments):
    t, S = md.correlation.shifted_correlation(
        partial(md.correlation.revised_alpha_parameter),
        com, average=False, description='rough', segments=segments)
    t, T = md.correlation.shifted_correlation(
        partial(md.correlation.revised_alpha_parameter),
        com, average=True, description='rough', segments=segments)

    m2 = S[:,:,1].mean(axis=0)
    m4 = S[:,:,2].mean(axis=0) # here the moments from each segment are averaged

    d = 3
    S = (m4 / ((1 + 2 / d) * m2 ** 2)) - 1 # finally alpha_2 is calculated

    S[0] = 0
    time, result = md.correlation.shifted_correlation(partial(md.correlation.non_gaussian_parameter),com, segments= segments)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{resname}_alpha")
    ax.plot(time, np.column_stack([result, S]))

    ax.set_xscale("log")  
    #ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$non-gauss$")
    plt.show()
    raise NotImplementedError
    fig.savefig(workdir + "/analysis/graphs/newnonGauss_" + resname)
    df = pd.DataFrame({'# t in ps': t, 'nonGauss': S})
    df.to_csv(workdir + "/analysis/xvg_files/newnonGauss_" + resname +".xvg", sep=' ', index=False)


def main():
    trajectory = get_coordinates(workdir)
    com = center_of_masses(trajectory, resname)
    non_Gauss(workdir, trajectory, com, resname, segments)
    
if __name__ == '__main__':
    main()
