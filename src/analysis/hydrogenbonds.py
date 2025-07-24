import numpy as np
import mdtraj as md


class HydrogenBonds:
    def __init__(self, traj: md.Trajectory, distance_cutoff=0.35, angle_cutoff=30.0):
        """
        Detect hydrogen bonds using custom geometric criteria.

        :param traj: MDTraj trajectory
        :param distance_cutoff: Max donor-acceptor distance (nm)
        :param angle_cutoff: Max D-H···A angle (degrees)
        """
        
        HNOIdxs = traj.topology.select("element H or element O or element N")
        HNOTop = traj.topology.subset(HNOIdxs)

        HNOPairs = traj.topology.select_pairs('element H', 'element O or element N')
        HNODists = md.compute_distances(traj, HNOPairs)

        print(HNODists)
        '''
        for bond in HNOTop.bonds:
            if 'H' in bond.atom1.name or 'H' in bond.atom2.name:
                print(bond.atom1.residue)
        '''