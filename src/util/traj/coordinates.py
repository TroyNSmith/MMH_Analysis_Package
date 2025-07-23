import mdtraj as md
from mdtraj import Trajectory
import numpy as np


class CoordIO:
    @staticmethod
    def Universe(traj: str, top: str = None) -> np.ndarray:


    @staticmethod
    def load_traj(traj: str, top: str = None) -> Trajectory:
        """
        :param trajectory: Path to trajectory in .xtc, .trr, ... format.
        :param topology: Path to topology in .pdb, .gro, ... format.
        :return Universe: Trajectory/topology as mdtraj object.
        """
        if ".xtc" in traj:
            return CoordIO._read_xtc(traj, top)

    @staticmethod
    def _read_xtc(traj: str, top: str) -> Trajectory:
        """
        Returns an object containing topology and trajectory information.

        :param trajectory: Path to trajectory in .xtc format.
        :param topology: Path to topology in .gro format.
        :return Universe: Trajectory/topology as mdtraj object.
        """
        return md.load(traj, top=top)
