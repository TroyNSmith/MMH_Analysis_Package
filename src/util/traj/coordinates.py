import mdtraj as md
from mdtraj import Trajectory
import numpy as np


class CoordIO:
    @staticmethod
    def load_traj(traj: str, top: str = None) -> Trajectory:
        """
        :param trajectory: Path to trajectory in .xtc, .trr, ... format.
        :param topology: Path to topology in .pdb, .gro, ... format.
        :return Universe: Trajectory/topology as mdtraj object.
        """
        if ".xtc" in traj:
            if top is None:
                raise SystemError(".xtc file format requires that a topology file (.gro, .pdb) also be provided.")
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

def GatherCOM(Universe: md.Trajectory) -> md.Trajectory:
    traj = Universe

    residue_names = [residue.name for residue in traj.topology.residues]
    group_indices = [[atom.index for atom in residue.atoms] for residue in traj.topology.residues]

    masses = np.array([atom.element.mass for atom in traj.topology.atoms])

    def compute_com(traj, group_indices, masses):
        n_frames = traj.n_frames
        n_groups = len(group_indices)
        com_xyz = np.zeros((n_frames, n_groups, 3), dtype=np.float32)

        for g, indices in enumerate(group_indices):
            group_mass = masses[indices]
            total_mass = group_mass.sum()
            weighted_pos = traj.xyz[:, indices, :] * group_mass[None, :, None]
            com_xyz[:, g, :] = weighted_pos.sum(axis=1) / total_mass

        return com_xyz

    com_xyz = compute_com(traj, group_indices, masses)

        # Build fake topology with one atom per residue
    top = md.Topology()
    chain = top.add_chain()
    for i, resname in enumerate(residue_names):
        res = top.add_residue(resname, chain)
        atom = top.add_atom(f"COM_{i}", md.element.carbon, res)

    # Check number of atoms before passing to Trajectory
    assert top.n_atoms == com_xyz.shape[1], f"Topology atom count {top.n_atoms} != COM shape {com_xyz.shape[1]}"

    # Construct new trajectory
    com_traj = md.Trajectory(xyz=com_xyz, topology=top, time=traj.time)
    
    for atom in com_traj.topology.atoms:
        print(atom.name)

    return com_traj