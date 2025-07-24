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
                raise SystemError(
                    ".xtc file format requires that a topology file (.gro, .pdb) also be provided."
                )
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


def GenCOM(Universe: md.Trajectory, Selection: str = "all") -> md.Trajectory:
    """
    Generate a center-of-mass trajectory for selected residues using a selection string.

    :param Universe: Input trajectory (mdtraj.Trajectory).
    :param Selection: MDTraj-style selection string, e.g., 'resname OCT', 'not resname PORE'.
    :return comUniverse: New mdtraj.Trajectory object with COM coordinates per selected residue.
    """
    selAtomIdxs = Universe.topology.select(Selection)
    
    selRes = [
        res for res in Universe.topology.residues
        if any(atom.index in selAtomIdxs for atom in res.atoms)
    ]

    grIdxs = [[atom.index for atom in res.atoms if atom.index in selAtomIdxs]
              for res in selRes]
    
    grIdxs = [grp for grp in grIdxs if grp]
    selRes = [res for res, grp in zip(selRes, grIdxs) if grp]
    Masses = np.array([atom.element.mass for atom in Universe.topology.atoms])

    def _comCalc(Universe, grIdxs, masses):
        nFrames = Universe.n_frames
        nGroups = len(grIdxs)
        comXYZ = np.zeros((nFrames, nGroups, 3), dtype=np.float32)

        for g, Idxs in enumerate(grIdxs):
            gMass = masses[Idxs]
            tMass = gMass.sum()
            weightedPos = Universe.xyz[:, Idxs, :] * gMass[None, :, None]
            comXYZ[:, g, :] = weightedPos.sum(axis=1) / tMass

        return comXYZ

    comXYZ = _comCalc(Universe, grIdxs, Masses)

    tempTop = md.Topology()
    Chain = tempTop.add_chain()
    for Res in selRes:
        newRes = tempTop.add_residue(Res.name, Chain)
        tempTop.add_atom(f"COM_{Res.index}", md.element.carbon, newRes)

    assert tempTop.n_atoms == comXYZ.shape[1], (
        f"Topology atom count {tempTop.n_atoms} != COM shape {comXYZ.shape[1]}"
    )

    comUniverse = md.Trajectory(
        xyz=comXYZ,
        topology=tempTop,
        time=Universe.time,
        unitcell_angles=Universe.unitcell_angles,
        unitcell_lengths=Universe.unitcell_lengths,
    )

    return comUniverse