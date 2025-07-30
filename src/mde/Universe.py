import MDAnalysis as mda
import os

from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba


class HydrogenBondAnalysis:
    def __init__(
        self,
        SimulationDirectory: str,
        Trajectory: str,
        Topology: str,
        OutputDirectory: str = None,
        ResName: str = None,
        Atoms: list = None,
        SelectDonors: str = "type O or type N",
        SelectHydros: str = "type H",
        SelectAcceps: str = "type O or type N",
        AngleCutoff: int = 150,
        DistanceCutoff: float = 3.5,
        OverwriteExisting: bool = False,
    ):
        self.Universe = mda.Universe(
            os.path.join(SimulationDirectory, Topology),
            os.path.join(SimulationDirectory, Trajectory),
        )

        if ResName is not None:
            if Atoms is None:
                raise ValueError(
                    "Two atoms from the topology must be specified when performing residue analysis to facilitate appropriate center of mass calculations."
                )
            assert len(Atoms) == 2, (
                "Two atoms corresponding to the designated resname must be provided."
            )
            self.Atoms = Atoms
            self.ResName = ResName

        self.OverwriteExisting = OverwriteExisting

        if OutputDirectory is None:
            self.OutputDirectory = os.path.join(
                SimulationDirectory,
                "Analysis/Data_files",
                f"{ResName}_{Atoms[0]}_{Atoms[1]}" if ResName is not None else "",
            )
        else:
            self.OutputDirectory = OutputDirectory

        if not os.path.exists(self.OutputDirectory):
            os.makedirs(self.OutputDirectory)

        self.HBonds = hba(
            universe=self.Universe,
            donors_sel=SelectDonors,
            hydrogens_sel=SelectHydros,
            acceptors_sel=SelectAcceps,
            d_a_cutoff=DistanceCutoff,
            d_h_a_angle_cutoff=AngleCutoff,
            update_selections=False,
        ).run()
