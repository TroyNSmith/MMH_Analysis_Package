import numpy as np
import mdevaluate as mde
import os

from functools import partial
from util.Backup import BackupFile


class TrajectoryAnalysis:
    def __init__(
        self,
        SimulationDirectory: str,
        Trajectory: str,
        Topology: str,
        OutputDirectory: str = None,
        nSegments: int = 100,
        Resname: str = None,
        Atoms: str = None,
        OverwriteExisting: bool = False,
    ):
        self.Coordinates = mde.open(
            directory=SimulationDirectory, trajectory=Trajectory, topology=Topology
        )

        if Resname is not None:
            assert len(Atoms) == 2, (
                "Two atoms corresponding to the designated resname must be provided."
            )
            self.Resname = Resname
            self.Coordinates = self._centers_of_masses()

        self.nSegments = nSegments
        self.OverwriteExisting = OverwriteExisting

        if OutputDirectory is None:
            self.OutputDirectory = os.path.join(
                SimulationDirectory,
                "Analysis/Data_files",
                f"{Resname}_{Atoms[0]}_{Atoms[1]}" if Resname is not None else "",
            )
        else:
            self.OutputDirectory = OutputDirectory

        if not os.path.exists(self.OutputDirectory):
            os.makedirs(self.OutputDirectory)

    def _centers_of_masses(self) -> mde.coordinates.Coordinates:
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

        return center_of_masses(
            self.Coordinates,
            atoms=self.Coordinates.subset(
                residue_name=self.Resname
            ).atom_subset.indices,
        ).nojump

    def _multi_radial_selector(
        Atoms: mde.coordinates.CoordinateFrame,
        rBins: np.ndarray,
    ) -> list:
        """
        Sorts atoms in Trajectory into radial bins.

        :param Trajectory: CoordinateFrame object from mdevaluate.
        :param rBins: Radial bins for sorting.
        :return Indices: Sorted list of atom indices by radial bin.
        """
        Indices = []
        for i in range(len(rBins) - 1):
            index = mde.coordinates.selector_radial_cylindrical(
                Atoms, r_min=rBins[i], r_max=rBins[i + 1]
            )
            Indices.append(index)
        return Indices

    def AveISF(
        self,
        qLength: float = None,
        Resolved: bool = False,
        nBins: int = 0,
        Diameter: float = 0,
        Buffer: float = 0.1,
    ):
        """
        Calculates incoherent scattering function based on an averaged shifted correlation.

        :param qLength: Length of the scattering vector, q.
        :param Resolved: Whether to provide radially binned results.
        :param nBins: Number of radial bins for radially resolved analysis.
        :param Diameter: Diameter of the analyzed system.
        :param Buffer: Distance to buffer the radius beyond the diameter of system.
        """
        if qLength is None:
            qLength = self.RDF()

        if Resolved:
            assert Diameter > 0.0, (
                "System diameter must be provided for radially resolved results."
            )
            assert nBins > 0, "Please specify a quantity of radial bins to use."

            Radius = Diameter / 2 + Buffer

            rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

            Filename = f"ISF/ISF_resolved_{nBins}bins.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    return

            Header = f"# q = {round(qLength, 3)} # Time / ps," + ",".join(
                [
                    f"{rBins[i]:.2f} to {rBins[i + 1]:.2f} nm"
                    for i in range(len(rBins) - 1)
                ]
            )

            t, Results = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=qLength),
                self.Coordinates,
                selector=partial(
                    TrajectoryAnalysis._multi_radial_selector, rBins=rBins
                ),
                segments=self.nSegments,
                skip=0.0,
            )

            OutArray = t
            for col in Results:
                OutArray = np.column_stack([OutArray, col])
        else:
            Filename = "ISF/ISF.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    return

            Header = f"# q = {round(qLength, 3)} # Time / ps, ISF"
            t, Results = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=qLength),
                self.Coordinates,
                segments=self.nSegments,
                skip=0.0,
            )

            OutArray = np.column_stack([t, Results])

        try:
            np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
        except FileNotFoundError:
            os.makedirs(os.path.join(self.OutputDirectory, "ISF"))
            np.savetxt(
                OutputFile,
                OutArray,
                header=Header,
                delimiter=",",
            )

    def AveMSD(
        self,
        Axes: str = "all",
        Resolved: bool = False,
        nBins: int = 0,
        Diameter: float = 0.0,
        Buffer: float = 0.1,
    ):
        """
        Calculates mean square displacement based on an averaged shifted correlation.

        :param Axes: Axes to perform analysis on separated by "and". Options: "all", "xy", "xz", "yz", "x", "y", "z". Example: "all and xy".
        :param Resolved: Whether to provide radially binned results.
        :param nBins: Number of radial bins for radially resolved analysis.
        :param Diameter: Diameter of the analyzed system.
        :param Buffer: Distance to buffer the radius beyond the diameter of system.
        """
        Axes = [a.strip().lower() for a in Axes.split("and")]

        for Axis in Axes:
            if Resolved:
                assert Diameter > 0.0, (
                    "System diameter must be provided for radially resolved results."
                )
                assert nBins > 0, "Please specify a quantity of radial bins to use."

                Radius = Diameter / 2 + Buffer

                rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

                Filename = f"MSD/MSD_resolved_{Axis}.csv"
                OutputFile = os.path.join(self.OutputDirectory, Filename)

                if os.path.exists(OutputFile):
                    if self.OverwriteExisting:
                        BackupFile(OutputFile)
                    else:
                        continue

                Header = "Time / ps," + ",".join(
                    [
                        f"{rBins[i]:.2f} to {rBins[i + 1]:.2f} nm"
                        for i in range(len(rBins) - 1)
                    ]
                )

                t, Results = mde.correlation.shifted_correlation(
                    partial(mde.correlation.msd, axis=Axis),
                    self.Coordinates,
                    selector=partial(
                        TrajectoryAnalysis._multi_radial_selector, rBins=rBins
                    ),
                    segments=self.nSegments,
                    skip=0.1,
                    average=True,
                )

                OutArray = t
                for col in Results:
                    OutArray = np.column_stack([OutArray, col])
            else:
                Filename = f"MSD/MSD_{Axis}.csv"
                OutputFile = os.path.join(self.OutputDirectory, Filename)

                if os.path.exists(OutputFile):
                    if self.OverwriteExisting:
                        BackupFile(OutputFile)
                    else:
                        continue

                Header = "Time / ps, MSD"
                t, Results = mde.correlation.shifted_correlation(
                    partial(mde.correlation.msd, axis=Axis),
                    self.Coordinates,
                    segments=self.nSegments,
                )
                OutArray = np.column_stack([t, Results])

            try:
                np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
            except FileNotFoundError:
                os.makedirs(os.path.join(self.OutputDirectory, "MSD"))
                np.savetxt(
                    OutputFile,
                    OutArray,
                    header=Header,
                    delimiter=",",
                )

    def nonGauss(self):
        """
        Calculates non-Gaussian displacements based on centers of masses.

        Args:
            com: NumPy array containing the center of mass coordinates.
            segments: Number of segments to divide the trajectory into.

        Returns:
            NDArray: NumPy array containing the non-Gaussian displacements and corresponding time series.
        """

        Filename = "Etc/nonGauss.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        Header = "Time / ps, Displacement"

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                return

        t, Result = mde.correlation.shifted_correlation(
            mde.correlation.non_gaussian_parameter,
            self.Coordinates,
            segments=self.nSegments,
        )

        OutArray = np.column_stack([t, Result])

        try:
            np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
        except FileNotFoundError:
            os.makedirs(os.path.join(self.OutputDirectory, "Etc"))
            np.savetxt(
                OutputFile,
                OutArray,
                header=Header,
                delimiter=",",
            )

    def RDF(self, rMax: float = 3.0, nBins: int = 1000, ReturnQ: bool = True) -> float:
        """
        Computes a radial distribution function (RDF) to a distance rMax.

        :param rMax: RDF radial distance cut-off value in nm.
        :param nBins: Number of segments of the radius to consider.
        :param ReturnQ: Whether to return the magnitude of the scattering vector, q.

        :return qLength: Magnitude of the scattering vector, q.
        """

        Filename = "RDF/Inter_RDF.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        Header = "r / nm, G(r)"

        Bins = np.arange(0, rMax, rMax / nBins)
        Results = mde.distribution.time_average(
            partial(mde.distribution.rdf, bins=Bins),
            self.Coordinates,
            segments=self.nSegments,
            skip=0.01,
        )
        OutArray = np.column_stack([Bins[:-1], Results])

        if not os.path.exists(OutputFile):
            try:
                np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
            except FileNotFoundError:
                os.makedirs(os.path.join(self.OutputDirectory, "RDF"))
                np.savetxt(
                    OutputFile,
                    OutArray,
                    header=Header,
                    delimiter=",",
                )
        else:
            if self.OverwriteExisting:
                BackupFile(OutputFile)
                np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")

        if ReturnQ:
            YMaxIdx = np.argmax(OutArray[:, 1])
            XAtMax = OutArray[YMaxIdx, 0]
            qLength = 2 * np.pi / XAtMax

            return qLength

    def vanHove(
        self,
        Diameter: float = 0.0,
        Translational: bool = False,
        Rotational: bool = False,
        nBins: int = 0,
        Buffer: float = 0.1,
    ):
        """
        Calculates mean square displacement based on an averaged shifted correlation.

        :param Diameter: Diameter of the analyzed system.
        :param Resolved: Whether to provide radially binned results.
        :param nBins: Number of radial bins for radially resolved analysis.
        :param Buffer: Distance to buffer the radius beyond the diameter of system.
        """
        if Translational:
            assert Diameter > 0.0, (
                "System diameter must be provided for radially resolved results."
            )
            assert nBins > 0, "Please specify a quantity of radial bins to use."

            Radius = Diameter / 2 + Buffer

            rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

            Filename = f"Etc/TranslVanHove_{nBins}bins.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    return

            Header = "Time / ps," + ",".join(
                [
                    f"{rBins[i]:.2f} to {rBins[i + 1]:.2f} nm"
                    for i in range(len(rBins) - 1)
                ]
            )

            t, Results = mde.correlation.shifted_correlation(
                partial(mde.correlation.van_hove_self, bins=rBins),
                self.Coordinates,
                segments=self.nSegments,
                skip=0.1,
            )

            OutArray = t
            for col in Results.T:
                OutArray = np.column_stack([OutArray, col])

        elif Rotational:
            assert Diameter > 0.0, (
                "System diameter must be provided for radially resolved results."
            )
            assert nBins > 0, "Please specify a quantity of radial bins to use."

            Radius = Diameter / 2 + Buffer

            rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

            Filename = f"Etc/TranslVanHove_{nBins}bins.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    return

            Header = "Time / ps," + ",".join(
                [
                    f"{rBins[i]:.2f} to {rBins[i + 1]:.2f} nm"
                    for i in range(len(rBins) - 1)
                ]
            )



            OutArray = t
            for col in Results.T:
                OutArray = np.column_stack([OutArray, col])

        try:
            np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
        except FileNotFoundError:
            os.makedirs(os.path.join(self.OutputDirectory, "Etc"))
            np.savetxt(
                OutputFile,
                OutArray,
                header=Header,
                delimiter=",",
            )
        
        except UnboundLocalError:
            raise UnboundLocalError(
                "Please designate a type of van Hove function to analyze (translational or rotational)."
            )
