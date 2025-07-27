import numpy as np
import mdevaluate as mde
import os
import warnings

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
        ResName: str = None,
        Atoms: list = None,
        OverwriteExisting: bool = False,
    ):
        self.Coordinates = mde.open(
            directory=SimulationDirectory, trajectory=Trajectory, topology=Topology
        )

        if ResName is not None:
            if Atoms is None:
                raise ValueError(
                    "Two atoms from the topology must be specified when performing residue analysis to facilitate appropriate center of mass calculations."
                )
            assert len(Atoms) == 2, (
                "Two atoms corresponding to the designated resname must be provided."
            )
            self.ResName = ResName
            self.Vectors = self._Vectors(Atoms=Atoms)
            self.Coordinates = self._CentersOfMasses()

        self.nSegments = nSegments
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

    def _CentersOfMasses(self) -> mde.coordinates.Coordinates:
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
                residue_name=self.ResName
            ).atom_subset.indices,
        ).nojump

    def _Vectors(self, Atoms: list):
        """
        Return residual vectors pointing from atom 1 to atom 2.
        """
        Atom1Idxs = self.Coordinates.subset(
            atom_name=Atoms[0], residue_name=self.ResName
        ).atom_subset.indices

        Atom2Idxs = self.Coordinates.subset(
            atom_name=Atoms[1], residue_name=self.ResName
        ).atom_subset.indices

        Vectors = mde.coordinates.vectors(
            self.Coordinates,
            atom_indices_a=Atom1Idxs,
            atom_indices_b=Atom2Idxs,
            normed=True,
        )

        return Vectors

    def _MultiRadialSelector(
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

    def _SaveCSV(self, OutputFile: str, OutArray: np.ndarray, Header: str = None):
        """
        Saves a numpy array as a .csv file.

        :param OutputFile: File path to output destination (include filename with .csv extension).
        :param OutArray: Numpy array to be saved at output.
        :param Header: Header row for .csv file as a comma separated string.
        """
        if ".csv" not in OutputFile:
            warnings.warn(
                f"Please include the output file name and .csv extension in {OutputFile}. \
                              Saving file as {os.path.join(OutputFile, 'Out.csv')} instead."
            )
            OutputFile = os.path.join(OutputFile, "Out.csv")

        try:
            np.savetxt(OutputFile, OutArray, header=Header, delimiter=",")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(OutputFile))
            np.savetxt(
                OutputFile,
                OutArray,
                header=Header,
                delimiter=",",
            )

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
            qLength = self.InterRDF()

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
                selector=partial(TrajectoryAnalysis._MultiRadialSelector, rBins=rBins),
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

        self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

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
                        TrajectoryAnalysis._MultiRadialSelector, rBins=rBins
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

            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

    def Chi4Susceptibility(self, qLength: float = None):
        """
        Calculates fourth-order susceptibility based on provided coordinates.

        :param qLength: Length of the scattering vector, q.
        """
        if qLength is None:
            qLength = self.InterRDF()

        Filename = "Etc/Chi4.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                return

        Header = "t / ps, Chi4, Chi4 (Rolling Average)"

        t, Results = mde.correlation.shifted_correlation(
            partial(mde.correlation.isf, q=qLength),
            self.Coordinates,
            average=False,
            segments=50,
        )

        RawResults = len(self.Coordinates[0]) * Results.var(axis=0) * 1e5
        SmoothedResults = mde.utils.moving_average(RawResults, 5)

        OutArray = np.column_stack([t[2:-2], RawResults[2:-2], SmoothedResults])

        self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

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

        self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

    def InterRDF(self, rMax: float = 3.0, nBins: int = 1000, ReturnQ: bool = True) -> float:
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
            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)
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

            def _vanHoveAngleDist(start, end, aBins):
                ScalarProduct = (start * end).sum(axis=-1)
                Angle = np.arccos(ScalarProduct)
                Angle = Angle[(Angle >= 0) * (Angle <= np.pi)]
                Histogram, _ = np.histogram(Angle * 360 / (2 * np.pi), aBins)
                return 1 / len(start) * Histogram

            aBins = np.linspace(0, 180, 361)
            x = aBins[1:] - (aBins[1] - aBins[0]) / 2

            Filename = "Etc/RotVanHove.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    return

            x = aBins[1:] - (aBins[1] - aBins[0]) / 2

            Time, Result = mde.correlation.shifted_correlation(
                function=partial(_vanHoveAngleDist, aBins=aBins),
                frames=self.Vectors,
                segments=self.nSegments,
            )
            t = np.array([t_i for t_i in Time for entry in x])
            Angle = np.array([entry for t_i in Time for entry in x])

            OutArray = np.column_stack([t, Angle, Result.flatten()])

            Header = "t / ps, Angle / degrees, Result"

        else:
            raise ValueError(
                "Please designate either translational or rotational for van Hove analysis."
            )

        try:
            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)
        except UnboundLocalError:
            raise UnboundLocalError(
                "Please designate a type of van Hove function to analyze (translational or rotational)."
            )

    def zAxisAlignment(self):
        """
        Calculates residual vectors' alignment with Z axis.
        """

        Filename = "Etc/zAxis_Alignment.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                return

        Header = "t / ps, Angle / degrees, Result"

        zVector = [0, 0, 1]
        aBins = np.linspace(0, 180, 361)
        x = aBins[1:] - (aBins[1] - aBins[0]) / 2

        def _Angles(start, end, zVector, aBins):
            Angle = np.arccos((start * zVector).sum(axis=-1))
            Angle = Angle[(Angle >= 0) * (Angle <= np.pi)]
            Histogram, _ = np.histogram(Angle * 360 / (2 * np.pi), aBins)
            return 1 / len(start) * Histogram

        Time, Result = mde.correlation.shifted_correlation(
            partial(_Angles, zVector=zVector, aBins=aBins),
            self.Vectors,
            segments=self.nSegments,
        )

        t = np.array([t_i for t_i in Time for entry in x])
        Angle = np.array([entry for t_i in Time for entry in x])

        OutArray = np.column_stack([t, Angle, Result.flatten()])

        self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

    def zAxisRadialPos(self):
        """
        Calculates Z-axis radial positions.
        """

        Filename = "Etc/zAxis_Radial_Positions.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                return
            
        Header = "t / ps, Angle / degrees, Result"
            
        rBins = np.linspace(-1, 1, 201)
        x = rBins[1:] - (rBins[1] - rBins[0]) / 2

        def z_comp(start, end, rBins):
            VectorsLengths = np.linalg.norm(start, axis=1)
            zComponent = start[:, 2] / VectorsLengths
            Histogram, _ = np.histogram(zComponent, rBins)
            return 1 / len(start) * Histogram

        Time, Result = mde.correlation.shifted_correlation(
            partial(z_comp, bins=rBins), self.Vectors, segments=self.nSegments
        )

        t = np.array([t_i for t_i in Time for entry in x])
        Angle = np.array([entry for t_i in Time for entry in x])

        OutArray = np.column_stack([t, Angle, Result.flatten()])

        self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)