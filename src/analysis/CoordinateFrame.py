import numpy as np
import matplotlib.pyplot as plt
import mdevaluate as mde
import os
import pandas as pd
import warnings

from cycler import cycler
from functools import partial
from util.Backup import BackupFile


class TrajectoryAnalysis:
    def __init__(
        self,
        SimulationDirectory: str,
        Trajectory: str,
        Topology: str,
        PlottingMode: str,
        OutputDirectory: str = None,
        nSegments: int = 100,
        ResName: str = None,
        Atoms: list = None,
        OverwriteExisting: bool = False,
    ):
        self.InitCoordinates = mde.open(
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
            self.Atoms = Atoms
            self.ResName = ResName
            self.Vectors = self._Vectors()
            self.Coordinates = self._CentersOfMasses()

        else:
            self.Coordinates = self.InitCoordinates

        self.nSegments = nSegments
        self.OverwriteExisting = OverwriteExisting
        self.PlottingMode = TrajectoryAnalysis._match_plotting_mode(PlottingMode)

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
            self.InitCoordinates,
            atoms=self.InitCoordinates.subset(
                residue_name=self.ResName
            ).atom_subset.indices,
        ).nojump

    def _Vectors(self):
        """
        Return residual vectors pointing from atom 1 to atom 2.
        """
        Atom1Idxs = self.InitCoordinates.subset(
            atom_name=self.Atoms[0], residue_name=self.ResName
        ).atom_subset.indices

        Atom2Idxs = self.InitCoordinates.subset(
            atom_name=self.Atoms[1], residue_name=self.ResName
        ).atom_subset.indices

        Vectors = mde.coordinates.vectors(
            self.InitCoordinates,
            atom_indices_a=Atom1Idxs,
            atom_indices_b=Atom2Idxs,
            normed=True,
        )

        return Vectors

    @staticmethod
    def _MultiRadialSelector(
        Atoms: mde.coordinates.CoordinateFrame,
        rBins: np.ndarray,
    ) -> list:
        """
        Sorts atoms in Trajectory into radial bins.

        :param Trajectory: CoordinateFrame object from mdevaluate.
        :param rBins: Radial bins for sorting.
        :return Idxs: Sorted list of atom indices by radial bin.
        """
        Idxs = []
        for i in range(len(rBins) - 1):
            Idx = mde.coordinates.selector_radial_cylindrical(
                Atoms, r_min=rBins[i], r_max=rBins[i + 1]
            )
            Idxs.append(Idx)
        return Idxs

    @staticmethod
    def _LinePlot(
        PlotName: str,
        XData: pd.DataFrame,
        YData: pd.DataFrame,
        XAxisLabel: str = "",
        XScale: str = "linear",
        YAxisLabel: str = "",
        YScale: str = "linear",
        UseHeaders: bool = True,
        PlotSize: tuple = (8, 6),
        AxisFontSize: float = 16,
        TickFontSize: float = 14,
        ColorCycler: list = None,
    ):
        if len(YData.columns.to_list()) <= 1:
            UseHeaders = False

        if ColorCycler is None:
            ColorCycler = cycler(
                "color",
                [
                    "#CC6677",
                    "#332288",
                    "#DDCC77",
                    "#117733",
                    "#88CCEE",
                    "#882255",
                    "#44AA99",
                    "#999933",
                    "#AA4499",
                    "#77AADD",
                    "#EE8866",
                    "#EEDD88",
                    "#FFAABB",
                    "#99DDFF",
                    "#44BB99",
                    "#BBCC33",
                    "#AAAA00",
                    "#DDDDDD",
                ],
            )

        Labels = YData.columns if UseHeaders else None
        plt.rc("axes", prop_cycle=ColorCycler)

        fig, ax = plt.subplots(figsize=PlotSize)

        ax.plot(XData, YData, label=Labels)

        ax.set_xlabel(XAxisLabel, fontsize=AxisFontSize)
        ax.set_xscale(XScale)
        ax.tick_params(axis="x", labelsize=TickFontSize)

        ax.set_ylabel(YAxisLabel, fontsize=AxisFontSize)
        ax.set_yscale(YScale)
        ax.tick_params(axis="y", labelsize=TickFontSize)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        fig.tight_layout()

        fig.savefig(PlotName)

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

    def _match_plotting_mode(PlottingMode: str):
        input = PlottingMode.lower().strip()
        if input.startswith("off"):
            return "Off"
        elif input.startswith("init"):
            return "Initial"
        elif input.startswith("over"):
            return "Overwrite existing"
        else:
            raise ValueError(
                f"Unknown plotmode '{input}'. Use -pm / --plotmode 'Off', 'Initial', or 'Overwrite existing'."
            )
        
    def _check_for_existing(self, file_pth: str) -> bool:
        """
        Check if a file is existing and determine whether to (re)generate it.

        :param file_pth: Path to the queried file.
        :return generate_file: Boolean 
        """
        generate_file = False
        if os.path.exists(file_pth):
            if self.OverwriteExisting:
                BackupFile(file_pth)
            else:
                generate_file = True
        
        return generate_file

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
            qLength = self.RDF(Mode="COM")

        SkipCalculation = False

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
                    SkipCalculation = True

            if not SkipCalculation:
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
                        TrajectoryAnalysis._MultiRadialSelector, rBins=rBins
                    ),
                    segments=self.nSegments,
                    skip=0.0,
                )

                OutArray = t
                for col in Results:
                    OutArray = np.column_stack([OutArray, col])

                self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        else:
            Filename = "ISF/ISF.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    SkipCalculation = True

            if not SkipCalculation:
                Header = f"# q = {round(qLength, 3)} # Time / ps, ISF"
                t, Results = mde.correlation.shifted_correlation(
                    partial(mde.correlation.isf, q=qLength),
                    self.Coordinates,
                    segments=self.nSegments,
                    skip=0.0,
                )

                OutArray = np.column_stack([t, Results])

                self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel=r"$t$ / ps",
                XScale="log",
                YAxisLabel=r"F(Q, t)",
                UseHeaders=True,
                PlotSize=(8, 6),
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

        SkipCalculation = False

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
                        SkipCalculation = True

                if not SkipCalculation:
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

                    self._SaveCSV(
                        OutputFile=OutputFile, OutArray=OutArray, Header=Header
                    )

            else:
                Filename = f"MSD/MSD_{Axis}.csv"
                OutputFile = os.path.join(self.OutputDirectory, Filename)

                if os.path.exists(OutputFile):
                    if self.OverwriteExisting:
                        BackupFile(OutputFile)
                    else:
                        SkipCalculation = True

                if not SkipCalculation:
                    Header = "Time / ps, MSD"

                    t, Results = mde.correlation.shifted_correlation(
                        partial(mde.correlation.msd, axis=Axis),
                        self.Coordinates,
                        segments=self.nSegments,
                    )

                    OutArray = np.column_stack([t, Results])

                    self._SaveCSV(
                        OutputFile=OutputFile, OutArray=OutArray, Header=Header
                    )

            if self.PlottingMode != "Off":
                PlotName = os.path.splitext(OutputFile)[0] + ".png"

                if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                    return

                DataIn = pd.read_csv(OutputFile)

                TrajectoryAnalysis._LinePlot(
                    PlotName,
                    DataIn.iloc[:, 0],
                    DataIn.iloc[:, 1:],
                    XAxisLabel=r"$\mathbf{\mathit{t}}$ / ps",
                    XScale="log",
                    YAxisLabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                    YScale="log",
                    UseHeaders=True,
                    PlotSize=(8, 6),
                )

    def Chi4Susceptibility(self, qLength: float = None):
        """
        Calculates fourth-order susceptibility.

        :param qLength: Length of the scattering vector, q.
        """

        SkipCalculation = False

        Filename = "Etc/Chi4.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                SkipCalculation = True

        if not SkipCalculation:
            if qLength is None:
                qLength = self.RDF()

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

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel=r"$t$ / ps",
                XScale="log",
                YAxisLabel=r"$\chi_4 \cdot 10^{-5}$",
                UseHeaders=True,
                PlotSize=(8, 6),
            )

    def nonGauss(self):
        """
        Calculates non-Gaussian displacements.
        """

        SkipCalculation = False

        Filename = "Etc/nonGauss.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                SkipCalculation = True

        if not SkipCalculation:
            Header = "Time / ps, Displacement"

            t, Result = mde.correlation.shifted_correlation(
                mde.correlation.non_gaussian_parameter,
                self.Coordinates,
                segments=self.nSegments,
            )

            OutArray = np.column_stack([t, Result])

            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel=r"$t$ / ps",
                XScale="log",
                YAxisLabel="Non-Gaussian Displacement",
                UseHeaders=True,
                PlotSize=(8, 6),
            )

    def RadialDensity(
        self,
        Diameter: float,
        Groups: list = None,
        nBins: int = 100,
        Buffer: float = 0.1,
    ):
        """
        Calculates radial density functions for atom, residue pairs specified in Groups.

        :param Diameter: Diameter of the analyzed system.
        :param Groups: A list of atom, residue pairs to perform analysis on. Example: [['O01', 'OCT'], ['CO0', 'OCT'], ['ATOM', 'RES'], ...]
        :param nBins: Number of bins to divide the radius into.
        :param Buffer: Distance to buffer the radius beyond the diameter of system.
        """

        SkipCalculation = False

        Filename = "RDF/Radial_Densities.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                SkipCalculation = True

        if not SkipCalculation:
            assert Diameter > 0.0, (
                "System diameter must be provided for radially resolved results."
            )

            Radius = Diameter / 2 + Buffer
            rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

            r = 0.5 * (rBins[:-1] + rBins[1:])

            Header = "r / nm," + ",".join(
                [f"{Groups[i][1]}:{Groups[i][0]}" for i in range(len(Groups))]
            )

            Results = []
            if Groups is not None:
                for AtomName, ResName in Groups:
                    Result = mde.distribution.time_average(
                        partial(mde.distribution.radial_density, bins=rBins),
                        self.InitCoordinates.subset(
                            atom_name=AtomName, residue_name=ResName
                        ),
                        segments=self.nSegments,
                        skip=0.01,
                    )
                    Results.append(Result)

            OutArray = r
            for Result in Results:
                OutArray = np.column_stack([OutArray, Result])

            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel="r / nm",
                YAxisLabel=r"Number Density / counts \cdot nm$^{-3}$",
                UseHeaders=True,
                PlotSize=(8, 6),
            )

    def RDF(
        self, rMax: float = 2.5, ReturnQ: bool = True, Mode: str = "Total"
    ) -> float:
        """
        Computes a radial distribution function (RDF) to a distance rMax.

        :param rMax: RDF radial distance cut-off value in nm.
        :param nBins: Number of segments of the radius to consider.
        :param ReturnQ: Whether to return the magnitude of the scattering vector, q.
        :param Mode: "COM" | "Total" | "Intra" | "Inter"

        :return qLength: Magnitude of the scattering vector, q.
        """
        if Mode == "COM":
            Filename = "RDF/COM_RDF.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            Header = "r / nm, G(r)"

            Bins = np.arange(0, rMax, 0.01)
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

            if self.PlottingMode != "Off":
                PlotName = os.path.splitext(OutputFile)[0] + ".png"

                if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                    return

                DataIn = pd.read_csv(OutputFile)

                TrajectoryAnalysis._LinePlot(
                    PlotName,
                    DataIn.iloc[:, 0],
                    DataIn.iloc[:, 1:],
                    XAxisLabel="r / nm",
                    YAxisLabel="g(r)",
                    UseHeaders=True,
                    PlotSize=(8, 6),
                )

        else:
            assert len(self.Atoms) == 2, (
                "Two atoms and their corresponding residue(s) must be specified when performing non-COM RDFS."
            )

            SkipCalculation = False

            Filename = f"RDF/{Mode}_RDF.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    SkipCalculation = True

            if not SkipCalculation:
                Header = "r / nm, G(r)"

                Atom1Coords = self.InitCoordinates.subset(
                    atom_name=self.Atoms[0], residue_name=self.ResName
                )

                Atom2Coords = self.InitCoordinates.subset(
                    atom_name=self.Atoms[1], residue_name=self.ResName
                )

                Bins = np.arange(0, rMax, 0.01)
                Results = mde.distribution.time_average(
                    function=partial(mde.distribution.rdf, bins=Bins, Mode=Mode),
                    coordinates=Atom1Coords,
                    coordinates_b=Atom2Coords,
                    segments=self.nSegments,
                )
                OutArray = np.column_stack([Bins[:-1], Results])

                self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

            if self.PlottingMode != "Off":
                PlotName = os.path.splitext(OutputFile)[0] + ".png"

                if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                    return

                DataIn = pd.read_csv(OutputFile)

                TrajectoryAnalysis._LinePlot(
                    PlotName,
                    DataIn.iloc[:, 0],
                    DataIn.iloc[:, 1:],
                    XAxisLabel="r / nm",
                    YAxisLabel="g(r)",
                    UseHeaders=True,
                    PlotSize=(8, 6),
                )

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
        :param Translational: Whether to perform translational van Hove analysis.
        :param Rotational: Whether to perform rotational van Hove analysis.
        :param Resolved: Whether to provide radially binned results.
        :param nBins: Number of radial bins for radially resolved analysis.
        :param Buffer: Distance to buffer the radius beyond the diameter of system.
        """
        if not Translational or Rotational:
            raise ValueError(
                "Please indicate whether to perform translational or rotational van Hove analysis."
            )

        if Translational:
            assert Diameter > 0.0, (
                "System diameter must be provided for radially resolved results."
            )
            assert nBins > 0, "Please specify a quantity of radial bins to use."

            SkipCalculation = False

            Filename = f"Etc/TranslVanHove_{nBins}bins.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    SkipCalculation = True

            if not SkipCalculation:
                Radius = Diameter / 2 + Buffer

                rBins = np.arange(0.0, Radius + Radius / nBins, Radius / nBins)

                Time, Results = mde.correlation.shifted_correlation(
                    partial(mde.correlation.van_hove_self, bins=rBins),
                    self.Coordinates,
                    segments=self.nSegments,
                    skip=0.1,
                )

                Header = "r / nm," + ",".join([f"{t:.2f} ps" for t in Time])

                r_bincenters = np.array(
                    [0.5 * (rBins[i] + rBins[i + 1]) for i in range(len(rBins) - 1)]
                )

                OutArray = np.column_stack([r_bincenters, Results.T])

                self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

            if self.PlottingMode != "Off":
                PlotName = os.path.splitext(OutputFile)[0] + ".png"

                if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                    return

                DataIn = pd.read_csv(OutputFile)

                TrajectoryAnalysis._LinePlot(
                    PlotName,
                    DataIn.iloc[:, 0],
                    DataIn.iloc[:, 1:],
                    XAxisLabel="r / nm",
                    YAxisLabel="F(r, t)",
                    UseHeaders=True,
                    PlotSize=(8, 6),
                )

        if Rotational:

            def _vanHoveAngleDist(start, end, aBins):
                ScalarProduct = (start * end).sum(axis=-1)
                Angle = np.arccos(ScalarProduct)
                Angle = Angle[(Angle >= 0) * (Angle <= np.pi)]
                Histogram, _ = np.histogram(Angle * 360 / (2 * np.pi), aBins)
                return 1 / len(start) * Histogram

            SkipCalculation = False

            Filename = "Etc/RotVanHove.csv"
            OutputFile = os.path.join(self.OutputDirectory, Filename)

            if os.path.exists(OutputFile):
                if self.OverwriteExisting:
                    BackupFile(OutputFile)
                else:
                    SkipCalculation = True

            if not SkipCalculation:
                aBins = np.linspace(0, 180, 361)

                Time, Result = mde.correlation.shifted_correlation(
                    function=partial(_vanHoveAngleDist, aBins=aBins),
                    frames=self.Vectors,
                    segments=self.nSegments,
                )

                Header = "Angle / degrees," + ",".join([f"{t:.2f}" for t in Time])

                a_bincenters = np.array(
                    [0.5 * (aBins[i] + aBins[i + 1]) for i in range(len(aBins) - 1)]
                )

                OutArray = np.column_stack([a_bincenters, Result.T])

                self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

            if self.PlottingMode != "Off":
                PlotName = os.path.splitext(OutputFile)[0] + ".png"

                if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                    return

                DataIn = pd.read_csv(OutputFile)

                TrajectoryAnalysis._LinePlot(
                    PlotName,
                    DataIn.iloc[:, 0],
                    DataIn.iloc[:, 1:],
                    XAxisLabel=r"$\varphi$",
                    YAxisLabel=r"F($\varphi$, t)",
                    UseHeaders=True,
                    PlotSize=(8, 6),
                )

    def zAxisAlignment(self):
        """
        Calculates residual vectors' alignment with Z axis.
        """
        SkipCalculation = False

        Filename = "Etc/zAxis_Alignment.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                SkipCalculation = True

        if not SkipCalculation:
            Header = "t / ps, Angle / degrees, Result"

            zVector = [0, 0, 1]
            aBins = np.linspace(0, 180, 361)

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

            Header = "Angle / degrees," + ",".join([f"{t:.2f}" for t in Time])

            a_bincenters = np.array(
                [0.5 * (aBins[i] + aBins[i + 1]) for i in range(len(aBins) - 1)]
            )

            OutArray = np.column_stack([a_bincenters, Result.T])

            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel=r"$\varphi$",
                YAxisLabel=r"S($\varphi$, t)",
                UseHeaders=True,
                PlotSize=(8, 6),
            )

    def zAxisRadialPos(self):
        """
        Calculates Z-axis radial positions.
        """
        SkipCalculation = False

        Filename = "Etc/zAxis_Radial_Positions.csv"
        OutputFile = os.path.join(self.OutputDirectory, Filename)

        if os.path.exists(OutputFile):
            if self.OverwriteExisting:
                BackupFile(OutputFile)
            else:
                SkipCalculation = True

        if not SkipCalculation:
            rBins = np.linspace(-1, 1, 201)

            def z_comp(start, end, rBins):
                VectorsLengths = np.linalg.norm(start, axis=1)
                zComponent = start[:, 2] / VectorsLengths
                Histogram, _ = np.histogram(zComponent, rBins)
                return 1 / len(start) * Histogram

            Time, Result = mde.correlation.shifted_correlation(
                partial(z_comp, rBins=rBins), self.Vectors, segments=self.nSegments
            )

            Header = "r / nm," + ",".join([f"{t:.2f}" for t in Time])

            r_bincenters = np.array(
                [0.5 * (rBins[i] + rBins[i + 1]) for i in range(len(rBins) - 1)]
            )

            OutArray = np.column_stack([r_bincenters, Result.T])

            self._SaveCSV(OutputFile=OutputFile, OutArray=OutArray, Header=Header)

        if self.PlottingMode != "Off":
            PlotName = os.path.splitext(OutputFile)[0] + ".png"

            if self.PlottingMode == "Initial" and os.path.exists(PlotName):
                return

            DataIn = pd.read_csv(OutputFile)

            TrajectoryAnalysis._LinePlot(
                PlotName,
                DataIn.iloc[:, 0],
                DataIn.iloc[:, 1:],
                XAxisLabel=r"r / nm",
                YAxisLabel="S(r, t)",
                UseHeaders=True,
                PlotSize=(8, 6),
            )
