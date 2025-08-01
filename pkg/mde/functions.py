import mde.mdevaluate as mde
import numpy as np
import pandas as pd

from datetime import datetime
from functools import partial
from pathlib import Path

from ..functions import plotting
from .utils.coordinates import multi_radial_selector
from .utils.types import BaseAnalysis


class Chi4Susceptibility(BaseAnalysis):
    def calculate(self, q_magnitude: float) -> pd.DataFrame:
        """
        :param q_magnitude: Magnitude of the scattering vector (q).
        """
        print("Calculating fourth order susceptibility...")

        self._params.q_magnitude = q_magnitude

        column_labels = "t / ps, Chi4, Chi4 (Rolling Average)"

        times, results = mde.correlation.shifted_correlation(
            partial(mde.correlation.isf, q=q_magnitude),
            self._coords,
            average=False,
            segments=self._num_segments,
            skip=self._skip,
        )

        adjust_results = len(self._coords[0]) * results.var(axis=0) * 1e5
        smooth_results = mde.utils.moving_average(adjust_results, 5)
        df = pd.DataFrame(
            np.column_stack([times[2:-2], adjust_results[2:-2], smooth_results]),
            columns=column_labels,
        )
        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label=r"$t$ / ps",
            x_axis_scale="log",
            y_axis_label=r"$\chi_4 \cdot 10^{-5}$",
        )


class IncoherentScatteringFunction(BaseAnalysis):
    def calculate(
        self,
        q_magnitude: float,
        radially_resolved: bool = False,
        pore_diameter: float = None,
        num_bins: int = 10,
        radius_buffer=0.1,
    ) -> pd.DataFrame:
        """
        Calculate the incoherent scattering function.

        :param radially_resolved: Generate radially resolved results.
        :param num_radial_bins: Integer number of cylinders to use in radially resolved analysis.
        :param diameter: Diameter of the pore being radially binned.
        :param radius_buffer: Distance to include beyond the pore diameter for binning.
        """
        print("Calculating ISF...")

        if radially_resolved:
            assert pore_diameter > 0.0, (
                "Pore diameter must be provided for radially resolved results."
            )

            self._set_radial_params(
                radially_resolved, pore_diameter, num_bins, radius_buffer
            )

            radius = pore_diameter / 2 + radius_buffer
            radial_bins = np.arange(0.0, radius + radius / num_bins, radius / num_bins)

            column_labels = ["Time / ps"] + [
                f"{radial_bins[i]:.2f} to {radial_bins[i + 1]:.2f} nm"
                for i in range(len(radial_bins) - 1)
            ]

            times, results = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=self._q_magnitude),
                self._coords,
                selector=partial(multi_radial_selector, radial_bins=radial_bins),
                segments=self._num_segments,
                skip=self._skip,
            )

            df = pd.DataFrame(np.column_stack([times, *results]), columns=column_labels)

        else:
            column_labels = ["Time / ps", "ISF"]

            times, results = mde.correlation.shifted_correlation(
                partial(mde.correlation.isf, q=self._q_magnitude),
                self._coords,
                segments=self._num_segments,
                skip=self._skip,
            )

            df = pd.DataFrame(np.column_stack([times, results]), columns=column_labels)

        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label="t / ps",
            x_axis_scale="log",
            y_axis_label=r"$F(\|\mathbf{q}\|, t)$",
        )


class MeanSquareDisplacement(BaseAnalysis):
    def calculate(
        self,
        axes: str = "all",
        radially_resolved: bool = False,
        pore_diameter: float = 0.0,
        num_bins: int = 10,
        radius_buffer=0.1,
    ) -> pd.DataFrame:
        """
        :param axes: String indicating selections separated by 'and'. Options: 3-d box ("all") | 2-d planes ("xy", "xz", "yz") | 1-d lines ("x", "y", "z")
        :param radially_resolved: Generate radially resolved results.
        :param pore_diameter: Diameter of the pore being radially binned.
        :param num_bins: Number of cylinders to use in radially resolved analysis.
        :param radius_buffer: Distance to include beyond the pore diameter for binning.
        """
        results = []
        axes = [a.strip().lower() for a in axes.split("and")]
        for axis in axes:
            if radially_resolved:
                assert pore_diameter > 0.0, (
                    "Provide pore diameter for radially resolved results."
                )
                assert num_bins > 0, "Specify a quantity of bins to use."

                self._set_radial_params(
                    radially_resolved, pore_diameter, num_bins, radius_buffer
                )

                radius = pore_diameter / 2 + radius_buffer

                bins = np.arange(0.0, radius + radius / num_bins, radius / num_bins)

                column_labels = "t / ps," + ",".join(
                    [
                        f"{bins[i]:.2f} to {bins[i + 1]:.2f} nm"
                        for i in range(len(bins) - 1)
                    ]
                )

                times, result = mde.correlation.shifted_correlation(
                    partial(mde.correlation.msd, axis=axis),
                    self._coords,
                    selector=partial(multi_radial_selector, rBins=bins),
                    segments=self._num_segments,
                    skip=self._skip,
                    average=True,
                )

                results.append(result)

            else:
                column_labels = "t / ps, MSD"

                times, result = mde.correlation.shifted_correlation(
                    partial(mde.correlation.msd, axis=axis),
                    self._coords,
                    segments=self._num_segments,
                )

                results.append(result)

        df = pd.DataFrame(np.column_stack([times, results]), columns=column_labels)
        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label="g(r)",
        )


class NonGaussianDisplacement(BaseAnalysis):
    def calculate(self) -> pd.DataFrame:
        print("Calculating fourth order susceptibility...")

        column_labels = "t / ps, Chi4, Chi4 (Rolling Average)"

        times, results = mde.correlation.shifted_correlation(
            partial(mde.correlation.isf, q=self._q_magnitude),
            self._coords,
            average=False,
            segments=self._num_segments,
            skip=self._skip,
        )

        adjust_results = len(self._coords[0]) * results.var(axis=0) * 1e5
        smooth_results = mde.utils.moving_average(adjust_results, 5)

        df = pd.DataFrame(
            np.column_stack([times[2:-2], adjust_results[2:-2], smooth_results]),
            columns=column_labels,
        )
        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label=r"$t$ / ps",
            x_axis_scale="log",
            y_axis_label="Non-Gaussian Displacement",
        )


class RadialDensityFunction(BaseAnalysis):
    def calculate(
        self,
        pore_diameter: float,
        groups: list[list[str]],
        num_bins: int = 10,
        radius_buffer: float = 0.1,
    ) -> pd.DataFrame:
        """
        :param groups: A list containing atom-residue pairs with the format (list("ATOM", "RESIDUE"), list("ATOM", "RESIDUE"), ...)
        """
        assert pore_diameter > 0.0, (
            "Pore diameter must be provided for radially resolved results."
        )
        assert len(groups) > 0, "Designate at least one atom, residue pair."

        self._params.groups = ", ".join(
            [f"{groups[i][1]}:{groups[i][0]}" for i in range(len(groups))]
        )
        self._set_radial_params(False, pore_diameter, num_bins, radius_buffer)

        radius = pore_diameter / 2 + radius_buffer
        bins = np.arange(0.0, radius + radius / num_bins, radius / num_bins)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        column_labels = "r / nm," + ",".join(
            [f"{groups[i][1]}:{groups[i][0]}" for i in range(len(groups))]
        )

        results = []
        if groups is not None:
            for atom, residue in groups:
                result = mde.distribution.time_average(
                    partial(mde.distribution.radial_density, bins=bins),
                    self.InitCoordinates.subset(atom_name=atom, residue_name=residue),
                    segments=self._num_segments,
                    skip=self._skip,
                )
                results.append(result)

        df = pd.DataFrame(
            np.column_stack([bin_centers, *results]), columns=column_labels
        )
        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def rdf_plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label=r"${\rho_{N}(r)} / {\mathrm{counts} \cdot \mathrm{cm}^{-3}}$",
        )


class RadialDistributionFunction(BaseAnalysis):
    @property
    def q_magnitude(self):
        y_max_idx = self._results_df.iloc[:, 1].idxmax()
        x_at_max_y = self._results_df.iloc[y_max_idx, 0]
        q_magnitude = 2 * np.pi / x_at_max_y
        return q_magnitude

    def calculate(
        self,
        mode: str = "total",
        res_name: str = None,
        atoms: list = None,
        max_search_radius: float = 2.5,
    ) -> pd.DataFrame:
        """
        Calculate the incoherent scattering function.

        :param mode: "total" | "intra" | "inter"
        """
        print(f"Calculating {mode} RDF...")

        bins = np.arange(0, max_search_radius, 0.01)
        column_labels = ["r / nm", "G(r)"]

        if self._coords_type == "com":
            results = mde.distribution.time_average(
                partial(mde.distribution.rdf, bins=bins),
                self._coords,
                segments=self._num_segments,
                skip=0.01,
            )
            df = pd.DataFrame(
                np.column_stack([bins[:-1], results]), columns=column_labels
            )
            self._results.q_magnitude = round(float(self.q_magnitude), 3)

        else:
            assert res_name is not None, "Provide a residue name for intra/inter RDFs."
            assert len(atoms) == 2, (
                "Atoms must be provided for non center of mass RDFs."
            )

            self._params.analysis_mode = mode
            self._params.res_name = res_name
            self._params.atoms = str(f"{atoms[0]}, {atoms[1]}")

            atom_1_coords = self._coords.subset(
                atom_name=atoms[0], residue_name=res_name
            )

            atom_2_coords = self._coords.subset(
                atom_name=atoms[1], residue_name=res_name
            )

            results = mde.distribution.time_average(
                function=partial(mde.distribution.rdf, bins=bins, mode=mode),
                coordinates=atom_1_coords,
                coordinates_b=atom_2_coords,
                segments=self._num_segments,
            )

            df = pd.DataFrame(
                np.column_stack([bins[:-1], results]), columns=column_labels
            )

        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label="g(r)",
        )


class vanHoveSelfCorrelation(BaseAnalysis):
    def calculate(
        self,
        mode: str,
        num_points: int = 10,
        pore_diameter: float = 0,
        num_bins: int = 1000,
        radius_buffer: float = 0.1,
    ):
        """
        :param mode: "rotational" or "translational"
        :param num_points: Number of timeshifts for which the shifted correlation will be calculated (num_points - 1 cols in resulting df).
        :param pore_diameter: Pore diameter (nm).
        :param num_bins: Number of cylinders to use in radially resolved analysis.
        :param radius_buffer: Distance (nm) to include beyond the pore diameter for binning.
        """
        if mode.lower() == "translational":
            assert pore_diameter > 0.0, (
                "Provide pore diameter for translational van Hove analysis."
            )

            self._set_radial_params(False, pore_diameter, num_bins, radius_buffer)

            radius = pore_diameter / 2 + radius_buffer

            try:
                bin_radius = radius / num_bins
                bins = np.arange(0.0, radius + bin_radius, bin_radius)
            except ZeroDivisionError:
                raise ZeroDivisionError("Indicate a number of bins > 0.")

            times, results = mde.correlation.shifted_correlation(
                partial(mde.correlation.van_hove_self, bins=bins),
                self._coords,
                segments=self._num_segments,
                skip=self._skip,
                points=num_points,
            )

            column_labels = "r / nm," + ",".join([f"{t / 1000:.0f} ns" for t in times])

            bin_centers = np.array(
                [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
            )

            df = pd.DataFrame(
                np.column_stack([bin_centers, results.T]), columns=column_labels
            )

        elif mode.lower() == "rotational":
            assert self._coords_type == "vectors", (
                "Use vectorized coordinates for rotational van Hove analysis."
            )

            def _angle_dist(start, end, bins):
                scalar_product = (start * end).sum(axis=-1)
                angle = np.arccos(scalar_product)
                angle = angle[(angle >= 0) * (angle <= np.pi)]
                histogram, _ = np.histogram(angle * 360 / (2 * np.pi), bins)
                return 1 / len(start) * histogram

            bins = np.linspace(0, 180, num_bins)

            times, results = mde.correlation.shifted_correlation(
                function=partial(_angle_dist, aBins=bins),
                frames=self._coords,
                segments=self._num_segments,
                skip=self._skip,
                points=8,
            )

            column_labels = "angle / degrees," + ",".join(
                [f"{t / 1000:.0f} ns" for t in times]
            )

            bin_centers = np.array(
                [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
            )

            df = pd.DataFrame(
                np.column_stack([bin_centers, results.T]), columns=column_labels
            )

        else:
            raise NotImplementedError(
                "Indicate either 'rotational' or 'translational' for van Hove self correlation distributions."
            )

        self._results_df = df
        self._metadata.analysis_last_performed = datetime.now()
        return df

    def _plot_func(df: pd.DataFrame, out_path: Path):
        plotting.plot_line(
            output_path=out_path,
            x_data=df.iloc[:, 0],
            y_data=df.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label="g(r)",
        )
