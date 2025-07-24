import mdtraj as md
import numpy as np
from numpy.typing import NDArray


class Scattering:
    """
    Class for calculating radial distribution functions and related scattering behavior.
    """

    @staticmethod
    def RDF(
        Universe: md.Trajectory, Selection1: str, Selection2: str, RetQ: bool = False
    ) -> tuple[NDArray, NDArray, float]:
        """
        Compute RDF for pairs in every frame of trajectory using mdtraj. Return the magnitude of the scattering vector (q) if option is toggled.

        :param Universe: mdtraj.Trajectory object.
        :param Selection1: mdtraj-style selection string, e.g., 'resname OCT', 'not resname PORE'.
        :param Selection2: mdtraj-style selection string, e.g., 'resname OCT', 'not resname PORE'.
        :param RetQ: (False) Return the magnitude of the associated scattering vector.
        :return RadialBins: Numpy array with radial bins in nm.
        :return RadialDist: Numpy array with radial distribution results.
        :return magScatteringVec: (Optional) Magnitude of the associated scattering vector.
        """
        Pairs = Universe.topology.select_pairs(Selection1, Selection2)
        RadialBins, RadialDist = md.compute_rdf(Universe, Pairs)

        if RetQ:
            magScatteringVec = Scattering._magScatteringVector(RadialBins, RadialDist)
            return RadialBins, RadialDist, magScatteringVec

        else:
            return RadialBins, RadialDist

    def _magScatteringVector(RadialBins: NDArray, RadialDist: NDArray) -> float:
        """
        Identify the scattering vector (q) for a radial distribution.

        :param RadialBins: Numpy array with radial bins in nm.
        :param RadialDist: Numpy array with radial distribution results.
        :return magScatteringVec: Length of the scattering vector (q).
        """
        MaxGIdx = np.argmax(RadialDist)

        return RadialBins[MaxGIdx]

    @staticmethod
    def ShiftedISF(
        Universe: md.Trajectory,
        magScatteringVec: float,
        Segments: int = 10,
        Window: float = 0.5,
        Skip: float = None,
        Average: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Calculate the incoherent intermediate scattering function for an mdtraj.Trajectory object using a shifted window correlation.

        :param Universe: mdtraj.Trajectory object.
        :param magScatteringVec: Scattering vector magnitude (nm^-1).
        :param Segments: Number of segments (start times) over which to average.
        :param Window: Fraction of trajectory used per correlation segment.
        :param Skip: Fraction to skip at the beginning of the trajectory.
        :param Average: If True, return averaged ISF; else return all results.

        Returns:
            tuple: (times, isf_data)
                - times: array of time differences (in ps)
                - isf_data: array of ISF values (1D if averaged, 2D if not)
        """
        nFrames = Universe.n_frames
        if Skip is None:
            Skip = (
                Universe._slice.start / nFrames if hasattr(Universe, "_slice") else 0
            )
        assert Window + Skip < 1, "Window + Skip must be < 1"

        StartIndices = np.unique(
            np.linspace(
                nFrames * Skip,
                nFrames * (1 - Window),
                num=Segments,
                endpoint=False,
                dtype=int,
            )
        ).astype(int)

        nCorrFrames = int(nFrames * Window)

        LogIndices = np.unique(
            np.logspace(0, np.log10(nCorrFrames), num=100, dtype=int)
        )
        LogIndices = LogIndices[LogIndices < nCorrFrames]

        Results = []

        for sIdx in StartIndices:
            sXYZ = Universe.xyz[sIdx]
            segISF = []

            for Offset in LogIndices:
                nIdx = sIdx + Offset
                if nIdx >= nFrames:
                    continue
                tXYZ = Universe.xyz[nIdx]       #tXYZ = Target coords

                if Universe.unitcell_lengths is not None:
                    Edges = Universe.unitcell_lengths[nIdx]
                    Δ = tXYZ - sXYZ
                    Δ -= (
                        np.round(Δ / Edges[np.newaxis, :])
                        * Edges[np.newaxis, :]
                    )
                else:
                    Δ = tXYZ - sXYZ

                Dist = np.linalg.norm(Δ, axis=1)
                Scat = np.sinc(Dist * magScatteringVec / np.pi).mean()
                segISF.append(Scat)

            Results.append(segISF)

        Times = Universe.time[LogIndices] - Universe.time[0]
        AllISF = np.array(Results)

        if Average:
            AvgISF = AllISF.mean(axis=0)
            return Times, AvgISF
        else:
            return Times, AllISF
