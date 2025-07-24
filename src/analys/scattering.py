import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray


class Scattering:
    """
    Class for calculating radial distribution functions and related scattering behavior.
    """

    @staticmethod
    def RDF(
        Universe: md.Trajectory, Pairs: ArrayLike, RetQ: bool = False
    ) -> tuple[NDArray, NDArray, float]:
        """
        Compute RDF for pairs in every frame of trajectory using mdtraj. Return the magnitude of the scattering vector (q) if option is toggled.

        :param Universe: mdtraj.Trajectory object.
        :param Pairs: ArrayLike with all atom pairs for selections of interest.
        :param RetQ: (False) Return the magnitude of the associated scattering vector.
        :return RadialBins: Numpy array with radial bins in nm.
        :return RadialDist: Numpy array with radial distribution results.
        :return q: (Optional) Magnitude of the associated scattering vector.
        """
        RadialBins, RadialDist = md.compute_rdf(Universe, Pairs)

        if RetQ:
            magScatteringVec = Scattering._magScatteringVector(
                RadialBins, RadialDist
            )
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

    def ShiftedISF(
        Universe: md.Trajectory,
        magScatteringVec: float,
        segments: int = 10,
        window: float = 0.5,
        skip: float = None,
        average: bool = False,
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
        num_frames = Universe.n_frames
        if skip is None:
            skip = Universe._slice.start / num_frames if hasattr(Universe, "_slice") else 0
        assert window + skip < 1, "window + skip must be < 1"

        # Time segment start indices
        start_indices = np.unique(
            np.linspace(
                num_frames * skip,
                num_frames * (1 - window),
                num=segments,
                endpoint=False,
                dtype=int,
            )
        ).astype(int)

        # Number of frames per window
        num_corr_frames = int(num_frames * window)

        # Logarithmic time steps
        log_indices = np.unique(
            np.logspace(0, np.log10(num_corr_frames), num=100, dtype=int)
        )
        log_indices = log_indices[log_indices < num_corr_frames]

        all_isfs = []

        for start_idx in start_indices:
            start_xyz = Universe.xyz[start_idx]  # shape: (n_atoms, 3)
            segment_isf = []

            for offset in log_indices:
                idx = start_idx + offset
                if idx >= num_frames:
                    continue
                target_xyz = Universe.xyz[idx]

                # Apply minimum image convention if box info is available
                if Universe.unitcell_lengths is not None:
                    box_lengths = Universe.unitcell_lengths[idx]
                    delta = target_xyz - start_xyz
                    delta -= (
                        np.round(delta / box_lengths[np.newaxis, :])
                        * box_lengths[np.newaxis, :]
                    )
                else:
                    delta = target_xyz - start_xyz

                distance = np.linalg.norm(delta, axis=1)
                isf_val = np.sinc(distance * magScatteringVec / np.pi).mean()
                segment_isf.append(isf_val)

            all_isfs.append(segment_isf)

        # Align time axis (in ps)
        times = Universe.time[log_indices] - Universe.time[0]
        isf_array = np.array(all_isfs)

        if average:
            isf_mean = isf_array.mean(axis=0)
            return times, isf_mean
        else:
            return times, isf_array
