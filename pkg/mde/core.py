import mdevaluate as mde

from .functions import IncoherentScatteringFunction, RadialDistributionFunction
from .utils.coordinates import centers_of_masses
from .utils.types import BaseAnalysis
from .utils.basemodels import Params


class MDEvaluateAnalysis:
    def __init__(self, sim_directory: str, topology: str, trajectory: str):
        """
        Initializes mdevaluate objects for trajectory analysis.

        :param sim_directory: String path to directory containing topology and trajectory files.
        :param topology: Name of MD topology file.
        :param trajectory: Name of MD trajectory file.
        """
        self._initial_coords = mde.open(
            directory=sim_directory, trajectory=trajectory, topology=topology
        )
        self._params = Params()

    @property
    def _has_centers_of_masses(self):
        return hasattr(self, "_centers_of_masses")

    def assign_centers_of_masses(self, res_name: str):
        self._params.res_name = res_name
        self._centers_of_masses = centers_of_masses(self._initial_coords, res_name)

    def get_analysis(self, analysis_type: str, coord_type: str, **kwargs) -> BaseAnalysis:
        analysis_map = {
            "isf": IncoherentScatteringFunction,
            "rdf": RadialDistributionFunction,
            # "msd": MeanSquaredDisplacement,
            # "rdf": RadialDistributionFunction,
        }

        cls = analysis_map.get(analysis_type.lower())
        if cls is None:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        if coord_type == "all":
            coords = self._initial_coords
        elif coord_type == "com":
            if not self._has_centers_of_masses:
                raise LookupError("Could not find centers of masses coordinates.")
            coords = self._centers_of_masses
        elif coord_type == "vectors":
            raise NotImplementedError
        
        self._params.coordinate_type = coord_type

        return cls(coords=coords, coords_type=coord_type, params=self._params, **kwargs)
