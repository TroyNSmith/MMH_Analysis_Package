class Analyze:
    def __init__(self, sim_directory: str, topology: str, trajectory: str):
        """
        Initializes mdevaluate objects for trajectory analysis.

        :param sim_directory: String path to directory containing topology and trajectory files.
        :param topology: Name of MD topology file.
        :param trajectory: Name of MD trajectory file.
        """
        return NotImplementedError

    def get_analysis(self, analysis_type: str, **kwargs):
        analysis_map = {
            #hbond heatmaps
        }

        return NotImplementedError