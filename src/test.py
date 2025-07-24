import click
from util.traj import coordinates
from analysis import scattering, hydrogenbonds
import matplotlib.pyplot as plt

GRO_FILE = "src/test_files/adk_oplsaa/adk_oplsaa.gro"
XTC_FILE = "src/test_files/adk_oplsaa/adk_oplsaa.xtc"


@click.group()
def analyze():
    """Main CLI group for MD analysis."""
    pass


@analyze.command()
@click.option(
    "-tp",
    "--Topology",
    "Topology",
    default=GRO_FILE,
    help="Path string to topology file.",
)
@click.option(
    "-tr",
    "--Traj",
    "Trajectory",
    default=XTC_FILE,
    help="Path string to trajectory file.",
)
@click.option("-m", "--Min", "Minimal", is_flag=True, help="Enable minimal analysis.")
@click.option(
    "-s1",
    "--Selection1",
    "Selection1",
    default=None,
    help="Selection string for first reference group in analyses.",
)
@click.option(
    "-s2",
    "--Selection2",
    "Selection2",
    default=None,
    help="Selection string for second reference group in analyses.",
)
def bulk(
    Topology: str,
    Trajectory: str,
    Minimal: bool,
    Selection1: str,
    Selection2: str = None,
):
    """
    Test function to load a trajectory.
    :param Topology: Path string to topology file.
    :param Trajectory: Path string to trajectory file.
    :param Minimal: (False) Whether to run minimal analysis.
    :param Selection1: Selection string for first reference group.
    :param Selection2: Selection string for second reference group.
    """
    Universe = coordinates.CoordIO.load_traj(Trajectory, Topology)

    HBonds = hydrogenbonds.HydrogenBonds(Universe)


if __name__ == "__main__":
    analyze()
