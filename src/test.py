import click
from util.traj import coordinates, selections
from analys import scattering
from functools import partial

# Test files
GRO_FILE = "test_files/adk_oplsaa/adk_oplsaa.gro"
XTC_FILE = "test_files/adk_oplsaa/adk_oplsaa.xtc"


@click.group()
def analyze():
    """Main CLI group."""
    pass


@analyze.command()
@click.option(
    "-top", "--topology", default=GRO_FILE, help="Path string to topology file."
)
@click.option(
    "-traj", "--trajectory", default=XTC_FILE, help="Path string to trajectory file."
)
@click.option("-min", "--minimal", is_flag=True, help="Enable minimal analysis.")
@click.option(
    "-g1",
    "--group1",
    default=None,
    help="Selection string for first reference group in analyses.",
)
@click.option(
    "-g2",
    "--group2",
    default=None,
    help="Selection string for second reference group in analyses.",
)
def bulk(
    topology: str, trajectory: str, minimal: bool, group1: str, group2: str = None
):
    """
    Test function to load a trajectory.
    :param topology: Path string to topology file.
    :param trajectory: Path string to trajectory file.
    :param minimal: (False) Whether to run minimal analysis.
    :param group2: Selection string for first reference group.
    :param group2: Selection string for second reference group.
    """
    Universe = coordinates.CoordIO.load_traj(trajectory, topology)
    Pairs = selections.select_pairs(Universe, group1, group2)
    traj_com = coordinates.GatherCOM(Universe)
    RadialBins, RadialDist, magScatteringVector = scattering.Scattering.RDF(Universe, Pairs, RetQ = True)
    a, b = scattering.Scattering.ShiftedISF(traj_com, magScatteringVector, average = True)
    
    print(a, b)

if __name__ == "__main__":
    analyze()