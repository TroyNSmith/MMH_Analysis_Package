import click
from pathlib import Path
from util.traj import reader, selections
from traj import rdf

# Test files
DATA_DIR = Path("tests/adk_oplsaa")
GRO_FILE = DATA_DIR / "adk_oplsaa.gro"
XTC_FILE = DATA_DIR / "adk_oplsaa.xtc"


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
    Topology: str, Trajectory: str, Minimal: bool, Group1: str, Group2: str = None
):
    """
    Test function to load a trajectory.
    :param topology: Path string to topology file.
    :param trajectory: Path string to trajectory file.
    :param minimal: (False) Whether to run minimal analysis.
    :param group1: Selection string for first reference group.
    :param group2: Selection string for second reference group.
    """
    Universe = reader.load_traj(Trajectory, Topology)
    Pairs = selections.select_pairs(Universe, Group1, Group2)
    R, G_R = rdf.rdf(Universe, Pairs)
    

if __name__ == "__main__":
    analyze()
