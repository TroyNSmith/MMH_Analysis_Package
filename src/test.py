import click
from pathlib import Path
from util.traj import reader
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
@click.option("--topology", default=GRO_FILE, help="Path string to topology file.")
@click.option("--trajectory", default=XTC_FILE, help="Path string to trajectory file.")
@click.option("--minimal", is_flag=True, help="Enable minimal analysis.")
@click.option("--selection", default=None, help="Selection string for analyses.")
def bulk(topology: str, trajectory: str, minimal: bool, selection: str):
    """Test function to load a trajectory."""
    Universe = reader.load_traj(trajectory, topology)
    radii, g_r = rdf.inter_rdf(Universe, "type O")
    print(radii, g_r)
    

if __name__ == "__main__":
    analyze()
