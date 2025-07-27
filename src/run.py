import click

from analysis.CoordinateFrame import TrajectoryAnalysis


@click.group()
def cli():
    """Main CLI entry point."""
    pass


@click.command()
@click.option(
    "-dir",
    "--directory",
    "SimulationDirectory",
    help="Path to simulation directory containing trajectory and topology files. This directory will also be used to dump output files unless otherwise specified.",
)
@click.option(
    "-tr",
    "--trajectory",
    "Trajectory",
    help="Name of the trajectory file located within -d / --directory.",
)
@click.option(
    "-tp",
    "--topology",
    "Topology",
    help="Name of the topology file located within -d / --directory.",
)
@click.option(
    "-res",
    "--resname",
    "Resname",
    help="Name of the residue to be analyzed.",
    default=None,
)
@click.option(
    "-a",
    "--atoms",
    "Atoms",
    nargs=2,
    help="Two atoms for analysis located on the residue specified in -res / --resname.",
    default=None,
)
@click.option(
    "-s",
    "--segments",
    "nSegments",
    help="Number of starting points to use with correlation functions.",
    type=click.IntRange(10, 1000, clamp=True),
    default=100,
)
@click.option(
    "-d",
    "--diameter",
    "Diameter",
    help="Diameter of the system in nm.",
    type=click.FloatRange(0, 1000, clamp=False),
    default=None,
)
@click.option(
    "-q",
    "--qLength",
    "qLength",
    help="Magnitude of the scatterign vector, q.",
    default=None,
)
@click.option(
    "-o",
    "--out",
    "OutputDirectory",
    help="Designated path for analysis output files.",
    default=None,
)
@click.option(
    "-ow",
    "--overwrite",
    "Overwrite",
    help="Overwrite existing analysis files.",
    is_flag=True,
    default=False,
)
def Pore(
    SimulationDirectory: str,
    Trajectory: str,
    Topology: str,
    nSegments: int,
    Resname: str,
    Atoms: str,
    Diameter: float,
    qLength: float,
    OutputDirectory: str,
    Overwrite: bool,
):
    CoordFrameAnalysis = TrajectoryAnalysis(
        SimulationDirectory=SimulationDirectory,
        Trajectory=Trajectory,
        Topology=Topology,
        nSegments=nSegments,
        ResName=Resname,
        Atoms=Atoms,
        OutputDirectory=OutputDirectory,
        OverwriteExisting=Overwrite,
    )

    CoordFrameAnalysis.AveMSD(Axes='all and x and xy')                                          # Mean square displacement in universe (total), x-direction, and xy-plane
    CoordFrameAnalysis.AveMSD(Resolved=True, Axes='all and xy', nBins=7, Diameter=Diameter)     # Radially resolved mean square displacement in universe (total) and xy-plane
    CoordFrameAnalysis.RDF(ReturnQ=False)                                                       # Radial distribution function
    CoordFrameAnalysis.AveISF(qLength=qLength)                                                  # Overall ISF
    CoordFrameAnalysis.AveISF(qLength=qLength, Resolved=True, nBins=3, Diameter=Diameter)       # Radially binned ISF (3 bins)
    CoordFrameAnalysis.AveISF(qLength=qLength, Resolved=True, nBins=10, Diameter=Diameter)      # Radially binned ISF (10 bins)
    CoordFrameAnalysis.nonGauss()                                                               # Non-Gaussian Displacement Statistics
    CoordFrameAnalysis.vanHove(Translational=True, Diameter=Diameter, nBins=15)                 # Translational van Hove dynamics
    CoordFrameAnalysis.vanHove(Rotational=True)                 # Translational van Hove dynamics

cli.add_command(Pore)

if __name__ == "__main__":
    cli()
