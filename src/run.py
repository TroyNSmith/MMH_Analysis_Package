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
    "ResName",
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
@click.option(
    "-pm",
    "--plotmode",
    "PlottingMode",
    type=click.Choice(["Off", "Initial", "Overwrite existing"], case_sensitive=False),
    default="Initial",
    help="Whether to plot generated data. Options: 'Off' | 'Initial' | 'Overwrite existing'",
)
def Pore(
    SimulationDirectory: str,
    Trajectory: str,
    Topology: str,
    nSegments: int,
    ResName: str,
    Atoms: str,
    Diameter: float,
    qLength: float,
    OutputDirectory: str,
    Overwrite: bool,
    PlottingMode: str,
):
    CoordFrameAnalysis = TrajectoryAnalysis(
        SimulationDirectory=SimulationDirectory,
        Trajectory=Trajectory,
        Topology=Topology,
        nSegments=nSegments,
        ResName=ResName,
        Atoms=Atoms,
        OutputDirectory=OutputDirectory,
        OverwriteExisting=Overwrite,
        PlottingMode=PlottingMode,
    )

    CoordFrameAnalysis.AveMSD(
        Axes="all and x and xy"
    )  # Mean square displacement in universe (total), x-direction, and xy-plane
    CoordFrameAnalysis.AveMSD(
        Resolved=True, Axes="all and xy", nBins=7, Diameter=Diameter
    )  # Radially resolved mean square displacement in universe (total) and xy-plane
    CoordFrameAnalysis.RDF(ReturnQ=False, Mode="COM")    # Radial distribution function
    CoordFrameAnalysis.RDF(ReturnQ=False, Mode="Total")  # Radial distribution function
    CoordFrameAnalysis.RDF(ReturnQ=False, Mode="Intra")  # Radial distribution function
    CoordFrameAnalysis.RDF(ReturnQ=False, Mode="Inter")  # Radial distribution function
    CoordFrameAnalysis.AveISF(qLength=qLength)  # Overall ISF
    CoordFrameAnalysis.AveISF(
        qLength=qLength, Resolved=True, nBins=3, Diameter=Diameter
    )  # Radially binned ISF (3 bins)
    CoordFrameAnalysis.AveISF(
        qLength=qLength, Resolved=True, nBins=10, Diameter=Diameter
    )  # Radially binned ISF (10 bins)
    CoordFrameAnalysis.nonGauss()  # Non-Gaussian Displacement Statistics
    CoordFrameAnalysis.vanHove(
        Translational=True, Diameter=Diameter, nBins=15
    )  # Translational van Hove dynamics
    CoordFrameAnalysis.vanHove(Rotational=True)  # Rotational van Hove dynamics
    CoordFrameAnalysis.Chi4Susceptibility()
    CoordFrameAnalysis.zAxisAlignment()
    CoordFrameAnalysis.zAxisRadialPos()
    CoordFrameAnalysis.RadialDensity(
        Groups=[
            [Atoms[0], ResName],
            [Atoms[1], ResName],
            ["NL", "LNK"],
            ["OEE", "ETH"],
            ["NV", "VAN"],
            ["OVE", "VAN"],
            ["OVH", "VAN"],
        ],
        nBins=150,
        Diameter=Diameter,
    )


cli.add_command(Pore)

if __name__ == "__main__":
    cli()
