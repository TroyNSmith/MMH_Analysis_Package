import click

from pathlib import Path

from mde.core import MDEvaluateAnalysis


@click.group()
def cli():
    """Main CLI entry point."""
    pass


@click.command()
@click.option(
    "-i",
    "--input",
    "sim_dir",
    help="Path to simulation directory containing trajectory and topology files. This directory will also be used to dump output files unless otherwise specified.",
)
@click.option(
    "-tr",
    "--trajectory",
    "trajectory",
    help="Name of the trajectory file located within -d / --directory.",
)
@click.option(
    "-tp",
    "--topology",
    "topology",
    help="Name of the topology file located within -d / --directory.",
)
@click.option(
    "-r",
    "--resname",
    "res_name",
    help="Name of the residue to be analyzed.",
    default=None,
)
@click.option(
    "-a",
    "--atoms",
    "atoms",
    nargs=2,
    help="Two atoms for analysis located on the residue specified in -res / --resname.",
    default=None,
)
@click.option(
    "-s",
    "--segments",
    "num_segments",
    help="Number of starting points to use with correlation functions.",
    type=click.IntRange(10, 1000, clamp=True),
    default=100,
)
@click.option(
    "-d",
    "--diameter",
    "pore_diameter",
    help="Diameter of the pore in nm.",
    type=click.FloatRange(0, 1000, clamp=False),
    default=None,
)
@click.option(
    "-q",
    "--q",
    "q_magnitude",
    type=float,
    help="Magnitude of the scattering vector, q.",
    default=None,
)
@click.option(
    "-o",
    "--out",
    "output_dir",
    help="Designated path for analysis output files.",
    default=None,
)
@click.option(
    "-ov",
    "--override",
    "override",
    help="Force re-run of calculations.",
    is_flag=True,
)
def Run(
    sim_dir: Path,
    topology: str,
    trajectory: str,
    num_segments: int,
    res_name: str,
    atoms: list,
    pore_diameter: float,
    q_magnitude: float,
    output_dir: Path,
    override: bool,
):
    analyzer = MDEvaluateAnalysis(sim_dir, topology, trajectory)
    analyzer.assign_centers_of_masses(res_name)

    isf_analysis = analyzer.get_analysis(
        "ISF", "com", num_segments=num_segments, q_magnitude=q_magnitude
    )

    if isf_analysis.should_run(output_dir, override):
        isf_analysis.calculate()
        isf_analysis.save(output_dir)
        isf_analysis.plot(output_dir)
    else:
        print("ISF analysis skipped (already exists). Use --override to force re-run.")

    rdf_analysis = analyzer.get_analysis(
        "RDF", "com", num_segments=num_segments, 
    )

    if rdf_analysis.should_run(output_dir, override):
        rdf_analysis.calculate()
        rdf_analysis.save(output_dir)
        #rdf_analysis.plot(output_dir)
    else:
        print("RDF analysis skipped (already exists). Use --override to force re-run.")

    rdf_analysis = analyzer.get_analysis(
        "RDF", "all", num_segments=num_segments, 
    )

    if rdf_analysis.should_run(output_dir, override):
        rdf_analysis.calculate(res_name=res_name, atoms=atoms)
        rdf_analysis.save(output_dir)
        #rdf_analysis.plot(output_dir)
        rdf_analysis.calculate(res_name=res_name, atoms=atoms, mode='intra')
        rdf_analysis.save(output_dir)
        #rdf_analysis.plot(output_dir)
    else:
        print("RDF analysis skipped (already exists). Use --override to force re-run.")

    rdf_analysis = analyzer.get_analysis(
        "RDF", "all", num_segments=num_segments, 
    )


cli.add_command(Run)

if __name__ == "__main__":
    cli()
