import click
import MDAnalysis as mda
import numpy as np
import os
import pandas as pd

from . import mdevaluate as mde
from openpyxl import Workbook
from pathlib import Path

from .functions import dynamics
from .functions.coordinates import centers_of_masses, vectorize_residue
from .functions.plotting import plot_line
from .functions import new_structural as structural
from .functions.utils import log_analysis_yaml


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
    "-o",
    "--output",
    "out_dir",
    help="Path to dump output files from analysis.",
    default=None,
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
    default=500,
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
    "q_val",
    type=float,
    help="Magnitude of the scattering vector, q.",
    default=None,
)
@click.option(
    "-ov",
    "--override",
    "override",
    help="Force re-run of calculations.",
    is_flag=True,
)
@click.option(
    "-po",
    "--plot_only",
    "plot_only",
    help="Skip calculations and regenerate plots of existing data.",
    is_flag=True,
)
def Run(
    sim_dir: Path,
    out_dir: Path,
    trajectory: str,
    topology: str,
    res_name: str,
    atoms: list[str],
    num_segments: int,
    pore_diameter: float,
    q_val: float,
    override: bool,
    plot_only: bool,
):
    # ==============================
    # ===== Step 1: Initialize =====
    # ==============================
    mde_coords = mde.open(
        directory=sim_dir, topology=topology, trajectory=trajectory, nojump=True
    )
    mde_vectors = vectorize_residue(
        mde_coords=mde_coords, res_name=res_name, atoms=atoms
    )
    mde_com = centers_of_masses(mde_coords=mde_coords, res_name=res_name)

    mde_atom_1_coords = mde_coords.subset(atom_name=atoms[0], residue_name=res_name)

    mde_atom_2_coords = mde_coords.subset(atom_name=atoms[1], residue_name=res_name)

    mda_coords = mda.Universe(
        os.path.join(sim_dir, topology), os.path.join(sim_dir, trajectory)
    )

    dir_out = (
        os.path.join(out_dir, "Analysis", f"{res_name}_{atoms[0]}_ {atoms[1]}")
        if out_dir is not None
        else os.path.join(sim_dir, "Analysis", f"{res_name}_{atoms[0]}_{atoms[1]}")
    )

    yaml_out = Path(os.path.join(dir_out, "log.yaml"))
    xlsx_out = Path(os.path.join(dir_out, "results.xlsx"))

    if not xlsx_out.exists():
        wb = Workbook()
        wb.save(xlsx_out)

    imported_params = {
        "trajectory": trajectory,
        "topology": topology,
        "res_name": res_name,
        "atoms": atoms,
        "num_segments": num_segments,
        "pore_diameter": pore_diameter,
        "q_magnitude": q_val,
    }

    # =================================================
    # ===== Step 2: Radial distribution functions =====
    # =================================================
    os.makedirs(os.path.join(dir_out, "RDF"), exist_ok=True)

    # ===== Total (centers of masses)
    csv_out_total = os.path.join(dir_out, "RDF", "RDF_total.csv")
    if not os.path.exists(csv_out_total) or override:
        print("Calculating total centers of masses RDF...")
        rdf_total = structural.radial_distribution_function(
            coords_1=mde_com,
            num_segments=num_segments,
            column_label="Centers of Masses",
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Centers of Masses RDF (total)",
            file_path=csv_out_total,
            parameters=imported_params,
        )
        rdf_total.to_csv(csv_out_total, index=False)

    # ===== Magnitude of q vector
    if q_val is None:
        rdf_total = pd.read_csv(csv_out_total)
        y_max_idx = rdf_total.iloc[:, 1].idxmax()
        x_at_max_y = rdf_total.iloc[y_max_idx, 0]
        q_val = float(round(2 * np.pi / x_at_max_y, 3))
        imported_params["q_magnitude"] = q_val

    # ===== Intramolecular
    csv_out_intra = os.path.join(dir_out, "RDF", "RDF_intra.csv")
    if not os.path.exists(csv_out_total) or override:
        print("Calculating intramolecular RDF...")
        rdf_intra = structural.radial_distribution_function(
            coords_1=mde_atom_1_coords,
            coords_2=mde_atom_2_coords,
            num_segments=num_segments,
            mode="intra",
            column_label=f"Intra {atoms[0]}:{atoms[1]}",
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Intra RDF",
            file_path=csv_out_intra,
            parameters=imported_params,
        )
        rdf_intra.to_csv(csv_out_intra, index=False)

    # ===== Intermolecular (atom 1)
    csv_out_inter_1 = os.path.join(dir_out, "RDF", f"RDF_inter_{atoms[0]}.csv")
    if not os.path.exists(csv_out_total) or override:
        print("Calculating intermolecular RDF #1...")
        rdf_inter_1 = structural.radial_distribution_function(
            coords_1=mde_atom_1_coords,
            coords_2=mde_atom_1_coords,
            num_segments=num_segments,
            mode="inter",
            column_label=f"Inter {atoms[0]}:{atoms[0]}",
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name=f"Inter RDF ({atoms[0]})",
            file_path=csv_out_inter_1,
            parameters=imported_params,
        )
        rdf_inter_1.to_csv(csv_out_inter_1, index=False)

    # ===== Intermolecular (atom 2)
    csv_out_inter_2 = os.path.join(dir_out, "RDF", f"RDF_inter_{atoms[1]}.csv")
    if not os.path.exists(csv_out_total) or override:
        print("Calculating intermolecular RDF #2...")
        rdf_inter_2 = structural.radial_distribution_function(
            coords_1=mde_atom_2_coords,
            coords_2=mde_atom_2_coords,
            num_segments=num_segments,
            mode="inter",
            column_label=f"Inter {atoms[1]}:{atoms[1]}",
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name=f"Inter RDF ({atoms[1]})",
            file_path=csv_out_inter_2,
            parameters=imported_params,
        )
        rdf_inter_2.to_csv(csv_out_inter_2, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "RDF", "RDF.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            rdf_total = pd.read_csv(csv_out_total)
            rdf_intra = pd.read_csv(csv_out_intra)
            rdf_inter_1 = pd.read_csv(csv_out_inter_1)
            rdf_inter_2 = pd.read_csv(csv_out_inter_2)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        msd_merged = pd.concat(
            [
                rdf_total,
                rdf_intra.iloc[:, 1],
                rdf_inter_1.iloc[:, 1],
                rdf_inter_2.iloc[:, 1],
            ],
            axis=1,
        )

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            msd_merged.to_excel(writer, sheet_name="RDF", index=False)

        plot_line(
            output_path=png_out,
            x_data=msd_merged.iloc[:, 0],
            y_data=msd_merged.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label="g(r)",
        )

    # ============================================
    # ===== Step 3: Mean square displacement =====
    # ============================================
    os.makedirs(os.path.join(dir_out, "MSD"), exist_ok=True)

    # ===== Residual MSD (all directions)
    csv_out_all = os.path.join(dir_out, "MSD", "MSD_all.csv")
    if not os.path.exists(csv_out_all) or override:
        print("Calculating total MSD...")
        msd_all = dynamics.mean_square_displacement(
            coords=mde_com, num_segments=num_segments
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="MSD (all)",
            file_path=csv_out_all,
            parameters=imported_params,
        )
        msd_all.to_csv(csv_out_all, index=False)

    # ===== Residual MSD (z-axis)
    csv_out_z = os.path.join(dir_out, "MSD", "MSD_z.csv")
    if not os.path.exists(csv_out_z) or override:
        print("Calculating z-axis MSD...")
        msd_z = dynamics.mean_square_displacement(
            coords=mde_com, axis="z", num_segments=num_segments
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="MSD (z-axis)",
            file_path=csv_out_z,
            parameters=imported_params,
        )
        msd_z.to_csv(csv_out_z, index=False)

    # ===== Residual MSD (xy-plane)
    csv_out_xy = os.path.join(dir_out, "MSD", "MSD_xy.csv")
    if not os.path.exists(csv_out_xy) or override:
        print("Calculating xy-plane MSD...")
        msd_xy = dynamics.mean_square_displacement(
            coords=mde_com, axis="xy", num_segments=num_segments
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="MSD (xy-plane)",
            file_path=csv_out_xy,
            parameters=imported_params,
        )
        msd_xy.to_csv(csv_out_xy, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "MSD", "MSD.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            msd_all = pd.read_csv(csv_out_all)
            msd_z = pd.read_csv(csv_out_z)
            msd_xy = pd.read_csv(csv_out_xy)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        msd_merged = pd.concat([msd_all, msd_z.iloc[:, 1], msd_xy.iloc[:, 1]], axis=1)

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            msd_merged.to_excel(writer, sheet_name="MSD", index=False)

        plot_line(
            output_path=png_out,
            x_data=msd_merged.iloc[:, 0],
            y_data=msd_merged.iloc[:, 1:],
            x_axis_scale="log",
            x_axis_label=r"$\mathbf{\mathit{t}}$ / ps",
            y_axis_scale="log",
            y_axis_label=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
        )

    # ==============================================================
    # ===== Step 4: Radially resolved mean square displacement =====
    # ==============================================================
    csv_out_resolved = os.path.join(dir_out, "MSD", "MSD_resolved.csv")
    if not os.path.exists(csv_out_resolved) or override:
        print("Calculating radially resolved MSD...")
        msd_resolved = dynamics.mean_square_displacement(
            coords=mde_com,
            num_segments=num_segments,
            radially_resolved=True,
            pore_diameter=pore_diameter,
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="MSD (radially resolved)",
            file_path=csv_out_resolved,
            parameters=imported_params,
        )
        msd_resolved.to_csv(csv_out_resolved, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "MSD", "MSD_resolved.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            msd_resolved = pd.read_csv(csv_out_resolved)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            msd_resolved.to_excel(writer, sheet_name="MSD Resolved", index=False)

        plot_line(
            output_path=png_out,
            x_data=msd_resolved.iloc[:, 0],
            y_data=msd_resolved.iloc[:, 1:],
            x_axis_scale="log",
            x_axis_label=r"$\mathbf{\mathit{t}}$ / ps",
            y_axis_scale="log",
            y_axis_label=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
        )

    # ==================================================
    # ===== Step 5: Incoherent scattering function =====
    # ==================================================
    os.makedirs(os.path.join(dir_out, "ISF"), exist_ok=True)

    # ===== Residual ISF
    csv_out_isf = os.path.join(dir_out, "ISF", "ISF.csv")
    if not os.path.exists(csv_out_isf) or override:
        print("Calculating ISF...")
        isf_total = dynamics.incoherent_scattering_function(
            coords=mde_com,
            q_val=q_val,
            num_segments=num_segments,
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="ISF (total)",
            file_path=csv_out_isf,
            parameters=imported_params,
        )
        isf_total.to_csv(csv_out_isf, index=False)

    # ===== Residual ISF (resolved)
    csv_out_isf_resolved = os.path.join(dir_out, "ISF", "ISF_resolved.csv")
    if not os.path.exists(csv_out_isf_resolved) or override:
        print("Calculating resolved ISF...")
        isf_resolved = dynamics.incoherent_scattering_function(
            coords=mde_com,
            q_val=q_val,
            num_segments=num_segments,
            radially_resolved=True,
            pore_diameter=pore_diameter,
            num_bins=5,
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="ISF (resolved)",
            file_path=csv_out_isf_resolved,
            parameters=imported_params,
        )
        isf_resolved.to_csv(csv_out_isf_resolved, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "ISF", "ISF.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            isf_total = pd.read_csv(csv_out_isf)
            isf_resolved = pd.read_csv(csv_out_isf_resolved)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        isf_merged = pd.concat([isf_total, isf_resolved.iloc[:, 1:]], axis=1)

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            isf_merged.to_excel(writer, sheet_name="ISF", index=False)

        plot_line(
            output_path=png_out,
            x_data=isf_merged.iloc[:, 0],
            y_data=isf_merged.iloc[:, 1:],
            x_axis_scale="log",
            x_axis_label="t / ps",
            y_axis_scale="log",
            y_axis_label=f"ISF(q={q_val:.1f}, t)",
        )

    # ==================================================
    # ===== Step 6: Rotational correlation coeffs. =====
    # ==================================================
    os.makedirs(os.path.join(dir_out, "Etc"), exist_ok=True)

    # ===== Residual ISF
    csv_out_rotat_correl = os.path.join(dir_out, "Etc", "Rotational_correlation_coeffs.csv")
    if not os.path.exists(csv_out_rotat_correl) or override:
        print("Calculating rotational correlation coefficients...")
        rotat_correl = dynamics.rotational_correlations(
            vectors = mde_vectors,
            num_segments=num_segments
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Rotational correlation coefficients",
            file_path=csv_out_rotat_correl,
            parameters=imported_params,
        )
        rotat_correl.to_csv(csv_out_rotat_correl, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "Etc", "Rotational_correlation_coeffs.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            rotat_correl = pd.read_csv(csv_out_rotat_correl)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            rotat_correl.to_excel(writer, sheet_name="Rotational correlation coefficients", index=False)

        plot_line(
            output_path=png_out,
            x_data=rotat_correl.iloc[:, 0],
            y_data=rotat_correl.iloc[:, 1:],
            x_axis_label="t / ps",
            x_axis_scale="log",
            y_axis_label=r"$F_n(t)$",
        )

    # ========================================================
    # ===== Step 7: Rotational van Hove Self-Correlation =====
    # ========================================================
    os.makedirs(os.path.join(dir_out, "Etc"), exist_ok=True)

    # ===== Rotational van Hove
    csv_out_rotat_van_Hove = os.path.join(dir_out, "Etc", "Rotational_van_Hove.csv")
    if not os.path.exists(csv_out_rotat_van_Hove) or override:
        print("Calculating rotational van Hove self-correlation...")
        rotat_van_Hove = dynamics.van_hove_rotation(
            vectors = mde_vectors,
            num_segments=num_segments
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Rotational van Hove",
            file_path=csv_out_rotat_van_Hove,
            parameters=imported_params,
        )
        rotat_van_Hove.to_csv(csv_out_rotat_van_Hove, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "Etc", "Rotational_van_Hove.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            rotat_van_Hove = pd.read_csv(csv_out_rotat_van_Hove)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            rotat_van_Hove.to_excel(writer, sheet_name="Rotational van Hove", index=False)

        plot_line(
            output_path=png_out,
            x_data=rotat_van_Hove.iloc[:, 0],
            y_data=rotat_van_Hove.iloc[:, 2:],
            x_axis_label="θ / degrees",
            y_axis_label="F(θ, t)",
        )

    # ===========================================================
    # ===== Step 7: Translational van Hove Self-Correlation =====
    # ===========================================================
    os.makedirs(os.path.join(dir_out, "Etc"), exist_ok=True)

    # ===== Translational van Hove
    csv_out_trans_van_Hove = os.path.join(dir_out, "Etc", "Translational_van_Hove.csv")
    if not os.path.exists(csv_out_trans_van_Hove) or override:
        print("Calculating translational van Hove self-correlation...")
        trans_van_Hove = dynamics.van_hove_translation(
            coords=mde_com,
            num_segments=num_segments,
            pore_diameter=pore_diameter,
            num_bins=250
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Translational van Hove",
            file_path=csv_out_trans_van_Hove,
            parameters=imported_params,
        )
        trans_van_Hove.to_csv(csv_out_trans_van_Hove, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "Etc", "Translational_van_Hove.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            trans_van_Hove = pd.read_csv(csv_out_trans_van_Hove)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            trans_van_Hove.to_excel(writer, sheet_name="Translational van Hove", index=False)

        plot_line(
            output_path=png_out,
            x_data=trans_van_Hove.iloc[:, 0],
            y_data=trans_van_Hove.iloc[:, 2:],
            x_axis_label="r / nm",
            y_axis_label="F(r, t)",
        )

    # ==========================================
    # ===== Step 7: Non-Gaussian Parameter =====
    # ==========================================
    os.makedirs(os.path.join(dir_out, "Etc"), exist_ok=True)

    # ===== Translational van Hove
    csv_out_non_gauss = os.path.join(dir_out, "Etc", "non_Gaussian_parameter.csv")
    if not os.path.exists(csv_out_non_gauss) or override:
        print("Calculating non-Gaussian parameters...")
        non_gauss_param = dynamics.non_gaussian_parameter(
            coords=mde_com,
            num_segments=num_segments,
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Non-Gaussian parameter",
            file_path=csv_out_non_gauss,
            parameters=imported_params,
        )
        non_gauss_param.to_csv(csv_out_non_gauss, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "Etc", "non_Gaussian_parameter.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            non_gauss_param = pd.read_csv(csv_out_non_gauss)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            non_gauss_param.to_excel(writer, sheet_name="Non-Gaussian Parameter", index=False)

        plot_line(
            output_path=png_out,
            x_data=non_gauss_param.iloc[:, 0],
            y_data=non_gauss_param.iloc[:, 1:],
            x_axis_label="t / ns",
            x_axis_scale='log',
            y_axis_label="NGP(t)",
        )

    # ==========================================
    # ===== Step 7: Chi-4 Susceptibility =====
    # ==========================================
    os.makedirs(os.path.join(dir_out, "Etc"), exist_ok=True)

    # ===== Chi-4 Susceptibility
    csv_out_chi_4 = os.path.join(dir_out, "Etc", "Chi_4_susceptibility.csv")
    if not os.path.exists(csv_out_chi_4) or override:
        print("Calculating chi-4 susceptibility...")
        chi_4 = dynamics.chi_4_susceptibility(
            coords=mde_com,
            q_val=q_val,
            num_segments=num_segments,
        )
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Chi-4 susceptibility",
            file_path=csv_out_chi_4,
            parameters=imported_params,
        )
        chi_4.to_csv(csv_out_chi_4, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "Etc", "Chi_4_susceptibility.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            chi_4 = pd.read_csv(csv_out_chi_4)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            chi_4.to_excel(writer, sheet_name="Chi-4 Susceptibility", index=False)

        plot_line(
            output_path=png_out,
            x_data=chi_4.iloc[:, 0],
            y_data=chi_4.iloc[:, 1:],
            x_axis_label="t / ns",
            x_axis_scale='log',
            y_axis_label=fr"$χ_4(q={q_val:.1f}, t)$",
        )

    # ============================================
    # ===== Step 8: Spatial density function =====
    # ============================================
    os.makedirs(os.path.join(dir_out, "RDF"), exist_ok=True)

    # ===== Spatial density function
    csv_out_sdf = os.path.join(dir_out, "RDF", "Spatial_density.csv")
    if not os.path.exists(csv_out_sdf) or override:
        print("Calculating spatial density functions...")

        res_atom_pairs = {
            res_name: atoms,
            "LNK": ["NL"],
            "ETH": ["OEE"],
            "VAN": ["NV", "OVE", "OVH"],
        }

        sdf = structural.spatial_density_function(
            coords=mde_coords,
            res_atom_pairs=res_atom_pairs,
            pore_diameter=pore_diameter
        )
        imported_params["res_atom_pairs"] = res_atom_pairs
        log_analysis_yaml(
            log_path=yaml_out,
            analysis_name="Spatial density function",
            file_path=csv_out_sdf,
            parameters=imported_params,
        )
        del imported_params["res_atom_pairs"]
        sdf.to_csv(csv_out_sdf, index=False)

    # ===== Plot and save to .xlsx
    png_out = os.path.join(dir_out, "RDF", "Spatial_density.png")
    if not os.path.exists(png_out) or override or plot_only:
        try:
            sdf = pd.read_csv(csv_out_sdf)

        except FileNotFoundError:
            raise FileNotFoundError("CSV files must be generated prior to plotting.")

        with pd.ExcelWriter(
            xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            sdf.to_excel(writer, sheet_name="Spatial density function", index=False)

        plot_line(
            output_path=png_out,
            x_data=sdf.iloc[:, 0],
            y_data=sdf.iloc[:, 1:],
            x_axis_label="r / nm",
            y_axis_label=r"Number density / Units $\cdot$ nm$^3$",
        )

    # ========================================
    # ===== Step 8: Hydrogen bond counts =====
    # ========================================
    os.makedirs(os.path.join(dir_out, "RDF"), exist_ok=True)

    # ===== Spatial density function
    csv_out_sdf = os.path.join(dir_out, "RDF", "Spatial_density.csv")
    if not os.path.exists(csv_out_sdf) or override:
"""

# ===== Step 17: Initialize the hydrogen bond information =====
        if not os.path.exists(f'{workdir}/analysis/data_files/HBonds/All_HBonds_NEW.csv') or overwrite:
            pbar.set_postfix(step="Initializing Hydrogen Bond Information")
            hyd_bonds, unique_pairs = hbonds.hbonds(mdaUniverse=mda_universe, residue=residue, pore=True)
            np.savetxt(f'{workdir}/analysis/data_files/HBonds/All_HBonds_NEW.csv', hyd_bonds, delimiter=',', fmt='%d,%d,%d,%s,%s,%d,%d,%s,%s,%.3f,%.2f', header='Frame,Donor_Index,Donor_ResID,Donor_ResName,Donor_Atom_Name,Acceptor_Index,Acceptor_ResID,Acceptor_ResName,Acceptor_Atom_Name,Distance,Angle')
            try:
                np.savetxt(f'{workdir}/analysis/data_files/HBonds/Unique_HBonds_IDs.csv', unique_pairs, delimiter=',', fmt='%d,%d,%s,%s,%d,%d,%d,%s,%s,%d', header='Donor_Index,Donor_ResID,Donor_ResName,Donor_Atom_Name,Hydrogen_Index,Acceptor_Index,Acceptor_ResName,Acceptor_ResID,Acceptor_Atom_Name,Number_of_Occurrences')
            except ValueError:
                pass
            counts = hbonds.hbond_counts(hbonds=hyd_bonds, residue=residue)

            for pair, count in counts.items():
                helpers.log_info(workdir, f'{pair}: {count}')

            print("Finding Clusters...")
            _, stats = hbonds.find_clusters(hbonds=hyd_bonds, filename=f'{workdir}/analysis/data_files/HBonds/Clusters_{residue}.txt')

            if not stats == None:
                helpers.log_info(workdir, stats)
                with open(f'{workdir}/analysis/data_files/HBonds/Clusters_Summary_{residue}.txt', 'w') as f:
                    f.write(stats)
                    f.close()

        pbar.update(1)

# ===== Step 18: Generate hydrogen bonds heatmap =====
        #if not os.path.exists(f'{workdir}/analysis/graphs/HBonds/HBonds_heatmap_{residue}_{atom1}_{atom2}.png') or overwrite:
        if 1 == 1:
            pbar.set_postfix(step="Calculating hydrogen bonds heatmap")
            hbonds.hbonds_heatmap(mda_universe=mda_universe, residue=residue, atom1=atom1, atom2=atom2, pore_diameter=pore_inf['D'], workdir=workdir, pore=True)
        pbar.update(1)

# ===== Step 19: Generate positional heatmap for any hydrogen bonding species =====
        #if not os.path.exists(f'{workdir}/analysis/graphs/HBonds_heatmap_{atom1}.png') or overwrite:
        if 1 == 1:
            pbar.set_postfix(step="Calculating positional heatmaps")
            structural.radial_distances(mda_universe=mda_universe, workdir = workdir, pore_diameter=pore_inf['D'], exclusions='resname PORE')
        pbar.update(1)

# ===== Step 20: Generate end to end distance =====
        if not os.path.exists(f'{workdir}/analysis/graphs/End_to_End/End_to_end_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating end-to-end distances")
            structural.end_to_end(mda_universe=mda_universe, residue=residue, workdir=workdir)
        pbar.update(1)

# ===== Step 21: Calculate dihedral angles =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Dihedrals/dihedrals.png') or overwrite:
            pbar.set_postfix(step="Gathering Dihedral Angles")
            structural.dihedrals(workdir=workdir)
        pbar.update(1)

# ===== Step 22: Calculate radii of gyration =====
        if not os.path.exists(f'{workdir}/analysis/graphs/Gyration/gyration_summary.txt') or overwrite:
            pbar.set_postfix(step="Gathering Radii of Gyration")
            structural.gyration(workdir=workdir)
        pbar.update(1)

# ===== Complete! =====
        pbar.set_postfix(step="Completed Full Analysis!")
        pbar.update(1)

def minimal(args_dict: dict)-> None:
    '''
    Performs minimal analysis on a pore simulation (including MSD, RDF, and ISF).

    Args:
        args_dict: A dictionary containing all of the imported arguments from main.py
        ver_dir  : Name of the verification folder.
    '''
    workdir = args_dict['workdir']
    segments = args_dict['segments']
    residue = args_dict['residue']
    atom1 = args_dict['atoms'][0]
    atom2 = args_dict['atoms'][1]
    q_val = args_dict['q_value'] if 'q_value' in args_dict.keys() else 0
    overwrite = args_dict['overwrite']
        
    with tqdm(total=4, desc="Minimal Analysis Progress", unit="step") as pbar:
# ===== Step 1: Initialize =====
        pbar.set_postfix(step="Initializing")           # Sets the progress bar step seen on the right hand side of the terminal interface
        # Loop through possible topology and trajectory names (add more if necessary; avoids issues with naming inconsistencies).
        topologies = ["md.tpr", "run.tpr"]
        trajectories = ["out/traj.xtc", "out/out.xtc"]

        mde_trajectory = None
    
        for tpr in topologies:
            for xtc in trajectories:
                print(workdir, mde_trajectory)
                try:
                    mde_trajectory = mde.open(directory=workdir, topology=tpr, trajectory=xtc, nojump=True)
                    mde_vectors = handle.handle_vectors(mde_trajectory=mde_trajectory, residue=residue, atom1=atom1, atom2=atom2)
                    com = helpers.center_of_masses(trajectory=mde_trajectory, residue=residue)
                    print(f"Opened with: topology={tpr}, trajectory={xtc}")
                    break       # Break if successful
                except FileNotFoundError:
                    continue    # Continue if unsuccessful
            if mde_trajectory is not None:
                break           # Break if successful
        if mde_trajectory is None:
            raise FileNotFoundError("No valid combination of topology and trajectory files found.\nPlease ensure that the files exist and/or the file name is listed in the possible topology/trajectory names (bulk.py Step 1)")
        
        # Loop through possible configuration and trajectory names (add more if necessary)
        configurations = ["out/out.gro", "out/md.gro"]
        trajectories = ["out/traj.xtc", "out/out.xtc"]

        mda_universe = None

        for gro in configurations:
            for xtc in trajectories:
                print(f'{workdir}/{gro}')
                try:
                    mda_universe = mda.Universe(f'{workdir}/{gro}', f'{workdir}/{xtc}')
                    print(f"Opened with: configuration={gro}, trajectory={xtc}")
                    break       # Break if successful
                except FileNotFoundError:
                    continue    # Continue if unsuccessful
            if mda_universe is not None:
                break           # Break if successful
        if mda_universe is None:
            raise FileNotFoundError("No valid combination of topology and trajectory files found.\nPlease ensure that the files exist and/or the file name is listed in the possible topology/trajectory names (bulk.py Step 1)")
        
        pore_inf = {}
        pore_dir = next((dir for dir in workdir.split('/') if 'pore_D' in dir), None)
        if pore_dir:
            for inf in re.findall(r'([DLWSEAV])(\d+\.?\d*)', pore_dir): # re.findall will search a string for a specific pattern and return all of the matches as a list.
                pore_inf[inf[0]] = float(inf[1]) if '.' in inf[1] else int(inf[1])
        helpers.log_info(workdir, pore_inf, f'\n@ ANALYSIS RESULTS', )  # Log extracted informati      

        pbar.update(1)          # Update the progress bar after initializing

# ===== Step 2: Calculate the mean square displacement =====
        if not os.path.exists(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png') or overwrite:
            pbar.set_postfix(step="Calculating average MSD")
            msd = dynamics.average_msd(coords=com, num_segments=segments, pore=True)
            plotting.plot_line(f'{workdir}/analysis/graphs/MSD/MSD_{residue}.png', msd[:,0], msd[:,1], msd[:,2], xlabel=r"$\mathbf{\mathit{t}}$ / ps",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$", xscale='log', legend=True, handles=['Average', 'Z-Direction'])
            np.savetxt(f'{workdir}/analysis/data_files/MSD/MSD_{residue}.csv', msd, delimiter=',', header='Time / ps, Average MSD, Average Z-MSD')
        pbar.update(1)

# ===== Step 3: Calculate the RDF / q constant =====
        if q_val == 0 or overwrite:
            pbar.set_postfix(step="Calculating RDF")
            rdf, q_val, g_max = structural.rdf_com(com=com, segments=1000)
            helpers.log_info(workdir, f'Value for q constant: {q_val}', f'Value for g max: {g_max}')
            plotting.plot_line(f'{workdir}/analysis/graphs/RDF/rdf_{residue}_{atom1}_{atom2}.png', rdf[:,0], rdf[:,1], xlabel='r / nm', ylabel='g(r)')
            np.savetxt(f'{workdir}/analysis/data_files/RDF/rdf_{residue}_{atom1}_{atom2}.csv', rdf, delimiter=',', header='r / nm, g(r)')
        pbar.update(1)

# ===== Step 4: Calculate the ISF =====
        if not os.path.exists(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png') or overwrite or q_val > 0:
            pbar.set_postfix(step="Calculating ISF")
            isf = dynamics.incoherent_scattering_function(coords=com, q_val=q_val, num_segments=segments, pore_diameter=pore_inf['D']/2)
            plotting.plot_line(f'{workdir}/analysis/graphs/ISF/ISF_{residue}_{atom1}_{atom2}.png', isf[:,0], isf[:,1], isf[:,2], isf[:,3], xlabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$",
                               ylabel=r"<r$^2$> / nm$^2$$\cdot$ps$^{-1}$", xscale='log', legend=True, handles=['All', 'Wall', 'Center'])
            np.savetxt(f'{workdir}/analysis/data_files/ISF/ISF_{residue}_{atom1}_{atom2}.csv', isf, delimiter=',', header='Time / ps, All, Wall, Center')
        pbar.update(1)

# ===== Complete! =====
        pbar.set_postfix(step="Completed Minimal Analysis!")
        pbar.update(1)
"""

cli.add_command(Run)

if __name__ == "__main__":
    cli()
