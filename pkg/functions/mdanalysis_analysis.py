import MDAnalysis as mda
import numpy as np
import pandas as pd

from collections import defaultdict
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba
from pathlib import Path


def identify_h_bonds(
    mda_coords: mda.Universe, res_name: str = None
) -> tuple[pd.DataFrame, hba]:
    """
    Gathers hydrogen bond data for a given residue as both an acceptor and a donor through MDAnalysis.

    Args:
        mdaUniverse : Universe object initialized by MDAnalysis.
        start       : Starting frame of analysis.
        stop        : Ending frame of analysis.

    Returns:
        NDArray     : NumPy array containing hydrogen bond information.
    """
    hbonds = hba(
        universe=mda_coords,
        d_a_cutoff=3.5,  # Change to 3.5 A if you wish to match GMX default parameters or 3.0 A for MDAnalysis default parameters.
        d_h_a_angle_cutoff=150,  # Same as GMX default parameters
        update_selections=False,  # No dynamic selections
    ).run()

    data_organized = np.column_stack(
        [
            hbonds.results.hbonds[:, 0].astype(int),
            hbonds.results.hbonds[:, 1].astype(int),
            mda_coords.atoms[hbonds.results.hbonds[:, 1].astype(int)].resids,
            mda_coords.atoms[hbonds.results.hbonds[:, 1].astype(int)].resnames,
            mda_coords.atoms[hbonds.results.hbonds[:, 1].astype(int)].names,
            hbonds.results.hbonds[:, 3].astype(int),
            mda_coords.atoms[hbonds.results.hbonds[:, 3].astype(int)].resids,
            mda_coords.atoms[hbonds.results.hbonds[:, 3].astype(int)].resnames,
            mda_coords.atoms[hbonds.results.hbonds[:, 3].astype(int)].names,
            hbonds.results.hbonds[:, 4],
            hbonds.results.hbonds[:, 5],
        ]
    )

    column_labels = [
        "Frame",
        "Donor_index",
        "Donor_resid",
        "Donor_resname",
        "Donor_atom",
        "Acceptor_index",
        "Acceptor_resid",
        "Acceptor_resname",
        "Acceptor_atom",
        "Distance",
        "Angle",
    ]

    df = pd.DataFrame(data_organized, columns=column_labels)

    if res_name is not None:
        mask = (df["Donor_resname"] == res_name) | (df["Acceptor_resname"] == res_name)
        df = df[mask]

    return df, hbonds


def analyze_h_bonds(h_bonds_df: pd.DataFrame, xlsx_out: Path) -> dict:
    results = {}

    # === Total and average hydrogen bonds per frame
    hbonds_per_frame = h_bonds_df.groupby("Frame").size()
    num_hbonds = len(h_bonds_df)
    results["H bonds (total)"] = num_hbonds

    avg_hbonds = hbonds_per_frame.mean()
    std_hbonds = hbonds_per_frame.std()
    results["Average H bonds (per frame)"] = f"{avg_hbonds:.1f} +/- {std_hbonds:.1f}"

    # === Intramolecular hydrogen bonds
    intra_hbonds = h_bonds_df[h_bonds_df["Donor_index"] == h_bonds_df["Acceptor_index"]]
    num_intra = len(intra_hbonds)
    results["Intramolecular H bonds (total)"] = num_intra

    intra_counts_per_frame = intra_hbonds.groupby("Frame").size()
    avg_intra = intra_counts_per_frame.mean()
    std_intra = intra_counts_per_frame.std()
    results["Average intramolecular H bonds (per frame)"] = (
        f"{avg_intra:.1f} +/- {std_intra:.1f}"
    )

    # === Intermolecular hydrogen bonds
    inter_hbonds = h_bonds_df[h_bonds_df["Donor_index"] != h_bonds_df["Acceptor_index"]]
    num_inter = len(inter_hbonds)
    results["Intermolecular H bonds (total)"] = num_inter

    inter_counts_per_frame = inter_hbonds.groupby("Frame").size()
    avg_inter = inter_counts_per_frame.mean()
    std_inter = inter_counts_per_frame.std()
    results["Average intermolecular H bonds (per frame)"] = (
        f"{avg_inter:.1f} +/- {std_inter:.1f}"
    )

    # === Unique donor-acceptor pair counts
    unique_pair_counts = (
        h_bonds_df.groupby(
            ["Donor_resname", "Donor_atom", "Acceptor_resname", "Acceptor_atom"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    with pd.ExcelWriter(
        xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        unique_pair_counts.to_excel(writer, sheet_name="Unique pairs", index=False)

    for row in unique_pair_counts.itertuples(index=False):
        key = f"{row.Donor_resname}/{row.Donor_atom} to {row.Acceptor_resname}/{row.Acceptor_atom} (total)"
        results[key] = row.count

    summary_df = pd.DataFrame(results.items(), columns=["Category", "Result"])
    with pd.ExcelWriter(
        xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    return results


def hbonds_heatmap(
    mda_coords: mda.Universe, h_bonds_df: pd.DataFrame, pore_diameter: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    
    x_bins = round(pore_diameter * 100)
    y_bins = round(pore_diameter * 100)

    individual_data = defaultdict(list)  # {(resname, atom): [(x, y), ...]}

    for ts in mda_coords.trajectory:
        frame = ts.frame
        h_bonds_frame = h_bonds_df[h_bonds_df["Frame"] == frame]

        for row in h_bonds_frame.itertuples(index=False):
            for type in ["Donor", "Acceptor"]:
                atom_index = int(getattr(row, f"{type}_index"))
                resname = getattr(row, f"{type}_resname")
                atom = getattr(row, f"{type}_atom")
                position = mda_coords.atoms[atom_index].position

                x = position[0] / 10.0  # Å to nm
                y = position[1] / 10.0
                individual_data[f"{resname}_{atom}"].append((x, y))

    heatmaps = {}
    total_heatmap = np.zeros((x_bins, y_bins))

    for (resname_atom, coords) in individual_data.items():
        if not coords:
            continue

        x_vals, y_vals = zip(*coords)
        heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[x_bins, y_bins])
        heatmaps[resname_atom] = heatmap
        total_heatmap += heatmap

    heatmaps["All_h_bonds"] = total_heatmap

    x_mesh, y_mesh = np.meshgrid(xedges[:-1], yedges[:-1]) # Create meshgrid once for all plots

    return x_mesh, y_mesh, heatmaps

def find_clusters(h_bonds_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    all_clusters = []

    def __neighbor_search(index: int, frame_df: pd.DataFrame, visited: np.ndarray):
        donor_idx = frame_df.iloc[index]["Donor_index"]
        donor_resid = frame_df.iloc[index]["Donor_resid"]
        acceptor_idx = frame_df.iloc[index]["Acceptor_index"]
        acceptor_resid = frame_df.iloc[index]["Donor_index"]

        neighbors = []

        for i in range(len(frame_df)):
            if not visited[i]:  # Only consider non-visited bonds
                if (
                    frame_df.iloc[i]["Donor_index"] == acceptor_idx or frame_df.iloc[i]["Acceptor_index"] == donor_idx
                ):  # Donor acts as an acceptor or acceptor acts as a donor
                    neighbors.append(i)

                if (
                    frame_df.iloc[i]["Donor_resid"] == donor_resid or frame_df.iloc[i]["Acceptor_resid"] == acceptor_resid
                ):  # Check for atoms in the same residue being donor/acceptor
                    neighbors.append(i)

        return neighbors

    def __explore_cluster(start_index: int, frame_df: pd.DataFrame, visited: np.ndarray):
        cluster_resids = set()  # Store unique residue IDs for the cluster
        to_visit = [start_index]

        while to_visit:
            current_idx = to_visit.pop()
            if visited[current_idx]:
                continue
            visited[current_idx] = True

            donor_resid = frame_df.iloc[current_idx]["Donor_resid"]
            cluster_resids.add(donor_resid)  # Add the donor to the cluster
            acceptor_resid = frame_df.iloc[current_idx]["Acceptor_resid"]
            cluster_resids.add(acceptor_resid)  # Add the acceptor to the cluster

            neighbors = __neighbor_search(current_idx, frame_df, visited)
            to_visit.extend(neighbors)

        return cluster_resids

    for frame in sorted(h_bonds_df["Frame"].unique()):  # Loop over unique frames
        frame_df = h_bonds_df[h_bonds_df["Frame"] == frame]

        visited = np.zeros(len(frame_df), dtype=bool)  # Visited array for this frame

        for i in range(len(frame_df)):
            if not visited[i]:  # If bond hasn't been visited, start a cluster search
                cluster_resids = __explore_cluster(i, frame_df, visited)
                if (
                    len(cluster_resids) > 2
                ):  # Only consider clusters with more than 1 bond
                    all_clusters.append({
                        "Frame": frame,
                        "Participants": [int(r) for r in cluster_resids]
                    })

    clusters = pd.DataFrame(all_clusters)

    # Statistics
    clusters["Size"] = clusters["Participants"].apply(len)
    frame_counts = clusters.groupby("Frame").size()
    cluster_sizes = clusters["Size"]

    stats = {
        "Number of clusters (total)": len(clusters),
        "Average clusters (per frame)": f"{frame_counts.mean():.2f} ± {frame_counts.std():.2f}",
        "Average cluster size": f"{cluster_sizes.mean():.2f} ± {cluster_sizes.std():.2f}"
    }

    # Drop "Size" column before returning
    clusters = clusters.drop(columns=["Size"])

    return clusters, stats