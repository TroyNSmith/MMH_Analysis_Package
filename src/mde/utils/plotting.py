import pandas as pd
import matplotlib.pyplot as plt

from cycler import cycler
from pathlib import Path
from typing import Callable

from .output import SummaryLogger

def plot_from_log(
    module_name: str,
    output_directory: Path,
    current_params: dict,
    plot_func: Callable[[pd.DataFrame, Path], None],
):
    """
    Searches the log for matching params and plots any associated outputs
    that haven't already been plotted.

    :param module_name: Name of the analysis module (e.g. "ISF").
    :param output_directory: Path to the output directory containing the log.
    :param current_params: Dict of parameters to match in the log.
    :param plot_func: A function that accepts (dataframe, output_path).
    """
    logger = SummaryLogger(output_directory)
    log_dir = Path(output_directory)

    matched_entries = [
        entry for entry in logger.data.get(module_name, [])
        if entry.get("params") == current_params
    ]

    if not matched_entries:
        raise ValueError("No matching parameters found in log.")

    for entry in matched_entries:
        outputs = entry.get("metadata", {}).get("outputs", [])
        for csv_path_str in outputs:
            csv_path = log_dir / csv_path_str
            if not csv_path.exists():
                continue

            png_path = csv_path.with_suffix(".png")
            if png_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
                plot_func(df, png_path)
            except Exception as e:
                print(f"Could not plot {csv_path.name}: {e}")

def plot_line(
    output_path: str,
    x_data: pd.DataFrame,
    y_data: pd.DataFrame,
    x_axis_label: str = "",
    x_axis_scale: str = "linear",
    y_axis_label: str = "",
    y_axis_scale: str = "linear",
    use_df_headers: bool = True,
    show_legend: bool = False,
    plot_size: tuple = (8, 6),
    axis_font_size: float = 16,
    tick_font_size: float = 14,
    color_cycler: list = None,
):

    if len(y_data.columns.to_list()) <= 1:
        use_df_headers = False

    if color_cycler is None:
        color_cycler = cycler("color", ["#CC6677", "#332288", "#DDCC77", "#117733"])

    Labels = y_data.columns if use_df_headers else None
    plt.rc("axes", prop_cycle=color_cycler)

    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(x_data, y_data, label=Labels)

    ax.set_xlabel(x_axis_label, fontsize=axis_font_size)
    ax.set_xscale(x_axis_scale)
    ax.tick_params(axis="x", labelsize=tick_font_size)

    ax.set_ylabel(y_axis_label, fontsize=axis_font_size)
    ax.set_yscale(y_axis_scale)
    ax.tick_params(axis="y", labelsize=tick_font_size)

    if show_legend and Labels is not None:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
