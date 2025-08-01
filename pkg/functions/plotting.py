import pandas as pd
import matplotlib.pyplot as plt

from cycler import cycler
from pathlib import Path


def plot_line(
    output_path: Path,
    x_data: pd.DataFrame,
    y_data: pd.DataFrame,
    x_axis_label: str = "",
    x_axis_scale: str = "linear",
    y_axis_label: str = "",
    y_axis_scale: str = "linear",
    use_df_headers: bool = True,
    show_legend: bool = True,
    plot_size: tuple = (8, 6),
    axis_font_size: float = 16,
    tick_font_size: float = 14,
    color_cycler: list = None,
):
    if len(y_data.columns.to_list()) <= 1:
        use_df_headers = False

    if color_cycler is None:
        color_cycler = cycler(
            "color",
            [
                "#CC6677",
                "#332288",
                "#DDCC77",
                "#117733",
                "#88CCEE",
                "#882255",
                "#44AA99",
                "#999933",
                "#AA4499",
                "#77AADD",
                "#EE8866",
                "#EEDD88",
                "#FFAABB",
                "#99DDFF",
                "#44BB99",
                "#BBCC33",
                "#AAAA00",
                "#DDDDDD",
            ],
        )

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
