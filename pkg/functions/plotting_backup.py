# Imports #
# ------- #
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any
# ------- #
try:
    plt.style.use('MPL_Styles/ForPapers.mplstyle')
except FileNotFoundError:
    pass

def plot_line(output: str,
              x: NDArray,
              *y: NDArray,
              **kwargs: Any,
              ) -> None:
    """
    Plots a line plot with multiple y-values (if provided), allowing customization via **kwargs.

    Args:
        output: Location and name of output file.
        x: Array of x-values.
        *y: One or more arrays of y-values to be plotted.
        **kwargs: Additional keyword arguments for customizing the plot, such as:
            - title (str): Title of the plot.
            - xlabel (str): Label for the x-axis.
            - ylabel (str): Label for the y-axis.
            - legend (bool): Whether to display a legend.
            - xscale (str): Scaling for the x-axis ('linear', 'log', etc.).
            - yscale (str): Scaling for the y-axis ('linear', 'log', etc.).
            - handles (list of str): List of legend labels corresponding to each y array.
            - ncols (int): Number of columns in legend.

    Example:
        plot_line(x, y1, y2, title="My Plot", xlabel="Time", ylabel="Value", linestyle="--", legend=True, label=["Sine", "Cosine"])
    """
    legend = kwargs.get('legend', False)

    # Get the labels from kwargs (default to Line 1, Line 2, etc. if not provided)
    labels = kwargs.get('handles', [f'Line {i+1}' for i in range(len(y))])
    
    plt.clf()

    # Create the plot for each y
    for i, y_vals in enumerate(y):
        try:
            for col in range(y_vals.shape[1]):
                plt.plot(x, y_vals[:,col], label=labels[col])
        except IndexError or AttributeError or UnboundLocalError:
            plt.plot(x, y_vals, label=labels[i])

    # Set the title and axis labels if provided
    if 'title' in kwargs:    
        plt.title(kwargs['title'])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])

    # Set the x and y axis scales if provided
    if 'xscale' in kwargs:
        plt.xscale(kwargs['xscale'])
    if 'yscale' in kwargs:
        plt.yscale(kwargs['yscale'])

    # Show the legend if requested
    if legend:
        if 'ncols' in kwargs:
            plt.legend(ncols=kwargs['ncols'], loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_scatter(output: str,
                 x: NDArray,
                 *y: NDArray,
                 **kwargs: Any,
                 ) -> None:
    """
    Plots a line plot with multiple y-values (if provided), allowing customization via **kwargs.

    Args:
        output: Location and name of output file.
        x: Array of x-values.
        *y: One or more arrays of y-values to be plotted.
        **kwargs: Additional keyword arguments for customizing the plot, such as:
            - title (str): Title of the plot.
            - xlabel (str): Label for the x-axis.
            - ylabel (str): Label for the y-axis.
            - legend (bool): Whether to display a legend.
            - xscale (str): Scaling for the x-axis ('linear', 'log', etc.).
            - yscale (str): Scaling for the y-axis ('linear', 'log', etc.).
            - handles (list of str): List of legend labels corresponding to each y array.
            - ncols (int): Number of columns in legend.

    Example:
        plot_line(x, y1, y2, title="My Plot", xlabel="Time", ylabel="Value", linestyle="--", legend=True, label=["Sine", "Cosine"])
    """
    legend = kwargs.get('legend', False)

    # Get the labels from kwargs (default to Line 1, Line 2, etc. if not provided)
    labels = kwargs.get('handles', [f'Line {i+1}' for i in range(len(y) + 1)])
    
    plt.clf()

    # Create the plot for each y
    for i, y_vals in enumerate(y):
        try:
            for col in range(y_vals.shape[1]):
                plt.scatter(x, y_vals[:,col], label=labels[col], linewidths=0.2, s=10)
        except IndexError or AttributeError or UnboundLocalError:
            plt.scatter(x, y_vals, label=labels[i], linewidths=0.2, s=10)
    # Set the title and axis labels if provided
    if 'title' in kwargs:    
        plt.title(kwargs['title'])
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])

    # Set the x and y axis scales if provided
    if 'xscale' in kwargs:
        plt.xscale(kwargs['xscale'])
    if 'yscale' in kwargs:
        plt.yscale(kwargs['yscale'])

    # Show the legend if requested
    if legend:
        if 'ncols' in kwargs:
            plt.legend(ncols=kwargs['ncols'], loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.legend(ncols=1, loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.close()
