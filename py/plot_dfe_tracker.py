import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from matplotlib.ticker import FuncFormatter


def backward_propagate(dfe_t: list, dfe_0: list):
    """
    Backward propagate the beneficial DFE from the last day to the first day.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :return: list of floats, list of floats
    """
    # The last day's beneficial DFE is the same as the last day's DFE
    bdfe_t = list(enumerate(dfe_t))
    bdfe_t = [x for x in bdfe_t if x[1] >= 0]
    bdfe_t_inds = [x[0] for x in bdfe_t]
    bdfe_t_fits = [x[1] for x in bdfe_t]
    # Now filter dfe_0 to only the elements with indices in bdfe_t_inds
    propagated_bdfe_t = [dfe_0[i] for i in bdfe_t_inds]
    return bdfe_t_fits, propagated_bdfe_t


def forward_propagate(dfe_t: list, dfe_0: list):
    """
    Forward propagate the deleterious DFE from the first day to the last day.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :return: list of floats, list of floats
    """
    bdfe_0 = list(enumerate(dfe_0))
    bdfe_0 = [x for x in bdfe_0 if 0 <= x[1]]  # Filter out all the elements with negative fitness effects
    bdfe_0_inds = [x[0] for x in bdfe_0]
    bdfe_0_fits = [x[1] for x in bdfe_0]
    # Now filter dfe_t to only the elements with indices in bdfe_0_inds
    propagated_bdfe_0 = [dfe_t[i] for i in bdfe_0_inds]
    return bdfe_0_fits, propagated_bdfe_0


def plot_3d_histogram(ax: plt.axes, data_front, data_back, bins, title, const_border, scale_by_front=True):
    """
    Plot a 3D histogram on a given axis.
    :param ax: plt.axes
    :param data_front: list of floats
    :param data_back: list of floats
    :param bins: int
    :param title: str
    :param const_border: float
    :param scale_by_front: bool
    :return: None
    """
    # If const border is non zero, create
    edges_const = np.linspace(start=-const_border, stop=const_border, num=bins)
    if const_border > 0:
        if scale_by_front:
            hist_front, edges = np.histogram(data_front, bins=edges_const)
            hist_back, _ = np.histogram(data_back, bins=edges_const)
        else:
            hist_back, edges = np.histogram(data_back, bins=edges_const)
            hist_front, _ = np.histogram(data_front, bins=edges_const)
    else:
        if scale_by_front:
            hist_front, edges = np.histogram(data_front, bins=bins)
            hist_back, _ = np.histogram(data_back, bins=edges)
        else:
            hist_back, edges = np.histogram(data_back, bins=bins)
            hist_front, _ = np.histogram(data_front, bins=edges)

    xpos = (edges[:-1] + edges[1:]) / 2
    ypos_front = np.zeros_like(xpos)
    ypos_back = np.ones_like(xpos)
    zpos = np.zeros_like(xpos)
    dx = dy = np.diff(edges)
    dz_front = hist_front
    dz_back = hist_back

    ax.bar3d(xpos, ypos_front, zpos, dx, dy, dz_front, zsort='average', color='grey', alpha=0.6, label='Anc')
    ax.bar3d(xpos, ypos_back, zpos, dx, dy, dz_back, zsort='average', color='blue', alpha=0.6, label='Evol')

    ax.set_yticks([])  # Remove yticks
    ax.view_init(elev=10, azim=-90)  # Set the view angle
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}'.format(x * 1e4)))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}'.format(x)))

    ax.set_title(title, fontname='DejaVu Serif')
    ax.set_xlabel('Fitness effect (x 1e-4)', fontname='DejaVu Serif')
    ax.set_zlabel('Frequency', fontname='DejaVu Serif')
    ax.legend(prop={'family': 'DejaVu Serif'})

    # Add vertical dotted line at x=0 with specified color
    ax.plot([0, 0], [0, 1], [0, max(max(dz_front), max(dz_back))], linestyle='dotted', color='purple')


if __name__ == '__main__':

    # Define command line options
    parser = argparse.ArgumentParser(description='Plot the evolution of the members of the BDFE through time.')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('bins', type=int, default=30, help='n_bins for histogram')
    parser.add_argument('border', type=float, default=0.0, help='Scales x axis of histograms')
    parser.add_argument('dir_name', type=str, help='Name of directory data is in')
    parser.add_argument('init_day', type=int, default=0, help='Initial day for tracking')
    parser.add_argument('dfe_days_increments', nargs='*', type=int, default=[0], help='Days for DFE as increments from init_day')
    args = parser.parse_args()

    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Create the main directory if it doesn't exist
    dir_name = args.dir_name
    main_dir = os.path.join(base_dir, 'dfe_tracker_plots_' + dir_name)
    os.makedirs(main_dir, exist_ok=True)
    times = [0]
    times.extend(args.dfe_days_increments)  # Add the days we want to track the DFE for
    times = [t+args.init_day for t in times]    # Add the initial day
    bins = args.bins
    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i, dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, dir_name)
        mut_order, mut_times, _ = dfe.pull_mut_hist(i, dir_name)
        # dfes will hold the dfe data for each strain at designated times
        # len(dfes) = n_samples
        dfes = []
        # Then add strains we chose to track lineages for
        bac_data = dfe.pull_bac_data(i, dir_name)
        # enum_pops is a list of tuples (index, population) sorted by population in descending order
        enum_pops = list(enumerate(bac_data))
        enum_pops.sort(key=lambda x: x[1], reverse=True)
        # n := the number of lineages we track. Cannot be larger than  number of survivors at end of simulation.
        n = min(args.n_samples, len(bac_data))
        for j in range(n):
            dfes_strain = []
            strain, n_bac = enum_pops[j]
            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha = dfe.build_alpha(alpha0s, mut_order_strain_t)
                dfe_t = dfe.compute_dfe(alpha, his, Jijs)
                dfes_strain.append(dfe_t)
            dfes.append(dfes_strain)
        rep = "replicate" + str(i)
        # Now dfes first axis is the strain number, second axis is the day
        # Can delete temp vars
        del mut_order_strain_t, dfe_t, dfes_strain, alpha, his, Jijs
        for j in range(len(dfes)):
            strain = "lineage_" + str(j)
            strain_dir = os.path.join(main_dir, rep, f"strain_{strain}")
            os.makedirs(strain_dir, exist_ok=True)
            dfe_strain_j = dfes[j]
            dfe_0 = dfe_strain_j[0]
            for k in range(1, len(times)):
                # Create a histogram for each dfe_t
                t = times[k]
                dfe_t = dfe_strain_j[k]
                # Backward propagate the beneficial DFE
                bdfe_t_fits, propagated_bdfe_t = backward_propagate(dfe_t, dfe_0)
                # Forward propagate the deleterious DFE
                bdfe_0_fits, propagated_bdfe_0 = forward_propagate(dfe_t, dfe_0)
                fig = plt.figure(figsize=(12, 6))

                # Remove the n_remove largest values from the data to make the plot more readable
                n_remove = 1
                bdfe_t_fits = dfe.remove_n_largest(bdfe_t_fits, n_remove)
                propagated_bdfe_t = dfe.remove_n_largest(propagated_bdfe_t, n_remove)
                bdfe_0_fits = dfe.remove_n_largest(bdfe_0_fits, n_remove)
                propagated_bdfe_0 = dfe.remove_n_largest(propagated_bdfe_0, n_remove)

                ax1 = fig.add_subplot(121, projection='3d')
                plot_3d_histogram(ax1, bdfe_0_fits, propagated_bdfe_0, bins=bins, title=f"Forward DFE at day {t}",
                                  const_border=args.border, scale_by_front=False)

                ax2 = fig.add_subplot(122, projection='3d')
                plot_3d_histogram(ax2, propagated_bdfe_t, bdfe_t_fits, bins=bins, title=f"Backward DFE at day {t}",
                                  const_border=args.border, scale_by_front=True)

                plt.suptitle(f"strain number: {strain}, day: {t}")
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()
