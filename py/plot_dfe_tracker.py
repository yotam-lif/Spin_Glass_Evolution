import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def backward_propagate(dfe_t: list, dfe_0: list, beneficial=True):
    """
    Backward propagate the DFE from the last day to the first day,
    based on whether beneficial or deleterious mutations are selected.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :param beneficial: bool, whether to track beneficial or deleterious mutations
    :return: list of floats, list of floats
    """
    bdfe_t = [(i, dfe_t[i]) for i in range(len(dfe_t)) if (dfe_t[i] >= 0 if beneficial else dfe_t[i] <= 0)]

    bdfe_t_inds = [x[0] for x in bdfe_t]
    bdfe_t_fits = [x[1] for x in bdfe_t]

    propagated_bdfe_t = [dfe_0[i] for i in bdfe_t_inds]

    return bdfe_t_fits, propagated_bdfe_t


def forward_propagate(dfe_t: list, dfe_0: list, beneficial=True):
    """
    Forward propagate the DFE from the first day to the last day,
    based on whether beneficial or deleterious mutations are selected.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :param beneficial: bool, whether to track beneficial or deleterious mutations
    :return: list of floats, list of floats
    """
    bdfe_0 = [(i, dfe_0[i]) for i in range(len(dfe_0)) if (dfe_0[i] >= 0 if beneficial else dfe_0[i] <= 0)]

    bdfe_0_inds = [x[0] for x in bdfe_0]
    bdfe_0_fits = [x[1] for x in bdfe_0]

    propagated_bdfe_0 = [dfe_t[i] for i in bdfe_0_inds]

    return bdfe_0_fits, propagated_bdfe_0


def plot_2d_histogram(ax: plt.axes, data_front, data_back, bins, title, scale_by_front=True):
    """
    Plot a 2D histogram on a given axis using seaborn for improved aesthetics.
    :param ax: plt.axes
    :param data_front: list of floats
    :param data_back: list of floats
    :param bins: int
    :param title: str
    :param scale_by_front: bool
    :return: None
    """
    if scale_by_front:
        sns.histplot(data_front, bins=bins, kde=False, stat="density", color="grey", label='Anc', ax=ax, alpha=0.7)
        sns.histplot(data_back, bins=bins, kde=False, stat="density", color="purple", label='Evol', ax=ax, alpha=0.5)
    else:
        sns.histplot(data_back, bins=bins, kde=False, stat="density", color="purple", label='Evol', ax=ax, alpha=0.5)
        sns.histplot(data_front, bins=bins, kde=False, stat="density", color="grey", label='Anc', ax=ax, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Fitness effect (x 1e-4)')
    ax.set_ylabel('Frequency')
    ax.legend()


if __name__ == '__main__':

    # Define command line options
    parser = argparse.ArgumentParser(description='Plot the evolution of the members of the DFE through time.')
    parser.add_argument('n_exps', type=int, help='Number of experiments')
    parser.add_argument('n_samples', type=int, help='Number of samples')
    parser.add_argument('bins', type=int, help='Number of bins for histogram')
    parser.add_argument('dir_name', type=str, help='Name of directory where data is located')
    parser.add_argument('--beneficial', action='store_true', help='Track beneficial mutations (default)')
    parser.add_argument('--deleterious', action='store_false', dest='beneficial', help='Track deleterious mutations')
    parser.add_argument('init_day', type=int, help='Initial day for tracking')
    parser.add_argument('dfe_days_increments', nargs='*', type=int, help='Days for DFE as increments from init_day')
    args = parser.parse_args()

    # Prepare directories
    current_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    dir_name = args.dir_name
    main_dir = os.path.join(base_dir, dir_name, 'dfe_tracker_plots')
    os.makedirs(main_dir, exist_ok=True)

    # Initialize times to track
    times = [args.init_day + t for t in [0] + args.dfe_days_increments]

    # Process each experiment
    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i, dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, dir_name)
        mut_order, mut_times, _ = dfe.pull_mut_hist(i, dir_name)
        bac_data = dfe.pull_bac_data(i, dir_name)
        enum_pops = sorted(enumerate(bac_data), key=lambda x: x[1], reverse=True)
        n = min(args.n_samples, len(bac_data))

        dfes = []
        for j in range(n):
            strain, _ = enum_pops[j]
            dfes_strain = []
            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha = dfe.build_alpha(alpha0s, mut_order_strain_t)
                dfe_t = dfe.compute_dfe(alpha, his, Jijs)
                dfes_strain.append(dfe_t)
            dfes.append(dfes_strain)

        rep = f"replicate_{i}"
        for j, dfe_strain_j in enumerate(dfes):
            strain_dir = os.path.join(main_dir, rep, f"strain_{j}")
            os.makedirs(strain_dir, exist_ok=True)
            dfe_0 = dfe_strain_j[0]
            for k in range(1, len(times)):
                t = times[k]
                dfe_t = dfe_strain_j[k]

                # Backward propagate the DFE
                bdfe_t_fits, propagated_bdfe_t = backward_propagate(dfe_t, dfe_0, beneficial=args.beneficial)
                # Forward propagate the DFE
                bdfe_0_fits, propagated_bdfe_0 = forward_propagate(dfe_t, dfe_0, beneficial=args.beneficial)

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                # Plot Forward DFE, scaled by front
                plot_2d_histogram(ax[0], bdfe_0_fits, propagated_bdfe_0, bins=args.bins,
                                  title=f"Forward DFE at day {t}", scale_by_front=True)

                # Plot Backward DFE, scaled by back
                plot_2d_histogram(ax[1], propagated_bdfe_t, bdfe_t_fits, bins=args.bins,
                                  title=f"Backward DFE at day {t}", scale_by_front=False)

                plt.suptitle(f"Strain number: {j}, Day: {t}")
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()
