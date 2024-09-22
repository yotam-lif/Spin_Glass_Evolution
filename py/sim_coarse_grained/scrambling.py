import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import common as cmn


def plot_2d_histogram(ax: plt.Axes, data_front, data_back, data_dfe, bins, title, scale_by_front=True):
    """
    Plot a 2D histogram on a given axis using seaborn for improved aesthetics.

    :param ax: plt.Axes
    :param data_front: list of floats (ancestral or earlier DFE data).
    :param data_back: list of floats (current DFE data).
    :param data_dfe: list of floats (DFE data to add as third histogram in blue).
    :param bins: int, number of bins for histograms.
    :param title: str, title for the plot.
    :param scale_by_front: bool, whether to scale by front data or back data for the order of plotting.
    :return: None.
    """
    stat = "probability"
    if scale_by_front:
        sns.histplot(data_front, bins=bins, kde=False, stat=stat, color="grey", label='Anc', ax=ax, alpha=0.7)
        sns.histplot(data_dfe, bins=bins, kde=False, stat=stat, color="blue", label='DFE Evol', ax=ax, alpha=0.2)
        sns.histplot(data_back, bins=bins, kde=False, stat=stat, color="purple", label='Evol', ax=ax, alpha=0.5)
    else:
        sns.histplot(data_back, bins=bins, kde=False, stat=stat, color="purple", label='Evol', ax=ax, alpha=0.5)
        sns.histplot(data_dfe, bins=bins, kde=False, stat=stat, color="blue", label='DFE Anc', ax=ax, alpha=0.2)
        sns.histplot(data_front, bins=bins, kde=False, stat=stat, color="grey", label='Anc', ax=ax, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Fitness effect (x 1e-4)')
    ax.set_ylabel('Frequency')
    ax.legend()


if __name__ == '__main__':

    # Parameters
    L = 4000  # Length of the genome
    rho = 0.05  # Sparsity parameter
    delta = 0.005
    beta = 0.75
    rank_final = 100  # Rank at which to stop the simulation
    bins = 50  # Number of bins for histogram
    sig_h = np.sqrt(1 - beta) * delta
    sig_J = np.sqrt(beta) * delta / np.sqrt(L * rho) / 2
    n_times = 10  # Number of time points to track

    # Create main output directory for scrambling plots
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.join(base_dir, "scrambling_plots")
    beneficial_dir = os.path.join(main_dir, "scrambling_beneficial_plots")
    deleterious_dir = os.path.join(main_dir, "scrambling_deleterious_plots")

    # Create subdirectories for beneficial and deleterious plots
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(beneficial_dir, exist_ok=True)
    os.makedirs(deleterious_dir, exist_ok=True)

    # Run a single simulation
    alpha0, his, Jijs, mut_hist = cmn.run_simulation_sswm(L, rho, rank_final, sig_h, sig_J)
    alpha_ts = cmn.build_alpha_t(alpha0, mut_hist)
    dfes, lfs = cmn.build_dfes_lfs(alpha_ts, his, Jijs)

    # Initialize times to track
    times = np.linspace(0, len(dfes) - 1, n_times, dtype=int)

    # Process the strain over time for both beneficial and deleterious mutations
    for beneficial in [True, False]:
        dfe_0 = dfes[0]  # DFE at the first time point (initial day)
        output_dir = beneficial_dir if beneficial else deleterious_dir  # Choose directory

        for t in times:
            dfe_t = dfes[t]

            # Backward propagate the DFE
            bdfe_t_fits, propagated_bdfe_t = cmn.backward_propagate(dfe_t, dfe_0, beneficial=beneficial)
            # Forward propagate the DFE
            bdfe_0_fits, propagated_bdfe_0 = cmn.forward_propagate(dfe_t, dfe_0, beneficial=beneficial)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Plot Forward DFE, scaled by front, with DFE added in blue
            plot_2d_histogram(ax[0], bdfe_0_fits, propagated_bdfe_0, dfe_t, bins=bins,
                              title=f"Forward DFE after {t} mutations ({'Beneficial' if beneficial else 'Deleterious'})",
                              scale_by_front=True)

            # Plot Backward DFE, scaled by back, with DFE added in blue
            plot_2d_histogram(ax[1], propagated_bdfe_t, bdfe_t_fits, dfe_0, bins=bins,
                              title=f"Backward DFE after {t} mutations ({'Beneficial' if beneficial else 'Deleterious'})",
                              scale_by_front=False)

            plt.suptitle(f"#mutations: {t} ({'Beneficial' if beneficial else 'Deleterious'})")
            plt.savefig(os.path.join(output_dir, f"scrambling_nmuts_{t}.png"), dpi=300)
            plt.close()
