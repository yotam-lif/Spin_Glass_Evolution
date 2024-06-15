import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde, levy_stable
from scipy.optimize import curve_fit
import numpy as np


# Function to fit the stable distribution
def stable_pdf(x, alpha, beta, loc, scale):
    return levy_stable.pdf(x, alpha, beta, loc=loc, scale=scale)


if __name__ == '__main__':
    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('res_pos', type=float, default=10**-3, help='Positive resolution')
    parser.add_argument('res_neg', type=float, default=5*10**-3, help='Negative resolution')
    parser.add_argument('n_bins', type=int, default=100, help='Number of bins for the DFE histogram')
    parser.add_argument('dfe_days', nargs='*', type=int, default=[0], help='Days for DFE')
    args = parser.parse_args()

    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(base_dir, 'dfe_plots')
    os.makedirs(main_dir, exist_ok=True)
    times = args.dfe_days

    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size)
        mut_order, mut_times, dom_strain_mut_order = dfe.pull_mut_hist(i)
        ranks, rank_times = dfe.pull_ranks(i)
        # dfes will hold the dfe data
        dfes = []

        # Calculate the DFE for the dominant strain at given days
        dfes_dom = []
        for t in times:
            mut_order_t = dom_strain_mut_order[t]
            alpha = dfe.build_alpha(alpha0s, mut_order_t)
            dfe_t = dfe.compute_dfe(alpha, his, Jijs)
            dfes_dom.append(dfe_t)
        dfes.append(dfes_dom)

        # Add strains we chose to track lineages for
        bac_data = dfe.pull_bac_data(i)
        enum_pops = sorted(enumerate(bac_data), key=lambda x: x[1], reverse=True)
        n = min(args.n_samples, len(bac_data))

        for j in range(n):
            dfes_strain = []
            strain, _ = enum_pops[j]
            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha = dfe.build_alpha(alpha0s, mut_order_strain_t)
                dfe_t = dfe.compute_dfe(alpha, his, Jijs)
                dfes_strain.append(dfe_t)
            dfes.append(dfes_strain)

        # The first element in dfes is the "dominant strain" DFE
        rep = f"replicate{i}"
        for j, dfe_strain_j in enumerate(dfes):
            strain = "dominant" if j == 0 else f"lineage_{j}"
            strain_dir = os.path.join(main_dir, rep, f"strain_{strain}")
            os.makedirs(strain_dir, exist_ok=True)

            for k, dfe_t in enumerate(dfe_strain_j):
                t = times[k]

                # Create a histogram for each dfe_t
                plt.hist(dfe_t, bins=args.n_bins, density=True, label='Raw Data')

                # Fit a KDE to the histogram data
                kde = gaussian_kde(dfe_t, bw_method='silverman')
                x = np.linspace(min(dfe_t), max(dfe_t), 1000)
                kde_values = kde(x)

                # Fit KDE data to stable distribution with initial guesses
                initial_guess = [1.5, -1.0, -0.01, 0.005]
                params, _ = curve_fit(stable_pdf, x, kde_values, p0=initial_guess, bounds=((0, -1, -np.inf, 0), (2, 1, np.inf, np.inf)))

                # Extract fitted parameters
                alpha, beta, loc, scale = params

                # Calculate fitted stable distribution values
                fitted_stable_values = stable_pdf(x, alpha, beta, loc, scale)

                # Plot KDE
                plt.plot(x, kde_values, label='KDE')

                # Plot fitted stable distribution
                plt.plot(x, fitted_stable_values, linestyle='--', label='Fitted stable distribution')
                title = f"strain number: {strain}, day: {t}"
                # if t in rank_times:
                #     ind = rank_times.index(t)
                #     title += f", rank: {ranks[ind]}"
                plt.legend()
                plt.title(title)
                plt.xlabel('Fitness effect')
                plt.ylabel('Frequency')

                # Save the plot in the strain's directory
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()

                # Print the fitted parameters
                print(f"Fitted stable distribution parameters for {strain}, day {t}: alpha={alpha}, beta={beta}, loc={loc}, scale={scale}")