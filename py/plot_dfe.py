import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde, levy_stable
from scipy.optimize import curve_fit
import numpy as np


# Function to fit the stable distribution
def stable_pdf(x, _alpha, _beta, _loc, _scale):
    return levy_stable.pdf(x, _alpha, _beta, loc=_loc, scale=_scale)


def exponential_pdf(x, _lambda):
    return _lambda * np.exp(-x * _lambda)


def positive_gaussian_pdf(x, _A, _lambda):
    return _A * np.exp(- (x ** 2) * _lambda)


if __name__ == '__main__':
    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, help='Number of experiments')
    parser.add_argument('n_samples', type=int, help='Number of samples')
    parser.add_argument('n_bins', type=int, help='Number of bins for the DFE histogram')
    parser.add_argument('dir_name', type=str, help='Name of directory data is in')
    parser.add_argument('--fit', action='store_true', help='Fit the DFE to a stable distribution')
    parser.add_argument('--no-fit', dest='fit', action='store_false',
                        help='Do not fit the DFE to a stable distribution')
    parser.add_argument('dfe_days', nargs='+', type=int, help='Days for DFE')
    parser.set_defaults(fit=False)
    args = parser.parse_args()

    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(base_dir, 'dfe_plots_' + args.dir_name)
    os.makedirs(main_dir, exist_ok=True)
    times = args.dfe_days

    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i, args.dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, args.dir_name)
        mut_order, mut_times, dom_strain_mut_order = dfe.pull_mut_hist(i, args.dir_name)

        # dfes will hold the dfe data
        dfes = []

        # Add strains we chose to track lineages for
        bac_data = dfe.pull_bac_data(i, args.dir_name)
        # enum_pops is a list of tuples (strain, bac_data[strain]) sorted by bac_data[strain]
        # For purposes of plotting dominant strains first
        enum_pops = sorted(enumerate(bac_data), key=lambda x: x[1], reverse=True)
        # We will only plot the DFE for the first n_samples strains
        n = min(args.n_samples, len(bac_data))
        # Calculate the DFE for the strains we chose to track lineages for
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
                # dfe_strain_j[k] is the DFE for strain j at day k
                t = times[k]
                # Create a histogram for each dfe_t
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                counts, bins, _ = plt.hist(dfe_t, bins=args.n_bins, density=True, label='Raw Data')
                # Fit a KDE to the histogram data
                kde = gaussian_kde(dfe_t, bw_method='silverman')
                x = np.linspace(min(dfe_t), max(dfe_t), 1000)
                kde_values = kde(x)
                # Plot KDE
                plt.plot(x, kde_values, label='KDE')
                # Calculate rank
                strain_ind = enum_pops[j][0]
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain_ind], mut_times[strain_ind], t)
                alpha_t = dfe.build_alpha(alpha0s, mut_order_strain_t)
                rank = dfe.compute_rank(alpha_t, his, Jijs)
                # Fit KDE data to stable distribution with initial guesses
                if args.fit:
                    initial_guess = [1.5, -1.0, -0.01, 0.005]
                    params, _ = curve_fit(stable_pdf, x, kde_values, p0=initial_guess,
                                          bounds=((0, -1, -np.inf, 0), (2, 1, np.inf, np.inf)))
                    # Extract fitted parameters
                    alpha, beta, loc, scale = params
                    # Calculate fitted stable distribution values
                    fitted_stable_values = stable_pdf(x, alpha, beta, loc, scale)
                    # Plot fitted stable distribution
                    plt.plot(x, fitted_stable_values, linestyle='--', label='Fitted stable distribution')
                    print(
                        f"Fitted stable distribution parameters for {strain}, day {t}: alpha={alpha}, beta={beta}, loc={loc}, scale={scale}")

                # Add vertical lines
                mean_val = np.mean(dfe_t)
                max_index = np.argmax(counts)
                max_val = bins[max_index]
                plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=1)
                plt.axvline(max_val, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(0, color='orange', linestyle='dashed', linewidth=1)

                # Annotate the vertical lines
                ylim = plt.ylim()
                plt.text(mean_val, ylim[1] * 0.1, f'Mean: {mean_val:.2f}', rotation=90, verticalalignment='bottom')
                plt.text(max_val, ylim[1] * 0.1, f'Max: {max_val:.2f}', rotation=90, verticalalignment='bottom')

                plt.legend()
                plt.title(f"strain number: {strain}, day: {t}, rank: {rank}")
                plt.xlabel('Fitness effect')
                plt.ylabel('Frequency')

                # Plot the beneficial tail histogram and fit to an exponential distribution
                plt.subplot(1, 2, 2)
                beneficial_dfe_t = [x for x in dfe_t if x > 0]
                if len(beneficial_dfe_t) > 0:
                    counts_ben, bins_ben, _ = plt.hist(beneficial_dfe_t, bins=int(args.n_bins/2), density=True, alpha=0.6, label='Beneficial tail')
                    kde_beneficial = gaussian_kde(beneficial_dfe_t, bw_method='silverman')
                    x_beneficial = np.linspace(min(beneficial_dfe_t), max(beneficial_dfe_t), 1000)
                    kde_beneficial_values = kde_beneficial(x_beneficial)
                    plt.plot(x_beneficial, kde_beneficial_values, label='KDE')

                    # Fit to exponential
                    initial_guess = [1]
                    params, _ = curve_fit(exponential_pdf, x_beneficial, kde_beneficial_values, p0=initial_guess)
                    _lambda = params[0]
                    fitted_exponential_values = exponential_pdf(x_beneficial, _lambda)
                    plt.plot(x_beneficial, fitted_exponential_values, linestyle='--', label='Fitted exponential')

                    # Calculate chi-squared value
                    observed_beneficial = kde_beneficial_values
                    expected_beneficial = exponential_pdf(x_beneficial, _lambda)
                    chi_squared_beneficial = round(sum((observed_beneficial - expected_beneficial) ** 2 / expected_beneficial), 2)

                    # Annotate the chi-squared value and lambda on the plot
                    plt.text(0.95, 0.5, f'Chi squared: {chi_squared_beneficial}', transform=plt.gca().transAxes,
                             verticalalignment='top', horizontalalignment='right')
                    plt.text(0.95, 0.5, f'λ: {_lambda:.2f}', transform=plt.gca().transAxes,
                             verticalalignment='top', horizontalalignment='right')

                    plt.legend()
                    plt.title(f'Beneficial Tail Histogram and Fit (strain: {strain}, day: {t})')
                    plt.xlabel('Fitness effect')
                    plt.ylabel('Frequency')

                # Save the plot in the strain's directory
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()
