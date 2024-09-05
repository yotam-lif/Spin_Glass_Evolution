import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import levy_stable
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
    # Set seaborn style
    sns.set_style('whitegrid')

    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, help='Number of experiments')
    parser.add_argument('n_samples', type=int, help='Number of samples')
    parser.add_argument('n_bins', type=int, help='Number of bins for the DFE histogram')
    parser.add_argument('dir_name', type=str, help='Name of directory data is in')
    parser.add_argument('--fit', action='store_true', help='Fit the DFE to a stable distribution')
    parser.add_argument('--no-fit', dest='fit', action='store_false', help='Do not fit the DFE to a stable distribution')
    parser.add_argument('dfe_days', nargs='+', type=int, help='Days for DFE')
    parser.set_defaults(fit=False)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(base_dir, args.dir_name, 'dfe_plots')
    os.makedirs(main_dir, exist_ok=True)
    times = args.dfe_days

    for i in range(args.n_exps):
        alpha0s, his, Jijs = dfe.pull_env(i, args.dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, args.dir_name)
        mut_order, mut_times, dom_strain_mut_order = dfe.pull_mut_hist(i, args.dir_name)

        dfes = []
        bac_data = dfe.pull_bac_data(i, args.dir_name)
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

        rep = f"replicate{i}"
        for j, dfe_strain_j in enumerate(dfes):
            strain = "dominant" if j == 0 else f"lineage_{j}"
            strain_dir = os.path.join(main_dir, rep, f"strain_{strain}")
            os.makedirs(strain_dir, exist_ok=True)

            for k, dfe_t in enumerate(dfe_strain_j):
                t = times[k]
                plt.figure(figsize=(12, 6))

                # Left subplot - DFE Histogram and KDE using sns.histplot
                plt.subplot(1, 2, 1)
                ax1 = sns.histplot(dfe_t, bins=args.n_bins, kde=True, label='DFE', color='blue', alpha=0.6, stat='density')

                strain_ind = enum_pops[j][0]
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain_ind], mut_times[strain_ind], t)
                alpha_t = dfe.build_alpha(alpha0s, mut_order_strain_t)
                rank = dfe.compute_rank(alpha_t, his, Jijs)

                if args.fit:
                    kde_line = ax1.lines[0]
                    x = kde_line.get_xdata()
                    kde_values = kde_line.get_ydata()

                    initial_guess = [1.5, -1.0, -0.01, 0.005]
                    params, _ = curve_fit(stable_pdf, x, kde_values, p0=initial_guess,
                                          bounds=((0, -1, -np.inf, 0), (2, 1, np.inf, np.inf)))
                    alpha, beta, loc, scale = params
                    fitted_stable_values = stable_pdf(x, alpha, beta, loc, scale)
                    plt.plot(x, fitted_stable_values, linestyle='--', label='Fitted stable distribution', color='green')
                    print(f"Fitted stable distribution parameters for {strain}, day {t}: alpha={alpha}, beta={beta}, loc={loc}, scale={scale}")

                mean_val = np.mean(dfe_t)
                # Add vertical lines with labels
                plt.axvline(mean_val, color='green', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
                plt.axvline(0, color='black', linestyle='dashed', linewidth=1, label=r'$\Delta = 0$')

                plt.legend()
                plt.title(f"strain number: {strain}, day: {t}, rank: {rank}")
                plt.xlabel(r'$\Delta$')
                plt.ylabel(r'P($\Delta$)')

                # Right subplot - Beneficial tail histogram and fit to exponential
                plt.subplot(1, 2, 2)
                beneficial_dfe_t = [x for x in dfe_t if x > 0]
                if len(beneficial_dfe_t) > 0:
                    ax2 = sns.histplot(beneficial_dfe_t, bins=int(args.n_bins / 2), kde=True, label='Beneficial tail', color='blue', alpha=0.6)

                    initial_guess = [1]
                    counts_ben, bins_ben = np.histogram(beneficial_dfe_t, bins=int(args.n_bins / 2), density=True)
                    bin_centers = (bins_ben[:-1] + bins_ben[1:]) / 2
                    params, _ = curve_fit(exponential_pdf, bin_centers, counts_ben, p0=initial_guess)
                    _lambda = params[0]

                    x_beneficial = np.linspace(min(beneficial_dfe_t), max(beneficial_dfe_t), 1000)
                    fitted_exponential_values = exponential_pdf(x_beneficial, _lambda)
                    plt.plot(x_beneficial, fitted_exponential_values, linestyle='--', label='Fitted exponential', color='red')

                    chi_squared_beneficial = round(sum((counts_ben - exponential_pdf(bin_centers, _lambda)) ** 2 / exponential_pdf(bin_centers, _lambda)), 2)

                    plt.text(0.95, 0.5, f'Chi squared: {chi_squared_beneficial}', transform=plt.gca().transAxes,
                             verticalalignment='top', horizontalalignment='right')
                    plt.text(0.95, 0.7, f'λ: {_lambda:.2f}', transform=plt.gca().transAxes,
                             verticalalignment='top', horizontalalignment='right')

                    plt.legend()
                    plt.title(f'Beneficial Tail Histogram and Fit (strain: {strain}, day: {t})')
                    plt.xlabel('Fitness effect')
                    plt.ylabel('Frequency')

                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()
