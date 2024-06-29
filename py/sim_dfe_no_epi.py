import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import os


# The idea behind this script is as follows:
# No need to simulate full experiment to get dfe evolution of no epistasis case.
# Instead, we can simulate the dfe evolution directly.
# At start of original simulation we sample h_is and ~ half have positive effects, dependent on sign of alpha_i.
# Evolution "eats" the positive effects iteratively dependent on their magnitude, so we can simulate this directly.
# Just sample L random "initial" alpha_i * h_i values. These are the alphai_x_hi at t=0.
# This is just sampling L gaussian random variables, as alpha_i is just a sign and they are equally probable.
# Then each time step we choose in a biased way (magnitude of h_i) an alpha_i to flip.
# If alpha_x_hi is negative, this can "evolve" to match. We "eat" these members and turn them positive.
# We just do the uncomplicated thing and "eat" the positives, and then dfe is 2* alpha_x_hi instead of -2 * alpha_x_hi.

def exponent(x, _lambda):
    return _lambda * np.exp(-x * _lambda)


def run_simulation(L, times, dir_path):
    # Build DFE vector
    alphai_x_hi = np.random.normal(0, 10, L)

    # Initialize a dictionary to store DFE at specified times
    saved_DFEs = {}

    for flip in range(1, L + 1):
        # Build choice_probs
        choice_probs = np.where(alphai_x_hi >= 0, alphai_x_hi, 0)

        # Normalize choice_probs
        sum_probs = choice_probs.sum()
        if sum_probs > 0:
            choice_probs /= sum_probs

        # Adjust to ensure sum of choice_probs is exactly 1
        sum_probs = choice_probs.sum()
        if not np.isclose(sum_probs, 1.0, atol=1e-15):
            adjustment = 1.0 - sum_probs
            # Find the index with the largest positive value
            max_prob_index = np.argmax(choice_probs)
            choice_probs[max_prob_index] += adjustment

        # Choose index based on probabilities
        chosen_index = np.random.choice(np.arange(L), p=choice_probs)

        # Flip the chosen index in DFE
        alphai_x_hi[chosen_index] *= -1

        # Save DFE at specified times
        if flip in times:
            saved_DFEs[flip] = 2 * alphai_x_hi.copy()

    # Print histogram for each saved DFE
    # Save figures to dir_path
    # Calculate mean of histogram, plot it as vertical axis and write value as text
    # Do the same for max of the histogram
    for t in times:
        if t in saved_DFEs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Main DFE Histogram
            counts, bins, _ = ax1.hist(saved_DFEs[t], bins=50, density=True, alpha=0.6)
            ax1.set_title(f'DFE Histogram at mut={t}')
            ax1.set_xlabel('Fitness effect')
            ax1.set_ylabel('Frequency')

            mean = float(np.mean(saved_DFEs[t]))
            ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1)
            ax1.text(mean, 0.1, f'Mean: {mean:.2f}', rotation=90)

            max_index = np.argmax(counts)
            max_value = float(bins[max_index])
            ax1.axvline(max_value, color='r', linestyle='dashed', linewidth=1)
            ax1.text(max_value, 0.1, f'Max: {max_value:.2f}', rotation=90)

            # Add vertical line at x=0
            ax1.axvline(0, color='g', linestyle='dashed', linewidth=1)

            # Beneficial Tail Histogram and Exponential Fit
            beneficial_data = [x for x in saved_DFEs[t] if x >= 0]
            if len(beneficial_data) > 0:
                counts_ben, bins_ben, _ = ax2.hist(beneficial_data, bins=30, density=True, alpha=0.6, color='g')
                ax2.set_title(f'Beneficial Tail at mut={t}')
                ax2.set_xlabel('Fitness effect')
                ax2.set_ylabel('Frequency')

                # Fit to exponential
                bin_centers = (bins_ben[:-1] + bins_ben[1:]) / 2
                params, _ = curve_fit(exponent, bin_centers, counts_ben, p0=[1])
                fitted_data = exponent(bin_centers, *params)
                ax2.plot(bin_centers, fitted_data, 'r--', label='Exponential Fit')

                # Calculate chi-squared value, rounded to 2 significant digits after decimal point
                chi_squared = np.sum((counts_ben - fitted_data) ** 2 / fitted_data)
                
                ax2.text(0.95, 0.95, f'Chi-squared: {chi_squared:.2f}', transform=ax2.transAxes,
                         verticalalignment='top', horizontalalignment='right')
                ax2.text(0.95, 0.90, f'λ: {params[0]:.2f}', transform=ax2.transAxes,
                         verticalalignment='top', horizontalalignment='right')

            plt.tight_layout()
            plt.savefig(f'{dir_path}/dfe_hist_t_{t}.png')
            plt.close()


# Example usage
L = 10000  # Length of the DFE vector
times = [1, 1000, 2000, 3000, 3500, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900]  # Times to save the DFE vector
dir_name = "sim_dfe_no_epi_plots"

base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

run_simulation(L, times, dir_path)
