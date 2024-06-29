import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def exponent(x, _lambda):
    return _lambda * np.exp(-x * _lambda)


def build_Jijs(L, rho, sig_J):
    """
    :param L: Length of the genome
    :param rho: Sparsity parameter
    :return:  Symmetric Jij matrix with zeros on the diagonal
    """
    Jijs = np.zeros((L, L))
    # Double loop iterate on all upper triangle elements (no diagonal)
    for i in range(L):
        for j in range(i + 1, L):
            Jij = np.random.normal(0, sig_J)
            # Choose with probability rho to insert non-zero value
            if np.random.rand() < rho:
                Jijs[i, j] = Jij
                Jijs[j, i] = Jij
    return Jijs


def compute_fit_slow(alpha, his, Jijs, F_off=0.0):
    """
    :param alpha: The genome
    :param his: The hi values
    :param Jijs: The Jij values
    :param F_off: The offset of the fitness
    :return: The fitness of the genome alpha given the hi and Jij, via full slow computation.
    """
    return alpha @ his + 0.5 * (alpha @ Jijs @ alpha) - F_off


def compute_fitness_delta_mutant(alpha, hi, Jalpha, k):
    """
    Computes the change in fitness for a mutant at index k.

    Args:
        alpha (np.ndarray): The alpha vector.
        hi (np.ndarray): The hi vector.
        Jalpha (np.ndarray): The J*alpha vector.
        k (int): The index of the mutant.

    Returns:
        float: The computed fitness delta.
    """
    return - 2 * alpha[k] * (hi[k] + 2 * Jalpha[k])


def compute_DFE(alpha, hi, Jijs):
    """
    Computes the DFE for the given genome.

    Args:
        alpha (np.ndarray): The alpha vector.
        hi (np.ndarray): The hi vector.
        Jalpha (np.ndarray): The J*alpha vector.

    Returns:
        np.ndarray: The computed DFE.
    """
    Jalpha = Jijs @ alpha
    DFE = np.zeros(len(alpha))
    for k in range(len(alpha)):
        DFE[k] = compute_fitness_delta_mutant(alpha, hi, Jalpha, k)
    return DFE


def run_simulation(L, dir_path, rho, rank_final, sig_h, sig_J):
    # Build DFE vector
    alpha = np.random.choice([-1, 1], L)
    his = np.random.normal(0, sig_h, L)
    Jijs = build_Jijs(L, rho, sig_J)
    init_fit = compute_fit_slow(alpha, his, Jijs)
    # We need F_off to be such that init_fit - F_off = 1
    F_off = init_fit - 1

    # Initialize a dictionary to store DFE at specified times
    saved_DFEs = {}
    saved_ranks = {}
    DFE = compute_DFE(alpha, his, Jijs)
    rank = len([x for x in DFE if x >= 0])
    mut = 0

    while rank > rank_final:
        # Build choice_probs
        choice_probs = np.where(DFE >= 0, DFE, 0)

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
        alpha[chosen_index] *= -1

        # Compute new DFE & Update rank
        DFE = compute_DFE(alpha, his, Jijs)
        rank = len([x for x in DFE if x >= 0])

        # Save DFE at specified times
        if mut % 100 == 0:
            saved_DFEs[mut] = DFE.copy()
            saved_ranks[mut] = rank
            print(mut, rank)

        mut += 1

    # Save the last DFE
    mut -= 1
    saved_DFEs[mut] = DFE.copy()
    saved_ranks[mut] = rank
    muts = np.append(np.arange(start=0, stop=mut, step=100), mut)

    # Print histogram for each saved DFE
    # Save figures to dir_path
    # Calculate mean of histogram, plot it as vertical axis and write value as text
    # Do the same for max of the histogram
    for mut in muts:
        if mut in saved_DFEs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Main DFE Histogram
            counts, bins, _ = ax1.hist(saved_DFEs[mut], bins=50, density=True, alpha=0.6)
            ax1.set_title(f'DFE Histogram at t={mut}, rank={saved_ranks[mut]}')
            ax1.set_xlabel('Fitness effect')
            ax1.set_ylabel('Frequency')

            mean = float(np.mean(saved_DFEs[mut]))
            ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1)
            max_index = np.argmax(counts)
            max_value = float(bins[max_index])
            ax1.axvline(max_value, color='r', linestyle='dashed', linewidth=1)

            # Add vertical line at x=0
            ax1.axvline(0, color='g', linestyle='dashed', linewidth=1)

            # Ensure text is within plot limits
            ylim = ax1.get_ylim()
            ax1.text(mean, ylim[1] * 0.8, f'Mean: {mean:.3f}', rotation=90, verticalalignment='bottom')
            ax1.text(max_value, ylim[1] * 0.6, f'Max: {max_value:.3f}', rotation=90, verticalalignment='bottom')

            # Beneficial Tail Histogram and Exponential Fit
            beneficial_data = [x for x in saved_DFEs[mut] if x >= 0]
            if len(beneficial_data) > 0:
                counts_ben, bins_ben, _ = ax2.hist(beneficial_data, bins=20, density=True, alpha=0.6, color='g')
                ax2.set_title(f'Beneficial Tail at mut={mut}, rank={saved_ranks[mut]}')
                ax2.set_xlabel('Fitness effect')
                ax2.set_ylabel('Frequency')

                # Fit to exponential
                bin_centers = (bins_ben[:-1] + bins_ben[1:]) / 2
                params, _ = curve_fit(exponent, bin_centers, counts_ben, p0=[0.01])
                fitted_data = exponent(bin_centers, *params)
                ax2.plot(bin_centers, fitted_data, 'r--', label='Exponential Fit')

                ax2.text(0.95, 0.90, f'λ: {params[0]:.2f}', transform=ax2.transAxes,
                         verticalalignment='top', horizontalalignment='right')

            plt.tight_layout()
            plt.savefig(f'{dir_path}/dfe_hist_mut_{mut}.png')
            plt.close()


# Example usage
L = 5000  # Length of the genome
rho = 0.05  # Sparsity parameter
delta = 0.005
beta = 0.25
rank_final = 1  # Rank to reach before stopping the simulation
dir_name = "sim_dfe_epi_plots"

sig_h = np.sqrt(1-beta) * delta
sig_J = np.sqrt(beta) * delta / np.sqrt(L*rho) / 2

base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

run_simulation(L, dir_path, rho, rank_final, sig_h, sig_J)
