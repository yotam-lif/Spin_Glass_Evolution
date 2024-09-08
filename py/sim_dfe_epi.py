import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os


def exponent(x, _lambda):
    return _lambda * np.exp(-x * _lambda)


def build_Jijs(L, rho, sig_J):
    Jijs = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            Jij = np.random.normal(0, sig_J)
            if np.random.rand() < rho:
                Jijs[i, j] = Jij
                Jijs[j, i] = Jij
    return Jijs


def compute_fit_slow(alpha, his, Jijs, F_off=0.0):
    return alpha @ his + alpha @ Jijs @ alpha - F_off


def compute_fitness_delta_mutant(alpha, hi, f_i, k):
    return -1 * alpha[k] * (hi[k] + 2 * f_i[k])


def compute_local_fields(alpha, Jijs):
    return alpha @ Jijs


def compute_DFE(alpha, hi, Jijs):
    f_i = compute_local_fields(alpha, Jijs)
    DFE = np.zeros(len(alpha))
    for k in range(len(alpha)):
        DFE[k] = compute_fitness_delta_mutant(alpha, hi, f_i, k)
    return DFE, 2 * f_i


def clonal_lambda(s: float, alpha: float) -> float:
    return (1 / s) * np.exp(-alpha * s) * (s + (1 / alpha))


def choice_function(DFE: np.array, regime: int, alpha: float) -> np.array:
    if regime == 0:
        return DFE
    elif regime == 1:
        return np.array([s * clonal_lambda(s, alpha) if s > 0 else s for s in DFE])
    else:
        raise ValueError(f"Invalid regime: {regime}")


def run_simulation(L, dir_path, rho, rank_final, sig_h, sig_J, regime, alpha_ci):
    alpha = np.random.choice([-1, 1], L)
    his = np.random.normal(0, sig_h, L)
    Jijs = build_Jijs(L, rho, sig_J)

    saved_DFEs = {}
    saved_lfs = {}
    saved_ranks = {}
    zero_counts = {}
    DFE, f_i = compute_DFE(alpha, his, Jijs)
    rank = len([x for x in DFE if x >= 0])
    mut = 0

    while rank > rank_final:
        beneficial_probs = np.where(DFE >= 0, DFE, 0)
        choice_probs = choice_function(beneficial_probs, regime, alpha_ci)
        sum_probs = choice_probs.sum()
        if sum_probs > 0:
            choice_probs /= sum_probs

        sum_probs = choice_probs.sum()
        if not np.isclose(sum_probs, 1.0, atol=1e-15):
            adjustment = 1.0 - sum_probs
            max_prob_index = np.argmax(choice_probs)
            choice_probs[max_prob_index] += adjustment

        chosen_index = np.random.choice(np.arange(L), p=choice_probs)
        alpha[chosen_index] *= -1
        DFE, f_i = compute_DFE(alpha, his, Jijs)
        rank = len([x for x in DFE if x >= 0])

        if mut % 100 == 0:
            saved_DFEs[mut] = DFE.copy()
            saved_lfs[mut] = f_i.copy()
            saved_ranks[mut] = rank
            print(mut, rank)

        mut += 1

    mut -= 1
    saved_DFEs[mut] = DFE.copy()
    saved_lfs[mut] = f_i.copy()
    saved_ranks[mut] = rank
    muts = np.append(np.arange(start=0, stop=mut, step=100), mut)

    for mut in muts:
        if mut in saved_DFEs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Find the bin that contains 0 and count values in that bin
            counts, bin_edges = np.histogram(saved_DFEs[mut], bins=50)
            zero_bin_index = np.digitize(0, bin_edges) - 1  # Find the bin that contains 0
            zero_bin_count = counts[zero_bin_index]  # Get the count of values in the bin containing 0
            zero_counts[mut] = zero_bin_count

            # Left plot: DFE and Local Fields histograms with KDE
            sns.histplot(saved_DFEs[mut], bins=50, kde=True, ax=ax1, alpha=0.4, color='blue', label='DFE')
            sns.histplot(saved_lfs[mut], bins=50, kde=True, ax=ax1, alpha=0.4, color='grey', label='Local Fields')

            # Plot vertical lines for means of DFE and Local Fields in the left plot
            mean_DFE = np.mean(saved_DFEs[mut])

            ax1.axvline(mean_DFE, color='blue', linestyle='dashed', linewidth=1, label=f'DFE Mean: {mean_DFE:.3f}')
            ax1.axvline(0, color='g', linestyle='dashed', linewidth=1, label='0')

            ax1.set_title(f'#mutations={mut}, rank={saved_ranks[mut]}')
            ax1.set_xlabel('Fitness effect / Local Fields')
            ax1.set_ylabel("Counts")
            ax1.legend()

            # Right plot: DFE and Absolute Local Fields histograms with KDE (peach color for abs local fields)
            abs_local_fields = -1 * np.abs(saved_lfs[mut])
            sns.histplot(saved_DFEs[mut], bins=50, stat="density", ax=ax2, alpha=0.4, color='blue', label='DFE')
            sns.histplot(abs_local_fields, bins=50, stat="density", ax=ax2, alpha=0.4, color='grey', label='-|LFs|')

            # Plot vertical lines for means of DFE and Abs Local Fields in the right plot
            mean_abs_lf = np.mean(abs_local_fields)
            ax2.axvline(mean_DFE, color='blue', linestyle='dashed', linewidth=1, label=f'DFE Mean: {mean_DFE:.3f}')
            ax2.axvline(mean_abs_lf, color='grey', linestyle='dashed', linewidth=1, label=f'-|LF| Mean: {mean_abs_lf:.3f}')
            ax2.axvline(0, color='g', linestyle='dashed', linewidth=1, label='0')

            ax2.set_title(f'#mutations={mut}, rank={saved_ranks[mut]}')
            ax2.set_xlabel('Fitness effect / Local Fields')
            ax1.set_ylabel("Density")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(f'{dir_path}/dfe_hist_mut_{mut}.png', dpi=300)
            plt.close()

    # Plot the zero counts
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(list(zero_counts.keys()), list(zero_counts.values()))
    ax.set_xlabel('#Mutations')
    ax.set_ylabel('P(h=0)')
    plt.savefig(f'{dir_path}/zero_counts.png', dpi=300)
    plt.close()


# Example usage
L = 4000  # Length of the genome
rho = 0.05  # Sparsity parameter
delta = 0.01
beta = 1.0
rank_final = 1  # Rank to reach before stopping the simulation
dir_name = "sim_dfe_epi_plots"
regime = 0  # 0 is SSWM, 1 is CI
alpha = 1

sig_h = np.sqrt(1 - beta) * delta
sig_J = np.sqrt(beta) * delta / np.sqrt(L * rho) / 2

base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

run_simulation(L, dir_path, rho, rank_final, sig_h, sig_J, regime, alpha)
