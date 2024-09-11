import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import common as cmn
from py.sim_coarse_grained.common import calculate_rank

# Parameters
L = 5000  # Length of the genome
rho = 0.05  # Sparsity parameter
delta = 0.005
beta = 1.0
rank_final = 1  # Rank to reach before stopping the simulation
dir_name = "dfe_lfs_gap_plots"
bins = 50
sig_h = np.sqrt(1 - beta) * delta
sig_J = np.sqrt(beta) * delta / np.sqrt(L * rho) / 2
n_times = 10

# Create directory for plots
base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

# Run a single simulation
alpha0, his, Jijs, mut_hist = cmn.run_simulation_sswm(L, rho, rank_final, sig_h, sig_J)
alpha_ts = cmn.build_alpha_t(alpha0, mut_hist)
dfes, lfs = cmn.build_dfes_lfs(alpha_ts, his, Jijs)
fit_hist = cmn.build_fit_hist(alpha_ts, his, Jijs)
ranks = [cmn.calculate_rank(x) for x in dfes]

# Generate time points: indices including first (0) and last (len(alpha_ts)-1)
times = np.linspace(0, len(alpha_ts) - 1, n_times, dtype=int)

# Plot histograms for different time points
for t in times:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot DFE and Local Fields histograms with KDE
    sns.histplot(dfes[t], bins=bins, kde=True, ax=ax1, alpha=0.4, color='blue', label='DFE')
    sns.histplot(-2 * lfs[t], bins=bins, kde=True, ax=ax1, alpha=0.4, color='grey', label='Local Fields')

    # Plot vertical lines for means
    mean_DFE = np.mean(dfes[t])
    ax1.axvline(mean_DFE, color='blue', linestyle='dashed', linewidth=1, label=f'DFE Mean: {mean_DFE:.3f}')
    ax1.axvline(0, color='g', linestyle='dashed', linewidth=1, label='0')

    ax1.set_title(f'#mutations={t}; Rank={ranks[t]}')
    ax1.set_xlabel('Fitness effect / Local Fields')
    ax1.set_ylabel("Counts")
    ax1.legend()

    # Plot DFE and Absolute Local Fields histograms
    abs_local_fields = -2 * np.abs(lfs[t])
    sns.histplot(dfes[t], bins=bins, stat="density", ax=ax2, alpha=0.4, color='blue', label='DFE')
    sns.histplot(abs_local_fields, bins=bins, stat="density", ax=ax2, alpha=0.4, color='grey', label='-|LFs|')

    # Plot vertical lines for means
    mean_abs_lf = np.mean(abs_local_fields)
    ax2.axvline(mean_DFE, color='blue', linestyle='dashed', linewidth=1, label=f'DFE Mean: {mean_DFE:.3f}')
    ax2.axvline(mean_abs_lf, color='grey', linestyle='dashed', linewidth=1, label=f'-|LF| Mean: {mean_abs_lf:.3f}')
    ax2.axvline(0, color='g', linestyle='dashed', linewidth=1, label='0')

    ax2.set_title(f'#mutations={t}; Rank={ranks[t]}')
    ax2.set_xlabel('Fitness effect / Local Fields')
    ax2.set_ylabel("Density")
    ax2.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(f'{dir_path}/dfe_hist_time_{t}.png', dpi=300)
    plt.close()
