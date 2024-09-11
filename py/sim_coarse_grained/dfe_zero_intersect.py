import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import common as cmn

# Parameters
L = 10000  # Length of the genome
rho = 0.05  # Sparsity parameter
delta = 0.005
beta = 1.0
rank_final = 1  # Rank to reach before stopping the simulation
n_sims = 5  # Number of simulations
n_times = 50
bins = 70
sig_h = np.sqrt(1 - beta) * delta
sig_J = np.sqrt(beta) * delta / np.sqrt(L * rho) / 2
dir_name = "dfe_zero_intersect_plots"

# Create directory for plots
base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

zero_counts_sims = []
fitness_sims = []

# Run multiple simulations
for sim in range(n_sims):
    alpha0, his, Jijs, mut_hist = cmn.run_simulation_sswm(L, rho, rank_final, sig_h, sig_J)
    alpha_ts = cmn.build_alpha_t(alpha0, mut_hist)
    dfes, lfs = cmn.build_dfes_lfs(alpha_ts, his, Jijs)
    fit_hist = cmn.build_fit_hist(alpha_ts, his, Jijs)

    # Track P(h=0) and fitness for each time point
    zero_counts = []
    fitness = []
    times = np.linspace(0, len(dfes) - 1, n_times, dtype=int)

    for t in times:
        counts, bin_edges = np.histogram(dfes[t], bins=bins)

        # Ensure 0 is within the bin range
        if bin_edges[0] <= 0 <= bin_edges[-1]:
            zero_bin_index = np.digitize(0, bin_edges) - 1  # Find the bin that contains 0

            # Ensure the bin index is valid (not out of bounds)
            if 0 <= zero_bin_index < len(counts):
                zero_bin_count = counts[zero_bin_index]  # Get the count of values in the bin containing 0
            else:
                zero_bin_count = 0
        else:
            zero_bin_count = 0

        zero_counts.append(zero_bin_count)
        fitness.append(fit_hist[t])

    zero_counts_sims.append(zero_counts)
    fitness_sims.append(fitness)

# Plot P(h=0) vs Mutations for each simulation on the same graph
fig, ax1 = plt.subplots(figsize=(10, 6))

times = np.linspace(0, len(alpha_ts), n_times, dtype=int)

for sim_idx, zero_counts in enumerate(zero_counts_sims):
    sns.lineplot(x=times, y=zero_counts, label=f'Simulation {sim_idx + 1}', ax=ax1)

ax1.set_xlabel('# Mutations')
ax1.set_ylabel('P(h=0)')
ax1.set_title('P(h=0) Over Time for Multiple Simulations')
ax1.legend()

# Save the plot for P(h=0) vs Mutations
plt.tight_layout()
plt.savefig(f'{dir_path}/p_h0_vs_mutations.png', dpi=300)
plt.close()

# Plot P(h=0) vs Fitness for each simulation on the same graph
fig, ax2 = plt.subplots(figsize=(10, 6))

for sim_idx in range(n_sims):
    sns.lineplot(x=fitness_sims[sim_idx], y=zero_counts_sims[sim_idx], label=f'Simulation {sim_idx + 1}', ax=ax2)

ax2.set_xlabel('Fitness')
ax2.set_ylabel('P(h=0)')
ax2.set_title('P(h=0) vs Fitness for Multiple Simulations')
ax2.legend()

# Save the plot for P(h=0) vs Fitness
plt.tight_layout()
plt.savefig(f'{dir_path}/p_h0_vs_fitness.png', dpi=300)
plt.close()
