import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import common as cmn

# Parameters
L = 4000  # Length of the genome
rho = 0.05  # Sparsity parameter
delta = 0.005
beta = 0.75
rank_plot = 500  # Rank at which to collect the DFEs
n_sims = 8  # Number of simulations
bins = 75
sig_h = np.sqrt(1 - beta) * delta
sig_J = np.sqrt(beta) * delta / np.sqrt(L * rho) / 2
dir_name = "dfe_ridgeplot"

# Create directory for plots
base_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(base_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)

# Initial conditions (shared across simulations)
alpha0 = np.random.choice([-1, 1], L)
his = np.random.normal(0, sig_h, L)
Jijs = cmn.build_Jijs(L, rho, sig_J)

# List to store DFEs for each simulation at the desired rank
dfes_at_rank = []

# Run multiple simulations with rank_final set to rank_plot
for sim in range(n_sims):
    alpha0, his, Jijs, mut_hist = cmn.run_simulation_sswm(L, rho, rank_plot, sig_h, sig_J)
    alpha_ts = cmn.build_alpha_t(alpha0, mut_hist)
    dfes, lfs = cmn.build_dfes_lfs(alpha_ts, his, Jijs)

    # Collect the DFE at the end of the simulation (rank_plot)
    dfes_at_rank.append(dfes[-1])

# Convert DFEs into a DataFrame for Seaborn plotting
dfes_data = []
for i, dfe in enumerate(dfes_at_rank):
    dfes_data.extend([(i+1, value) for value in dfe])

dfes_df = pd.DataFrame(dfes_data, columns=["Simulation", "DFE"])

# Set up the plot style using Seaborn's ridge plot style
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(n_sims, rot=-.25, light=.7)
g = sns.FacetGrid(dfes_df, row="Simulation", hue="Simulation", aspect=5, height=1.5, palette=pal)

# Draw the densities in a ridge plot style
g.map(sns.kdeplot, "DFE", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "DFE", clip_on=False, color="w", lw=2, bw_adjust=0.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, 0.2, f'Clone {label}', fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "DFE")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.5)  # Overlap the plots

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

# Set the x-axis limits to shorten the range
# g.set(xlim=(-0.02, 0.005))  # Adjust these values based on the range of your DFEs

# Save the plot
plt.savefig(f'{dir_path}/ridge_plot_dfe.png', dpi=300)
plt.close()
