import argparse
import os
import dfe_common as dfe
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams["font.family"] = "sans-serif"  # Set the default font family to sans-serif

if __name__ == '__main__':
    # Define command line options
    parser = argparse.ArgumentParser(description='Plot the local fields evolution of strains through time.')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('dir_name', type=str, help='Name of directory data is in')
    parser.add_argument('init_day', type=int, default=0, help='Initial day for tracking')
    parser.add_argument('final_day', type=int, default=0, help='Final day for tracking')
    parser.add_argument('density', type=float, help='Density for calculating n_plus')
    args = parser.parse_args()

    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Create the main directory if it doesn't exist
    dir_name = args.dir_name
    main_dir = os.path.join(base_dir, dir_name, 'n_flip_plots')
    os.makedirs(main_dir, exist_ok=True)

    days = args.final_day - args.init_day
    time_steps = int(days * args.density) + 1
    times = np.linspace(args.init_day, args.final_day, time_steps)

    sns.set(style="whitegrid")  # Set the seaborn style for the plots

    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i, dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, dir_name)
        mut_order, mut_times, _ = dfe.pull_mut_hist(i, dir_name)
        # Add strains we chose to track lineages for
        bac_data = dfe.pull_bac_data(i, dir_name)
        # enum_pops is a list of tuples (index, population) sorted by population in descending order
        enum_pops = list(enumerate(bac_data))
        enum_pops.sort(key=lambda x: x[1], reverse=True)
        # n := the number of lineages we track. Cannot be larger than number of survivors at end of simulation.
        n = min(args.n_samples, len(bac_data))
        rep_dir = os.path.join(main_dir, f'replicate{i}')
        os.makedirs(rep_dir, exist_ok=True)

        for j in range(n):
            strain, n_bac = enum_pops[j]
            strain_dir = os.path.join(rep_dir, f'strain_lineage_{j}')
            os.makedirs(strain_dir, exist_ok=True)

            # Gather indices for tracking
            mut_order_strain_t0 = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], args.init_day)

            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                flips = np.zeros(len(alpha0s))
                for gene in mut_order_strain_t:
                    flips[gene] += 1
                for gene in mut_order_strain_t0:
                    flips[gene] -= 1

                # Create scatter plot with seaborn
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=np.arange(len(flips)), y=flips, color='blue', s=20)  # Smaller marker size (s=20)
                plt.xlabel('Gene Index', fontsize=16)
                plt.ylabel('Number of Flips', fontsize=16)
                plt.title(f'Strain {j}, Time {int(t)}', fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

                # Set y-axis limits and ticks to integer values >= 1
                plt.ylim(0.8, max(1, flips.max()) + 1)  # Ensure that the plot starts from 1 on the y-axis
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Only integer ticks

                # Count the occurrences of each y-value and annotate the plot
                flip_counts = Counter(flips)
                for flip_value, count in flip_counts.items():
                    if flip_value >= 1:
                        plt.text(len(alpha0s) - 1, flip_value, f'$n_f/L ={count/len(alpha0s):.3f}$',
                                 verticalalignment='top', fontsize=12, color='black')

                # Save the plot to a file with higher resolution
                plot_filename = os.path.join(strain_dir, f'flips_time_{int(t)}.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close()

    print(f'Plots saved in directory: {main_dir}')
