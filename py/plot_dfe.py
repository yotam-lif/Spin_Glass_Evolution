import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('res_pos', type=float, default=10**-3, help='Positive resolution')
    parser.add_argument('res_neg', type=float, default=5*10**-3, help='Negative resolution')
    parser.add_argument('differentiate_bins', type=int, default=0, help='To differentiate bins or not')
    parser.add_argument('n_bins', type=int, default=100, help='Number of bins for the DFE histogram')
    parser.add_argument('dfe_days', nargs='*', type=int, default=[0], help='Days for DFE')
    args = parser.parse_args()

    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Create the main directory if it doesn't exist
    main_dir = os.path.join(base_dir, 'dfe_plots')
    os.makedirs(main_dir, exist_ok=True)
    times = args.dfe_days
    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size)
        mut_order, mut_times, dom_strain_mut_order = dfe.pull_mut_hist(i)
        # dfes will hold the dfe data
        # len(dfes) = n_samples + 1, where the first element is the dominant strain
        dfes = []
        # Now calculate the DFE for the dominant strain at given days
        dfes_dom = []
        for t in times:
            mut_order_t = dom_strain_mut_order[t]
            # Because this is the dominant strain at time t, no need to build its mutation sequence up to t.
            alpha = dfe.build_alpha(alpha0s, mut_order_t)
            dfe_t = dfe.compute_dfe(alpha, his, Jijs)
            dfes_dom.append(dfe_t)
        dfes.append(dfes_dom)
        # Then add strains we chose to track lineages for
        bac_data = dfe.pull_bac_data(i)
        # enum_pops is a list of tuples (index, population) sorted by population in descending order
        enum_pops = list(enumerate(bac_data))
        enum_pops.sort(key=lambda x: x[1], reverse=True)
        # n := the number of lineages we track. Cannot be larger than  number of survivors at end of simulation.
        n = min(args.n_samples, len(bac_data))
        for j in range(n):
            dfes_strain = []
            strain, n_bac = enum_pops[j]
            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha = dfe.build_alpha(alpha0s, mut_order_strain_t)
                dfe_t = dfe.compute_dfe(alpha, his, Jijs)
                dfes_strain.append(dfe_t)
            dfes.append(dfes_strain)
        # The first element in dfes is the "dominant strain" dfe (not truly a strain).
        # The rest are the strains we chose to track lineages for.
        rep = "replicate" + str(i)
        for j in range(len(dfes)):
            # Create a directory for each strain if it doesn't exist
            if j == 0:
                strain = "dominant"
            else:
                strain = "lineage_" + str(j)
            strain_dir = os.path.join(main_dir, rep, f"strain_{strain}")
            os.makedirs(strain_dir, exist_ok=True)
            dfe_strain_j = dfes[j]
            for k in range(len(dfe_strain_j)):
                # Create a histogram for each dfe_t
                t = times[k]
                dfe_t = dfe_strain_j[k]
                if args.differentiate_bins:
                    bin_data, x_vals = dfe.create_bin_data(dfe_t, args.res_neg, args.res_pos)
                    plt.stairs(values=bin_data, edges=x_vals)
                else:
                    plt.hist(dfe_t, bins=args.n_bins, density=True)
                plt.title(f"strain number: {strain}, day: {t}")
                plt.xlabel('Fitness effect')
                plt.ylabel('Frequency')
                # Save the plot in the strain's directory
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()  # Close the plot to free up memory


