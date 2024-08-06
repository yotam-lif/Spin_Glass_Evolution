import dfe_common as dfe
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rc
import scienceplots

# Use the 'science' style from the scienceplots package
plt.style.use(['science', 'no-latex'])

# Enable LaTeX for text rendering in the plots
rc('text', usetex=True)
rc('font', family='serif')


def bdfe_indices(dfe_t):
    """
    Return indices of genes with beneficial effects.
    :param dfe_t: np.array
    :return: np.Array
    """
    bdfe_t = list(enumerate(dfe_t))
    bdfe_t = [x for x in bdfe_t if x[1] > 0]
    return [x[0] for x in bdfe_t]


def n_plus_value(alpha_t, Jij, rho, bdf_t0_inds):
    """
    Calculate the n_plus value.
    Calculated by summing per row the number of positive elements in the row.
    Then, average over all rows.
    :param alpha_t: np.Array
    :param Jij: np.Array
    :param rho: float
    :param bdf_t0_inds: np.Array
    :return:
    """
    # Take only the alpha values of beneficial genes
    reduced_alpha_t = alpha_t[bdfe_t0_inds]
    # Build aJa, such that rows are not zero only for beneficial mutations
    reduced_Jij = Jij[bdfe_t0_inds, :]
    # Build Delta_ij. Delta_i = Sum over j of Delta_ij is the fitness effect of mutation i
    # {Delta}_ij = -2 * alpha_i * alpha_j * J_ij
    # Rows are not zero only for beneficial mutations, but columns are not zero for all mutations
    Dij = -2 * np.outer(reduced_alpha_t, alpha_t) * reduced_Jij
    L_ben = len(bdf_t0_inds)
    L = len(alpha_t)
    # Count the number of positive elements in each row, should be about half of the non-zero elements.
    # We care about the deviation from this half value.
    vec = np.sum(Dij > 0, axis=1)
    vec = vec / (L * rho) - 0.5
    # Average over all *beneficial* genes, notice the L and not the true length of the vector
    return sum(vec) / L_ben


if __name__ == '__main__':
    # Define command line options
    parser = argparse.ArgumentParser(description='Plot the local fields evolution of strains through time.')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('dir_name', type=str, help='Name of directory data is in')
    parser.add_argument('rho', type=float, default=0, help='Simulation rho value')
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
    main_dir = os.path.join(base_dir, f'dfe_tracker_plots_{dir_name}')
    os.makedirs(main_dir, exist_ok=True)
    days = args.final_day - args.init_day
    time_steps = int(days * args.density) + 1
    times = np.linspace(args.init_day, args.final_day, time_steps)

    for i in range(args.n_exps):
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i, dir_name)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size, dir_name)
        mut_order, mut_times, _ = dfe.pull_mut_hist(i, dir_name)
        # Then add strains we chose to track lineages for
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
            n_plus_values = []

            # Gather indices for tracking
            mut_order_strain_t0 = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], args.init_day)
            alpha_t0 = dfe.build_alpha(alpha0s, mut_order_strain_t0)
            # Build the beneficial DFE at t0
            dfe_t0 = dfe.compute_dfe(alpha_t0, his, Jijs)
            bdfe_t0_inds = bdfe_indices(dfe_t0)

            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha_t = dfe.build_alpha(alpha0s, mut_order_strain_t)
                # Take only the alpha values of beneficial genes
                masked_alpha_t = np.where(np.isin(np.arange(alpha_t.size), bdfe_t0_inds), alpha_t, 0)
                n_plus = n_plus_value(alpha_t, Jijs, args.rho, bdfe_t0_inds)
                n_plus_values.append(n_plus)

            # Plotting n_plus values over time
            plt.figure(figsize=(10, 6))
            plt.plot(times, n_plus_values, 'bo-')
            plt.xlabel('t')
            plt.ylabel(r'$ N_+ / \rho L - \frac{1}{2}$')
            plt.title(f'Strain {strain} ; Replicate {i}, $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            plt.grid(True)

            graph_path = os.path.join(strain_dir, f'n_plus_{args.init_day}_{args.final_day}.png')
            plt.savefig(graph_path, dpi=300)
            plt.close()

    print("Plots have been saved.")
