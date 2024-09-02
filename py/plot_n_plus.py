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

def n_plus_lfs(alpha_t, Jij, rho, bdfe_t0_indices):
    """
    Calculate the local fields for a given time step.
    :param alpha_t: np.Array
    :param Jij: np.Array
    :param rho: float
    :param bdfe_t0_indices: np.Array
    :return: np.Array
    """
    L = len(alpha_t)
    L_ben = len(bdfe_t0_indices)
    if L_ben == 0:
        return 0
    reduced_alpha_t = alpha_t[bdfe_t0_indices]
    reduced_Jij = Jij[bdfe_t0_indices, :]
    # We want a matrix f_ij := J_ij * alpha_j
    f_ij = reduced_Jij * reduced_alpha_t[:, np.newaxis]
    vec = np.sum(f_ij > 0, axis=1)
    vec = np.abs(vec / (L * rho) - 0.5)
    # Average over all *beneficial* genes, notice the L and not the true length of the vector
    return sum(vec) / L_ben

def mean_lf_value(alpha_t, Jij, bdfe_t0_inds):
    """
    Calculate the mean local field value for a given time step.
    :param alpha_t: np.Array
    :param Jij: np.Array
    :param bdfe_t0_inds: np.Array
    :return: float
    """
    reduced_alpha_t = alpha_t[bdfe_t0_inds]
    reduced_Jij = Jij[bdfe_t0_inds, :]
    f_i = np.sum(reduced_Jij * reduced_alpha_t[:, np.newaxis], axis=0)
    return np.mean(np.abs(f_i))


def n_plus_value(alpha_t, Jij, rho, bdfe_t0_inds):
    """
    Calculate the n_plus value.
    Calculated by summing per row the number of positive elements in the row.
    Then, average over all rows.
    :param alpha_t: np.Array
    :param Jij: np.Array
    :param rho: float
    :param bdfe_t0_inds: np.Array
    :return:
    """
    if len(bdfe_t0_inds) == 0:
        return 0, 0
    # Take only the alpha values of beneficial genes
    reduced_alpha_t = alpha_t[bdfe_t0_inds]
    # Build aJa, such that rows are not zero only for beneficial mutations
    reduced_Jij = Jij[bdfe_t0_inds, :]
    # Build Delta_ij. Delta_i = Sum over j of Delta_ij is the fitness effect of mutation i
    # {Delta}_ij = -2 * alpha_i * alpha_j * J_ij
    # Rows are not zero only for beneficial mutations, but columns are not zero for all mutations
    Dij = np.outer(reduced_alpha_t, alpha_t) * reduced_Jij
    L_ben = len(bdfe_t0_inds)
    L = len(alpha_t)
    # Count the number of positive elements in each row, should be about half of the non-zero elements.
    # We care about the deviation from this half value.
    vec = np.sum(Dij > 0, axis=1)
    vec = vec / (L * rho) - 0.5
    # Average over all *beneficial* genes, notice the L and not the true length of the vector
    return sum(vec) / L_ben, np.max(np.abs(vec))


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
    main_dir = os.path.join(base_dir, dir_name, 'n_plus_plots')
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
            n_plus_flipped_values = []
            n_plus_non_flipped_values = []
            n_plus_lfs_values = []
            mean_lf_values = []

            # Gather indices for tracking
            mut_order_strain_t0 = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], args.init_day)
            alpha_t0 = dfe.build_alpha(alpha0s, mut_order_strain_t0)
            # Build the beneficial DFE at t0
            dfe_t0 = dfe.compute_dfe(alpha_t0, his, Jijs)
            bdfe_t0_inds = bdfe_indices(dfe_t0)

            for t in times:
                mut_order_strain_t = dfe.build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha_t = dfe.build_alpha(alpha0s, mut_order_strain_t)
                # Split the n_plus calculation into 2 parts: the alpha_i that flipped from t_0 and the ones that didn't
                flipped_indices = np.where(alpha_t != alpha_t0)[0]
                non_flipped_indices = np.where(alpha_t == alpha_t0)[0]

                # Flipped beneficial indices
                bdfe_flipped_indices = np.intersect1d(flipped_indices, bdfe_t0_inds)
                # Non-flipped beneficial indices
                bdfe_non_flipped_indices = np.intersect1d(non_flipped_indices, bdfe_t0_inds)

                n_plus_flipped = n_plus_value(alpha_t, Jijs, args.rho, bdfe_flipped_indices)
                n_plus_non_flipped = n_plus_value(alpha_t, Jijs, args.rho, bdfe_non_flipped_indices)

                n_plus_flipped_values.append(n_plus_flipped)
                n_plus_non_flipped_values.append(n_plus_non_flipped)

                # Calculate n_plus_lfs for this time step
                n_plus_lfs_value = n_plus_lfs(alpha_t, Jijs, args.rho, bdfe_t0_inds)
                n_plus_lfs_values.append(n_plus_lfs_value)

                # Calculate mean local field value for this time step
                mean_lf_value_t = mean_lf_value(alpha_t, Jijs, bdfe_t0_inds)
                mean_lf_values.append(mean_lf_value_t)

            # Figure 1: n_plus flipped values and max values in separate subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

            # First subplot: Mean n_plus for flipped alpha values
            ax1.plot(times[1:], [x[0] for x in n_plus_flipped_values][1:], 'bo-', label='Mean')
            ax1.set_xlabel('t')
            ax1.set_ylabel(r'$\langle N_+ \rangle / \rho L - \frac{1}{2}$')
            ax1.set_title(f'Strain {strain} ; Replicate {i} (Flipped - Mean), $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax1.grid(True)

            # Second subplot: Max n_plus for flipped alpha values
            ax2.plot(times[1:], [x[1] for x in n_plus_flipped_values][1:], 'go-', label='Max')
            ax2.set_xlabel('t')
            ax2.set_ylabel(r'$\max(|N_+ / \rho L - \frac{1}{2}|)$')
            ax2.set_title(f'Strain {strain} ; Replicate {i} (Flipped - Max), $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax2.grid(True)

            # Save the figure with both subplots
            graph_path = os.path.join(strain_dir, f'n_plus_flipped_{args.init_day}_{args.final_day}.png')
            plt.savefig(graph_path, dpi=300)
            plt.close()

            # Figure 2: n_plus non-flipped values and max values in separate subplots
            fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))

            # First subplot: Mean n_plus for non-flipped alpha values
            ax3.plot(times, [x[0] for x in n_plus_non_flipped_values], 'ro-', label='Mean')
            ax3.set_xlabel('t')
            ax3.set_ylabel(r'$\langle N_+ \rangle / \rho L - \frac{1}{2}$')
            ax3.set_title(f'Strain {strain} ; Replicate {i} (Non-Flipped - Mean), $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax3.grid(True)

            # Second subplot: Max n_plus for non-flipped alpha values
            ax4.plot(times, [x[1] for x in n_plus_non_flipped_values], 'go-', label='Max')
            ax4.set_xlabel('t')
            ax4.set_ylabel(r'$\max(|N_+ / \rho L - \frac{1}{2}|)$')
            ax4.set_title(f'Strain {strain} ; Replicate {i} (Non-Flipped - Max), $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax4.grid(True)

            # Save the figure with both subplots
            graph_path = os.path.join(strain_dir, f'n_plus_non_flipped_{args.init_day}_{args.final_day}.png')
            plt.savefig(graph_path, dpi=300)
            plt.close()

            # Figure 3: n_plus_lfs and mean_lf_value side by side in the same figure
            fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

            # First subplot: n_plus_lfs values over time
            ax5.plot(times, n_plus_lfs_values, 'go-')
            ax5.set_xlabel('t')
            ax5.set_ylabel(r'$|\langle N_+^{\text{lfs}} \rangle / \rho L - \frac{1}{2}|$')
            ax5.set_title(f'Strain {strain} ; Replicate {i}, Local Fields, $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax5.grid(True)

            # Second subplot: mean_lf_value over time
            ax6.plot(times, mean_lf_values, 'mo-')
            ax6.set_xlabel('t')
            ax6.set_ylabel(r'$\langle |f_i| \rangle$')
            ax6.set_title(f'Strain {strain} ; Replicate {i}, Mean LF, $t_0 = {args.init_day}$, $t_f = {args.final_day}$')
            ax6.grid(True)

            # Save the combined figure with both subplots
            graph_path_lfs = os.path.join(strain_dir, f'n_plus_lfs_mean_lf_{args.init_day}_{args.final_day}.png')
            plt.savefig(graph_path_lfs, dpi=300)
            plt.close()

    print("Plots have been saved.")
