import os
import struct
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np


def read_bin_to_type(path, _type_bin, _type):
    """
    Reads a binary file and converts it to a list of specified type.

    Args:
        path (str): Path to the binary file.
        _type_bin (str): The format of the data in the binary file (e.g., 'd' for double).
        _type (type): The desired output type of the data (e.g., float).

    Returns:
        list: List of converted data.
    """
    try:
        with open(path, 'rb') as file:
            res = file.read()
            # Determine the number of elements in the file
            num_elems = len(res) // struct.calcsize(_type_bin)
            # Unpack the file content into a list of elements
            _format = f'{num_elems}{_type_bin}'
            data = struct.unpack(_format, res)
            # Convert the data to the specified type
            return [_type(x) for x in data]
    except FileNotFoundError:
        print(f"File not found: {path}")
        return []
    except IOError as e:
        print(f"An I/O error occurred: {e}")
        return []


# Returns 2D array, even if 2nd axis is only of single elements
def read_txt_file(path):
    """
    Reads a text file and returns a list of lists where each sublist
    contains elements from each line split by whitespace.

    Args:
        path (str): Path to the text file.

    Returns:
        list: List of lists with elements from each line.
    """
    try:
        with open(path, 'rt') as file:
            return [line.split() for line in file.read().splitlines()]
    except FileNotFoundError:
        print(f"File not found: {path}")
    except IOError as e:
        print(f"An I/O error occurred: {e}")


def read_txt_to_type(path, _type):
    """
    Reads a text file and converts its contents to the specified type.

    Args:
        path (str): Path to the text file.
        _type (type): The desired output type of the data.

    Returns:
        list: List of converted data.
    """
    return [[_type(x) for x in line] for line in read_txt_file(path)]


def pull_env(n_replicate: int):
    """
    Pulls the environment data for a specific replicate.

    Args:
        n_replicate (int): The replicate number.

    Returns:
        tuple: alpha0s, his, and Jijs as numpy arrays.
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    rep = "replicate" + str(n_replicate)
    J_path = os.path.join(base_dir, 'lenski_data', rep, 'Jijs.dat.bin')
    h_path = os.path.join(base_dir, 'lenski_data', rep, 'his.dat.bin')
    alpha0_path = os.path.join(base_dir, 'lenski_data', rep, 'alpha0s.dat')

    # Open and read
    Jijs = np.array(read_bin_to_type(J_path, 'd', float))
    his = np.array(read_bin_to_type(h_path, 'd', float))
    alpha0s = read_txt_file(alpha0_path)

    # Additional debug print
    if alpha0s is None:
        print(f"Failed to read alpha0s.dat file at {alpha0_path}")
    else:
        print(f"alpha0s: {alpha0s}")

    alpha0s = np.array([float(x[0]) for x in alpha0s])
    return alpha0s, his, Jijs


def pull_bac_data(n_replicate: int):
    """
    Pulls the bacteria data for a specific replicate.

    Args:
        n_replicate (int): The replicate number.

    Returns:
        list: List of bacteria data.
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    ld_path = os.path.join(base_dir, 'lenski_data')
    # Open & read Sim_data
    rep = "replicate" + str(n_replicate)
    sim_path = os.path.join(ld_path, 'sim_data.txt')
    f_sim = open(sim_path, 'rt')
    sim_dat = f_sim.read().splitlines()
    # Output_interval is 6th item
    n_days = int(sim_dat[3].split(": ", 2)[-1])
    # Now get the actual bac data
    bac_end = "bac_data." + str(n_days) + ".bin"
    bac_data_path = os.path.join(ld_path, rep, bac_end)
    bac_data = read_bin_to_type(bac_data_path, 'f', float)
    # The data is structured in the following way:
    #  for (0 < i < n_strains):
    #       data.push(nbac[i])
    #       data.push(fitness[i])
    # So return only even elements of bac_data
    return bac_data[::2]


def pull_mut_hist(n_replicate: int):
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    ld_path = os.path.join(base_dir, 'lenski_data')
    # Open & read Sim_data
    rep = "replicate" + str(n_replicate)
    sim_path = os.path.join(ld_path, 'sim_data.txt')
    f_sim = open(sim_path, 'rt')
    sim_dat = f_sim.read().splitlines()
    # Output_interval is 6th item
    n_days = int(sim_dat[3].split(": ", 2)[-1])
    L = int(sim_dat[0].split(": ", 2)[-1])
    # Now get the actual mutant data
    mut_end = "mut_data." + str(n_days) + ".bin"
    mut_data_path = os.path.join(ld_path, rep, mut_end)
    mut_data = read_bin_to_type(mut_data_path, 'f', int)
    # The data is structured in the following way:
    #  for (0 < i < n_strains):
    #       data.push(10*L + i + 1) - as a distinct seperator
    #       for (0 < j < mut_order[i].size()):
    #           data.push(mut_order[i][j])
    #           data.push(fitness effect of mut_order[i][j])
    #       data.push(rank of strain i at time of frame \ -L as placeholder if not time to compute rank)
    #       data.push(average fitness increment of strain i at time of frame \ -L ... )
    #
    # Now split first by separators
    sep_is = []
    for i in range(len(mut_data)):
        if mut_data[i] >= 10 * L:
            sep_is.append(i)
    mut_order = []
    for j in range(len(sep_is)):
        start = sep_is[j] + 1
        if j == len(sep_is) - 1:
            stop = len(mut_data)
        else:
            stop = sep_is[j + 1]
        arr = mut_data[start:stop]
        # Now for each array in mut_data_split we want the even numbered items minus last one
        # First take out last two items
        # TODO: Fix this somehow
        arr = arr[:-2]
        # Now add only even placed items
        mut_order.append(arr[::2])
    mut_times_path = os.path.join(ld_path, rep, "mut_times.dat")
    mut_times = read_txt_to_type(mut_times_path, int)
    dom_path = os.path.join(ld_path, rep, "dom_strain.dat")
    dom_strain = read_txt_to_type(dom_path, int)
    return mut_order, mut_times, dom_strain


def load_Jijs(Jij_arr: np.ndarray, L: int):
    """
    Takes the vector of Jijs as loaded from binary and
    converts it into a symmetric matrix.

    Args:
        Jij_arr (np.ndarray): The array of Jij values.
        L (int): The size of the matrix.

    Returns:
        np.ndarray: The symmetric matrix of Jijs.
    """
    Jijs = np.zeros((L, L))

    # Index for traversing the Jij_arr
    n_elements = 0

    # Fill the upper triangular part of the matrix
    for row in range(L):
        # num_elemnts is the number of elements in the upper triangle of
        # the matrix of row 'row' (excluding the diagonal)
        # Jijs[row, row + 1:] = ... is assignment to the upper triangle part of row 'row'
        # n_elements is the index of the first element of the row 'row' in Jij_arr & the number of elements
        # traversed so far
        num_elements = L - row - 1
        Jijs[row, row + 1:] = Jij_arr[n_elements:n_elements + num_elements]
        n_elements += num_elements

    # Add the transpose to get the full symmetric matrix
    Jijs += Jijs.T

    return Jijs


def compute_fitness(alpha: np.ndarray, hi: np.ndarray, Jij: np.ndarray):
    """
    Computes the fitness of a strain.

    Args:
        alpha (np.ndarray): The alpha vector.
        hi (np.ndarray): The hi vector.
        Jij (np.ndarray): The Jij matrix.

    Returns:
        float: The computed fitness value.
    """
    epi = Jij @ alpha
    epi = np.dot(alpha, epi)
    return np.dot(alpha, hi) + epi


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


def build_mut_series_t(mut_order_strain, mut_times_strain, t):
    """
    Builds the mutation series at time t given the mutation order and mutation times.

    Args:
        mut_order_strain (list): The mutation order for the strain.
        mut_times_strain (list): The mutation times for the strain.
        t (int): The time at which to compute the mutation series.

    Returns:
        list: The computed mutation series at time t.
    """
    # Get index j of largest mutation time that is smaller than t
    j = next((i for i, time in enumerate(mut_times_strain) if time >= t), len(mut_times_strain))
    return mut_order_strain[:j]


def build_alpha(alpha0, mut_series):
    """
    Computes the alpha vector given initial alpha vector and mutation series.

    Args:
        alpha0 (np.ndarray): The initial alpha vector.
        mut_series (list): The mutation series.

    Returns:
        np.ndarray: The computed alpha vector at time t.
    """
    alpha = alpha0.copy()  # To ensure the original alpha0 is not modified
    # Flip the sign of the first j mutations
    for m in mut_series:
        alpha[m] *= -1
    return alpha


def compute_dfe(alpha: np.ndarray, hi: np.ndarray, Jij: np.ndarray):
    """
    Computes the distribution of fitness effects for a given genome alpha.

    Args:
        alpha (np.ndarray): The alpha(t) vector (not to be confused with alpha0).
        hi (np.ndarray): The hi vector.
        Jij (np.ndarray): The Jij matrix.

    Returns:
        list: The DFE for the given strain at the given day.
    """
    Jalpha = Jij @ alpha
    dfe = []
    for k in range(alpha.size):
        delta_fit_k = compute_fitness_delta_mutant(alpha, hi, Jalpha, k)
        dfe.append(delta_fit_k)
    return dfe


def create_bin_data(data: list, res_neg: float, res_pos: float):
    """
    Creates binned data for a histogram.

    Args:
        data (list): The data to be binned.
        res_neg (float): The resolution for negative values.
        res_pos (float): The resolution for positive values.

    Returns:
        tuple: The binned data and the corresponding x values.
    """
    min_fd = min(data)
    max_fd = max(data)
    n_neg_bins = math.ceil(abs(min_fd / res_neg))
    n_pos_bins = math.ceil(abs(max_fd / res_pos))
    pos_bins = np.zeros(n_pos_bins)
    neg_bins = np.zeros(n_neg_bins)
    for fd in data:
        if fd <= 0:
            bin_ind = math.floor(fd / res_neg)
            neg_bins[bin_ind] += 1
        else:
            bin_ind = math.floor(abs(fd / res_pos))
            pos_bins[bin_ind] += 1
    # The largest bin in the negative bins corresponds to most negative number.
    # To get the true histogram need to flip it.
    # np.flip(neg_bins)
    # Normalize both positive and negative so that they are same scale,
    # Otherwise one with higher resolution is much smaller
    neg_bins /= neg_bins.max()
    pos_bins /= pos_bins.max()
    bin_data = np.concatenate((neg_bins, pos_bins))
    # Now create xvals data
    x_neg = np.arange(min_fd, 0, res_neg)
    x_pos = np.arange(0, max_fd + res_pos, res_pos)
    x_vals = np.concatenate((x_neg, x_pos))
    return bin_data, x_vals


if __name__ == '__main__':

    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    parser.add_argument('n_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('res_pos', type=float, default=10**-3, help='Positive resolution')
    parser.add_argument('res_neg', type=float, default=5*10**-3, help='Negative resolution')
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
        alpha0s, his, Jijs = pull_env(i)
        Jijs = load_Jijs(Jijs, alpha0s.size)
        mut_order, mut_times, dom_strain_mut_order = pull_mut_hist(i)
        # dfes will hold the dfe data
        # len(dfes) = n_samples + 1, where the first element is the dominant strain
        dfes = []
        # Now calculate the DFE for the dominant strain at given days
        dfes_dom = []
        for t in times:
            mut_order_t = dom_strain_mut_order[t]
            # Because this is the dominant strain at time t, no need to build its mutation sequence up to t.
            alpha = build_alpha(alpha0s, mut_order_t)
            dfe_t = compute_dfe(alpha, his, Jijs)
            dfes_dom.append(dfe_t)
        dfes.append(dfes_dom)
        # Then add strains we chose to track lineages for
        bac_data = pull_bac_data(i)
        # enum_pops is a list of tuples (index, population) sorted by population in descending order
        enum_pops = list(enumerate(bac_data))
        enum_pops.sort(key=lambda x: x[1], reverse=True)
        # n := the number of lineages we track. Cannot be larger than  number of survivors at end of simulation.
        n = min(args.n_samples, len(bac_data))
        for j in range(n):
            dfes_strain = []
            strain, n_bac = enum_pops[j]
            for t in times:
                mut_order_strain_t = build_mut_series_t(mut_order[strain], mut_times[strain], t)
                alpha = build_alpha(alpha0s, mut_order_strain_t)
                dfe_t = compute_dfe(alpha, his, Jijs)
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
                bin_data, x_vals = create_bin_data(dfe_t, args.res_neg, args.res_pos)
                plt.stairs(values=bin_data, edges=x_vals)
                plt.title(f"strain number: {strain}, day: {t}")
                plt.xlabel('Fitness effect')
                plt.ylabel('Frequency')
                # Save the plot in the strain's directory
                plt.savefig(os.path.join(strain_dir, f"dfe_day_{t}.png"))
                plt.close()  # Close the plot to free up memory


