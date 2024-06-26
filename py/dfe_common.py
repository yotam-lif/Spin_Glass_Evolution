import os
import struct
import math
import numpy as np
from scipy.sparse import csr_array


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


def pull_env(n_replicate: int, dir_name: str):
    """
    Pulls the environment data for a specific replicate.

    Args:
        n_replicate (int): The replicate number.
        dir_name (str): The name of the directory containing the data.

    Returns:
        tuple: alpha0s, his, and Jijs as numpy arrays.
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    rep = "replicate" + str(n_replicate)
    J_path = os.path.join(base_dir, dir_name, rep, 'Jijs.dat.bin')
    h_path = os.path.join(base_dir, dir_name, rep, 'his.dat.bin')
    alpha0_path = os.path.join(base_dir, dir_name, rep, 'alpha0s.dat')

    # Open and read
    Jijs = np.array(read_bin_to_type(J_path, 'd', float))
    his = np.array(read_bin_to_type(h_path, 'd', float))
    alpha0s = read_txt_file(alpha0_path)
    alpha0s = np.array([float(x[0]) for x in alpha0s])
    return alpha0s, his, Jijs


def pull_bac_data(n_replicate: int, dir_name: str):
    """
    Pulls the bacteria data for a specific replicate.

    Args:
        n_replicate (int): The replicate number.
        dir_name (str): The name of the directory containing the data.

    Returns:
        list: List of bacteria data.
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    ld_path = os.path.join(base_dir, dir_name)
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


def separate_mut_data(mut_data: list, L: int):
    """
    Separates the mutation data into individual strain data.

    Args:
        mut_data (list): The mutation data.
        L (int): The length of the genome.

    Returns:
        list: List of separated mutation data.
    """
    # The data is structured in the following way:
    #  for (0 < i < n_strains):
    #       data.push(10*L + i + 1) - as a distinct seperator
    #       * Now for each mutation in history of strain i: *
    #       for (0 < j < mut_order[i].size()):
    #           * Push site, fitness effect *
    #           data.push(mut_order[i][j])
    #           data.push(fitness effect of mut_order[i][j])
    #       * Push, if time for rank calc, rank, average fitness increment *
    #       * Otherwise push -L, -L as placeholders *
    #       data.push(rank of strain i at time of frame \ -L as placeholder if not time to compute rank)
    #       data.push(average fitness increment of strain i at time of frame \ -L ... )
    #
    # Now split first by separators
    sep_is = []
    res = []
    for i in range(len(mut_data)):
        if mut_data[i] >= 10 * L:
            sep_is.append(i)
    for j in range(len(sep_is)):
        start = sep_is[j] + 1
        if j == len(sep_is) - 1:
            stop = len(mut_data)
        else:
            stop = sep_is[j + 1]
        res.append(mut_data[start:stop])
    return res


def pull_mut_hist(n_replicate: int, dir_name: str):
    """
    Pulls the mutation history data for a specific replicate.

    Args:
        n_replicate (int): The replicate number.
        dir_name (str): The name of the directory containing the data.

    Returns:
        tuple: The mutation order, mutation times, and dominant strain data.
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    ld_path = os.path.join(base_dir, dir_name)
    # Open & read Sim_data
    rep = "replicate" + str(n_replicate)
    sim_dat = read_sim_data(dir_name)
    # Output_interval is 6th item
    n_days = int(sim_dat['ndays'])
    L = int(sim_dat['L'])
    # Now get the actual mutant data
    mut_end = "mut_data." + str(n_days) + ".bin"
    mut_data_path = os.path.join(ld_path, rep, mut_end)
    mut_data = read_bin_to_type(mut_data_path, 'f', int)
    mut_order = separate_mut_data(mut_data, L)
    # Now for each array in mut_order we want the even numbered items minus last one
    # First take out last two items
    mut_order = [arr[:-2] for arr in mut_order]
    # Now add only even placed items
    mut_order = [arr[::2] for arr in mut_order]
    mut_times_path = os.path.join(ld_path, rep, "mut_times.dat")
    mut_times = read_txt_to_type(mut_times_path, int)
    dom_path = os.path.join(ld_path, rep, "dom_strain.dat")
    # dom_strain = read_txt_to_type(dom_path, int)
    dom_strain = []
    return mut_order, mut_times, dom_strain


def load_Jijs(Jij_arr: np.ndarray, L: int, dir_name: str):
    """
    Takes the vector of Jijs as loaded from binary and
    converts it into a symmetric matrix.

    Args:
        Jij_arr (np.ndarray): The array of Jij values.
        L (int): The size of the matrix.
        dir_name (str): The name of the directory containing the data.

    Returns:
        np.ndarray: The symmetric matrix of Jijs.
    """
    # Initialize the matrix
    Jijs = np.zeros((L, L))

    # Index for traversing the Jij_arr
    n_elements = 0

    # If beta == 0 then Jijs is just a matrix of zeros
    sim_data = read_sim_data(dir_name)
    beta = float(sim_data['beta'])
    if beta == 0:
        return Jijs
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


def compute_fitness(alpha: np.ndarray, alpha0: np.ndarray, hi: np.ndarray, Jij: np.ndarray):
    """
    Computes the fitness of a strain.

    Args:
        alpha (np.ndarray): The alpha vector.
        alpha0 (np.ndarray): The alpha0 vector.
        hi (np.ndarray): The hi vector.
        Jij (np.ndarray): The Jij matrix.

    Returns:
        float: The computed fitness value.
    """
    F_0 = np.dot(alpha0, hi) + np.dot(alpha0, Jij @ alpha0)
    F_off = F_0 - 1
    epi = Jij @ alpha
    epi = np.dot(alpha, epi)
    return np.dot(alpha, hi) + epi - F_off


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


def compute_dfe(alpha: np.ndarray, hi: np.ndarray, Jij: np.ndarray, ben: bool = False):
    """
    Computes the distribution of fitness effects for a given genome alpha.

    Args:
        alpha (np.ndarray): The alpha(t) vector (not to be confused with alpha0).
        hi (np.ndarray): The hi vector.
        Jij (np.ndarray): The Jij matrix.
        ben (bool): Whether to consider only beneficial mutations.

    Returns:
        list: The DFE for the given strain at the given day.
    """
    Jalpha = Jij @ alpha
    dfe = []
    for k in range(alpha.size):
        delta_fit_k = compute_fitness_delta_mutant(alpha, hi, Jalpha, k)
        dfe.append(delta_fit_k)
    if ben:
        dfe = [x for x in dfe if x >= 0]
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


def remove_n_largest(data: list, n: int):
    """
    Remove the n largest absolute value elements from a list.
    :param data: list of floats
    :param n: int
    :return res: list
    """
    res = data.copy()
    for _ in range(n):
        max_val = max(res, key=abs)
        res.remove(max_val)
    return res


def read_sim_data(dir_name: str):
    """
    Reads a simulation data file and returns a dictionary with the parameters.

    :return: dict, dictionary with the parameters
    """
    # Determine the path to the current script
    current_script_path = os.path.abspath(__file__)
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Construct the relative paths
    ld_path = os.path.join(base_dir, dir_name)
    # Open & read Sim_data
    file_path = os.path.join(ld_path, 'sim_data.txt')
    sim_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                key, val = line.split(':', 1)
                sim_data[key.strip()] = val.strip()

    return sim_data


def compute_rank(alpha: np.ndarray, hi: np.ndarray, Jij: np.ndarray):
    """
    Computes the rank of a strain at time t.

    Args:
        alpha (np.ndarray): The alpha0 vector.
        hi (np.ndarray): The hi vector.
        Jij (np.ndarray): The Jij matrix.

    Returns:
        int: The computed rank of the strain at time t.
    """
    # Some variables
    L = len(alpha)
    J_alpha = Jij @ alpha
    rank = 0
    # Compute the rank of the strain
    for i in range(L):
        delta_i = compute_fitness_delta_mutant(alpha, hi, J_alpha, i)
        if delta_i > 0:
            rank += 1
    return rank
