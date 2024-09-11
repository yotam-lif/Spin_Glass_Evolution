import numpy as np

def exponent(x, _lambda):
    """
    Compute the exponential decay function.

    Parameters:
    x (float or np.ndarray): The input values.
    _lambda (float): The decay constant.

    Returns:
    np.ndarray: The exponential decay of x.
    """
    return _lambda * np.exp(-x * _lambda)


def build_Jijs(L, rho, sig_J):
    """
    Build the interaction matrix Jijs with off-diagonal random interactions.

    Parameters:
    L (int): The length of the genome (number of sites).
    rho (float): The sparsity parameter, controlling the probability of interactions.
    sig_J (float): The standard deviation of the normal distribution for interactions.

    Returns:
    np.ndarray: A symmetric LxL matrix with zero diagonal and normally distributed interactions.
    """
    Jijs = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            Jij = np.random.normal(0, sig_J)
            if np.random.rand() < rho:
                Jijs[i, j] = Jij
                Jijs[j, i] = Jij  # Ensuring symmetry
    return Jijs


def compute_fit_slow(alpha, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration alpha using full slow computation.

    Parameters:
    alpha (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration alpha.
    """
    return alpha @ his + alpha @ Jijs @ alpha - F_off


def compute_fitness_delta_mutant(alpha, hi, f_i, k):
    """
    Compute the fitness change for a mutant at site k.

    Parameters:
    alpha (np.ndarray): The genome configuration.
    hi (np.ndarray): The vector of site-specific fitness contributions.
    f_i (np.ndarray): The local fitness fields.
    k (int): The index of the mutation site.

    Returns:
    float: The change in fitness caused by a mutation at site k.
    """
    return -2 * alpha[k] * (hi[k] + f_i[k])


def compute_local_fields(alpha, Jijs):
    """
    Compute the local fitness fields for a given genome configuration.

    Parameters:
    alpha (np.ndarray): The genome configuration.
    Jijs (np.ndarray): The interaction matrix.

    Returns:
    np.ndarray: The local fitness fields.
    """
    return alpha @ Jijs


def compute_dfe(alpha, hi, Jijs):
    """
    Compute the distribution of fitness effects (DFE) and local fitness fields.

    Parameters:
    alpha (np.ndarray): The genome configuration.
    hi (np.ndarray): The vector of site-specific fitness contributions.
    Jijs (np.ndarray): The interaction matrix.

    Returns:
    tuple:
        - np.ndarray: The DFE (fitness change for all potential mutations).
        - np.ndarray: The local fitness fields.
    """
    f_i = compute_local_fields(alpha, Jijs)
    DFE = np.zeros(len(alpha))
    for k in range(len(alpha)):
        DFE[k] = compute_fitness_delta_mutant(alpha, hi, f_i, k)
    return DFE, f_i


def clonal_lambda(s, alpha):
    """
    Compute the clonal selection coefficient (lambda) for a given fitness effect.

    Parameters:
    s (float): The selection coefficient (fitness effect).
    alpha (float): The scaling parameter for clonal competition.

    Returns:
    float: The clonal lambda value.
    """
    return (1 / s) * np.exp(-alpha * s) * (s + (1 / alpha))


def calculate_rank(dfe):
    """
    Calculate the rank (number of non-negative fitness effects) of the DFE.

    Parameters:
    dfe (np.ndarray): The distribution of fitness effects (DFE).

    Returns:
    int: The number of non-negative fitness effects.
    """
    return len([x for x in dfe if x >= 0])


def compute_bdfe(dfe):
    """
    Compute the beneficial distribution of fitness effects (bDFE), i.e., only non-negative values.

    Parameters:
    dfe (np.ndarray): The DFE.

    Returns:
    np.ndarray: The beneficial DFE (bDFE).
    """
    return np.where(dfe >= 0, dfe, 0)


def choose_mut_site_sswm(dfe):
    """
    Choose the mutation site under the Strong Selection, Weak Mutation (SSWM) regime.

    Parameters:
    dfe (np.ndarray): The distribution of fitness effects.

    Returns:
    int: The index of the chosen mutation site.
    """
    choice_probs = compute_bdfe(dfe)
    choice_probs /= choice_probs.sum()  # Normalize the probabilities
    sum_probs = choice_probs.sum()
    if not np.isclose(sum_probs, 1.0, atol=1e-15):  # Correct rounding errors
        adjustment = 1.0 - sum_probs
        max_prob_index = np.argmax(choice_probs)
        choice_probs[max_prob_index] += adjustment
    return np.random.choice(np.arange(len(choice_probs)), p=choice_probs)


def run_simulation_sswm(L, rho, rank_final, sig_h, sig_J):
    """
    Run the SSWM simulation until the rank reaches the target rank.

    Parameters:
    L (int): Length of the genome.
    rho (float): Sparsity parameter for interaction matrix Jijs.
    rank_final (int): The target rank to stop the simulation.
    sig_h (float): Standard deviation for hi values (site-specific contributions).
    sig_J (float): Standard deviation for Jijs values (interaction strengths).

    Returns:
    tuple:
        - np.ndarray: The initial genome configuration (alpha0).
        - np.ndarray: The hi values.
        - np.ndarray: The Jijs matrix.
        - list: The mutation history (indices of mutation sites).
    """
    alpha = np.random.choice([-1, 1], L)
    alpha0 = alpha.copy()  # Save initial state
    his = np.random.normal(0, sig_h, L)
    Jijs = build_Jijs(L, rho, sig_J)

    mut_hist = []  # Store mutation history
    dfe, _ = compute_dfe(alpha, his, Jijs)
    rank = calculate_rank(dfe)
    num_muts = 0

    # Run the simulation until the rank reaches the desired value
    while rank > rank_final:
        chosen_index = choose_mut_site_sswm(dfe)
        alpha[chosen_index] *= -1  # Apply mutation
        mut_hist.append(chosen_index)
        num_muts += 1
        dfe, _ = compute_dfe(alpha, his, Jijs)
        rank = calculate_rank(dfe)
        # print(f"Rank: {rank}, Mutations: {num_muts}")

    return alpha0, his, Jijs, mut_hist


def build_alpha_t(alpha0, mut_hist):
    """
    Reconstruct the genome configuration at each step of the mutation history.

    Parameters:
    alpha0 (np.ndarray): The initial genome configuration.
    mut_hist (list): The list of mutation site indices.

    Returns:
    list of np.ndarray: A list of genome configurations at each step.
    """
    alpha_t = [alpha0.copy()]
    for mut in mut_hist:
        alpha = alpha_t[-1].copy()
        alpha[mut] *= -1
        alpha_t.append(alpha)
    return alpha_t


def build_dfes_lfs(alpha_ts, his, Jijs):
    """
    Compute the DFE and local fields for each genome configuration in alpha_ts.

    Parameters:
    alpha_ts (list of np.ndarray): A list of genome configurations over time.
    his (np.ndarray): The vector of site-specific fitness contributions.
    Jijs (np.ndarray): The interaction matrix.

    Returns:
    tuple:
        - list of np.ndarray: The DFEs over time.
        - list of np.ndarray: The local fitness fields over time.
    """
    dfes = []
    lfs = []
    for alpha in alpha_ts:
        dfe, lf = compute_dfe(alpha, his, Jijs)
        dfes.append(dfe)
        lfs.append(lf)
    return dfes, lfs


def build_fit_hist(alpha_ts, his, Jijs):
    """
    Compute the fitness history for each genome configuration in alpha_ts.

    Parameters:
    alpha_ts (list of np.ndarray): A list of genome configurations over time.
    his (np.ndarray): The vector of site-specific fitness contributions.
    Jijs (np.ndarray): The interaction matrix.

    Returns:
    list of float: The fitness values over time.
    """
    fit_hist = []
    fit_off = compute_fit_slow(alpha_ts[0], his, Jijs)
    for alpha in alpha_ts:
        fit = compute_fit_slow(alpha, his, Jijs, fit_off)
        fit_hist.append(fit)
    return fit_hist

def backward_propagate(dfe_t: list, dfe_0: list, beneficial=True):
    """
    Backward propagate the DFE from the last day to the first day,
    based on whether beneficial or deleterious mutations are selected.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :param beneficial: bool, whether to track beneficial or deleterious mutations
    :return: list of floats, list of floats
    """
    bdfe_t = [(i, dfe_t[i]) for i in range(len(dfe_t)) if (dfe_t[i] >= 0 if beneficial else dfe_t[i] <= 0)]

    bdfe_t_inds = [x[0] for x in bdfe_t]
    bdfe_t_fits = [x[1] for x in bdfe_t]

    propagated_bdfe_t = [dfe_0[i] for i in bdfe_t_inds]

    return bdfe_t_fits, propagated_bdfe_t


def forward_propagate(dfe_t: list, dfe_0: list, beneficial=True):
    """
    Forward propagate the DFE from the first day to the last day,
    based on whether beneficial or deleterious mutations are selected.
    :param dfe_t: list of floats
    :param dfe_0: list of floats
    :param beneficial: bool, whether to track beneficial or deleterious mutations
    :return: list of floats, list of floats
    """
    bdfe_0 = [(i, dfe_0[i]) for i in range(len(dfe_0)) if (dfe_0[i] >= 0 if beneficial else dfe_0[i] <= 0)]

    bdfe_0_inds = [x[0] for x in bdfe_0]
    bdfe_0_fits = [x[1] for x in bdfe_0]

    propagated_bdfe_0 = [dfe_t[i] for i in bdfe_0_inds]

    return bdfe_0_fits, propagated_bdfe_0

