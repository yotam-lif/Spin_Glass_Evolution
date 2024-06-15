import dfe_common as dfe
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    # Define command line options
    parser = argparse.ArgumentParser(description='Compute and plot the distribution of fitness effects (DFE).')
    parser.add_argument('n_exps', type=int, default=1, help='Number of experiments')
    args = parser.parse_args()
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(base_dir, 'fit_plots')
    os.makedirs(main_dir, exist_ok=True)
    ld_path = os.path.join(base_dir, 'lenski_data')
    # For each experiment, pull data
    for i in range(args.n_exps):
        rep = f"replicate{i}"
        rep_path = os.path.join(ld_path, rep)
        # Pull data
        alpha0s, his, Jijs = dfe.pull_env(i)
        Jijs = dfe.load_Jijs(Jijs, alpha0s.size)
        mut_order, mut_times, dom_strain_mut_order = dfe.pull_mut_hist(i)
        sim_params = dfe.read_sim_data()
        L = int(sim_params['L'])
        n_days = int(sim_params['ndays'])
        output_interval = int(sim_params['output_interval'])
        # Define the output times
        output_times = np.arange(0, n_days + 1, output_interval)
        # For each output day, read data
        F_t = []
        # Stupid quick fix I need to do because the final output
        dom_strain_mut_order.append(dom_strain_mut_order[-1])
        for t in output_times:
            alpha_dom_t = dfe.build_alpha(alpha0s, dom_strain_mut_order[t])
            F_t.append(dfe.compute_fitness(alpha_dom_t, alpha0s, his, Jijs))
        # Maximum fitness is sum of absolute values of hi
        F_0_true = np.dot(alpha0s, his) + np.dot(alpha0s, Jijs @ alpha0s)
        F_off = F_0_true - 1
        F_inf = np.sum(np.abs(his)) - F_off

        # Define the function to fit
        def no_epi(t, a):
            return F_inf - (F_inf - 1) / (1 + a*t)**2

        # Give range such that B, a > 0
        popt, pcov = curve_fit(no_epi, output_times, F_t, bounds=(0, np.inf))
        # Plot the data
        plt.plot(output_times, F_t, label='Data')
        plt.plot(output_times, no_epi(output_times, *popt), label='Fit')
        plt.xlabel('Time (days)')
        plt.ylabel('Fitness')
        # Write params on graph
        plt.text(0, 1.2, f'a = {popt[0]:.2f}', fontsize=10)
        # Calculate chi squared
        chi_sq = np.sum((np.array(F_t) - no_epi(output_times, *popt)) ** 2)
        plt.text(0, 1.0, f'Chi squared = {chi_sq:.2f}', fontsize=10)
        plt.title(f'Fitness trajectory of dominant strain')
        plt.legend()
        plt.savefig(os.path.join(main_dir, f'fit_{rep}.png'))
        plt.close()
