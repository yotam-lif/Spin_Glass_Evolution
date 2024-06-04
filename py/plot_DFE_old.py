import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os


# Return 2D array of dimensions [2*n_days, L] of type float
def read_data(path: str):
    f = open(path, 'rt')
    data_raw = f.readlines()
    # Now we have a 1D list of frames in f, seperated by "\n" (\n are deleted)
    data_str = []
    for frame in data_raw:
        data_str.append(frame.split(sep=" "))
    # Now 2D array of strings, first axis is frame, 2nd axis is fitness delta of mutation.
    # Dimensions are ndays X L
    data = []
    for frame in data_str:
        data.append([float(fd) for fd in frame])
    f.close()
    return data


# Need to separate the data of the different strains we are tracking
# Right now naive implementation is (tracked strain, dominant strain, tracked strain, dominant ...)
# Is 2D vector of dimensions [2*n_days, L], separate_data(data) return 2 lists of dimensions [n_days, L]
def separate_data(data: list):
    data_lineage = []
    data_dominant = []
    for i in range(len(data)):
        if i % 2 == 0:
            data_lineage.append(data[i])
        else:
            data_dominant.append(data[i])
    return data_lineage, data_dominant


# Need to make custom bins, where resolution/bin size is different in positive and negative sides.
# This is because we need higher resolution on positive side.
# Need to give x values as well.
def create_bin_data(data: list, res_neg: float, res_pos: float):
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


def create_test_data(n_days: int, L: int):
    res = []
    for _ in range(n_days):
        # data_lin
        pos = np.random.exponential(0.2, int(L/2))
        neg = -1 * np.random.exponential(1, int(L/2))
        res.append(np.concatenate((pos, neg)))
        # data_dom
        pos = np.random.exponential(0.2, int(L/2))
        neg = -1 * np.random.exponential(1, int(L/2))
        res.append(np.concatenate((pos, neg)))
    return res


# Correct argument passing is path_data(file path), res_neg, res_pos.
n = len(sys.argv)
print("Total arguments passed:", n)
path_data = sys.argv[1]
res_neg = float(sys.argv[2])
res_pos = float(sys.argv[3])

path_figs = "/Users/yotamlifschytz/Desktop/Spin_Glass_Evolution/Simulation/DFE_figs/"
dist_data = read_data(path_data)
dist_data_lin, dist_data_dom = separate_data(dist_data)
n_days = len(dist_data_lin)
# path_figs = os.path.abspath(os.path.join(path_figs, os.pardir))

for i in range(n_days):
    bin_data_lin, x_vals_lin = create_bin_data(dist_data_lin[i], res_neg, res_pos)
    plt.stairs(values=bin_data_lin, edges=x_vals_lin)
    plt.title("DFE of Tracked Strain; Day " + str(i))
    plt.xlabel("$ \Delta F $")
    plt.savefig(path_figs + "DFE_lin_" + str(i) + ".png", format='png')
    plt.close()

    bin_data_dom, x_vals_dom = create_bin_data(dist_data_dom[i], res_neg, res_pos)
    plt.stairs(values=bin_data_dom, edges=x_vals_dom)
    plt.title("DFE of Dominant Strain; Day " + str(i))
    plt.xlabel("$ \Delta F $")
    plt.savefig(path_figs + "DFE_dom_" + str(i) + ".png", format='png')
    plt.close()
