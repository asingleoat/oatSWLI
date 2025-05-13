#!/usr/bin/env python3

import numpy as np

a = np.load("fullframes.npy")
b = np.load("cutframes.npy")
frame_cut_factor = 10
b = b * frame_cut_factor

diff = a - b

offset = np.median(diff)

adjusted_diff = diff - offset

var = np.var(adjusted_diff)
std = np.std(adjusted_diff)
mean_abs = np.mean(np.abs(adjusted_diff))

print(f"Offset applied: {offset}")
print(f"Mean absolute difference (after offset): {mean_abs}")
print(f"Variance: {var}")
print(f"Standard deviation: {std}")
