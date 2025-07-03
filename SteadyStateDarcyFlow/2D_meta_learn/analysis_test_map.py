import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt
import os
import argparse
import fenics as fe


mpl.rcParams['font.family'] = 'Times New Roman'


meta_results_dir = "./RESULTS/"
results_dir = "./RESULTS-PAPER-MAP/"

error_unlearn = np.load(meta_results_dir + "complex_unlearned_error_list_test.npy")[0,:]
_, indices = np.unique(error_unlearn, return_index=True)
error_unlearn = error_unlearn[np.sort(indices)][:-1]

error_learned_f = np.load(meta_results_dir + "complex_learned_error_list_f_test.npy")[0,:]
_, indices = np.unique(error_learned_f, return_index=True)
error_learned_f = error_learned_f[np.sort(indices)][:-1]

print("IterNum of Unlearned Method = ", error_unlearn.shape)
print("IterNum of Learned_f Method = ", error_learned_f.shape)

print("Finall Relative Error = ", error_unlearn[-1])
print("Finall Relative Error = ", error_learned_f[-1])
import scienceplots
plt.style.use('science')
plt.rcParams['font.size'] = 15
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(error_unlearn, linestyle='none', marker='o', markersize=2)
plt.xlabel("Iterative Number")
plt.ylabel("Relative Error")
plt.title("Unlearned Error", pad=15)
plt.subplot(1, 2, 2)
plt.plot(error_learned_f, linestyle='none', marker='o', markersize=2)
plt.xlabel("Iterative Number")
plt.ylabel("Relative Error")
plt.title("Learned " + r"$f_m(\theta_1)$" + " Error", pad=15)
plt.savefig(results_dir + "complex_error_analysis.png", dpi=1000, bbox_inches='tight')
plt.savefig(results_dir + "complex_error_analysis.pdf", dpi=1000, bbox_inches='tight')
plt.close()