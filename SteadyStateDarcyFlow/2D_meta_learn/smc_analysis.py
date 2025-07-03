import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt
import os
import argparse
import fenics as fe
import torch

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import relative_error_simple
from SteadyStateDarcyFlow.MLcommon import GaussianFiniteRankTorch


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.4

meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
smc_results_dir = "./RESULTS/SMC/"
results_fig_table = "./RESULTS-PAPER-SMC/"
os.makedirs(results_fig_table, exist_ok=True)
results_fig_table_data = "./RESULTS-PAPER-SMC/DATA/"
os.makedirs(results_fig_table_data, exist_ok=True)

parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
parser.add_argument('--idx', type=int, default=0, help='index of the particles to analyze')
parser.add_argument('--basis_available', type=bool, default=False, help='whether the basis is available')
args = parser.parse_args()

env = args.env
idx = args.idx
basis_available = args.basis_available

## ----------------------------------------------------------------------------------------------------
## load particles unlearned
particles_unlearned = np.load(smc_results_dir + env + "_particles_unlearned_" + str(idx) + ".npy")
h_unlearned = np.load(smc_results_dir + env + "_h_list_unlearned_" + str(idx) + ".npy")
## load particles learned_f
particles_learned_f = np.load(smc_results_dir + env + "_particles_learned_f_" + str(idx) + ".npy")
h_learned_f = np.load(smc_results_dir + env + "_h_list_learned_f_" + str(idx) + ".npy")
## load particles learned_fS 
particles_learned_fS = np.load(smc_results_dir + env + "_particles_learned_fS_" + str(idx) + ".npy")
h_learned_fS = np.load(smc_results_dir + env + "_h_list_learned_fS_" + str(idx) + ".npy")  

h_unlearned_sum = np.cumsum(h_unlearned)
h_learned_f_sum = np.cumsum(h_learned_f)
h_learned_fS_sum = np.cumsum(h_learned_fS)

# Plotting the results
plt.figure() 
plt.plot(h_unlearned_sum, linestyle='none', marker='o', label='Unlearned', color='blue')
plt.plot(h_learned_f_sum, linestyle='none', marker='*', label='Learned f', color='orange')
plt.plot(h_learned_fS_sum, linestyle='none', marker='.', label='Learned fS', color='green')
plt.xlabel('Iteration')
plt.ylabel('h')
plt.title('SMC Analysis of h over Iterations')
plt.legend()
plt.grid()
plt.savefig(results_fig_table + env + "_smc_h_analysis_" + str(idx) + ".png", dpi=500, bbox_inches='tight')
plt.close()

## ----------------------------------------------------------------------------------------------------
## loading the mesh information of the model_params
num_points = int(100)
## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

path = "./RESULTS/FNO_results/"
param = np.array(np.load(path + env + "_result_" + str(idx) + ".npy"))
param = np.array(param[1])  # Extract the true function values
fun_truth = fe.Function(domain.function_space)
fun_truth.vector()[:] = np.array(param)

# calculate the mean of particles
mean_unlearned = np.mean(particles_unlearned, axis=0)
mean_learned_f = np.mean(particles_learned_f, axis=0)
mean_learned_fS = np.mean(particles_learned_fS, axis=0)
fun_mean_unlearned = fe.Function(domain.function_space)
fun_mean_learned_f = fe.Function(domain.function_space)
fun_mean_learned_fS = fe.Function(domain.function_space)
fun_mean_unlearned.vector()[:] = np.array(mean_unlearned)
fun_mean_learned_f.vector()[:] = np.array(mean_learned_f)
fun_mean_learned_fS.vector()[:] = np.array(mean_learned_fS)

# Plotting the results
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
fig = fe.plot(fun_truth)
plt.colorbar(fig)
plt.title("(a) True function")
plt.subplot(2, 2, 2)
fig = fe.plot(fun_mean_unlearned)
plt.colorbar(fig)
plt.title("(b) Estimated mean function (Unlearned)")
plt.subplot(2, 2, 3)
fig = fe.plot(fun_mean_learned_f)
plt.colorbar(fig)
plt.title("(c) Estimated mean function (Learned f)")
plt.subplot(2, 2, 4)
fig = fe.plot(fun_mean_learned_fS)
plt.colorbar(fig)
plt.title("(d) Estimated mean function (Learned fS)")
plt.tight_layout(pad=1, w_pad=0.5, h_pad=2)
plt.savefig(results_fig_table + env + "_smc_mean_analysis_" + str(idx) + ".png", dpi=500, bbox_inches='tight')
plt.close()

print("Relative error (Unlearned): ", relative_error_simple(fun_mean_unlearned, fun_truth))
print("Relative error (Learned f): ", relative_error_simple(fun_mean_learned_f, fun_truth))
print("Relative error (Learned fS): ", relative_error_simple(fun_mean_learned_fS, fun_truth))

## ----------------------------------------------------------------------------------------------------
## We project the function particles to the eigen-basis functions and do PCA to find the variables 
## to represent the function particles. Then draw the variances of the finite-dimensional variables. 
## ----------------------------------------------------------------------------------------------------

equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
        domain=domain, domain_=domain, alpha=0.1, beta=10, s=2
        )
MS = prior.Ms 

num_basis = 200
if basis_available is False:
    prior.calculate_eigensystem()
    proj_funs = []
    for ii in range(num_basis):
        tmp_fun = fe.Function(domain.function_space)
        tmp_fun.vector()[:] = prior.eigvec[:, ii]
        proj_funs.append(tmp_fun.vector()[:])
    
    np.save(results_fig_table_data + "proj_funs.npy", np.array(proj_funs))

proj_funs = np.load(results_fig_table_data + "proj_funs.npy")[50:100,:]

feature_truth = fun_truth.vector()[:]@MS@(proj_funs.T)
feature_unlearned = particles_unlearned@MS@(proj_funs.T)
feature_learned_f = particles_learned_f@MS@(proj_funs.T)
feature_learned_fS = particles_learned_fS@MS@(proj_funs.T)
# feature_truth = fun_truth.vector()[:5]
# feature_unlearned = particles_unlearned[:, :5]
# feature_learned_f = particles_learned_f[:, :5]
# feature_learned_fS = particles_learned_fS[:, :5]

# num_pca = 5
# pca = PCA(n_components=num_pca)  
# feature_unlearned = pca.fit_transform(feature_unlearned)[0]
# feature_learned_f = pca.fit_transform(feature_learned_f)[0]
# feature_learned_fS = pca.fit_transform(feature_learned_fS)[0]

mean_feature_unlearned = np.mean(feature_unlearned, axis=0)
mean_feature_learned_f = np.mean(feature_learned_f, axis=0)
mean_feature_learned_fS = np.mean(feature_learned_fS, axis=0)
std_feature_unlearned = np.std(feature_unlearned, axis=0)
std_feature_learned_f = np.std(feature_learned_f, axis=0)
std_feature_learned_fS = np.std(feature_learned_fS, axis=0)

num = mean_feature_unlearned.shape[0]

# Plotting the results
plt.figure(figsize=(17, 5))
plt.subplot(1, 3, 1)
plt.errorbar(np.arange(num), mean_feature_unlearned, yerr=2*std_feature_unlearned, fmt='o', label="Mean")
plt.scatter(np.arange(num), feature_truth, color='red', label='True')
plt.xlabel('Coefficients')
plt.legend()
plt.title('Unlearned')
plt.subplot(1, 3, 2)
plt.errorbar(np.arange(num), mean_feature_learned_f, yerr=2*std_feature_learned_f, fmt='o', label="Mean")
plt.scatter(np.arange(num), feature_truth, color='red', label='True')
plt.xlabel('Coefficients')
plt.legend()
plt.title('Learned f')
plt.subplot(1, 3, 3)
plt.errorbar(np.arange(num), mean_feature_learned_fS, yerr=2*std_feature_learned_fS, fmt='o', label="Mean")
plt.scatter(np.arange(num), feature_truth, color='red', label='True')
plt.xlabel('Coefficients')
plt.legend()
plt.title('Learned fS')
plt.savefig(results_fig_table + env + "_smc_std_analysis_" + str(idx) + ".png", dpi=500, bbox_inches='tight')
plt.close()













