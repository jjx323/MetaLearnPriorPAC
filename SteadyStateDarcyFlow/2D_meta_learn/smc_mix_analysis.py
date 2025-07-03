import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
import fenics as fe
import torch
from sklearn.decomposition import PCA

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import relative_error_simple
from SteadyStateDarcyFlow.MLcommon import GaussianFiniteRankTorch
from SteadyStateDarcyFlow.common import file_process


mpl.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
import scienceplots
plt.style.use('science')

meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
smc_results_dir = "./RESULTS/SMC_Mix/"
results_fig_table = "./RESULTS-PAPER-SMC-MIX/"
os.makedirs(results_fig_table, exist_ok=True)
results_fig_table_data = "./RESULTS-PAPER-SMC-MIX/DATA/"
os.makedirs(results_fig_table_data, exist_ok=True)

parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
parser.add_argument('--basis_available', type=bool, default=False, help='whether the basis is available')
args = parser.parse_args()

env = args.env
idxs = [0,1,2,3,4,5,6,7,8,9]
basis_available = args.basis_available

relative_errors_unlearned = []
relative_errors_learned_f = []
relavive_errors_learned_fS = []

if env == "complex":
    env_num = 1
elif env == "simple":
    env_num = 2

for idx in idxs:
    print("idx: ", idx)
    ## ----------------------------------------------------------------------------------------------------
    ## load the particles and the h_list
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

    # print(particles_unlearned.shape)
    ## ----------------------------------------------------------------------------------------------------
    ## draw the sum_h to validate the learned prior accelerate the SMC-MIX sampling algorithm
    ## ----------------------------------------------------------------------------------------------------

    h_unlearned_sum = np.cumsum(h_unlearned)
    h_learned_f_sum = np.cumsum(h_learned_f)
    h_learned_fS_sum = np.cumsum(h_learned_fS)

    # Plotting the results
    plt.rcParams['font.size'] = 7
    plt.figure() 
    plt.plot(h_unlearned_sum, linestyle='none', marker='o', label="Unlearned "+ r"$\mathbb{P}$", color='blue')
    plt.plot(h_learned_f_sum, linestyle='none', marker='*', label="Learned "+ r"$\mathbb{P}^{\theta}$", color='orange')
    plt.plot(h_learned_fS_sum, linestyle='none', marker='.', label="Learned "+ r"$\mathbb{P}_S^{\theta}$", color='green')
    plt.xlabel('Iteration number')
    plt.ylabel('Accumulated temperature')
    if env == "simple":
        ttt = "(a) Simple environment"
    elif idx%2 == 0:
        ttt = "(b) Complex environment (positive branch)"
    elif idx%2 == 1:
        ttt = "(c) Complex environment (negative branch)"
    plt.title(ttt)
    plt.legend()
    plt.savefig(results_fig_table + env + "_smc_h_analysis_" + str(idx) + ".png", dpi=1000, bbox_inches='tight')
    plt.close()

    ## ----------------------------------------------------------------------------------------------------
    ## draw mean function to validate the learned prior produce better mean estimation
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
    plt.figure(figsize=(22, 5))
    plt.subplot(1, 4, 1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(a) True function", pad=15)
    plt.subplot(1, 4, 2)
    fig = fe.plot(fun_mean_unlearned)
    plt.colorbar(fig)
    plt.title("(b) Posterior mean (unlearned "+ r"$\mathbb{P}$" + ")")
    plt.subplot(1, 4, 3)
    fig = fe.plot(fun_mean_learned_f)
    plt.colorbar(fig)
    plt.title("(c) Posterior mean (learned " + r"$\mathbb{P}^{\theta}$" + ")", pad=15)
    plt.subplot(1, 4, 4)
    fig = fe.plot(fun_mean_learned_fS)
    plt.colorbar(fig)
    plt.title("(d) Posterior mean (learned " + r"$\mathbb{P}_{S}^{\theta}$" + ")", pad=15)
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2)
    plt.savefig(results_fig_table + env + "_smc_mean_analysis_" + str(idx) + ".png", dpi=1000, bbox_inches='tight')
    plt.close()

    tmp1 = relative_error_simple(fun_mean_unlearned, fun_truth)
    tmp2 = relative_error_simple(fun_mean_learned_f, fun_truth)
    tmp3 = relative_error_simple(fun_mean_learned_fS, fun_truth)
    relative_errors_unlearned.append(tmp1)
    relative_errors_learned_f.append(tmp2)
    relavive_errors_learned_fS.append(tmp3)

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
    if basis_available == False and idx == 0:
        prior.calculate_eigensystem()
        proj_funs = []
        for ii in range(num_basis):
            tmp_fun = fe.Function(domain.function_space)
            tmp_fun.vector()[:] = prior.eigvec[:, ii]
            proj_funs.append(tmp_fun.vector()[:])
        
        np.save(results_fig_table_data + "proj_funs.npy", np.array(proj_funs))

    proj_funs = np.load(results_fig_table_data + "proj_funs.npy")[50:100, :]

    feature_truth = fun_truth.vector()[:]@MS@(proj_funs.T)
    feature_unlearned = particles_unlearned@MS@(proj_funs.T)
    feature_learned_f = particles_learned_f@MS@(proj_funs.T)
    feature_learned_fS = particles_learned_fS@MS@(proj_funs.T)

    mean_feature_unlearned = np.mean(feature_unlearned, axis=0)
    mean_feature_learned_f = np.mean(feature_learned_f, axis=0)
    mean_feature_learned_fS = np.mean(feature_learned_fS, axis=0)
    std_feature_unlearned = np.std(feature_unlearned, axis=0)
    std_feature_learned_f = np.std(feature_learned_f, axis=0)
    std_feature_learned_fS = np.std(feature_learned_fS, axis=0)

    num = mean_feature_unlearned.shape[0]

    # Plotting the results
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    plt.errorbar(np.arange(num), mean_feature_unlearned, yerr=3*std_feature_unlearned, fmt='o', label="Mean")
    plt.scatter(np.arange(num), feature_truth, color='red', label='True')
    plt.xlabel('Coefficients')
    plt.legend(loc='lower left')
    plt.title('(a) Unlearned (' + r'$\mathbb{P}$' + ')', pad=15)
    plt.subplot(1, 3, 2)
    plt.errorbar(np.arange(num), mean_feature_learned_f, yerr=3*std_feature_learned_f, fmt='o', label="Mean")
    plt.scatter(np.arange(num), feature_truth, color='red', label='True')
    plt.xlabel('Coefficients')
    plt.legend(loc='lower left')
    plt.title('(b) Learned (' + r'$\mathbb{P}^{\theta}$' + ')', pad=15)
    plt.subplot(1, 3, 3)
    plt.errorbar(np.arange(num), mean_feature_learned_fS, yerr=3*std_feature_learned_fS, fmt='o', label="Mean")
    plt.scatter(np.arange(num), feature_truth, color='red', label='True')
    plt.xlabel('Coefficients')
    plt.legend(loc='lower left')
    plt.title('(c) Learned (' + r'$\mathbb{P}_S^{\theta}$' + ')', pad=15)
    plt.savefig(results_fig_table + env + "_smc_std_analysis_" + str(idx) + ".png", dpi=1000, bbox_inches='tight')
    plt.close()             


avg_relative_error_unlearned = np.mean(relative_errors_unlearned)
avg_relative_error_learned_f = np.mean(relative_errors_learned_f)
avg_relative_error_learned_fS = np.mean(relavive_errors_learned_fS)
print("Average relative error unlearned: ", avg_relative_error_unlearned)
print("Average relative error learned f: ", avg_relative_error_learned_f)
print("Average relative error learned fS: ", avg_relative_error_learned_fS)

errors_file_names = results_fig_table + env + "_errors.txt"
file_process(errors_file_names)
with open(errors_file_names, 'a') as file:
    file.write("Average relative error unlearned: "+ str(avg_relative_error_unlearned) + '\n')  
    file.write("Average relative error learned f: "+ str(avg_relative_error_learned_f) + '\n')  
    file.write("Average relative error learned fS: "+ str(avg_relative_error_learned_fS) + '\n')  


