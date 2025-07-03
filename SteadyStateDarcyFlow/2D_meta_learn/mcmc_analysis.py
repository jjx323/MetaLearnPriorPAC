## import necessary packages
import numpy as np 
import fenics as fe
import torch 
import argparse
import matplotlib as mpl
import time
from sklearn.decomposition import PCA
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import NewtonCG
from core.sample import pCN
from core.misc import load_expre, relative_error_simple, construct_measurement_matrix

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss, file_process


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.4

## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="Analysis of the MCMC results for PCN")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
parser.add_argument('--idx_particle', type=int, default=0, help='index of the particles to analyze')
parser.add_argument('--data_available', type=bool, default=False, help='data_available = True or False')
args = parser.parse_args()

## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
results_fig_table = "./RESULTS-PAPER-MCMC/"
os.makedirs(results_fig_table, exist_ok=True)
results_fig_table_data = "./RESULTS-PAPER-MCMC/DATA/"
os.makedirs(results_fig_table_data, exist_ok=True)
path = "./RESULTS/FNO_results/"

env = args.env
idx_particle = args.idx_particle
data_available = args.data_available

if env == "complex":
    env_num = 1
elif env == "simple":
    env_num = 2

noise_level = 0.01

## Load the saved parameters
if env == "complex":
    kk_FNO = int(7300)  # env = 'complex'
    kk_f = int(7300)      # env = 'complex'
elif env == "simple":
    kk_FNO = int(7300)   # env = 'simple'
    kk_f = int(7300)     # env = 'simple'
else:
    raise ValueError("env should be complex or simple")


pcn_unlearn_dir = "./RESULTS/" + env + "_est_unlearn_pcn_" + str(idx_particle) + "/" 
pcn_learned_f_dir = "./RESULTS/" + env + "_estimates_learned_f_pcn_" + str(idx_particle) + "/"    
pcn_learned_fS_dir = "./RESULTS/" + env + "_estimates_learned_fS_pcn_" + str(idx_particle) + "/"  

num_points = int(100)
## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
domain_ = domain

fun_truth = fe.Function(domain.function_space)
fun_unlearn = fe.Function(domain.function_space)
fun_learned = fe.Function(domain.function_space)
fun_learned_f = fe.Function(domain.function_space)

small_n = equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )

## construct the basis function
if data_available == False:
    prior.calculate_eigensystem()
    proj_fun = []
    for ii in range(200):
        tmp_fun = fe.Function(domain.function_space)
        tmp_fun.vector()[:] = prior.eigvec[:, ii]
        proj_fun.append(tmp_fun.vector()[:])
    np.save(results_fig_table_data + "proj_funs.npy", np.array(proj_fun))

proj_fun = np.load(results_fig_table_data + "proj_funs.npy")


##-------------------------------------------------------------------------------------------
proj_fun = np.array(proj_fun).T
print("proj_fun.shape = ", proj_fun.shape)
Ms = prior.Ms

def load_chain_proj(dir, proj_fun=proj_fun, Ms=Ms, num_pca=3, idx_start=None, convert=True):
    if os.path.exists(dir):
        npy_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        if idx_start is None:
            idx_start = int(len(npy_files)/2)
        for idx in range(idx_start, len(npy_files)): ## just using the last half of samples
            # Load the numpy array from the file
            data = np.load(dir + "sample_" + str(idx) + ".npy", allow_pickle=True)
            if idx == idx_start:
                pcn_chain = data@(Ms@proj_fun)
            else:
                pcn_chain = np.concatenate((pcn_chain, data@(Ms@proj_fun)), axis=0)
        pcn_chain = np.array(pcn_chain)
        # for idx in range(proj_fun.shape[1]):
        if convert == True:
            pca = PCA(n_components=num_pca)  
            pcn_chain = pca.fit_transform(pcn_chain)
            chain = {}
            for idx in range(num_pca):
                chain["var_" + str(idx+1)] = pcn_chain[:, idx]
            data_chain = az.convert_to_inference_data(chain)
            return data_chain
        else:
            return pcn_chain
    else:
        raise FileNotFoundError(f"The directory {dir} does not exist.")
    

def load_chain(dir, idx_trace=[0, 1, 2], idx_start=None):
    if os.path.exists(dir):
        npy_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        if idx_start is None:
            idx_start = int(len(npy_files)/2)
        for idx in range(idx_start, len(npy_files)): ## just using the last half of samples
            # Load the numpy array from the file
            data = np.load(dir + "sample_" + str(idx) + ".npy", allow_pickle=True)
            if idx == idx_start:
                if idx_trace == None:
                    pcn_chain = data[:, :]
                else:
                    pcn_chain = data[:, idx_trace]
            else:
                if idx_trace == None:
                    pcn_chain = np.concatenate((pcn_chain, data[:, :]), axis=0)
                else:
                    pcn_chain = np.concatenate((pcn_chain, data[:, idx_trace]), axis=0)
        pcn_chain = np.array(pcn_chain)
        if idx_trace == None:
            data_chain = pcn_chain
        else:
            # Transform the chain to InferenceData format
            chain = {}
            for idx in idx_trace:
                chain["var_" + str(idx+1)] = pcn_chain[:, idx]
            data_chain = az.convert_to_inference_data(chain)
        return data_chain, idx_trace, dir
    else:
        raise FileNotFoundError(f"The directory {dir} does not exist.")
    

def load_chain_key_quantity(dir, Ms=Ms, idx_start=None):
    if os.path.exists(dir):
        npy_files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        if idx_start is None:
                idx_start = int(len(npy_files)/2)
        for idx in range(idx_start, len(npy_files)): ## just using the last half of samples
            # Load the numpy array from the file
            data = np.load(dir + "sample_" + str(idx) + ".npy", allow_pickle=True)
            if idx == idx_start:
                pcn_chain = np.diag(data@Ms@(data.T))   #np.sum(data, axis=1)
            else:
                pcn_chain = np.concatenate((pcn_chain, np.diag(data@Ms@(data.T))), axis=0)
        pcn_chain = np.array(pcn_chain)
        return pcn_chain
    else:
        raise FileNotFoundError(f"The directory {dir} does not exist.")
    
## ----------------------------------------------------------------------------------------------------------------
print("Start computing the posterior mean and std...")
param = np.array(np.load(path + env + "_result_" + str(idx_particle) + ".npy"))
param = np.array(param[1])  # Extract the true function values
fun_truth.vector()[:] = np.array(param)
if data_available == False:
    ## calculate the posterior mean
    unlearn_data, _, _ = load_chain(pcn_unlearn_dir, idx_trace=None)
    unlearn_mean = np.mean(unlearn_data, axis=0)
    unlearn_std = np.std(unlearn_data, axis=0)

    learned_f_data, _, _ = load_chain(pcn_learned_f_dir, idx_trace=None)
    learned_f_mean = np.mean(learned_f_data, axis=0)
    learned_f_std = np.std(learned_f_data, axis=0)

    learned_fS_data, _, _ = load_chain(pcn_learned_fS_dir, idx_trace=None)
    learned_fS_mean = np.mean(learned_fS_data, axis=0)
    learned_fS_std = np.std(learned_fS_data, axis=0)

    fun_unlearn.vector()[:] = np.array(unlearn_mean)
    fun_learned.vector()[:] = np.array(learned_f_mean)
    fun_learned_f.vector()[:] = np.array(learned_fS_mean)

    std_unlearn = fe.Function(domain.function_space)
    std_learned = fe.Function(domain.function_space)
    std_learned_f = fe.Function(domain.function_space)
    std_unlearn.vector()[:] = np.array(unlearn_std)
    std_learned.vector()[:] = np.array(learned_f_std)
    std_learned_f.vector()[:] = np.array(learned_fS_std)

    err1 = relative_error_simple(fun_unlearn, fun_truth)
    err2 = relative_error_simple(fun_learned, fun_truth)
    err3 = relative_error_simple(fun_learned_f, fun_truth)
    print("Relative error unlearn: ", err1)
    print("Relative error learned_f: ", err2)
    print("Relative error learned_fS: ", err3)
    file_process(results_fig_table + env + "_relative_error_" + str(idx_particle) + ".txt")
    with open(results_fig_table + env + "_relative_error_" + str(idx_particle) + ".txt", 'a') as f:
        f.write(f"Relative error unlearn: {err1}\n")
        f.write(f"Relative error learned_f: {err2}\n")
        f.write(f"Relative error learned_fS: {err3}\n")

    np.save(results_fig_table_data + env + "_pcn_unlearn_mean_" + str(idx_particle) + ".npy", unlearn_mean)
    np.save(results_fig_table_data + env + "_pcn_learned_f_mean_" + str(idx_particle) + ".npy", learned_f_mean)
    np.save(results_fig_table_data + env + "_pcn_learned_fS_mean_" + str(idx_particle) + ".npy", learned_fS_mean)
    np.save(results_fig_table_data + env + "_pcn_unlearn_std_" + str(idx_particle) + ".npy", unlearn_std)
    np.save(results_fig_table_data + env + "_pcn_learned_f_std_" + str(idx_particle) + ".npy", learned_f_std)    
    np.save(results_fig_table_data + env + "_pcn_learned_fS_std_" + str(idx_particle) + ".npy", learned_fS_std)

## load the posterior mean and std
unlearn_mean = np.load(results_fig_table_data + env + "_pcn_unlearn_mean_" + str(idx_particle) + ".npy")
learned_f_mean = np.load(results_fig_table_data + env + "_pcn_learned_f_mean_" + str(idx_particle) + ".npy")
learned_fS_mean = np.load(results_fig_table_data + env + "_pcn_learned_fS_mean_" + str(idx_particle) + ".npy")
fun_unlearn.vector()[:] = np.array(unlearn_mean)
fun_learned.vector()[:] = np.array(learned_f_mean)
fun_learned_f.vector()[:] = np.array(learned_fS_mean)

unlearn_std = np.load(results_fig_table_data + env + "_pcn_unlearn_std_" + str(idx_particle) + ".npy")
learned_f_std = np.load(results_fig_table_data + env + "_pcn_learned_f_std_" + str(idx_particle) + ".npy")
learned_fS_std = np.load(results_fig_table_data + env + "_pcn_learned_fS_std_" + str(idx_particle) + ".npy")
std_unlearn = fe.Function(domain.function_space)
std_learned = fe.Function(domain.function_space)
std_learned_f = fe.Function(domain.function_space)
std_unlearn.vector()[:] = np.array(unlearn_std)
std_learned.vector()[:] = np.array(learned_f_std)
std_learned_f.vector()[:] = np.array(learned_fS_std)


plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.4

if env == "simple":
    plt.figure(figsize=(22,5))
    plt.subplot(1,4,1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(a) True function")
    plt.subplot(1,4,2)
    fig = fe.plot(fun_unlearn)
    plt.colorbar(fig)
    plt.title("(b) Posterior mean (unlearned prior)")
    plt.subplot(1,4,3)
    fig = fe.plot(fun_learned)
    plt.colorbar(fig)
    plt.title("(c) Posterior mean (learned data-independent prior)")
    plt.subplot(1,4,4)
    fig = fe.plot(fun_learned_f)
    plt.colorbar(fig)
    plt.title("(d) Posterior mean (learned data-dependent prior)")
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2) 
    plt.savefig(results_fig_table + env + "_pcn_mean_" + str(idx_particle) + ".png", dpi=500)
    plt.close()

num = 100
points_x = np.linspace(0, 1, num).reshape(num, 1)
points_y = 0.0*np.ones((num, 1)) + 0.2
points = []
for idx in range(num):
    points.append((points_x[idx], points_y[idx]))
points = np.array(points).reshape(num, 2)
Mpoints = construct_measurement_matrix(points, domain.function_space)

if env == "simple":
    plt.figure(figsize=(17,5))
    plt.subplot(1,3,1)
    plt.plot(Mpoints@fun_truth.vector()[:])
    plt.plot(Mpoints@fun_unlearn.vector()[:])
    plt.plot(Mpoints@fun_unlearn.vector()[:] + 3*Mpoints@std_unlearn.vector()[:], '--', color='gray')
    plt.plot(Mpoints@fun_unlearn.vector()[:] - 3*Mpoints@std_unlearn.vector()[:], '--', color='gray')
    plt.subplot(1,3,2)
    plt.plot(Mpoints@fun_truth.vector()[:])
    plt.plot(Mpoints@fun_learned.vector()[:])
    plt.plot(Mpoints@fun_learned.vector()[:] + 3*Mpoints@std_learned.vector()[:], '--', color='gray')
    plt.plot(Mpoints@fun_learned.vector()[:] - 3*Mpoints@std_learned.vector()[:], '--', color='gray')
    plt.subplot(1,3,3)
    plt.plot(Mpoints@fun_truth.vector()[:])
    plt.plot(Mpoints@fun_learned_f.vector()[:])
    plt.plot(Mpoints@fun_learned_f.vector()[:] + 3*Mpoints@std_learned_f.vector()[:], '--', color='gray')
    plt.plot(Mpoints@fun_learned_f.vector()[:] - 3*Mpoints@std_learned_f.vector()[:], '--', color='gray')
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2) 
    plt.savefig(results_fig_table + env + "_pcn_mean_slice_" + str(idx_particle) + ".png", dpi=500)
    plt.close()

## ----------------------------------------------------------------------------------------------------------------
print("Start drawing the trace plot of the key quantities...")
if data_available == False:
    tmp1 = load_chain_key_quantity(pcn_unlearn_dir)
    tmp2 = load_chain_key_quantity(pcn_learned_f_dir)
    tmp3 = load_chain_key_quantity(pcn_learned_fS_dir)
    np.save(results_fig_table_data + env + "_pcn_unlearn_key_quantity_" + str(idx_particle) + ".npy", tmp1)
    np.save(results_fig_table_data + env + "_pcn_learned_f_key_quantity_" + str(idx_particle) + ".npy", tmp2)
    np.save(results_fig_table_data + env + "_pcn_learned_fS_key_quantity_" + str(idx_particle) + ".npy", tmp3)

tmp1 = np.load(results_fig_table_data + env + "_pcn_unlearn_key_quantity_" + str(idx_particle) + ".npy")
tmp2 = np.load(results_fig_table_data + env + "_pcn_learned_f_key_quantity_" + str(idx_particle) + ".npy")
tmp3 = np.load(results_fig_table_data + env + "_pcn_learned_fS_key_quantity_" + str(idx_particle) + ".npy")

plt.figure(figsize=(17,5))
plt.subplot(1,3,1)
plt.plot(tmp1, label="Unlearned Method")
plt.legend()
plt.subplot(1,3,2)
plt.plot(tmp2, label="Data-Independent Method")
plt.legend()
plt.subplot(1,3,3)  
plt.plot(tmp3, label="Data-Dependent Method")    
plt.legend()
plt.tight_layout(pad=1, w_pad=0.5, h_pad=2)
plt.savefig(results_fig_table + env + "_pcn_trace_key_quantity_" + str(idx_particle) + ".png", dpi=500)
plt.close()

if data_available == False:
    tmp1 = load_chain_key_quantity(pcn_unlearn_dir, idx_start=4)
    tmp2 = load_chain_key_quantity(pcn_learned_f_dir, idx_start=4)
    tmp3 = load_chain_key_quantity(pcn_learned_fS_dir, idx_start=4)
    np.save(results_fig_table_data + env + "_" + str(4) + "_pcn_unlearn_key_quantity_" + str(idx_particle) + ".npy", tmp1) 
    np.save(results_fig_table_data + env + "_" + str(4) + "_pcn_learned_f_key_quantity_" + str(idx_particle) + ".npy", tmp2)
    np.save(results_fig_table_data + env + "_" + str(4) + "_pcn_learned_fS_key_quantity_" + str(idx_particle) + ".npy", tmp3)

tmp1 = np.load(results_fig_table_data + env + "_" + str(4) + "_pcn_unlearn_key_quantity_" + str(idx_particle) + ".npy")
tmp2 = np.load(results_fig_table_data + env + "_" + str(4) + "_pcn_learned_f_key_quantity_" + str(idx_particle) + ".npy")
tmp3 = np.load(results_fig_table_data + env + "_" + str(4) + "_pcn_learned_fS_key_quantity_" + str(idx_particle) + ".npy")

means_key_quantities = np.array([np.mean(tmp1, axis=0), np.mean(tmp2, axis=0), np.mean(tmp3, axis=0)])
stds_key_quantities = np.array([np.std(tmp1, axis=0), np.std(tmp2, axis=0), np.std(tmp3, axis=0)])
tmp = fun_truth.vector()[:]@Ms@fun_truth.vector()
tmp = np.array([tmp, tmp, tmp])
plt.figure()
plt.errorbar(np.arange(3), means_key_quantities, yerr=2*stds_key_quantities, fmt='o', label="Estimated")
plt.scatter(np.arange(3), tmp, color='red', label='True')
plt.xlabel('Coefficients')
plt.legend()
plt.title('Key Quantities')
plt.savefig(results_fig_table + env + "_pcn_errorbar_key_quantity_" + str(idx_particle) + "_.png", dpi=500)
plt.close()

chain = {}
chain["trace_unlearn"] = tmp1
chain["trace_data-free_prior"] = tmp2
chain["trace_data-depend_prior"] = tmp3
pcn_chain_key = az.convert_to_inference_data(chain)
az.plot_autocorr(pcn_chain_key, max_lag=500)
plt.tight_layout()
plt.savefig(results_fig_table + env + "_pcn_autocorr_plot_key_" + str(idx_particle) + ".png", dpi=500)
plt.close()

## ----------------------------------------------------------------------------------------------------------------
print("Start drawing the trace plot of the projected PCA quantities...")
if data_available == False:
    ## load the chains
    pcn_unlearn_chain = load_chain_proj(pcn_unlearn_dir)
    pcn_learned_f_chain = load_chain_proj(pcn_learned_f_dir)
    pcn_learned_fS_chain = load_chain_proj(pcn_learned_fS_dir)
    np.save(results_fig_table_data + env + "_pcn_unlearn_chain_" + str(idx_particle) + ".npy", pcn_unlearn_chain.posterior.data_vars["var_1"])
    np.save(results_fig_table_data + env + "_pcn_learned_f_chain_" + str(idx_particle) + ".npy", pcn_learned_f_chain.posterior.data_vars["var_1"])
    np.save(results_fig_table_data + env + "_pcn_learned_fS_chain_" + str(idx_particle) + ".npy", pcn_learned_fS_chain.posterior.data_vars["var_1"])

## draw trace plot    
tmp1 = np.load(results_fig_table_data + env + "_pcn_unlearn_chain_" + str(idx_particle) + ".npy")[0]
tmp2 = np.load(results_fig_table_data + env + "_pcn_learned_f_chain_" + str(idx_particle) + ".npy")[0]
tmp3 = np.load(results_fig_table_data + env + "_pcn_learned_fS_chain_" + str(idx_particle) + ".npy")[0]
plt.figure(figsize=(17,5))
plt.subplot(1,3,1)
plt.plot(tmp1)
plt.title("Unlearned Method")
plt.subplot(1,3,2)
plt.plot(tmp2)
plt.title("Data-Independent Method")
plt.subplot(1,3,3)  
plt.plot(tmp3)    
plt.title("Data-Dependent Method")
plt.tight_layout(pad=1, w_pad=0.5, h_pad=2)
plt.savefig(results_fig_table + env + "_pcn_trace_projection_" + str(idx_particle) + ".png", dpi=500)
plt.close()

## ----------------------------------------------------------------------------------------------------------------
print("Start drawing the trace plot of the projected quantities...")
if data_available == False:
    ## load the chains
    pcn_unlearn_chain = load_chain_proj(pcn_unlearn_dir, convert=False)
    pcn_learned_f_chain = load_chain_proj(pcn_learned_f_dir, convert=False)
    pcn_learned_fS_chain = load_chain_proj(pcn_learned_fS_dir, convert=False)
    np.save(results_fig_table_data + env + "_notpca_pcn_unlearn_chain_" + str(idx_particle) + ".npy", pcn_unlearn_chain)
    np.save(results_fig_table_data + env + "_notpca_pcn_learned_f_chain_" + str(idx_particle) + ".npy", pcn_learned_f_chain)
    np.save(results_fig_table_data + env + "_notpca_pcn_learned_fS_chain_" + str(idx_particle) + ".npy", pcn_learned_fS_chain)

## draw trace plot    
ii = 100
tmp1 = np.load(results_fig_table_data + env + "_notpca_pcn_unlearn_chain_" + str(idx_particle) + ".npy")[:, ii]
tmp2 = np.load(results_fig_table_data + env + "_notpca_pcn_learned_f_chain_" + str(idx_particle) + ".npy")[:, ii]
tmp3 = np.load(results_fig_table_data + env + "_notpca_pcn_learned_fS_chain_" + str(idx_particle) + ".npy")[:, ii]
plt.figure(figsize=(17,5))
plt.subplot(1,3,1)
plt.plot(tmp1)
plt.title("Unlearned Method")
plt.subplot(1,3,2)
plt.plot(tmp2)
plt.title("Data-Independent Method")
plt.subplot(1,3,3)  
plt.plot(tmp3)    
plt.title("Data-Dependent Method")
plt.tight_layout(pad=1, w_pad=0.5, h_pad=2)
plt.savefig(results_fig_table + env + "_notpca_pcn_trace_projection_" + str(idx_particle) + ".png", dpi=500)
plt.close()

## calculate ESS
ess1 = az.ess(tmp1)
ess2 = az.ess(tmp2)
ess3 = az.ess(tmp3)
print(ess1)
print(ess2)
print(ess3)

if data_available == False:
    ## calculate the autocorrelation
    pcn_unlearn_chain = load_chain_proj(pcn_unlearn_dir, idx_start=4)
    pcn_learned_f_chain = load_chain_proj(pcn_learned_f_dir, idx_start=4)
    pcn_learned_fS_chain = load_chain_proj(pcn_learned_fS_dir, idx_start=4)

    az.plot_autocorr(pcn_unlearn_chain, max_lag=500)
    plt.tight_layout()
    plt.savefig(results_fig_table + env + "_pcn_autocorr_plot_unlearn_" + str(idx_particle) + ".png", dpi=500)
    plt.close()

    az.plot_autocorr(pcn_learned_f_chain, max_lag=500)
    plt.tight_layout()
    plt.savefig(results_fig_table + env + "_pcn_autocorr_plot_learned_f_" + str(idx_particle) + ".png", dpi=500)
    plt.close()

    az.plot_autocorr(pcn_learned_fS_chain, max_lag=500)
    plt.tight_layout()
    plt.savefig(results_fig_table + env + "_pcn_autocorr_plot_learned_fS_" + str(idx_particle) + ".png", dpi=500)
    plt.close()

    ## 计算ESS
    ess1 = az.ess(pcn_unlearn_chain)
    ess2 = az.ess(pcn_learned_f_chain)
    ess3 = az.ess(pcn_learned_fS_chain)
    file_process(results_fig_table + env + "_pcn_ESS_" + str(idx_particle) + ".txt")
    with open(results_fig_table + env + "_pcn_ESS_" + str(idx_particle) + ".txt", 'a') as f:
        f.write(f"ess_unlearn: {ess1}\n")
        f.write(f"ess_learned_f: {ess2}\n")
        f.write(f"ess_learned_fS: {ess3}\n")
        



