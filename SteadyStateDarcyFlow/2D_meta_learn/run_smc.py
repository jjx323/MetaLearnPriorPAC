## import necessary packages
import numpy as np 
import fenics as fe
import torch 
import argparse
import time
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import NewtonCG
from core.sample import pCN, SMC
from core.misc import load_expre, relative_error_simple, construct_measurement_matrix

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss, file_process


"""
In the following, we compare the performance by SMC with different prior measures.
"""

"""
Preparations for construct appropriate prior measures 
"""

## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
parser.add_argument('--newton_method', type=str, default="bicgstab", help='newton_method = "cg_my" or "bicgstab"')
parser.add_argument('--length_total', type=int, default=np.int64(1e5), help='length_total of MCMC')
parser.add_argument('--beta_pcn', type=float, default=1, help='beta of pCN')
parser.add_argument('--len_pcn_chain', type=int, default=100, help='length of pCN chain')
parser.add_argument('--num_particles', type=int, default=500, help='number of particles for SMC')
args = parser.parse_args()

num_particles = args.num_particles
len_pcn_chain = np.int64(args.len_pcn_chain)

## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
smc_results_dir = "./RESULTS/SMC/"
os.makedirs(smc_results_dir, exist_ok=True)
results_fig_table = "./RESULTS-PAPER-SMC/"
os.makedirs(results_fig_table, exist_ok=True)

env = args.env
if env == "complex":
    env_num = 1
    indexs = [2, 3]
elif env == "simple":
    env_num = 2
    indexs = [0]
with_hyper_prior = True
# with_hyper_prior = False
device = "cpu"
# device = "cuda"

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

pcn_unlearn_dir = "./RESULTS/" + env + "_estimates_unlearn_pcn/" 
pcn_learned_f_dir = "./RESULTS/" + env + "_estimates_learned_f_pcn/"   
pcn_learned_fS_dir = "./RESULTS/" + env + "_estimates_learned_fS_pcn/" 

num_points = int(100)
comput_unlearned = True
comput_learned_f = True
comput_learned_fS = True

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
path = "./RESULTS/FNO_results/"
domain_ = domain

test_dataset_x = np.load(meta_data_dir + "test_dataset_x_" + str(num_points) + ".npy")[::env_num]
test_dataset_y = np.load(meta_data_dir + "test_dataset_y_" + str(num_points) + ".npy")[::env_num]
n_test = test_dataset_x.shape[0]
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32)

f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

nnprior_mean = FNO2d(
    modes1=12, modes2=12, width=12, coordinates=domain.mesh.coordinates(),
    mode="residual"
    ).to(device)

## load results
if with_hyper_prior == True:
    if kk_FNO == -1:
        dir_nn = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"    
    else:
        dir_nn = meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_mean_prior"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    if kk_FNO == -1:
        loss_list = np.load(
            meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss_prior.npy"
            )
    else:
        loss_list = np.load(
            meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_loss_prior.npy"
            )
    prior_log_lam = np.load(
        meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_log_lam_prior.npy", 
        )
else:
    if kk_FNO == -1:
        dir_nn = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"
    else:
        dir_nn = meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_mean"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    if kk_FNO == -1:
        loss_list = np.load(
            meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss.npy"
            ) 
    else:
        loss_list = np.load(
            meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_loss.npy"
            )
    prior_log_lam = np.load(
        meta_results_dir + env + str(kk_FNO) + "_" + str(equ_nx) + "_meta_FNO_log_lam.npy", 
        )

fun_truth = fe.Function(domain.function_space)
fun_unlearn = fe.Function(domain.function_space)
fun_learned = fe.Function(domain.function_space)
fun_learned_f = fe.Function(domain.function_space)
sieve = False

"""
Sampling by using pCN for test examples with unlearned prior measures 
"""

prior_unlearn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2, sieve=sieve
    )
prior_unlearn.calculate_eigensystem()

estimates_unlearn = []
final_error_unlearn = []

unlearned_error_list = []

if comput_unlearned == True:
    # for idx in range(n_test):
    for idx in indexs:
        start_time = time.time()
        # param = test_model_params[idx].cpu().detach().numpy().flatten()
        # param = param[coor_transfer["d2v"]]
        tmp = np.array(np.load(path + env + "_result_" + str(idx) + ".npy"))
        param = np.array(tmp[1])  # Extract the true function values
        fun_truth.vector()[:] = np.array(param)
        equ_solver = EquSolver(
            domain_equ=domain, m=fe.Function(domain.function_space), f=f,
            points=test_dataset_x[idx,:].cpu().detach().numpy()
            )
        d_noisy = test_dataset_y[idx, :]

        noise_level_ = noise_level*max(test_dataset_y[idx,:])
        noise = NoiseGaussianIID(dim=len(test_dataset_y[idx,:]))
        noise.set_parameters(variance=noise_level_**2)

        ## setting the Model
        model = ModelDarcyFlow(
            d=d_noisy, domain_equ=domain, prior=prior_unlearn,
            noise=noise, equ_solver=equ_solver
            )
        
        smc = SMC(model, num_particles)
        smc.prepare()
        h0 = 1
        sum_h = 0.0
        num_layer = 0
        h_list = []
        beta = args.beta_pcn
        while 1:
            if sum_h >= 1-1e-5:
                break
            num_layer += 1
            smc.eval_potential_funs(potential_fun=model.loss_residual)
            h = smc.search_h(h=h0) 
            h_list.append(h)
            sum_h += h
            print("h = %4.15f" % h)
            print("sum_h = %4.15f" % sum_h)
            smc.resampling(h=h)
            def phi(u_vec):
                model.update_m(u_vec.flatten(), update_sol=True)
                val = sum_h*model.loss_residual()
                return val
            sampler = pCN(model.prior, phi, beta=beta, save_num=np.int64(1e4))
            smc.transition(sampler=sampler, len_chain=np.int64(len_pcn_chain), info_acc_rate=False)
            acc_mean = np.mean(smc.acc_rates)
            if acc_mean < 0.15:
                beta = beta/2
            elif acc_mean > 0.4:
                beta = min(beta*2, 1)
            print("acc_mean = %.5f" % acc_mean)
            print("beta = %.5f" % beta)
            print("num_layer = %d" % num_layer)
            h0 = min(3*h, 1-sum_h)

            est_mean = fe.Function(domain.function_space)
            est_mean.vector()[:] = np.mean(smc.particles, axis=0)
            err = relative_error_simple(est_mean, fun_truth)
            print("Relative error = %4.4f" % err)

        end_time = time.time()
        time_unlearn = end_time - start_time   
        particles = smc.particles.copy()
        np.save(
            smc_results_dir + env + "_particles_unlearned_" + str(idx) + ".npy", particles
            )
        np.save(
            smc_results_dir + env + "_h_list_unlearned_" + str(idx) + ".npy", np.array(h_list)
            )

"""
Sampling by using pCN for test examples with learned prior measures f(\theta)
"""

prior_learn_f = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2, sieve=sieve
    )
if comput_learned_f == True:
    prior_learn_f.calculate_eigensystem()

estimates_learn_f = []
final_error_learn_f = []

learned_error_list_f = []

learned_mean_f = np.load(
    meta_results_dir + env + str(kk_f) + "_" + str(equ_nx) + "_meta_prior_mean.npy"
    )
learned_var_f = np.load(
    meta_results_dir + env + str(kk_f) + "_" + str(equ_nx) + "_meta_prior_log_lam.npy"
    )

if comput_learned_f == True:
    for idx in indexs:
        start_time = time.time()
        # param = test_model_params[idx].cpu().detach().numpy().flatten()
        # param = param[coor_transfer["d2v"]]
        tmp = np.array(np.load(path + env + "_result_" + str(idx) + ".npy"))
        param = np.array(tmp[1])  # Extract the true function values
        fun_truth.vector()[:] = np.array(param)

        prior_learn_f.update_mean_fun(learned_mean_f)
        prior_learn_f.set_log_lam(learned_var_f)

        equ_solver = EquSolver(
            domain_equ=domain, m=fe.Function(domain.function_space), f=f,
            points=test_dataset_x[idx,:].cpu().detach().numpy()
            )
        d_noisy = test_dataset_y[idx, :]

        noise_level_ = noise_level*max(test_dataset_y[idx,:])
        noise = NoiseGaussianIID(dim=len(test_dataset_y[idx,:]))
        noise.set_parameters(variance=noise_level_**2)

        ## setting the Model
        model = ModelDarcyFlow(
            d=d_noisy, domain_equ=domain, prior=prior_learn_f,
            noise=noise, equ_solver=equ_solver
            )
        
        smc = SMC(model, num_particles)
        smc.prepare()
        h0 = 1
        sum_h = 0.0
        h_list = []
        num_layer = 0
        beta = args.beta_pcn
        while 1:
            if sum_h >= 1-1e-5:
                break
            num_layer += 1
            smc.eval_potential_funs(potential_fun=model.loss_residual)
            h = smc.search_h(h=h0) 
            h_list.append(h)
            sum_h += h
            print("h = %4.15f" % h)
            print("sum_h = %4.15f" % sum_h)
            smc.resampling(h=h)
            def phi(u_vec):
                model.update_m(u_vec.flatten(), update_sol=True)
                val = sum_h*model.loss_residual()
                return val
            sampler = pCN(model.prior, phi, beta=beta, save_num=np.int64(1e4))
            smc.transition(sampler=sampler, len_chain=np.int64(len_pcn_chain), info_acc_rate=False)
            acc_mean = np.mean(smc.acc_rates)
            if acc_mean < 0.15:
                beta = beta/2
            elif acc_mean > 0.4:
                beta = min(beta*2, 1)
            print("acc_mean = %.5f" % acc_mean)
            print("beta = %.5f" % beta)
            print("num_layer = %d" % num_layer)
            h0 = min(3*h, 1-sum_h)

            est_mean = fe.Function(domain.function_space)
            est_mean.vector()[:] = np.mean(smc.particles, axis=0)
            err = relative_error_simple(est_mean, fun_truth)
            print("Relative error = %4.4f" % err)

        end_time = time.time()
        time_learn_f = end_time - start_time    
        particles = smc.particles.copy()
        np.save(
            smc_results_dir + env + "_particles_learned_f_" + str(idx) + ".npy", particles
            )
        np.save(
            smc_results_dir + env + "_h_list_learned_f_" + str(idx) + ".npy", np.array(h_list)
            )

"""
Sampling by using pCN for test examples with learned prior measures f(S;\theta)
"""

prior_learn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2, sieve=sieve
    )
if comput_learned_fS == True:
    prior_learn.calculate_eigensystem()

estimates_learn = []
final_error_learn = []
learned_error_list = []

if comput_learned_fS == True:
    for idx in indexs:
        start_time = time.time()
        # param = test_model_params[idx].cpu().detach().numpy().flatten()
        # param = param[coor_transfer["d2v"]]
        tmp = np.array(np.load(path + env + "_result_" + str(idx) + ".npy"))
        param = np.array(tmp[1])  # Extract the true function values
        fun_truth.vector()[:] = np.array(param)

        # param = output_FNO[idx].cpu().detach().numpy().flatten()
        # param = param[coor_transfer["d2v"]]
        param = np.array(tmp[0])  # Extract the FNO predicted function values
        prior_learn.update_mean_fun(param)
        prior_learn.set_log_lam(prior_log_lam)

        equ_solver = EquSolver(
            domain_equ=domain, m=fe.Function(domain.function_space), f=f,
            points=test_dataset_x[idx,:].cpu().detach().numpy()
            )
        d_noisy = test_dataset_y[idx, :]

        noise_level_ = noise_level*max(test_dataset_y[idx,:])
        noise = NoiseGaussianIID(dim=len(test_dataset_y[idx,:]))
        noise.set_parameters(variance=noise_level_**2)

        ## setting the Model
        model = ModelDarcyFlow(
            d=d_noisy, domain_equ=domain, prior=prior_learn,
            noise=noise, equ_solver=equ_solver
            )
        
        smc = SMC(model, num_particles)
        smc.prepare()
        h0 = 1
        sum_h = 0.0
        h_list = []
        num_layer = 0
        beta = args.beta_pcn
        while 1:
            if sum_h >= 1-1e-5:
                break
            num_layer += 1
            smc.eval_potential_funs(potential_fun=model.loss_residual)
            h = smc.search_h(h=h0) 
            h_list.append(h)
            sum_h += h
            print("h = %4.15f" % h)
            print("sum_h = %4.15f" % sum_h)
            smc.resampling(h=h)
            def phi(u_vec):
                model.update_m(u_vec.flatten(), update_sol=True)
                val = sum_h*model.loss_residual()
                return val
            sampler = pCN(model.prior, phi, beta=beta, save_num=np.int64(1e4))
            smc.transition(sampler=sampler, len_chain=np.int64(len_pcn_chain), info_acc_rate=False)
            acc_mean = np.mean(smc.acc_rates)
            if acc_mean < 0.15:
                beta = beta/2
            elif acc_mean > 0.4:
                beta = min(beta*2, 1)
            print("acc_mean = %.5f" % acc_mean)
            print("beta = %.5f" % beta)
            print("num_layer = %d" % num_layer)
            h0 = min(3*h, 1-sum_h)

            est_mean = fe.Function(domain.function_space)
            est_mean.vector()[:] = np.mean(smc.particles, axis=0)
            err = relative_error_simple(est_mean, fun_truth)
            print("Relative error = %4.4f" % err)

        end_time = time.time()  
        time_learn_fS = end_time - start_time
        particles = smc.particles.copy()
        np.save(
            smc_results_dir + env + "_particles_learned_fS_" + str(idx) + ".npy", particles
            )
        np.save(
            smc_results_dir + env + "_h_list_learned_fS_" + str(idx) + ".npy", np.array(h_list)
            )
    

tmp = env + "_smc_computation_time.txt"
time_file_path = os.path.join(results_fig_table, tmp)
with open(time_file_path, "w") as f:
    f.write(f"time_unlearn: {time_unlearn}\n")
    f.write(f"time_learn_f: {time_learn_f}\n")
    f.write(f"time_learn_fS: {time_learn_fS}\n")






