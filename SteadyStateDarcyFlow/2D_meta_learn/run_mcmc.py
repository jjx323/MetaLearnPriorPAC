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
from core.sample import pCN
from core.misc import load_expre, relative_error_simple, construct_measurement_matrix

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss, file_process


"""
In the following, we compare the performance by MCMC with different prior measures.
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
parser.add_argument('--length_total', type=int, default=np.int64(1e6), help='length_total of MCMC')
parser.add_argument('--beta_pcn', type=float, default=0.05, help='beta of pCN')
args = parser.parse_args()

sieve = False

## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
results_fig_table = "./RESULTS-PAPER/"
os.makedirs(results_fig_table, exist_ok=True)

env = args.env
if env == "complex":
    env_num = 1
elif env == "simple":
    env_num = 2
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

num_points = int(100)
comput_unlearned = True
comput_learned_f = True
comput_learned_fS = True

if env == "simple":
    n_start, n_end = 0, 1
elif env == "complex":
    n_start, n_end = 0, 2
else:
    raise ValueError("env should be complex or simple")

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

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
global num_

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
    for idx in range(n_start, n_end):
        pcn_unlearn_dir = "./RESULTS/" + env + "_est_unlearn_pcn_" + str(idx) + "/" 
        
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
        ## define the function phi used in pCN
        def phi(u_vec):
            model.update_m(u_vec.flatten(), update_sol=True)
            return model.loss_residual()
        
        pcn_dir = pcn_unlearn_dir
        pcn = pCN(model.prior, phi, beta=args.beta_pcn, save_num=np.int64(1e4), path=pcn_dir)
        num_ = 0
        class CallBack(object):
            def __init__(self, num_=0, function_space=domain.function_space, truth=fun_truth,
                         length_total=args.length_total, phi=phi):
                self.num_ = num_
                self.fun = fe.Function(function_space)
                self.truth = truth
                self.num_fre = 1000
                self.length_total = length_total
                self.phi = phi
                
            def callback_fun(self, params):
                # params = [uk, iter_num, accept_rate, accept_num]
                num = params[1]
                if num % self.num_fre == 0:
                    print("-"*70)
                    print(str(idx+1) + "th test example", end='; ')
                    print('Iteration number = %d/%d' % (num, self.length_total), end='; ')
                    print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
                    print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
                    self.num_ = params[3]
                    print('Phi = %4.4f' % self.phi(params[0]))
                
        callback = CallBack()
        ## start sampling
        acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=args.length_total, callback=callback.callback_fun) 
        acc_rate_unlearn = acc_rate
        file_process(results_fig_table + env + '_acc_rate_pcn.txt')
        with open(results_fig_table + 'acc_rate_pcn.txt', 'a') as ff:
            ff.write(f"acc_rate_unlearn: {acc_rate_unlearn}\n")


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
    for idx in range(n_start, n_end):
        pcn_learned_f_dir = "./RESULTS/" + env + "_estimates_learned_f_pcn_" + str(idx) + "/"  
        
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
        ## define the function phi used in pCN
        def phi(u_vec):
            model.update_m(u_vec.flatten(), update_sol=True)
            return model.loss_residual()
        
        pcn_dir = pcn_learned_f_dir   
        pcn = pCN(model.prior, phi, beta=args.beta_pcn, save_num=np.int64(1e4), path=pcn_dir)
        num_ = 0
        class CallBack(object):
            def __init__(self, num_=0, function_space=domain.function_space, truth=fun_truth,
                         length_total=args.length_total, phi=phi):
                self.num_ = num_
                self.fun = fe.Function(function_space)
                self.truth = truth
                self.num_fre = 1000
                self.length_total = length_total
                self.phi = phi
                
            def callback_fun(self, params):
                # params = [uk, iter_num, accept_rate, accept_num]
                num = params[1]
                if num % self.num_fre == 0:
                    print("-"*70)
                    print(str(idx+1) + "th test example", end='; ')
                    print('Iteration number = %d/%d' % (num, self.length_total), end='; ')
                    print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
                    print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
                    self.num_ = params[3]
                    print('Phi = %4.4f' % self.phi(params[0]))
                
        callback = CallBack()
        ## start sampling
        acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=args.length_total, callback=callback.callback_fun)
        acc_rate_learned_f = acc_rate
        with open(results_fig_table + 'acc_rate_pcn.txt', 'a') as ff:
            ff.write(f"acc_rate_learned_f: {acc_rate_learned_f}\n")

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
    for idx in range(n_start, n_end):
        pcn_learned_fS_dir = "./RESULTS/" + env + "_estimates_learned_fS_pcn_" + str(idx) + "/"   
        
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
        ## define the function phi used in pCN
        def phi(u_vec):
            model.update_m(u_vec.flatten(), update_sol=True)
            return model.loss_residual()
        
        pcn_dir = pcn_learned_fS_dir
        pcn = pCN(model.prior, phi, beta=args.beta_pcn, save_num=np.int64(1e4), path=pcn_dir)
        num_ = 0
        class CallBack(object):
            def __init__(self, num_=0, function_space=domain.function_space, truth=fun_truth,
                         length_total=args.length_total, phi=phi):
                self.num_ = num_
                self.fun = fe.Function(function_space)
                self.truth = truth
                self.num_fre = 1000
                self.length_total = length_total
                self.phi = phi
                
            def callback_fun(self, params):
                # params = [uk, iter_num, accept_rate, accept_num]
                num = params[1]
                if num % self.num_fre == 0:
                    print("-"*70)
                    print(str(idx+1) + "th test example", end='; ')
                    print('Iteration number = %d/%d' % (num, self.length_total), end='; ')
                    print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
                    print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
                    self.num_ = params[3]
                    print('Phi = %4.4f' % self.phi(params[0]))
                
        callback = CallBack()
        ## start sampling
        acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=args.length_total, callback=callback.callback_fun)
        acc_rate_learned_fS = acc_rate
        with open(results_fig_table + 'acc_rate_pcn.txt', 'a') as ff:
            ff.write(f"acc_rate_learned_fS: {acc_rate_learned_fS}\n")
    








