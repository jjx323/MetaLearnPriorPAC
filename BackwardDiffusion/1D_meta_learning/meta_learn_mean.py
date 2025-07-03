## import necessary packages
import numpy as np
import fenics as fe
import torch
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt
import copy
import pickle
import argparse
import time 

## Add path to the parent directory
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

## Import necessary modules in our programs
from core.model import Domain1D
from core.noise import NoiseGaussianIID
from core.misc import eval_min_max
from BackwardDiffusion.common import EquSolver
from BackwardDiffusion.meta_common import GaussianElliptic2Learn
from BackwardDiffusion.meta_common import Gaussian1DFiniteDifference
from BackwardDiffusion.meta_common import PDEFun, PDEasNet, LossResidual

"""
In the following code, we implement the algorithm that learn the prior mean function directly.
This case corresponds to set the function f(S;\theta) = f(\theta) that did not depends on 
the new dataset. Specifically, the parameters $\theta$ are coefficients of the finite element
basis expansion: 
    f(\theta) = \sum_{k=1}^{N}\theta_k \phi_k(x),
where $\{\phi_k\}_{k=1}^{N}$ are basis functions. 
"""

## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="train prediction function f(theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
args = parser.parse_args()

## Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
os.makedirs(meta_results_dir, exist_ok=True)
env = args.env
with_hyper_prior = True
max_iter = 4000

print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

## domain for solving PDE
equ_nx = 70
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

noise_level = np.load("./DATA/noise_level.npy")

## load the meta_data. 
## meta_data_x: coordinates of measurement points; 
##              a list with elements of np.arrays with different length;
## meta_data_y: the noisy datasets obtained by different parameters generated from one environment distribution
##              a list with elements of np.arrays with different length;
with open(meta_data_dir + env + "_meta_data_x", 'rb') as f: 
    meta_data_x = pickle.load(f)
with open(meta_data_dir + env + "_meta_data_y", 'rb') as f: 
    meta_data_y = pickle.load(f)
n = len(meta_data_x)
T, num_steps = np.load(meta_data_dir + env + "_equation_parameters.npy")
num_steps = np.int64(num_steps)
coordinates = meta_data_x[0]
## in the present setting, for each parameter u, there is only one dataset 
m = meta_data_y[0].shape[0]
print("m = ", m)

## -----------------------------------------------------------------------
## loading the background true parameters for comparision
with open(meta_data_dir + env + "_meta_parameters", 'rb') as f: 
    u_meta = pickle.load(f)
u_meta = np.array(u_meta)
mesh_meta = fe.Mesh(meta_data_dir + env + '_saved_mesh_meta.xml')
V_meta = fe.FunctionSpace(mesh_meta, 'P', 1)

if env == 'simple':
    u_meta_fun = fe.Function(V_meta)
    u_meta_fun.vector()[:] = np.array(u_meta[0])
elif env == 'complex':
    u_meta_fun1 = fe.Function(V_meta)
    u_meta_fun2 = fe.Function(V_meta)
    u_meta_fun1.vector()[:] = np.array(u_meta[0])
    u_meta_fun2.vector()[:] = np.array(u_meta[1])
else:
    raise NotImplementedError("env should be simple or complex")

## -----------------------------------------------------------------------
## constructing equ_solver
if env == 'simple':
    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates]).T, m=u_meta_fun)
    sol = equ_solver.forward_solver()
    sol_fun = fe.Function(domain.function_space)
    sol_fun.vector()[:] = np.array(sol)
elif env == 'complex':
    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates]).T, m=u_meta_fun1)
    sol = equ_solver.forward_solver()
    sol_fun1 = fe.Function(domain.function_space)
    sol_fun1.vector()[:] = np.array(sol)

    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates]).T, m=u_meta_fun2)
    sol = equ_solver.forward_solver()
    sol_fun2 = fe.Function(domain.function_space)
    sol_fun2.vector()[:] = np.array(sol)
else:
    raise NotImplementedError("env should be simple or complex")

## -----------------------------------------------------------------------
## construct the base prior measure that is a Gaussian measure 
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

prior = GaussianElliptic2Learn(
    domain, alpha=0.01, a_fun=fe.Constant(1.0), theta=1.0, 
    mean_fun=None, tensor=False, boundary="Neumann"
    )
learn_var = True#False
prior.trans2learnable(learn_mean=True, learn_var=learn_var)

## -----------------------------------------------------------------------
## construct the hyper-prior measure that is a Gaussian measure 
alpha_hyper_prior, beta_hyper_prior = 0.0, 0.005

hyper_prior_mean = Gaussian1DFiniteDifference(
    nx=equ_nx+1, a_fun=alpha_hyper_prior, alpha=beta_hyper_prior
    )
hyper_prior_mean.trans2learnable(learn_mean=False, learn_a=False)

## the log_gamma is assumed to be a finite-dimensional Gaussian
def make_hyper_prior_log_gamma(mean_vec, weight=0.01):
    mean_vec = copy.deepcopy(mean_vec)
    weight = weight
    
    def hyper_prior_log_gamma(val):
        temp = val - mean_vec
        return weight*torch.sum(temp*temp)
    
    return hyper_prior_log_gamma

if learn_var == True:
    hyper_prior_log_gamma = make_hyper_prior_log_gamma(prior.log_gamma_learn, weight=0.001)

## -----------------------------------------------------------------------
## Set the noise 
noise_level_ = noise_level
noise = NoiseGaussianIID(dim=len(meta_data_y[0]))
noise.set_parameters(variance=noise_level_**2)
noise.to_tensor()

loss_residual = LossResidual(noise)
## -----------------------------------------------------------------------
## transfer the PDEs as a layer of the neural network that makes the loss.backward() useable
pde_fun = PDEFun.apply 

## L: the number of samples used to approximate Z_m(S, P_{S}^{\theta}) 
L = 5
np.save(meta_data_dir + "L_for_Zm", L)

## batch_size: only use batch_size number of datasets for each iteration 
batch_size = 20

if learn_var == True:
    optimizer = torch.optim.AdamW(
        [{"params": prior.mean_vec_learn, "lr": 0.01},
         {"params": prior.log_gamma_learn, "lr": 0.0}],
        #lr=0.2,
        weight_decay=0.00
        )  

if learn_var == False:
    optimizer = torch.optim.AdamW(
        [{"params": prior.mean_vec_learn, "lr": 0.01}],
        #lr=0.2,
        weight_decay=0.00
        )   


loss_list = []

weight_of_lnZ = n/((m+1)*batch_size)


def set_learning_rate(optimizer, group_index, new_lr):
    optimizer.param_groups[group_index]['lr'] = new_lr
    print(f"Learning rate of group {group_index} set to {new_lr}")

start_time = time.time()
for kk in range(max_iter):
    
    optimizer.zero_grad()
    lnZ = torch.zeros(1)
    
    batch = np.random.choice(np.arange(n), batch_size)
    panduan = 0
    for itr in batch:
        coordinates = meta_data_x[itr]
        
        ## -------------------------------------------------------------------
        ## Since the data dimension is different for different task, we need 
        ## different loss functions. 
        noise_level_ = noise_level
        noise = NoiseGaussianIID(dim=len(meta_data_y[itr]))
        noise.set_parameters(variance=noise_level_**2)
        noise.to_tensor()

        loss_residual = LossResidual(noise)
        ## -------------------------------------------------------------------
        ## for each dataset, we need to reconstruct equ_solver since the points are changed
        if panduan == 0:
            equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                                    points=np.array([coordinates]).T, m=None)
            pdeasnet = PDEasNet(pde_fun, equ_solver)
            panduan = 1
        else:
            pdeasnet.equ_solver.update_points(np.array([coordinates]).T)
        
        loss_res_L = torch.zeros(L)
        
        targets = torch.tensor(meta_data_y[itr], dtype=torch.float32)
        
        for ii in range(L):
            ul = prior.generate_sample_learn()
            preds = pdeasnet(ul)        
            loss_res_L[ii] = -loss_residual(preds, targets)
        
        ## torch.logsumexp is used to avoid possible instability of computations
        lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                  - torch.log(torch.tensor(L, dtype=torch.float32))
                  
    if with_hyper_prior == True:
        prior1 = hyper_prior_mean.evaluate_CM_inner(prior.mean_vec_learn)
        if learn_var == True:
            prior1 += hyper_prior_log_gamma(prior.log_gamma_learn)
        nlogP = -weight_of_lnZ*lnZ + prior1 
    else:
        nlogP = -weight_of_lnZ*lnZ
        
    nlogP.backward()
    
    if kk == 2000:
        set_learning_rate(optimizer, 0, 0.01)
        set_learning_rate(optimizer, 1, 0.001)
    if kk == 3000:
        set_learning_rate(optimizer, 0, 0.01)
        set_learning_rate(optimizer, 1, 0.001)   
            
    optimizer.step() 
    loss_list.append(nlogP.item())
    
    if kk % 10 == 0:
        if with_hyper_prior == True:
            print("Iter = %4d/%d, nlogP = %.4f, lnZ = %.4f, prior = %.4f" % 
                  (kk, max_iter, nlogP.item(), -weight_of_lnZ*lnZ.item(), prior1.item()))
        else:
            print("Iter = %4d/%d, nlogP = %.4f" % (kk, max_iter, nlogP.item()))
        fun = fe.Function(domain.function_space)
        fun.vector()[:] = np.array(prior.mean_vec_learn.cpu().detach().numpy())
        equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                                points=np.array([coordinates]).T, m=fun)
        sol_est = equ_solver.forward_solver()
        sol_fun_est = fe.Function(domain.function_space)
        sol_fun_est.vector()[:] = np.array(sol_est)
        
        if env == 'simple':
            min_val, max_val = eval_min_max([
                u_meta_fun.vector()[:], sol_fun.vector()[:], 
                fun.vector()[:], sol_fun_est.vector()[:]
                ])
            plt.figure()
            fe.plot(u_meta_fun, label="one_meta_u")
            fe.plot(sol_fun, label="sol_meta_u")
            fe.plot(fun, label="estimated_mean")
            fe.plot(sol_fun_est, label="sol_estimated_meta")
            plt.ylim([min_val-0.1, max_val+0.1])
            plt.legend()
        elif env == 'complex':
            min_val, max_val = eval_min_max([
                u_meta_fun1.vector()[:], sol_fun1.vector()[:], 
                u_meta_fun2.vector()[:], sol_fun2.vector()[:], 
                fun.vector()[:], sol_fun_est.vector()[:]
                ])
            plt.figure()
            fe.plot(u_meta_fun1, label="one_meta_u1")
            fe.plot(sol_fun1, label="sol_meta_u1")
            fe.plot(u_meta_fun2, label="one_meta_u2")
            fe.plot(sol_fun2, label="sol_meta_u2")
            fe.plot(fun, label="estimated_mean")
            fe.plot(sol_fun_est, label="sol_estimated_meta")
            plt.ylim([min_val-0.1, max_val+0.1])
            plt.legend()
        else:
            raise NotImplementedError("env should be simple or complex")
        if with_hyper_prior == True:
            os.makedirs(meta_results_dir + env + "_figs_prior/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs_prior/" + 'fig' + str(kk) + '.png')
        else:
            os.makedirs(meta_results_dir + env + "_figs/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs/" + 'fig' + str(kk) + '.png')
        plt.close()

        del nlogP, lnZ, prior1, loss_residual, loss_res_L


## results
if with_hyper_prior == True:
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_mean_prior", 
        prior.mean_vec_learn.detach().numpy()
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_loss_prior", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_log_gamma_prior", 
        prior.log_gamma_learn.detach().numpy()
        )
else:
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_mean", 
        prior.mean_vec_learn.detach().numpy()
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_loss", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_log_gamma", 
        prior.log_gamma_learn.detach().numpy()
        )

# ## draw
# u_meta_fun = fe.Function(V_meta)
# u_meta_fun.vector()[:] = np.array(u_meta[0])
#
# plt.figure()
# uu = fe.Function(domain.function_space)
# for itr in range(len(u_meta)):
#     u_meta_fun.vector()[:] = np.array(u_meta[itr])
#     uu = fe.interpolate(u_meta_fun, domain.function_space)
#     plt.plot(uu.vector()[:], alpha=0.1)
# plt.plot(prior.mean_vec_learn.detach().numpy())
# plt.show()

end_time = time.time()
print("Time used: %.2f seconds" % (end_time - start_time))
if env == "simple":
    with open('record_time_simple.txt', 'a') as file:
        file.write(env  + ' mean ' + str(end_time - start_time) + '\n')  # 追加到文件末尾
if env == "complex":
    with open('record_time_complex.txt', 'a') as file:
        file.write(env  + ' mean ' + str(end_time - start_time) + '\n')  # 追加到文件末尾





