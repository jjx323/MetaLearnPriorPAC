## import necessary packages
import numpy as np
import fenics as fe
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt
import torch
import copy
import time
import argparse

import os
import sys
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.noise import NoiseGaussianIID
from core.misc import load_expre
from core.misc import construct_measurement_matrix

from SteadyStateDarcyFlow.common import EquSolver
from SteadyStateDarcyFlow.MLcommon import GaussianElliptic2Torch, \
        GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual, PriorFun, LpLoss


## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
parser.add_argument('--k_start', type=int, default=0, help='k_start is the initial iteration number')
args = parser.parse_args()

## Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
os.makedirs(meta_results_dir, exist_ok=True)

env = args.env

if env == "complex":
    env_num = 1
elif env == "simple":
    env_num = 2
with_hyper_prior = True
# with_hyper_prior = False
max_iter = int(7500)
device = "cpu"
# device = "cuda"

noise_level = 0.01
num = np.arange(1, 100)
num_points = []
for idx_num in num:
    num_points.append(idx_num*25)
num_points = int(num_points[3])
print("num_points: ", num_points)

k_start = args.k_start # if we need not load previous data, we just need to set k_start = int(0)
print("k_start = ", k_start)
print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
## d2v is used to transfer the grid coordinates. 
d2v = np.array(fe.dof_to_vertex_map(domain.function_space), dtype=np.int64)
v2d = np.array(fe.vertex_to_dof_map(domain.function_space), dtype=np.int64)
coor_transfer = {"d2v": d2v, "v2d": v2d}

## ----------------------------------------------------------------------------
## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

def trans_model_data(model_params, V1, V2):
    n, ll = model_params.shape
    params = np.zeros((n, V2.dim()))
    fun1 = fe.Function(V1)
    for itr in range(n):
        fun1.vector()[:] = np.array(model_params[itr, :])
        fun2 = fe.interpolate(fun1, V2)
        params[itr, :] = np.array(fun2.vector()[:])
    return np.array(params)

## load the train model, train dataset x and train dataset y 
train_model_params = np.load(meta_data_dir + "train_model_params.npy")[::env_num]
train_dataset_x = np.load(meta_data_dir + "train_dataset_x_" + str(num_points) + ".npy")[::env_num]
train_dataset_y = np.load(meta_data_dir + "train_dataset_y_" + str(num_points) + ".npy")[::env_num]
## n is the number of the training data
n = train_dataset_x.shape[0]
m = train_dataset_y.shape[1]

train_model_params = trans_model_data(
    train_model_params, V_truth, domain.function_space
    )

## ----------------------------------------------------------------------------
f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)


train_model_params = torch.tensor(train_model_params, dtype=torch.float32)
train_model_params = train_model_params[:, v2d].reshape(n, equ_nx+1, equ_nx+1)
train_dataset_x = torch.tensor(train_dataset_x, dtype=torch.float32)


test_model_params = np.load(meta_data_dir + "test_model_params.npy")[::env_num]
test_dataset_x = np.load(meta_data_dir + "test_dataset_x_" + str(num_points) + ".npy")[::env_num]
test_dataset_y = np.load(meta_data_dir + "test_dataset_y_" + str(num_points) + ".npy")[::env_num]

n_test = test_dataset_x.shape[0]

test_model_params = trans_model_data(
    test_model_params, V_truth, domain.function_space
    )

test_model_params = torch.tensor(test_model_params, dtype=torch.float32)
test_model_params = test_model_params[:, v2d].reshape(n_test, equ_nx+1, equ_nx+1)
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32)

## -----------------------------------------------------------------------
## construct the base prior measure that is a Gaussian measure 
small_n = equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior.calculate_eigensystem()
if k_start > 0:
    prior.mean_vec = np.load(
        meta_results_dir + env + str(k_start) + "_" + str(equ_nx) + "_meta_prior_mean.npy"
        )
    prior.log_lam = np.load(
        meta_results_dir + env + str(k_start) + "_" + str(equ_nx) + "_meta_prior_log_lam.npy"
        )
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)
prior.learnable_mean() 
prior.learnable_loglam()

## construct interpolation matrix
coor = domain_.mesh.coordinates()
# v2d = fe.vertex_to_dof_map(self.domain_.function_space)
d2v_ = fe.dof_to_vertex_map(domain_.function_space)
## full to small matrix
f2sM = construct_measurement_matrix(coor[d2v_], domain.function_space).todense()
f2sM = torch.tensor(f2sM, dtype=torch.float32).to(torch.device(device))

coor = domain.mesh.coordinates()
# v2d = fe.vertex_to_dof_map(self.domain.function_space)
d2v_ = fe.dof_to_vertex_map(domain.function_space)
## small to full matrix
s2fM = construct_measurement_matrix(coor[d2v_], domain_.function_space).todense()
s2fM = torch.tensor(s2fM, dtype=torch.float32).to(torch.device(device))
mesh_transfer = {"s2fM": s2fM, "f2sM": f2sM}

hyper_prior_mean_ = GaussianElliptic2Torch( 
    domain=domain, alpha=0.1, a_fun=fe.Constant(0.01), theta=1.0, 
    boundary="Neumann"
    )
hyper_prior_mean_.trans2torch(device=device)
hyper_prior_mean = PriorFun.apply


def make_hyper_prior_loglam(mean_vec, weight=0.01):
    mean_vec = copy.deepcopy(mean_vec)
    weight = weight
    
    def hyper_prior_loglam(val):
        temp = val - mean_vec
        return weight*torch.sum(temp*temp)
    
    return hyper_prior_loglam
    
hyper_prior_loglam = make_hyper_prior_loglam(prior.log_lam, weight=0.01)

## randomly select some datasets (for repeatable, we fix the selection here)
rand_idx = [0, 1, 2, 3]

## -----------------------------------------------------------------------
## Set the noise 
noise_level_ = noise_level*max(train_dataset_y[0])
noise = NoiseGaussianIID(dim=len(train_dataset_y[0]))
noise.set_parameters(variance=noise_level_**2)
noise.to_torch(device=device)

loss_residual = LossResidual(noise)

equ_params = {"domain": domain, "f": f, "pde_fun": PDEFun.apply}

loss_lp = LpLoss()

def eva_grad_norm(model):
    total_norm = []
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = torch.abs(p.grad.detach().data).max()
        total_norm.append(param_norm)
    total_norm = max(total_norm)
    return total_norm


## -----------------------------------------------------------------------
## transfer the PDEs as a layer of the neural network that makes the loss.backward() useable
pde_fun = PDEFun.apply 

## L: the number of samples used to approximate Z_m(S, P_{S}^{\theta}) 
L = 10

## batch_size: only use batch_size number of datasets for each iteration 
batch_size = 10

optimizer = torch.optim.AdamW(
    [{"params": prior.mean_vec_torch, "lr": 1e-3}, 
     {"params": prior.log_lam, "lr": 1e-3}],
    weight_decay=0.00
    )  

loss_list = []

weight_of_lnZ = n/((m+1)*batch_size)
fun = fe.Function(domain.function_space)  ## construct a function used for plotting
k_start = k_start + 1

start_time = time.time()
for kk in range(k_start, max_iter):

    optimizer.zero_grad()
    lnZ = torch.zeros(1).to(device)
    
    batch = np.random.choice(np.arange(n), batch_size)
    panduan = 0
    start = time.time()
    for itr in batch:
        points = train_dataset_x[itr, :]
        
        ## -------------------------------------------------------------------
        ## Since the data dimension is different for different task, we need 
        ## different loss functions. 
        noise_level_ = noise_level*max(train_dataset_y[itr,:])
        noise = NoiseGaussianIID(dim=len(train_dataset_y[itr,:]))
        noise.reset(dim=len(train_dataset_y[itr,:]))
        noise.set_parameters(variance=noise_level_**2)
        noise.to_torch(device=device)

        loss_residual = LossResidual(noise)
        loss_residual.update(noise)
        ## -------------------------------------------------------------------
        ## for each dataset, we need to reconstruct equ_solver since the points are changed
        if panduan == 0:
            equ_solver = EquSolver(
                domain_equ=domain, points=points, m=fe.Constant(0.0), f=f
                )

            pdeasnet = PDEasNet(pde_fun, equ_solver)
            panduan = 1
        else:
            pdeasnet.equ_solver.update_points(points)
        
        loss_res_L = torch.zeros(L).to(device)
        
        targets = torch.tensor(train_dataset_y[itr], dtype=torch.float32).to(device)
        
        for ii in range(L):
            ul = prior.generate_sample()
            ## By experiments, I think the functions provided by CuPy (solving
            ## Ax=b with A is a large sparse matrix) are not efficient compared 
            ## with the cpu version given by SciPy. 
            preds = pdeasnet(ul.cpu())
            preds = preds.to(targets.device)
            val = -loss_residual(preds, targets)
            loss_res_L[ii] = val
        
        ## torch.logsumexp is used to avoid possible instability of computations
        lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                  - torch.log(torch.tensor(L, dtype=torch.float32)).to(device)
                  
    if with_hyper_prior == True:
        # prior1 = hyper_prior_mean.evaluate_CM_inner(prior.mean_vec_torch)
        prior1 = hyper_prior_mean(prior.mean_vec_torch, hyper_prior_mean_)
        prior1 += hyper_prior_loglam(prior.log_lam)
        nlogP = -weight_of_lnZ*lnZ + prior1 
    else:
        nlogP = -weight_of_lnZ*lnZ
        
    nlogP.backward()
    
    if kk % 100 == 0:
        for g in optimizer.param_groups:
            g["lr"] = g["lr"]*1.0
            
    optimizer.step()
    end = time.time()
    
    if kk % 10 == 0:
        print("Iter = %4d/%d, nlogP = %.4f, term1 = %.4f, prior1 = %.4f" 
              % (kk, max_iter, nlogP.item(), (-weight_of_lnZ*lnZ).item(), \
                  prior1.item()))
        print("Time: ", end-start)
        fun.vector()[:] = np.array(prior.mean_vec_torch.cpu().detach().numpy())
        
        plt.figure()
        fig = fe.plot(fun)
        plt.colorbar(fig)
        os.makedirs(meta_results_dir + env + "_figs_prior/", exist_ok=True)
        plt.savefig(meta_results_dir + env + "_figs_prior/" + 'fig' + str(kk) + '.png')
        plt.close()

    if kk % 100 == 0 and  kk > 0:
        ## save results
        np.save(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_prior_mean", 
            prior.mean_vec_torch.cpu().detach().numpy()
            )
        np.save(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_prior_log_lam",
            prior.log_lam.cpu().detach().numpy()   
            )
        
    
    del nlogP


end_time = time.time()
print("Time used: %.2f seconds" % (end_time - start_time))
with open('record_time.txt', 'a') as file:
    file.write(env  + ' FNO ' + str(end_time - start_time) + '\n')  # 追加到文件末尾

























