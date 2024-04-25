## import necessary packages
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import time
import argparse

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing, print_my
from core.misc import construct_measurement_matrix, trans2spnumpy, spnumpy2sptorch, \
    sptorch2spnumpy, eval_min_max

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import GaussianElliptic2Torch, \
        GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual, PriorFun, \
        FNO2d, Dis2Fun, UnitNormalization, LpLoss, LossFun, HyperPrior, HyperPriorAll, \
        ForwardProcessNN, ForwardProcessPDE, ForwardPrior, AdaptiveLossFun


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
init_dis2funC = False

k_start = int(args.k_start) # if we need not load previous data, we just need to set k_start = int(0)
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

class Dis2FunC(object):
    def __init__(self, domainS, domainF, f, noise_level, dataset_x, dataset_y):
        self.equ_solver = EquSolver(
            domain_equ=domainS, m=fe.Function(domainS.function_space), f=f, 
            points=np.array([[0,0]])
            )
        
        self.noise_level = noise_level
        noise_level_ = self.noise_level*max(abs(dataset_y[0,:]))
        self.noise = NoiseGaussianIID(dim=len(dataset_y[0,:]))
        self.noise.set_parameters(variance=noise_level_**2)
        
        d_noisy = dataset_y[0, :]
        
        self.prior = GaussianFiniteRank(
            domain=domainS, domain_=domainS, alpha=0.1, beta=10, s=2
            )
        self.prior.calculate_eigensystem()
        
        self.model = ModelDarcyFlow(
            d=d_noisy, domain_equ=domainS, prior=self.prior,
            noise=self.noise, equ_solver=self.equ_solver
            )
        
        self.funS = fe.Function(domainS.function_space)
        self.domainF = domainF
        self.newton_cg = None
    
    def reset_points(self, points):
        self.model.update_S(points)
        
    def __call__(self, d, max_iter=2, cg_max=30):
        ## set optimizer NewtonCG
        self.model.update_d(d)
        
        noise_level_ = self.noise_level*max(abs(d))
        if self.newton_cg is None:
            self.model.noise = NoiseGaussianIID(dim=len(d))
        self.model.noise.set_parameters(variance=noise_level_**2)
        
        if self.newton_cg is None:
            self.newton_cg = NewtonCG(model=self.model)
        else:
            self.newton_cg.re_init()
            
        max_iter = max_iter
        
        loss_pre = self.model.loss()[0]
        for itr in range(max_iter):
            self.newton_cg.descent_direction(cg_max=cg_max, method='cg_my')
            # newton_cg.descent_direction(cg_max=30, method='bicgstab')
            # print(newton_cg.hessian_terminate_info)
            self.newton_cg.step(method='armijo', show_step=False)
            if self.newton_cg.converged == False:
                break
            loss = self.model.loss()[0]
            # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-3*loss:
                # print("Iteration stoped at iter = %d" % itr)
                break 
            loss_pre = loss
            
        self.funS.vector()[:] = np.array(self.newton_cg.mk.copy())
        funF = fe.interpolate(self.funS, self.domainF.function_space)
        
        return np.array(funF.vector()[:])
        
## Transfer discrete observation points to the functions defined on domain
equ_nxS = 20
domainS = Domain2D(nx=equ_nxS, ny=equ_nxS, mesh_type='P', mesh_order=1)

## real data
dis2fun = Dis2Fun(domain=domain, points=train_dataset_x[0, :], alpha=0.01)
train_dataset_Sy = []
for itr in range(n):
    dis2fun.reset_points(train_dataset_x[itr, :])
    Sy = dis2fun(train_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    train_dataset_Sy.append(Sy)
train_dataset_Sy = np.array(train_dataset_Sy)
np.save(meta_data_dir + "train_dataset_Sy_" + str(num_points), train_dataset_Sy)

## initial inversion data
if init_dis2funC == True:
    dis2funC = Dis2FunC(
        domainS=domainS, domainF=domain, f=f, noise_level=noise_level, 
        dataset_x=train_dataset_x, dataset_y=train_dataset_y
        )
    train_dataset_Sy0 = []
    for itr in range(n):
        dis2funC.reset_points(train_dataset_x[itr, :])
        Sy0 = dis2funC(train_dataset_y[itr], max_iter=10, cg_max=10)[v2d].reshape(equ_nx+1, equ_nx+1)
        Sy0 = Sy0.reshape(equ_nx+1, equ_nx+1)
        train_dataset_Sy0.append(Sy0)
        print("itr = ", itr)
    train_dataset_Sy0 = np.array(train_dataset_Sy0)
    np.save(meta_data_dir + "train_dataset_Sy0_" + str(num_points), train_dataset_Sy0)
train_dataset_Sy0 = np.load(meta_data_dir + "train_dataset_Sy0_" + str(num_points) + ".npy")[::env_num]
train_dataset_Sy0 = torch.tensor(train_dataset_Sy0, dtype=torch.float32)
normalize_train_Sy0 = UnitNormalization(train_dataset_Sy0)

train_model_params = torch.tensor(train_model_params, dtype=torch.float32)
train_model_params = train_model_params[:, v2d].reshape(n, equ_nx+1, equ_nx+1)
normalize_model_params = UnitNormalization(train_model_params)
train_dataset_x = torch.tensor(train_dataset_x, dtype=torch.float32)
train_dataset_Sy = torch.tensor(train_dataset_Sy, dtype=torch.float32)
normalize_train_Sy = UnitNormalization(train_dataset_Sy)

test_model_params = np.load(meta_data_dir + "test_model_params.npy")[::env_num]
test_dataset_x = np.load(meta_data_dir + "test_dataset_x_" + str(num_points) + ".npy")[::env_num]
test_dataset_y = np.load(meta_data_dir + "test_dataset_y_" + str(num_points) + ".npy")[::env_num]
n_test = test_dataset_x.shape[0]

test_model_params = trans_model_data(
    test_model_params, V_truth, domain.function_space
    )

## real data
test_dataset_Sy = []
for itr in range(n_test):
    dis2fun.reset_points(test_dataset_x[itr, :])
    Sy = dis2fun(test_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    test_dataset_Sy.append(Sy)
test_dataset_Sy = np.array(test_dataset_Sy)  
np.save(meta_data_dir + "test_dataset_Sy_" + str(num_points), test_dataset_Sy)

## initial inversion data
if init_dis2funC == True:
    test_dataset_Sy0 = []
    for itr in range(n_test):
        dis2funC.reset_points(test_dataset_x[itr, :])
        Sy0 = dis2funC(test_dataset_y[itr], max_iter=10, cg_max=10)[v2d].reshape(equ_nx+1, equ_nx+1)
        Sy0 = Sy0.reshape(equ_nx+1, equ_nx+1)
        test_dataset_Sy0.append(Sy0)
        print("itr = ", itr)
    test_dataset_Sy0 = np.array(test_dataset_Sy0)  
    np.save(meta_data_dir + "test_dataset_Sy0_" + str(num_points), test_dataset_Sy0)
test_dataset_Sy0 = np.load(meta_data_dir + "test_dataset_Sy0_" + str(num_points) + ".npy")[::env_num]
test_dataset_Sy0 = torch.tensor(test_dataset_Sy0, dtype=torch.float32)
normalize_test_Sy0 = UnitNormalization(test_dataset_Sy0)

test_model_params = torch.tensor(test_model_params, dtype=torch.float32)
test_model_params = test_model_params[:, v2d].reshape(n_test, equ_nx+1, equ_nx+1)
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32)
test_dataset_Sy = torch.tensor(test_dataset_Sy, dtype=torch.float32)
normalize_test_Sy = UnitNormalization(test_dataset_Sy)

## -----------------------------------------------------------------------
## construct the base prior measure that is a Gaussian measure 
small_n = equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
# prior.trans2torch(device=device)
# prior.learnable_mean() 
# prior.learnable_loglam()

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

## -----------------------------------------------------------------------
## Set the neural network for learning the prediction policy of the mean functions.
## The parameters are specified similar to the original paper proposed FNO.
nnprior_mean = FNO2d(
    modes1=12, modes2=12, width=12, coordinates=domain.mesh.coordinates(), 
    mode="residual"
    ).to(device)

print("Loading data ......")
## load data
loss_list = []
if k_start >= 1:
    kk = k_start
    if with_hyper_prior == True:
        dir_nn = meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_mean_prior"
        nnprior_mean.load_state_dict(torch.load(dir_nn))
        loss_pre = np.load(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_loss_prior.npy"
            )
        for loss in loss_pre:
            loss_list.append(loss)
        prior.set_log_lam(
            np.load(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_log_lam_prior.npy" 
            ))
        prior.trans2torch(device=device)
        prior.learnable_loglam()
    else:
        dir_nn = meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_mean"
        nnprior_mean.load_state_dict(torch.load(dir_nn))
        loss_pre = np.load(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_loss.npy"
            )
        for loss in loss_pre:
            loss_list.append(loss)
        prior.set_log_lam(
            np.load(
            meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_log_lam.npy" 
            ))
        prior.trans2torch(device=device)
        prior.learnable_loglam()
else:
    prior.trans2torch(device=device)
    prior.learnable_loglam()

## -----------------------------------------------------------------------

hyper_prior_mean = HyperPrior(
    measure=GaussianElliptic2Torch( 
        domain=domain, alpha=0.1, a_fun=fe.Constant(0.01), theta=1.0, 
        boundary="Neumann"
        ), 
    fun_norm=PriorFun.apply
    )
hyper_prior_mean.to_torch(device=device)


def make_hyper_prior_loglam(mean_vec, weight=0.01):
    mean_vec = copy.deepcopy(mean_vec)
    weight = weight
    
    def hyper_prior_loglam(val):
        temp = val - mean_vec
        return weight*torch.sum(temp*temp)
    
    return hyper_prior_loglam

hyper_prior_loglam = HyperPrior(
    measure=None, 
    fun_norm=make_hyper_prior_loglam(prior.log_lam, weight=0.01)
    )    

hyper_prior = HyperPriorAll([hyper_prior_mean, hyper_prior_loglam])

# hyper_prior = HyperPriorAll([hyper_prior_mean])

## randomly select some datasets (for repeatable, we fix the selection here)
rand_idx = [0, 1, 2, 3]#np.random.choice(n, 4)

## -----------------------------------------------------------------------
## Set the noise 
noise_level_ = noise_level*max(train_dataset_y[0])
noise = NoiseGaussianIID(dim=len(train_dataset_y[0]))
noise.set_parameters(variance=noise_level_**2)
noise.to_torch(device=device)

loss_residual = LossResidual(noise)

## L: the number of samples used to approximate Z_m(S, P_{S}^{\theta}) 
L = 10

## batch_size: only use batch_size number of datasets for each iteration 
batch_size = 20

weight_of_lnZ = n/((m+1)*batch_size)

normalize_model_params.to(torch.device(device))
normalize_train_Sy.to(torch.device(device))
normalize_test_Sy.to(torch.device(device))

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


forward_nn = ForwardProcessNN(
    nn_model=nnprior_mean, coor_transfer=coor_transfer, 
    normalize_data=normalize_train_Sy, normalize_param=normalize_model_params, 
    normalize_data0=normalize_train_Sy0, 
    # normalize_data=None, normalize_param=None, normalize_data0=None,
    device=device
    )

forward_pde = ForwardProcessPDE(
    noise=noise, equ_params=equ_params, prior=prior, mesh_transfer=mesh_transfer, 
    noise_level=noise_level, weight=weight_of_lnZ, L=10, device=device
    )

forward_prior = ForwardPrior(
    prior=hyper_prior, forward_nn=forward_nn, rand_idx=rand_idx, 
    dataset=train_dataset_Sy, is_log_lam=True, device=device,
    dataset0=train_dataset_Sy0
    )

forward_process = {"forward_nn": forward_nn, "forward_pde": forward_pde, \
                   "forward_prior": forward_prior}

loss_fun = LossFun(
    forward_process=forward_process, prior_base=prior, with_hyper_prior=True
    )

adaptive_loss_fun = AdaptiveLossFun(
    forward_process=forward_process
    )

optimizer = torch.optim.AdamW(
    [{"params": nnprior_mean.parameters(), "lr": 1e-3}, 
      {"params": prior.log_lam, "lr": 1e-3}],
    weight_decay=0.00
    )

equ1 = EquSolver(
    domain_equ=domain, points=np.array([[0,0]]), m=fe.Constant(0.0), f=f
    )

if k_start != 0:
    k_start = int(k_start + 1)

start = time.time()
print("Start learning ......")
for kk in range(k_start, max_iter):
    
    if kk > 100 and kk <= 500:
        optimizer.param_groups[0]["lr"] = 1e-4
    elif kk > 500:
        optimizer.param_groups[0]["lr"] = 1e-5
    
    optimizer.zero_grad()
    
    batch = np.random.choice(np.arange(n), batch_size)
    # print("obatch: ", batch)
    ## choose bad performance components 
    dataset_x = train_dataset_x[batch, :, :]
    dataset_y = train_dataset_y[batch, :]
    Sy = train_dataset_Sy[batch, :, :].to(torch.device(device))
    Sy0 = train_dataset_Sy0[batch, :, :].to(torch.device(device))
    adaptive_loss = adaptive_loss_fun(
        Sy=Sy, dataset_x=dataset_x, dataset_y=dataset_y, Sy0=Sy0
        )
    batch = batch[adaptive_loss >= np.median(adaptive_loss)]
    # print("res_loss: ", adaptive_loss)
    if kk % 10 == 0:
        print("----------------------------------")
        print("Current batch idx: ", batch)
    ## ----------------------------------
    dataset_x = train_dataset_x[batch, :, :]
    dataset_y = train_dataset_y[batch, :]
    Sy = train_dataset_Sy[batch, :, :].to(torch.device(device))
    Sy0 = train_dataset_Sy0[batch, :, :].to(torch.device(device))
    
    if loss_fun.with_hyper_prior == True:
        rand_idx = np.random.choice(n, 4)
        loss_all = loss_fun(
            Sy=Sy, dataset_x=dataset_x, dataset_y=dataset_y, rand_idx=rand_idx,
            Sy0=Sy0
            )
        nlogP, res_loss, prior_loss = loss_all
    else:
        nlogP = loss_fun(Sy=Sy, dataset_x=dataset_x, dataset_y=dataset_y)
    
    nn_params = [x for name, x in nnprior_mean.named_parameters()]
    nn.utils.clip_grad_norm_(nn_params, 1e4)
    nlogP.backward()
        
    optimizer.step() 
    loss_list.append(nlogP.item())
    
    if kk % 10 == 0:
        end = time.time()
        print("-"*50)
        print("Time: %.2f, lr_u: %.6f"  %  \
              (end-start, optimizer.param_groups[0]["lr"])
              )
        start = time.time()
        
        batch_show_error = np.arange(10)  # show errors for the same set of data
        # batch_show_error = batch # show errors for the current batch
        Sy = train_dataset_Sy[batch_show_error, :, :].to(torch.device(device))
        if nnprior_mean.mode == "residual":
            Sy0 = train_dataset_Sy0[batch_show_error, :, :].to(torch.device(device))
            output_FNO = forward_nn(Sy, Sy0)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
        else:
            output_FNO = forward_nn(Sy)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
        train_error = loss_lp(output_FNO.cpu(), train_model_params[batch_show_error, :, :])
        
        Sy = test_dataset_Sy.to(torch.device(device))
        if nnprior_mean.mode == "residual":
            Sy0 = test_dataset_Sy0.to(torch.device(device))
            output_FNO = forward_nn(Sy, Sy0)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
        else:
            output_FNO = forward_nn(Sy)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
        test_error = loss_lp(output_FNO.cpu(), test_model_params)
        
        print_str = "Iter = %4d/%d, nlogP = %.4f, term1 = %.4f, prior1 = %.4f," 
        print_str += "train_error = %.4f, test_error = %.4f" 
        print(print_str  % (kk, max_iter, nlogP.item(), res_loss.item(), \
                            prior_loss.item(), train_error.item(), test_error.item()))
        
        if env == "simple":
            draw_num = [0]
        elif env == "complex":
            draw_num = [5, 0]
        else:
            raise ValueError("env should be simple or complex")
            
        for it in draw_num:
            which_example = it
            # Syit = normalize_train_Sy.encode(Sy[which_example, :, :]).cpu().detach().numpy()
            draw_est = output_FNO[which_example,:,:].cpu().detach().numpy()
            draw_truth = test_model_params[which_example, :, :]
            sol_truth = test_dataset_y[which_example, :]
            
            points = test_dataset_x[which_example,:].cpu().detach().numpy()
            equ1.update_points(points)
            m_est = output_FNO.cpu().detach().numpy()[which_example, :]
            m_est = np.array(m_est).flatten()[coor_transfer["d2v"]]
            equ1.update_m(m_est)
            sol_est = np.array(equ1.S@equ1.forward_solver()).flatten()
            
            plt.ioff()
            plt.figure(figsize=(17, 17)) 
            plt.subplot(2,2,1)
            fig = plt.imshow(draw_truth)
            plt.colorbar(fig)
            plt.title("Meta u")
            plt.subplot(2,2,2)
            fig = plt.imshow(draw_est)
            plt.colorbar(fig)
            plt.title("Estimated Mean")   
            plt.subplot(2,2,3)
            plt.plot(prior.log_lam.cpu().detach().numpy(), linewidth=3, label="learned loglam")
            plt.plot(loglam, linewidth=1, label="loglam")
            plt.legend()
            plt.title("Log-Lam")
            plt.subplot(2,2,4)
            plt.plot(sol_truth[:100], label="sol_truh")
            plt.plot(sol_est[:100], label="sol_est")
            plt.legend() 
            plt.title("sol_compare")
    
            if with_hyper_prior == True:
                os.makedirs(meta_results_dir + env + str(it) + "_figs_prior/", exist_ok=True)
                plt.savefig(meta_results_dir + env + str(it) + "_figs_prior/" + 'fig' + str(kk) + '.png')
            else:
                os.makedirs(meta_results_dir + env + str(it) + "_figs/", exist_ok=True)
                plt.savefig(meta_results_dir + env + str(it) + "_figs/" + 'fig' + str(kk) + '.png')
            plt.close()
            
    if kk % 100 == 0 and  kk > 0:
        ## save results
        if with_hyper_prior == True:
            torch.save(
                nnprior_mean.state_dict(), meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_mean_prior"
                )
            np.save(meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_loss_prior", loss_list)
            np.save(
                meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_log_lam_prior", 
                prior.log_lam.cpu().detach().numpy()
                )
        else:
            torch.save(
                nnprior_mean.state_dict(), meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_mean"
                )
            np.save(meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_loss", loss_list)
            np.save(
                meta_results_dir + env + str(kk) + "_" + str(equ_nx) + "_meta_FNO_log_lam", 
                prior.log_lam.cpu().detach().numpy()
                )


## save results
if with_hyper_prior == True:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss_prior", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_lam_prior", 
        prior.log_lam.cpu().detach().numpy()
        )
else:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_lam", 
        prior.log_lam.cpu().detach().numpy()
        )



























