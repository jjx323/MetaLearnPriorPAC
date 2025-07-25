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
from NN_library import FNO1d, d2fun

"""
In the following code, we used different parameters L to see the results with different sampling numbers of Monte Carlo 
"""

## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="train prediction function f(theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
# Add the argument for max_iters
parser.add_argument('--max_iters', type=int, nargs='+', default=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], 
                    help='A list of integers representing max iterations.')
parser.add_argument("--Ls", type=int, nargs='+', default=[1, 5, 10, 15, 20], help='A list of integers representing Ls.')
args = parser.parse_args()

## Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
os.makedirs(meta_results_dir, exist_ok=True)
env = args.env
with_hyper_prior = True
max_iters = args.max_iters
Ls = args.Ls

print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

for L in Ls: 
    print("---------- max_iter = " + str(max_iters) + ", L = " + str(L) + " ----------")
    ## domain for solving PDE
    equ_nx = 70
    domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
    ## d2v is used to transfer the grid coordinates. 
    d2v = fe.dof_to_vertex_map(domain.function_space)
    ## gridx contains coordinates that are match the function values obtained by fun.vector()[:]
    ## More detailed illustrations can be found in Subsections 5.4.1 to 5.4.2 of the following book:
    ##     Hans Petter Langtangen, Anders Logg, Solving PDEs in Python: The FEniCS Tutorial I. 
    gridx = domain.mesh.coordinates()[d2v]
    ## transfer numpy.arrays to torch.tensor that are used as part of the input of FNO 
    gridx_tensor = torch.tensor(gridx, dtype=torch.float32)

    noise_level = np.load(meta_data_dir + "noise_level.npy")

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
    with open(meta_data_dir + env + "_meta_parameters", 'rb') as f1: 
        u_meta = pickle.load(f1)
    u_meta = copy.deepcopy(np.array(u_meta))
    mesh_meta = fe.Mesh(meta_data_dir + env + '_saved_mesh_meta.xml')
    V_meta = fe.FunctionSpace(mesh_meta, 'P', 1)
    ## -----------------------------------------------------------------------
    ## load test data
    with open(meta_data_dir + env + "_meta_data_x_test", 'rb') as f2: 
        meta_data_x_test = pickle.load(f2)
    with open(meta_data_dir + env + "_meta_data_y_test", 'rb') as f3: 
        meta_data_y_test = pickle.load(f3)
    with open(meta_data_dir + env + "_meta_parameters_test", 'rb') as f4: 
        u_meta_test = pickle.load(f4)
    u_meta_test = copy.deepcopy(np.array(u_meta_test))
    coordinates_test = meta_data_x_test[0]
    ## -----------------------------------------------------------------------
    if env == 'simple':
        u_meta_fun = fe.Function(V_meta)
        u_meta_fun.vector()[:] = np.array(u_meta[0])
        ## -------------------------------------------------
        u_meta_fun_test = fe.Function(V_meta)
        u_meta_fun_test.vector()[:] = np.array(u_meta_test[0])
    elif env == 'complex':
        u_meta_fun1 = fe.Function(V_meta)
        u_meta_fun2 = fe.Function(V_meta)
        u_meta_fun1.vector()[:] = np.array(u_meta[0])
        u_meta_fun2.vector()[:] = np.array(u_meta[1])
        ## -------------------------------------------------
        u_meta_fun1_test = fe.Function(V_meta)
        u_meta_fun2_test = fe.Function(V_meta)
        u_meta_fun1_test.vector()[:] = np.array(u_meta_test[0])
        u_meta_fun2_test.vector()[:] = np.array(u_meta_test[1])
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
        ## ------------------------------------------------------------------------
        equ_solver_test = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                                points=np.array([coordinates_test]).T, m=u_meta_fun_test)
        sol_test = equ_solver_test.forward_solver()
        sol_fun_test = fe.Function(domain.function_space)
        sol_fun_test.vector()[:] = np.array(sol_test)
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
        ## ------------------------------------------------------------------------
        equ_solver_test = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                                points=np.array([coordinates_test]).T, m=u_meta_fun1_test)
        sol_test1 = equ_solver_test.forward_solver()
        sol_fun1_test = fe.Function(domain.function_space)
        sol_fun1_test.vector()[:] = np.array(sol_test1)

        equ_solver_test = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                                points=np.array([coordinates_test]).T, m=u_meta_fun2_test)
        sol_test2 = equ_solver_test.forward_solver()
        sol_fun2_test = fe.Function(domain.function_space)
        sol_fun2_test.vector()[:] = np.array(sol_test2)
    else:
        raise NotImplementedError("env should be simple or complex")

    ## -------------------------------------------------------------------------
    analysis_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    Sy_train_list = []
    u_train_list = []
    for idx_train in analysis_list:
        Sy_train_list.append(d2fun(meta_data_y[idx_train], equ_solver))
        ff = fe.Function(V_meta)
        ff.vector()[:] = np.array(u_meta[idx_train])
        tmp = fe.project(ff, domain.function_space).vector()[:]
        u_train_list.append(tmp.copy())
    training_error_list = []

    Sy_test_list = []
    u_test_list = []
    for idx_test in analysis_list:
        Sy_test_list.append(d2fun(meta_data_y_test[idx_test], equ_solver_test))
        ff = fe.Function(V_meta)
        ff.vector()[:] = np.array(u_meta_test[idx_test])
        tmp = fe.project(ff, domain.function_space).vector()[:]
        u_test_list.append(tmp.copy())
    testing_error_list = []

    ## -----------------------------------------------------------------------
    ## construct the base prior measure that is a Gaussian measure 
    domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

    prior = GaussianElliptic2Learn(
        domain, alpha=0.01, a_fun=fe.Constant(1.0), theta=1.0, 
        mean_fun=None, tensor=False, boundary="Neumann"
        )
    learn_var = True#False
    prior.trans2learnable(learn_mean=False, learn_var=learn_var)

    ## -----------------------------------------------------------------------
    ## construct the hyper-prior measure that is a Gaussian measure 
    alpha_hyper_prior, beta_hyper_prior = 0.001, 0.007#0.1, 0.01

    hyper_prior_mean = Gaussian1DFiniteDifference(
        nx=equ_nx+1, a_fun=alpha_hyper_prior, alpha=beta_hyper_prior
        )
    hyper_prior_mean.trans2learnable(learn_mean=False, learn_a=False)

    def make_hyper_prior_log_gamma(mean_vec, weight=0.01):
        mean_vec = copy.deepcopy(mean_vec)
        weight = weight
        
        def hyper_prior_log_gamma(val):
            temp = val - mean_vec
            return weight*torch.sum(temp*temp)
        
        return hyper_prior_log_gamma

    if learn_var == True:
        hyper_prior_log_gamma = make_hyper_prior_log_gamma(0.0, weight=0.001)

    ## -----------------------------------------------------------------------
    ## Set the neural network for learning the prediction policy of the mean functions.
    ## The parameters are specified similar to the original paper proposed FNO.
    hidden_dim = np.load(meta_results_dir + "hidden_dim.npy")
    nnprior_mean = FNO1d(
        modes=15, width=hidden_dim
        )
    def nnprior_mean_boundary(u, g):
        output = nnprior_mean(u, g).reshape(-1)
        output[0], output[-1] = 0.0, 0.0 
        return output

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

    ## batch_size: only use batch_size number of datasets for each iteration 
    batch_size = 20

    if learn_var == True:
        optimizer = torch.optim.AdamW(
            [{"params": nnprior_mean.parameters(), "lr": 0.001},
            {"params": prior.log_gamma_learn, "lr": 0.0}],
            #lr=0.001,
            weight_decay=0.0
            )

    if learn_var == False:
        optimizer = torch.optim.AdamW(
            [{"params": nnprior_mean.parameters()}],
            lr=0.005,
            weight_decay=0.0
            )  

    def set_learning_rate(optimizer, group_index, new_lr):
        optimizer.param_groups[group_index]['lr'] = new_lr
        print(f"Learning rate of group {group_index} set to {new_lr}")

    loss_list = []
    learned_var_list = []

    weight_of_lnZ = n/((m+1)*batch_size)

    start_time = time.time()
    counter = 0
    for kk in range(max_iters[-1]):
        
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
            Sy = d2fun(meta_data_y[itr], equ_solver)
            prior.mean_vec_learn = nnprior_mean_boundary(Sy, gridx_tensor)
            
            for ii in range(L):
                ul = prior.generate_sample_learn()
                preds = pdeasnet(ul)        
                loss_res_L[ii] = -loss_residual(preds, targets)
                
            lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                    - torch.log(torch.tensor(L, dtype=torch.float32))
        
        if with_hyper_prior == True:
            prior1 = 0
            rand_idx = np.random.choice(n, 4)
            for itr in rand_idx:
                Sy = d2fun(meta_data_y[itr], equ_solver)
                tmp = nnprior_mean_boundary(Sy, gridx_tensor)
                prior1 +=  hyper_prior_mean.evaluate_CM_inner(tmp)          
            if learn_var == True:
                prior1 += hyper_prior_log_gamma(prior.log_gamma_learn)
            nlogP = -weight_of_lnZ*lnZ + prior1
        else:
            nlogP = -weight_of_lnZ*lnZ
        
        nlogP.backward()
        
        if kk == 1000:
            set_learning_rate(optimizer, 0, 0.001)
            set_learning_rate(optimizer, 1, 0.000)
        if kk == 2000:
            set_learning_rate(optimizer, 0, 0.001)
            set_learning_rate(optimizer, 1, 0.0005)   
                
        optimizer.step() 

        loss_list.append(nlogP.item())
        if learn_var == True:
            learned_var_list.append(prior.log_gamma_learn.detach().numpy())

        
        if kk % 10 == 0:
            del nlogP, lnZ, loss_res_L, loss_residual
        
        max_iter = max_iters[counter]
        if kk == max_iter:
            counter = counter + 1
            print("max_iter = ", max_iter, end="; ")
            print("L = ", L)
            ## results
            if learn_var == True:
                np.save(
                    meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_learned_var_prior",
                    np.array(learned_var_list)
                )
                plt.figure()
                plt.plot(np.array(learned_var_list), label="learned_var")
                plt.legend()
                plt.savefig(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_learned_var_prior.png")
                plt.close()
                
            if with_hyper_prior == True:
                torch.save(
                    nnprior_mean.state_dict(), meta_results_dir + env +
                    str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_mean_prior"
                )
                np.save(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) +
                        "_meta_FNO_loss_prior", loss_list)
                np.save(
                    meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_log_gamma_prior",
                    prior.log_gamma_learn.detach().numpy()
                )
            else:
                torch.save(
                    nnprior_mean.state_dict(), meta_results_dir + env +
                    str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_mean"
                )
                np.save(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_loss", loss_list)
                np.save(
                    meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_log_gamma",
                    prior.log_gamma_learn.detach().numpy()
                )

max_iter = max_iters[-1]
## results
if learn_var == True:
    np.save(
        meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_learned_var_prior",
        np.array(learned_var_list)
    )
    plt.figure()
    plt.plot(np.array(learned_var_list), label="learned_var")
    plt.legend()
    plt.savefig(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_learned_var_prior.png")
    plt.close()
    
if with_hyper_prior == True:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env +
        str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_mean_prior"
    )
    np.save(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) +
            "_meta_FNO_loss_prior", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_log_gamma_prior",
        prior.log_gamma_learn.detach().numpy()
    )
else:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env +
        str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_mean"
    )
    np.save(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_loss", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_log_gamma",
        prior.log_gamma_learn.detach().numpy()
    )







