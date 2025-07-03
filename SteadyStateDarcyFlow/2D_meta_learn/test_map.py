## import necessary packages
import numpy as np 
import fenics as fe
import torch 
import argparse

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import NewtonCG, GradientDescent
from core.misc import load_expre, str2bool, relative_error_simple

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss, file_process


"""
In the following, we select one sample of the positive-valued branch of the complex environment. 
Set the max_iter_num = 100000, to see why the unlearned and learned data-independent Bayesian models 
perform badly. 
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
parser.add_argument('--newton_method', type=str, default="cg_my", help='newton_method = "cg_my" or "bicgstab"')
args = parser.parse_args()

## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
results_fig_table = "./RESULTS-PAPER-MAP/"
os.makedirs(results_fig_table, exist_ok=True)
results_fig_table_data = "./RESULTS-PAPER-MAP/DATA/"
os.makedirs(results_fig_table_data, exist_ok=True)

env = "complex"
newton_method = args.newton_method

if env == "complex":
    env_num = 1
elif env == "simple":
    env_num = 2
with_hyper_prior = True
# with_hyper_prior = False
device = "cpu"

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
comput_learned_fS = False#True
cg_max = 200

sample_indexs = [3]
max_iter_num = 100000

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

"""
Evaluate MAP estimates for test examples with unlearned prior measures 
"""

prior_unlearn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2
    )
file_path = results_fig_table_data + "prior_unlearn_eigval.npy"
if os.path.isfile(file_path):
    eigval = np.load(results_fig_table_data + "prior_unlearn_eigval.npy") 
    eigvec = np.load(results_fig_table_data + "prior_unlearn_eigvec.npy")
    prior_unlearn.set_eigensystem(eigval, eigvec)
else:
    prior_unlearn.calculate_eigensystem()
    np.save(results_fig_table_data + "prior_unlearn_eigval.npy", prior_unlearn.lam) 
    np.save(results_fig_table_data + "prior_unlearn_eigvec.npy", prior_unlearn.eigvec_)

estimates_unlearn = []
final_error_unlearn = []

unlearned_error_list = []

if comput_unlearned == True:
    # for idx in range(n_test):
    for idx in sample_indexs:
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

        ## set optimizer NewtonCG
        newton_cg = NewtonCG(model=model)
        max_iter = max_iter_num

        newton_cg.re_init(prior_unlearn.mean_vec)

        loss_pre = model.loss()[0]
        tmp_list = np.zeros(max_iter + 1)
        fun_unlearn.vector()[:] = np.array(newton_cg.mk.copy())
        tmp_list[0] = relative_error_simple(fun_unlearn, fun_truth)
        for itr in range(1, max_iter):
            newton_cg.descent_direction(cg_max=cg_max, method=newton_method)
            print(newton_cg.hessian_terminate_info)
            newton_cg.step(method='armijo', show_step=False)
            if newton_cg.converged == False:
                break
            loss = model.loss()[0]
            print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-10*loss:
                print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss

            fun_unlearn.vector()[:] = np.array(newton_cg.mk.copy())
            tmp_list[itr] = relative_error_simple(fun_unlearn, fun_truth)
            print("Relative error = ", tmp_list[itr])

        # m_newton_cg = fe.Function(domain.function_space)
        # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())

        if itr != max_iter - 1:
            for idx1 in range(itr, max_iter):
                tmp_list[idx1] = tmp_list[itr-1]

        estimates_unlearn.append(np.array(newton_cg.mk.copy()))
        final_error_unlearn.append(loss)
        print("unlearn idx: ", idx)
        fun_unlearn.vector()[:] = np.array(estimates_unlearn[-1])
        unlearned_error_list.append(np.array(tmp_list))
        print("Unlearn Error: ", unlearned_error_list[-1][-1])

    unlearned_error_list = np.array(unlearned_error_list, dtype=np.float64)
    np.save(meta_results_dir + env + "_unlearned_error_list_test", np.array(unlearned_error_list))
    print("Saved unlearned Error")
    # np.save(meta_results_dir + env + "_unlearned_error_list", np.array(unlearned_error_list))
    # np.save(meta_results_dir + "estimates_unlearn_MAP", (estimates_unlearn, final_error_unlearn))


"""
Evaluate MAP estimates for test examples with learned prior measures f(\theta)
"""

prior_learn_f = GaussianFiniteRank(
        domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2
        )

file_path = results_fig_table_data + "prior_learn_f_eigval.npy"
if os.path.isfile(file_path):
    eigval = np.load(results_fig_table_data + "prior_learn_f_eigval.npy") 
    eigvec = np.load(results_fig_table_data + "prior_learn_f_eigvec.npy")
    prior_learn_f.set_eigensystem(eigval, eigvec)
else:
    prior_learn_f.calculate_eigensystem()
    print("eigensystem calculated")
    np.save(results_fig_table_data + "prior_learn_f_eigval.npy", prior_learn_f.lam) 
    np.save(results_fig_table_data + "prior_learn_f_eigvec.npy", prior_learn_f.eigvec_)

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
    for idx in sample_indexs:
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

        ## set optimizer NewtonCG
        newton_cg = NewtonCG(model=model)
        # gradient_descent = GradientDescent(model=model)
        max_iter = max_iter_num

        newton_cg.re_init(prior_learn_f.mean_vec)
        # gradient_descent.re_init(prior_learn_f.mean_vec*0)

        loss_pre = model.loss()[0]
        tmp_list = np.zeros(max_iter + 1)
        fun_learned_f.vector()[:] = np.array(newton_cg.mk.copy())
        # fun_learned_f.vector()[:] = np.array(gradient_descent.mk.copy())
        tmp_list[0] = relative_error_simple(fun_learned_f, fun_truth)
        for itr in range(1, max_iter):
            newton_cg.descent_direction(cg_max=cg_max, method=newton_method)
            print(newton_cg.hessian_terminate_info)
            newton_cg.step(method='armijo', show_step=False)
            # gradient_descent.descent_direction()
            # gradient_descent.step(method='armijo', show_step=False)
            print(np.linalg.norm(newton_cg.M@newton_cg.grad))
            loss = model.loss()[0]
            if newton_cg.converged == False:
                break
            print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-10*loss:
                print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss

            fun_learned_f.vector()[:] = np.array(newton_cg.mk.copy())
            # fun_learned_f.vector()[:] = np.array(gradient_descent.mk.copy())
            tmp_list[itr] = relative_error_simple(fun_learned_f, fun_truth)
            print("Relative error = ", tmp_list[itr])

            data_iter = model.S@equ_solver.forward_solver(fun_learned_f.vector()[:])
            d_error = np.linalg.norm(data_iter - d_noisy)/np.linalg.norm(d_noisy)
            print("Error of the data = ", d_error)

        if itr != max_iter - 1:
            for idx1 in range(itr, max_iter):
                tmp_list[idx1] = tmp_list[itr-1]

        estimates_learn_f.append(np.array(newton_cg.mk.copy()))
        # estimates_learn_f.append(np.array(gradient_descent.mk.copy()))
        final_error_learn_f.append(loss)
        print("Learned f(theta) idx: ", idx)
        fun_learned_f.vector()[:] = np.array(estimates_learn_f[-1])
        learned_error_list_f.append(np.array(tmp_list))
        print("Learned f(theta) Error: ", learned_error_list_f[-1][-1])

    learned_error_list_f = np.array(learned_error_list_f)
    print(learned_error_list_f)
    np.save(meta_results_dir + env + "_learned_error_list_f_test", np.array(learned_error_list_f))
    print("Saved Learned f(theta) Error")
    # np.save(meta_results_dir + env + "_learned_error_list_f", np.array(learned_error_list_f))
    # np.save(meta_results_dir + "estimates_learn_MAP_f", (estimates_learn_f, final_error_learn_f))


