## import necessary packages
import numpy as np
import fenics as fe
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib.pyplot as plt
import pickle
import torch
import time

## Add path to the parent directory
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

## Import necessary modules in our programs
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.optimizer import NewtonCG
from core.noise import NoiseGaussianIID
from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion, file_process, relative_error
from NN_library import FNO1d, d2fun


## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

## set data and result dir
data_dir = './DATA/'
meta_results_dir = './RESULTS/'
results_fig_table = './RESULTS-PAPER-MAP/'
os.makedirs(results_fig_table, exist_ok=True)
env = "simple"
noise_level = np.load(data_dir + "noise_level.npy")
newton_method = "cg_my"
# newton_method = "bicgstab"

## domain for solving PDE
equ_nx = 70
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
## save the mesh information 
os.makedirs(data_dir, exist_ok=True)
file_mesh = fe.File(data_dir + env + '_saved_mesh_meta_pCN.xml')
file_mesh << domain.function_space.mesh()

d2v = fe.dof_to_vertex_map(domain.function_space)
## gridx contains coordinates that are match the function values obtained by fun.vector()[:]
## More detailed illustrations can be found in Subsections 5.4.1 to 5.4.2 of the following book:
##     Hans Petter Langtangen, Anders Logg, Solving PDEs in Python: The FEniCS Tutorial I. 
gridx = domain.mesh.coordinates()[d2v]
## transfer numpy.arrays to torch.tensor that are used as part of the input of FNO 
gridx_tensor = torch.tensor(gridx, dtype=torch.float32)

error_file_names = results_fig_table + "errors_simple.txt"
file_process(error_file_names)
times_file_names = results_fig_table + "simple_record_time_map.txt"
file_process(times_file_names)

for max_iter in range(1, 8):
    error_posterior_mean = []
    save_posterior_mean = []
    save_prior_mean = []
    times_unlearned = []
    times_learned_f = []
    times_learned_fS = []

    for idx_ in range(0, 50):

        """
        load model parameters; load test samples 
        """
        with open(data_dir + env + "_meta_parameters_test", 'rb') as f:
            u_meta_test = pickle.load(f)
        u_meta_test = np.array(u_meta_test)
        mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
        V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
        u_meta_fun_test = fe.Function(V_meta)
        u_meta_fun_test.vector()[:] = np.array(u_meta_test[idx_])

        ## load the test data pairs
        with open(data_dir + env + "_meta_data_x_test", 'rb') as f:
            meta_data_x_test = pickle.load(f)
        with open(data_dir + env + "_meta_data_y_test", 'rb') as f:
            meta_data_y_test = pickle.load(f)
        T, num_steps = np.load(data_dir + env + "_equation_parameters_test.npy")
        num_steps = np.int64(num_steps)

        ## construct different equ_solvers for different model parameters that with
        ## different measurement data
        coordinates_test = meta_data_x_test[idx_]
        equ_solver = EquSolver(
            domain_equ=domain, T=T, num_steps=num_steps,
            points=np.array([coordinates_test]).T, m=u_meta_fun_test  ## test fun1
        )

        ## idx_p, idx_n indicate different branches.
        ## Transfer measured data for the two branches into functions.
        Sy = d2fun(meta_data_y_test[idx_], equ_solver)

        ## load results of f(S;\theta) obtained with hyperprior
        with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"
        hidden_dim = np.load(meta_results_dir + "hidden_dim.npy")
        nnprior_mean_MAP = FNO1d(
            modes=15, width=hidden_dim
        )
        nnprior_mean_MAP.load_state_dict(torch.load(with_dir))
        nnprior_mean_MAP.eval()

        mean_fS_MAP = fe.Function(domain.function_space)
        mean_fS_MAP.vector()[:] = np.array(
            nnprior_mean_MAP(Sy, gridx_tensor).reshape(-1).detach().numpy()
        )

        with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"

        """
        Construct different priors for testing 
        """
        ## 1. The prior measure without learning, set it as the initial prior measure
        ##    in the learning stage.

        ## alpha_prior and beta_prior are set as in the training stage
        alpha_prior = 0.01
        aa = 1.0
        boundary_condition = "Neumann"
        prior_measure = GaussianElliptic2(
            domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0,
            boundary=boundary_condition
        )

        ## 2. The prior measure with learned mean and variance by MLL(maximum likelihood)
        ##    algorithm that without hyper-prior.
        temp = np.load(meta_results_dir + env + str(equ_nx) + "_meta_mean_prior.npy")
        mean_MAP = fe.Function(domain.function_space)
        mean_MAP.vector()[:] = np.array(temp)
        log_gamma = np.load(meta_results_dir + env + str(equ_nx) + "_meta_log_gamma_prior.npy")
        alpha_prior_learn = np.exp(log_gamma)
        prior_measure_MAP = GaussianElliptic2(
            domain=domain, alpha=alpha_prior, a_fun=fe.Constant(alpha_prior_learn), theta=1.0,
            boundary=boundary_condition, mean_fun=mean_MAP
        )

        ## 3. The prior measure with learned mean and variance by MAP with hyper-prior.
        log_gamma = np.load(meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior.npy")
        alpha_prior_learn = np.exp(log_gamma)
        prior_measure_fS_MAP = GaussianElliptic2(
            domain=domain, alpha=alpha_prior, a_fun=fe.Constant(alpha_prior_learn), theta=1.0,
            boundary=boundary_condition, mean_fun=mean_fS_MAP
        )

        save_prior_mean.append([
            mean_MAP.vector()[:], mean_fS_MAP.vector()[:]
        ])

        dd = meta_data_y_test[idx_]  ## data of the model parameter that is above the x-axis

        ## Set the noise
        noise_level_ = noise_level
        noise = NoiseGaussianIID(dim=len(dd))
        noise.set_parameters(variance=noise_level_ ** 2)

        ## model1 with prior measure $\mathcal{N}(0, \mathcal{C}_0)$ and data d1
        model = ModelBackwarDiffusion(
            d=dd, domain_equ=domain, prior=prior_measure,
            noise=noise, equ_solver=equ_solver
        )

        model_MAP = ModelBackwarDiffusion(
            d=dd, domain_equ=domain, prior=prior_measure_MAP,
            noise=noise, equ_solver=equ_solver
        )

        model_fS_MAP = ModelBackwarDiffusion(
            d=dd, domain_equ=domain, prior=prior_measure_fS_MAP,
            noise=noise, equ_solver=equ_solver
        )

        """
        Calculate the posterior mean (same as the MAP in this case)
        """
        def calculate_MAP(model, max_iter):
            ## set optimizer NewtonCG
            newton_cg = NewtonCG(model=model)

            ## calculate the posterior mean
            loss_pre, _, _ = model.loss()
            for itr in range(max_iter):
                newton_cg.descent_direction(cg_max=1000, method=newton_method)
                newton_cg.step(method='armijo', show_step=False)
                loss, _, _ = model.loss()
                print("iter = %d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
                loss_pre = loss
            mk = newton_cg.mk.copy()

            return np.array(mk)

        start_time = time.time()
        mean_unlearn = fe.Function(domain.function_space)
        mean_unlearn.vector()[:] = np.array(calculate_MAP(model, max_iter))
        end_time = time.time()
        times_unlearned.append(end_time - start_time)
    
        start_time = time.time()
        mean_learn_MAP = fe.Function(domain.function_space)
        mean_learn_MAP.vector()[:] = np.array(calculate_MAP(model_MAP, max_iter))
        end_time = time.time()
        times_learned_f.append(end_time - start_time)

        # mean_learn_MLL = fe.Function(domain.function_space)
        start_time = time.time()
        mean_learn_fS_MAP = fe.Function(domain.function_space)
        mean_learn_fS_MAP.vector()[:] = np.array(calculate_MAP(model_fS_MAP, max_iter))
        end_time = time.time()
        times_learned_fS.append(end_time - start_time)

        mean_learn_fS_MLL = fe.Function(domain.function_space)

        if idx_ <= 50:
            save_posterior_mean.append([
                mean_unlearn.vector()[:], mean_learn_MAP.vector()[:],
                mean_learn_fS_MAP.vector()[:]
            ])

        err_L2 = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="L2")
        err_Max = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="Max")

        err_learn_MAP_L2 = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="L2")
        err_learn_MAP_Max = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="Max")
        err_learn_MLL_L2 = None #relative_error(mean_learn_MLL, u_meta_fun_test, domain, err_type="L2")
        err_learn_MLL_Max = None #relative_error(mean_learn_MLL, u_meta_fun_test, domain, err_type="Max")

        err_learn_fS_MAP_L2 = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="L2")
        err_learn_fS_MAP_Max = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="Max")
        err_learn_fS_MLL_L2 = None #relative_error(mean_learn_fS_MLL, u_meta_fun_test, domain, err_type="L2")
        err_learn_fS_MLL_Max = None #relative_error(mean_learn_fS_MLL, u_meta_fun_test, domain, err_type="Max")

        error_posterior_mean.append([
            err_L2, err_Max,
            err_learn_MAP_L2, err_learn_MAP_Max, err_learn_MLL_L2, err_learn_MLL_Max,
            err_learn_fS_MAP_L2, err_learn_fS_MAP_Max, err_learn_fS_MLL_L2,
            err_learn_fS_MLL_Max
        ])

    avg_time_unlearned = np.mean(times_unlearned)
    avg_time_learned_f = np.mean(times_learned_f) 
    avg_time_learned_fS = np.mean(times_learned_fS)
    avg_time_unlearned_str = env + "_" + str(max_iter) + "_optim_unlearned: "
    avg_time_learned_f_str = env + "_" + str(max_iter) + "_optim_learned_f: "
    avg_time_learned_fS_str = env + "_" + str(max_iter) + "_optim_learned_fS: "
    with open(times_file_names, 'a') as file:
        file.write(avg_time_unlearned_str + str(avg_time_unlearned) + '\n')  
        file.write(avg_time_learned_f_str+ str(avg_time_learned_f) + '\n')  
        file.write(avg_time_learned_fS_str+ str(avg_time_learned_fS) + '\n')

    error_posterior_mean = np.array(error_posterior_mean)

    unlearned_error = np.mean(error_posterior_mean[:, 0])
    std_unlearned = np.std(error_posterior_mean[:, 0])
    learned_f = np.mean(error_posterior_mean[:, 2])
    std_learned_f = np.std(error_posterior_mean[:, 2])
    learned_fS = np.mean(error_posterior_mean[:, 6])
    std_learned_fS = np.std(error_posterior_mean[:, 6])

    with open(error_file_names, "a") as f:
        formatted_string = f"iter_num = {max_iter:2d}, unlearned = {unlearned_error:.4f} +- {std_unlearned:.4f}, "
        formatted_string += f"learned_f = {learned_f:.4f} +- {std_learned_f:.4f}, "
        formatted_string += f"learned_fS_L2 = {learned_fS:.4f} +- {std_learned_fS:.4f}"
        formatted_string += "\n"
        f.write(formatted_string)

    save_posterior_mean = np.array(save_posterior_mean)

###----------------------------------------------------------------------------
idx1 = 0
idx2 = 4

import scienceplots
plt.style.use('science')
plt.rcParams['font.size'] = 15
plt.figure(figsize=(17, 5))
plt.subplot(1, 3, 1)
with open(data_dir + env + "_meta_parameters_test", 'rb') as f:
    env_samples = pickle.load(f)
env_samples = np.array(env_samples)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
for idx in [0, 20, 40, 60, 80]:
    tmp_fun.vector()[:] = np.array(env_samples[idx])
    fe.plot(tmp_fun)
plt.ylim([-12.5, 9.5])
plt.title("(a) Environment samples", pad=15)
plt.subplot(1, 3, 2)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(env_samples[idx1])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = save_prior_mean[idx1][0]
fe.plot(tmp_fun, label=r"Learned mean $f_m(\theta_1)$", color="blue", linestyle="--")
tmp_fun.vector()[:] = save_prior_mean[idx1][1]
fe.plot(tmp_fun, label=r"Learned mean $f_m(S;\theta_1)$", color="red", linestyle="-.")
plt.legend(loc="upper left")
plt.ylim([-10.5, 9])
plt.title("(b) Estimated prior means", pad=15)
plt.subplot(1, 3, 3)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(env_samples[idx2])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = save_prior_mean[idx2][0]
fe.plot(tmp_fun, label=r"Learned mean $f_m(\theta_1)$", color="blue", linestyle="--")
tmp_fun.vector()[:] = save_prior_mean[idx2][1]
fe.plot(tmp_fun, label=r"Learned mean $f_m(S;\theta_1)$", color="red", linestyle="-.")
plt.legend(loc="upper left")
plt.ylim([-2.5, 6])
plt.title("(c) Estimated prior means", pad=15)

plt.tight_layout(pad=0.8, w_pad=2, h_pad=3)
plt.savefig(results_fig_table + "ExampleSimple1.jpg", dpi=1000)
plt.savefig(results_fig_table + "ExampleSimple1.pdf")
plt.close("all")
