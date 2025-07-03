## import necessary packages
import numpy as np
import fenics as fe
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import torch

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

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.4

## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

## set data and result dir
data_dir = './DATA/'
meta_results_dir = './RESULTS/'
results_fig_table = './RESULTS-PAPER-MaxIter-L/'
os.makedirs(results_fig_table, exist_ok=True)
env = "simple"
noise_level = np.load(data_dir + "noise_level.npy")

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

max_iters = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
Ls = [1, 5, 10, 15, 20]
errors = np.zeros((len(max_iters), len(Ls)))
errors_std = np.zeros((len(max_iters), len(Ls)))

for idx_iter, max_iter in enumerate(max_iters):
    for idx_L, L in enumerate(Ls):
        mean_f = fe.Function(domain.function_space)
        error_tmp = []
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

            alpha_prior = 0.01
            aa = 1.0
            boundary_condition = "Neumann"
            prior_measure = GaussianElliptic2(
                domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0,
                boundary=boundary_condition
            )
            
            ## Transfer measured data for the two branches into functions.
            Sy = d2fun(meta_data_y_test[idx_], equ_solver)

            ## load results of f(S;\theta) obtained with hyperprior
            with_dir = meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_mean_prior"
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
            ## 3. The prior measure with learned mean and variance by MAP with hyper-prior.
            log_gamma = np.load(meta_results_dir + env + str(equ_nx) + "_" + str(max_iter) + "_" + str(L) + "_meta_FNO_log_gamma_prior.npy")
            alpha_prior_learn = np.exp(log_gamma)
            prior_measure_fS_MAP = GaussianElliptic2(
                domain=domain, alpha=alpha_prior, a_fun=fe.Constant(alpha_prior_learn), theta=1.0,
                boundary=boundary_condition, mean_fun=mean_fS_MAP
            )

            dd = meta_data_y_test[idx_]  ## data of the model parameter that is above the x-axis
            ## Set the noise
            noise_level_ = noise_level
            noise = NoiseGaussianIID(dim=len(dd))
            noise.set_parameters(variance=noise_level_ ** 2)

            """
            Calculate the posterior mean (same as the MAP in this case)
            """
            def calculate_MAP(model, max_iter):
                newton_cg = NewtonCG(model=model)

                ## calculate the posterior mean
                loss_pre, _, _ = model.loss()
                for itr in range(max_iter):
                    newton_cg.descent_direction(cg_max=1000, method="cg_my")
                    newton_cg.step(method='armijo', show_step=False)
                    loss, _, _ = model.loss()
                    print("iter = %d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
                    loss_pre = loss
                mk = newton_cg.mk.copy()

                return np.array(mk)
        
            model_MAP = ModelBackwarDiffusion(
                d=dd, domain_equ=domain, prior=prior_measure_fS_MAP,
                noise=noise, equ_solver=equ_solver
            )    
            # mean_f.vector()[:] = np.array(calculate_MAP(model_MAP, 2))
            # error_tmp.append(relative_error(mean_f, u_meta_fun_test, domain, err_type="L2"))
            error_tmp.append(relative_error(mean_fS_MAP, u_meta_fun_test, domain, err_type="L2"))
        errors[idx_iter, idx_L] = np.mean(error_tmp)
        errors_std[idx_iter, idx_L] = np.std(error_tmp)

np.save(results_fig_table + "errors", errors)
np.save(results_fig_table + "errors_std", errors_std)
print(errors)
errors = np.load(results_fig_table + "errors.npy")

plt.figure()
for idx_iter, max_iter in enumerate(max_iters):
    # plt.subplot(2, 2, idx_iter + 1)
    plt.plot(Ls, errors[idx_iter, :], marker='o', label="MaxIter=" + str(max_iter))
plt.xticks(ticks=Ls)
plt.xlabel("Number of Samples L")
plt.ylabel("Relative Errors")
plt.title("Relative Errors")
plt.legend()
plt.savefig(results_fig_table + "errors.pdf", bbox_inches='tight', dpi=500)
plt.savefig(results_fig_table + "errors.png", bbox_inches='tight', dpi=500)
plt.close()

file_process(results_fig_table + "errors.txt")
with open(results_fig_table + "errors.txt", "a") as f:
    f.write(str(errors))
    f.write("\n")
    f.write("\n")
    f.write(str(errors_std))























