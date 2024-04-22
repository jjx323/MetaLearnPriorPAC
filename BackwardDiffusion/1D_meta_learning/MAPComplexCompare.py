## import necessary packages
import numpy as np
import fenics as fe
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


## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

## set data and result dir
data_dir = './DATA/'
meta_results_dir = './RESULTS/'
env = "complex"
noise_level = np.load(data_dir  + "noise_level.npy")
newton_method = "bicgstab"

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

error_file_names = meta_results_dir + "errors_complex.txt"
file_process(error_file_names)

for max_iter in range(1, 8):

    error_posterior_mean = []
    save_truth = []
    save_learned_mean = []
    save_learned_mean_fS = []
    save_posterior_mean_unlearn = []
    save_posterior_mean = []
    save_posterior_mean_fS = []

    # for idx_ in range(0,5):
    for idx_ in range(100):

        """
        load model parameters; load test samples 
        """
        ## u_meta_fun_test1: first branch of the random model parameters
        ## u_meta_fun_test2: second branch of the random model parameters
        with open(data_dir + env + "_meta_parameters_test", 'rb') as f:
        # with open(data_dir + env + "_meta_parameters", 'rb') as f:
            u_meta_test_ = pickle.load(f)
        u_meta_test = np.array(u_meta_test_)
        mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
        V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
        u_meta_fun_test = fe.Function(V_meta)
        u_meta_fun_test.vector()[:] = np.array(u_meta_test[idx_])

        # if idx_ in save_nums:
        save_truth.append(u_meta_fun_test.vector()[:].copy())

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
            points=np.array([coordinates_test]).T, m=u_meta_fun_test ## test fun1
            )

        ## idx_p, idx_n indicate different branches.
        ## Transfer measured data for the two branches into functions.
        Sy = d2fun(meta_data_y_test[idx_], equ_solver)

        ## load results of f(S;\theta) obtained with hyperprior
        with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"

        nnprior_mean_MAP = FNO1d(
            modes=15, width=5
            )
        nnprior_mean_MAP.load_state_dict(torch.load(with_dir))
        nnprior_mean_MAP.eval()

        mean_fS_MAP = fe.Function(domain.function_space)
        tmp = nnprior_mean_MAP(Sy, gridx_tensor).reshape(-1).detach().numpy()
        mean_fS_MAP.vector()[:] = np.array(tmp)

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
        prior_measure_MAP = GaussianElliptic2(
            domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0,
            boundary=boundary_condition, mean_fun=mean_MAP
            )

        # if idx_ in save_nums:
        save_learned_mean.append(mean_MAP.vector()[:].copy())

        ## 3. The prior measure with learned mean and variance by MAP with hyper-prior.

        prior_measure_fS_MAP = GaussianElliptic2(
            domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0,
            boundary=boundary_condition, mean_fun=mean_fS_MAP
            )

        # if idx_ in save_nums:
        save_learned_mean_fS.append(mean_fS_MAP.vector()[:].copy())

        """
        There will be two test datasets corresponding to two branches of the random model parameters
        """
        dd = meta_data_y_test[idx_] ## data of the model parameter that is above the x-axis

        ## Set the noise
        noise_level_ = noise_level
        noise = NoiseGaussianIID(dim=len(dd))
        noise.set_parameters(variance=noise_level_**2)

        """
        There will be two models corresponding to two branches of the random model parameters
        """
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
            # ff = fe.Function(model.domain_equ.function_space)
            # ff.vector()[:] = 0.0
            ### ff = fe.interpolate(u_meta_fun_test1, model.domain_equ.function_space)
            # newton_cg = NewtonCG(model=model, mk=ff.vector()[:])

            newton_cg = NewtonCG(model=model)

            ## calculate the posterior mean
            loss_pre, _, _ = model.loss()
            for itr in range(max_iter):
                newton_cg.descent_direction(cg_max=1000, method=newton_method)
                newton_cg.step(method='armijo', show_step=False)
                loss, _, _ = model.loss()
                loss_pre = loss
            mk = newton_cg.mk.copy()

            return np.array(mk)

        mean_unlearn = fe.Function(domain.function_space)
        mean_unlearn.vector()[:] = np.array(calculate_MAP(model, max_iter))


        mean_learn_MAP = fe.Function(domain.function_space)
        mean_learn_MAP.vector()[:] = np.array(calculate_MAP(model_MAP, max_iter))

        mean_learn_fS_MAP = fe.Function(domain.function_space)
        mean_learn_fS_MAP.vector()[:] = np.array(calculate_MAP(model_fS_MAP, max_iter))

        save_posterior_mean_unlearn.append(mean_unlearn.vector()[:].copy())
        save_posterior_mean.append(mean_learn_MAP.vector()[:].copy())
        save_posterior_mean_fS.append(mean_learn_fS_MAP.vector()[:].copy())

        err_L2 = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="L2")
        err_Max = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="Max")

        err_learn_MAP_L2 = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="L2")
        err_learn_MAP_Max = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="Max")
        err_learn_MLL_L2 = 0
        err_learn_MLL_Max = 0

        err_learn_fS_MAP_L2 = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="L2")
        err_learn_fS_MAP_Max = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="Max")
        err_learn_fS_MLL_L2 = 0
        err_learn_fS_MLL_Max = 0

        error_posterior_mean.append([
            err_L2, err_Max,
            err_learn_MAP_L2, err_learn_MAP_Max, err_learn_MLL_L2, err_learn_MLL_Max,
            err_learn_fS_MAP_L2, err_learn_fS_MAP_Max, err_learn_fS_MLL_L2,
            err_learn_fS_MLL_Max
            ])

        print("Idx = ", idx_)

    error_posterior_mean = np.array(error_posterior_mean)

    unlearned_error = np.mean(error_posterior_mean[:,0])
    learned_f = np.mean(error_posterior_mean[:,2])
    learned_fS = np.mean(error_posterior_mean[:,6])

    with open(error_file_names, "a") as f:
        formatted_string = f"iter_num = {max_iter:2d}, unlearned = {unlearned_error:.4f}, "
        formatted_string += f"learned_f = {learned_f:.4f}, learned_fS_L2 = {learned_fS:.4f}"
        formatted_string += "\n"
        f.write(formatted_string)

###----------------------------------------------------------------------------
i1 = 14  # i1 should be an even number
i2 = 12  # i2 should be an even number
sample_draw_list = [0, 10, 20, 30, 40]

plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(17, 12))
plt.subplot(2, 3, 1)
with open(data_dir + env + "_meta_parameters_test", 'rb') as f:
    env_samples = pickle.load(f)
env_samples = np.array(env_samples)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
for idx in sample_draw_list:
    if idx % 2 == 0:
        tmp_fun.vector()[:] = np.array(env_samples[idx])
        fe.plot(tmp_fun)
plt.ylim([-13, 13])
plt.title("(a) Environment samples (Branch 1)")

plt.subplot(2, 3, 2)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(save_truth[i1])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = np.array(save_learned_mean[i1])
fe.plot(tmp_fun, label=r"Learned mean $f(\theta)$",
        color="blue", linestyle="--")
tmp_fun.vector()[:] = np.array(save_learned_mean_fS[i1])
fe.plot(tmp_fun, label=r"Learned mean $f(S;\theta)$",
        color="red", linestyle="-.")
plt.legend(loc='upper left')
plt.title("(b) Estimated prior means (Branch 1)")

plt.subplot(2, 3, 3)

mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(save_truth[i2])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = np.array(save_learned_mean[i2])
fe.plot(tmp_fun, label=r"Learned mean $f(\theta)$",
        color="blue", linestyle="--")
tmp_fun.vector()[:] = np.array(save_learned_mean_fS[i2])
fe.plot(tmp_fun, label=r"Learned mean $f(S;\theta)$",
        color="red", linestyle="-.")
plt.legend(loc='upper left')
plt.title("(c) Estimated prior means (Branch 1)")

plt.subplot(2, 3, 4)
with open(data_dir + env + "_meta_parameters_test", 'rb') as f:
    env_samples = pickle.load(f)
env_samples = np.array(env_samples)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
for idx in sample_draw_list:
    tmp_fun.vector()[:] = np.array(env_samples[idx+1])
    fe.plot(tmp_fun)
plt.ylim([-13, 13])
plt.title("(d) Environment samples (Branch 2)")

plt.subplot(2, 3, 5)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(save_truth[i1+1])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = np.array(save_learned_mean[i1+1])
fe.plot(tmp_fun, label=r"Learned mean $f(\theta)$",
        color="blue", linestyle="--")
tmp_fun.vector()[:] = np.array(save_learned_mean_fS[i1+1])
fe.plot(tmp_fun, label=r"Learned mean $f(S;\theta)$",
        color="red", linestyle="-.")
plt.legend(loc='lower left')
plt.title("(e) Estimated prior means (Branch 2)")

plt.subplot(2, 3, 6)

mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
tmp_fun.vector()[:] = np.array(save_truth[i2+1])
fe.plot(tmp_fun, label="Truth", color="black")
tmp_fun = fe.Function(domain.function_space)
tmp_fun.vector()[:] = np.array(save_learned_mean[i2+1])
fe.plot(tmp_fun, label=r"Learned mean $f(\theta)$",
        color="blue", linestyle="--")
tmp_fun.vector()[:] = np.array(save_learned_mean_fS[i2+1])
fe.plot(tmp_fun, label=r"Learned mean $f(S;\theta)$",
        color="red", linestyle="-.")
plt.legend(loc='lower left')
plt.title("(f) Estimated prior means (Branch 2)")

plt.tight_layout(pad=0.8, w_pad=2, h_pad=3)
os.makedirs(meta_results_dir + "figures/", exist_ok=True)
plt.savefig(meta_results_dir + "figures/ExampleComplex1.jpg", dpi=500)
plt.close("all")


