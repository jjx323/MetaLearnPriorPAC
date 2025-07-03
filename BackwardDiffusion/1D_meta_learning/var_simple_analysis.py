## import necessary packages
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import pickle
import torch
import matplotlib as mpl

## Add path to the parent directory
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

## Import necessary modules in our programs
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.optimizer import NewtonCG, GradientDescent
from core.noise import NoiseGaussianIID
from core.approximate_sample import LaplaceApproximate
from core.misc import construct_measurement_matrix
from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion, file_process, relative_error
from BackwardDiffusion.meta_common import GaussianElliptic2Learn
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
results_fig_table = './RESULTS-PAPER-MAP/'
os.makedirs(results_fig_table, exist_ok=True)
env = "simple"
noise_level = np.load(data_dir + "noise_level.npy")
newton_method = "cg_my"
# newton_method = "bicgstab"
# eigen_method = "double_pass"
eigen_method = "scipy_eigsh"

num_eigval = 50
idxes = [5]

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

max_iter = 10
error_posterior_mean = []
save_posterior_mean = []
save_prior_mean = []
for ii_, idx_ in enumerate(idxes):
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
    num_points = coordinates_test.shape[0]
    
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

    dd = meta_data_y_test[idx_]  ## data of the model parameter that is above the x-axis

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
        domain=domain, alpha=alpha_prior_learn, a_fun=fe.Constant(aa), theta=1.0,
        boundary=boundary_condition, mean_fun=mean_fS_MAP
    )

    save_prior_mean.append([
        mean_MAP.vector()[:], mean_fS_MAP.vector()[:]
    ])

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
        # newton_cg.re_init(gradient_descent.mk.copy())
        newton_cg.re_init(model.prior.mean_vec)
        ## calculate the posterior mean
        loss_pre, _, _ = model.loss()
        for itr in range(max_iter):
            newton_cg.descent_direction(cg_max=1000, method=newton_method)
            newton_cg.step(method='armijo', show_step=False)
            if newton_cg.converged == False:
                break
            loss, _, _ = model.loss()
            print("iter = %d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-10*loss:
                print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss
        mk = newton_cg.mk.copy()

        return np.array(mk)

    print("Calculate the posterior mean without learning ...")
    mean_unlearn = fe.Function(domain.function_space)
    mean_unlearn.vector()[:] = np.array(calculate_MAP(model, max_iter))

    print("Calculate the posterior mean with learned mean f...")
    mean_learn_f = fe.Function(domain.function_space)
    mean_learn_f.vector()[:] = np.array(calculate_MAP(model_MAP, max_iter))

    print("Calculate the posterior mean with learned mean fS...")
    mean_learn_fS = fe.Function(domain.function_space)
    mean_learn_fS.vector()[:] = np.array(calculate_MAP(model_fS_MAP, max_iter))

    print("Calculate the Laplace approximation without learning ...")
    appro_unlearn = LaplaceApproximate(model)
    appro_unlearn.calculate_eigensystem(num_eigval=num_eigval, method=eigen_method, cut_val=0)
    appro_unlearn.set_mean(mean_unlearn.vector()[:])

    print("Calculate the Laplace approximation with learned mean f...") 
    appro_learn_f = LaplaceApproximate(model_MAP)
    appro_learn_f.calculate_eigensystem(num_eigval=num_eigval, method=eigen_method, cut_val=0)
    appro_learn_f.set_mean(mean_learn_f.vector()[:])

    print("Calculate the Laplace approximation with learned mean fS...")
    appro_learn_fS = LaplaceApproximate(model_fS_MAP)
    appro_learn_fS.calculate_eigensystem(num_eigval=num_eigval, method=eigen_method, cut_val=0)
    appro_learn_fS.set_mean(mean_learn_fS.vector()[:])

    # specify some points in the domain 
    xx = np.linspace(0, 1, 100)
    points = []
    for ii in range(len(xx)):
        points.append((xx[ii], xx[ii]))
    SM = construct_measurement_matrix(np.array(points), model.domain_equ.function_space)
    # print(SM.shape)

    mean_unlearn_points = SM@mean_unlearn.vector()[:]
    mean_learn_f_points = SM@mean_learn_f.vector()[:]
    mean_learn_fS_points = SM@mean_learn_fS.vector()[:]

    print("Calculate the variance field...")
    std_unlearn_points = appro_unlearn.pointwise_variance_field(points, points)
    std_learn_f_points = appro_learn_f.pointwise_variance_field(points, points)
    std_learn_fS_points = appro_learn_fS.pointwise_variance_field(points, points)

    true_fun = fe.interpolate(u_meta_fun_test, model.domain_equ.function_space)
    true_fun_points = SM@true_fun.vector()[:]

    err_L2 = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="L2")
    err_learn_MAP_L2 = relative_error(mean_learn_f, u_meta_fun_test, domain, err_type="L2")
    err_learn_fS_MAP_L2 = relative_error(mean_learn_fS, u_meta_fun_test, domain, err_type="L2")
    print("Relative Error (unlearn): ", err_L2)
    print("Relative Error (learn f): ", err_learn_MAP_L2)   
    print("Relative Error (learn fS): ", err_learn_fS_MAP_L2)

    fun_f_points = SM@mean_MAP.vector()[:]
    fun_fS_points = SM@mean_fS_MAP.vector()[:]
    err_L2_f = relative_error(mean_MAP, u_meta_fun_test, domain, err_type="L2")
    err_L2_fS = relative_error(mean_fS_MAP, u_meta_fun_test, domain, err_type="L2")
    print("Relative Error (f): ", err_L2_f)
    print("Relative Error (fS): ", err_L2_fS)

    print("Ploting...")
    import scienceplots
    plt.style.use('science')
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(17, 5))
    xx = np.linspace(0, 1, len(mean_unlearn_points))
    plt.subplot(1, 3, 1)
    plt.plot(xx, mean_unlearn_points, label="Unlearn "+r"$\mathbb{P}$", color='red')
    plt.plot(xx, true_fun_points, label="True Function", linestyle='-.')
    plt.plot(xx, mean_unlearn_points + 2*np.diag(std_unlearn_points), '--', color='gray')
    plt.plot(xx, mean_unlearn_points - 2*np.diag(std_unlearn_points), '--', color='gray')
    plt.legend()
    plt.xlabel("Points")
    plt.title("(a) Unlearned "+ r"$\mathbb{P}$" + " Method", pad=15)
    plt.subplot(1, 3, 2)
    plt.plot(xx, mean_learn_f_points, label="Learn "+r"$\mathbb{P}^{\theta}$", color='red')
    plt.plot(xx, true_fun_points, label="True Function", linestyle='-.')
    plt.plot(xx, mean_learn_f_points + 2*np.diag(std_learn_f_points), '--', color='gray')
    plt.plot(xx, mean_learn_f_points - 2*np.diag(std_learn_f_points), '--', color='gray')
    plt.legend()
    plt.xlabel("Points")
    plt.title("(b) Learned " + r"$\mathbb{P}^{\theta}$" + " Method", pad=15)
    plt.subplot(1, 3, 3)
    plt.plot(xx, mean_learn_fS_points, label="Learn "+r"$\mathbb{P}_{S}^{\theta}$", color='red')
    plt.plot(xx, true_fun_points, label="True Function", linestyle='-.')
    plt.plot(xx, mean_learn_fS_points + 2*np.diag(std_learn_fS_points), '--', color='gray')
    plt.plot(xx, mean_learn_fS_points - 2*np.diag(std_learn_fS_points), '--', color='gray')
    plt.legend()
    plt.xlabel("Points")    
    plt.title("(c) Learned " + r"$\mathbb{P}_{S}^{\theta}$" + " Method", pad=15)
    plt.savefig(results_fig_table + env + "_appro_std_analysis_" + str(ii_) + ".png", dpi=1000, bbox_inches='tight')
    plt.savefig(results_fig_table + env + "_appro_std_analysis_" + str(ii_) + ".pdf", bbox_inches='tight')
    plt.close()


