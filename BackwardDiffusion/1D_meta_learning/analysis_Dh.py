## import necessary packages
import numpy as np
import fenics as fe
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 前设置为无图形界面模式
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
results_fig_table = './RESULTS-PAPER-Dh/'
os.makedirs(results_fig_table, exist_ok=True)
env = "complex"
noise_level = np.load(data_dir  + "noise_level.npy")

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

hidden_dims = [5, 10, 15, 20, 25, 30]
errors = []
for hidden_dim in hidden_dims:
    for idx_ in range(100):

        """
        load model parameters; load test samples 
        """
        errors_tmp = []
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
        with_dir = meta_results_dir + env + str(equ_nx) + "_" + str(hidden_dim) + "_" + "_meta_FNO_mean_prior"

        nnprior_mean_MAP = FNO1d(
            modes=15, width=hidden_dim
            )
        nnprior_mean_MAP.load_state_dict(torch.load(with_dir))
        nnprior_mean_MAP.eval()

        mean_fS_MAP = fe.Function(domain.function_space)
        tmp = nnprior_mean_MAP(Sy, gridx_tensor).reshape(-1).detach().numpy()
        mean_fS_MAP.vector()[:] = np.array(tmp)

        errors_tmp.append(relative_error(mean_fS_MAP, u_meta_fun_test, domain, err_type="L2"))
    
    errors.append(np.mean(errors_tmp))

print("Errors with hidden dim ", errors)
np.save(results_fig_table + "errors", errors)

file_process(results_fig_table + "errors.txt")
with open(results_fig_table + "errors.txt", "a") as f:
    f.write(str(errors))




