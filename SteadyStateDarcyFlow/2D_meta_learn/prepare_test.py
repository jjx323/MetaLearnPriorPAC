## import necessary packages
import numpy as np 
import fenics as fe
import torch 
import argparse

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import load_expre

from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss, file_process


## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
args = parser.parse_args()

## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
results_fig_table = "./RESULTS-PAPER/"
os.makedirs(results_fig_table, exist_ok=True)

env = args.env
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

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
## d2v is used to transfer the grid coordinates. 
d2v = np.array(fe.dof_to_vertex_map(domain.function_space), dtype=np.int64)
v2d = np.array(fe.vertex_to_dof_map(domain.function_space), dtype=np.int64)
coor_transfer = {"d2v": d2v, "v2d": v2d}

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
print("n = ", n, ", m = ", m)

train_model_params = trans_model_data(
    train_model_params, V_truth, domain.function_space
    )

## Transfer discrete observation points to the functions defined on domain
dis2fun = Dis2Fun(domain=domain, points=train_dataset_x[0, :], alpha=0.01)

train_dataset_Sy = []
for itr in range(n):
    dis2fun.reset_points(train_dataset_x[itr, :])
    Sy = dis2fun(train_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    train_dataset_Sy.append(Sy)
train_dataset_Sy = np.array(train_dataset_Sy)

train_model_params = torch.tensor(train_model_params, dtype=torch.float32)
train_model_params = train_model_params[:, v2d].reshape(n, equ_nx+1, equ_nx+1)
normalize_model_params = UnitNormalization(train_model_params)
train_dataset_x = torch.tensor(train_dataset_x, dtype=torch.float32)
train_dataset_Sy = torch.tensor(train_dataset_Sy, dtype=torch.float32)
normalize_train_Sy = UnitNormalization(train_dataset_Sy)

train_dataset_Sy0 = np.load(meta_data_dir + "train_dataset_Sy0_" + str(num_points) + ".npy")[::env_num]
train_dataset_Sy0 = torch.tensor(train_dataset_Sy0, dtype=torch.float32)
normalize_train_Sy0 = UnitNormalization(train_dataset_Sy0)


test_model_params = np.load(meta_data_dir + "test_model_params.npy")[::env_num]
test_dataset_x = np.load(meta_data_dir + "test_dataset_x_" + str(num_points) + ".npy")[::env_num]
test_dataset_y = np.load(meta_data_dir + "test_dataset_y_" + str(num_points) + ".npy")[::env_num]
# indices = list(range(0, test_dataset_x.shape[1], 5))
# test_dataset_x = test_dataset_x[:, indices, :]
# test_dataset_y = test_dataset_y[:, indices]
# print(test_dataset_y.shape, test_dataset_x.shape)
n_test = test_dataset_x.shape[0]

test_model_params = trans_model_data(
    test_model_params, V_truth, domain.function_space
    )

# ## Transfer discrete observation points to the functions defined on domain
dis2fun = Dis2Fun(domain=domain, points=test_dataset_x[0, :], alpha=0.01)

test_dataset_Sy = []
for itr in range(n_test):
    dis2fun.reset_points(test_dataset_x[itr, :])
    Sy = dis2fun(test_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    test_dataset_Sy.append(Sy)
test_dataset_Sy = np.array(test_dataset_Sy)

test_model_params = torch.tensor(test_model_params, dtype=torch.float32)
test_model_params = test_model_params[:, v2d].reshape(n_test, equ_nx+1, equ_nx+1)
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32)
test_dataset_Sy = torch.tensor(test_dataset_Sy, dtype=torch.float32)
normalize_test_Sy = UnitNormalization(test_dataset_Sy)

test_dataset_Sy0 = np.load(meta_data_dir + "test_dataset_Sy0_" + str(num_points) + ".npy")[::env_num]
test_dataset_Sy0 = torch.tensor(test_dataset_Sy0, dtype=torch.float32)
normalize_test_Sy0 = UnitNormalization(test_dataset_Sy0)

f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

small_n = equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)

nnprior_mean = FNO2d(
    modes1=12, modes2=12, width=12, coordinates=domain.mesh.coordinates(),
    mode="residual"
    ).to(device)

## load results
if with_hyper_prior == True:
    if kk_FNO == -1:
        dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean_prior"    
    else:
        dir_nn = meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_mean_prior"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    if kk_FNO == -1:
        loss_list = np.load(
            meta_results_dir + env + str(50) + "_meta_FNO_loss_prior.npy"
            )
    else:
        loss_list = np.load(
            meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_loss_prior.npy"
            )
    prior_log_lam = np.load(
        meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_log_lam_prior.npy", 
        )
else:
    if kk_FNO == -1:
        dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean"
    else:
        dir_nn = meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_mean"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    if kk_FNO == -1:
        loss_list = np.load(
            meta_results_dir + env + str(50) + "_meta_FNO_loss.npy"
            ) 
    else:
        loss_list = np.load(
            meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_loss.npy"
            )
    prior_log_lam = np.load(
        meta_results_dir + env + str(kk_FNO) + "_" + str(50) + "_meta_FNO_log_lam.npy", 
        )

forward_nn = ForwardProcessNN(
    nn_model=nnprior_mean, coor_transfer=coor_transfer, 
    normalize_data=normalize_train_Sy, normalize_param=normalize_model_params, 
    normalize_data0=normalize_train_Sy0,
    device=device
    )

loss_lp = LpLoss()

Sy = test_dataset_Sy.to(torch.device(device))
Sy0 = test_dataset_Sy0.to(torch.device(device))
output_FNO = forward_nn(Sy, Sy0)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
test_error = loss_lp(output_FNO.cpu(), test_model_params)
test_error = test_error.cpu().detach().numpy()
print("Test Error: ", test_error)

## save the output of the neural network

output_FNO_cpu = output_FNO.detach().cpu().numpy()
nr, nc = output_FNO_cpu[0].shape
path_FNO_results = meta_results_dir + "FNO_results/"
if not os.path.exists(path_FNO_results):
    os.makedirs(path_FNO_results)

for idx in range(len(output_FNO_cpu)):
    tmp1 = np.array(output_FNO_cpu[idx].reshape(nr*nc)[coor_transfer["d2v"]])
    tmp2 = np.array(test_model_params[idx].reshape(nr*nc)[coor_transfer["d2v"]])
    np.save(path_FNO_results + env + "_result_" + str(idx), [tmp1, tmp2])
    
del output_FNO_cpu, tmp1, tmp2
print("Outputs of FNO have been saved.")