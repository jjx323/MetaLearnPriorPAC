## loading necessary general libraries
import numpy as np
import pickle

import fenics as fe
import torch
import argparse

## loading modules in the core folder
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.misc import save_expre

## loading the EquSolver for solving diffusion equations
from BackwardDiffusion.common import EquSolver

## test = True: generate the testing data
## test = False: generate the training data
## env = "simple": generate data under the simple environment (only one branch)
## env = "complex": generate data under the complex environment (has two branches)
parser = argparse.ArgumentParser(description="generate data for diffusion problem")
parser.add_argument('--test_true', '-v', action='store_true', help='test = False or True')
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')

args = parser.parse_args()

test = args.test_true
env = args.env
assert env in ['simple', 'complex']

if test == False:
    ## Fix the random seeds
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
if test == True:
    ## Fix the random seeds
    np.random.seed(1243)
    torch.manual_seed(1243)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1243)

print("Test mode: ", test)
print("Environment is ", env)

DATA_DIR = './DATA/'  # set the data folder
os.makedirs(DATA_DIR, exist_ok=True)

# set mesh with 600 grid points
equ_nx = 600
# set the domain with Lagrange element of order 1
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
# set the noise level
noise_level = 0.1
np.save(DATA_DIR + "noise_level", noise_level)

## Set the parameters
##     T: the final time of the diffusion equation
##     num_steps: the number of step numbers for solving the diffusion equation
##     n: the number of training data
T = 0.01
num_steps = 20
n = 2000
num_points = 20
meta_data = []
clean_data = []
coordinates_list = []
u_list = []

def generate_data(u=None, itr=0, T=0.1, num_steps=20, env='simple', num_points=5):
    beta = np.random.normal(0.5, 0.5)
    a = np.random.uniform(5, 15) 
    b = np.random.normal(0, 0.1)
    c = np.random.normal(4, 1)
    
    if u is None:
        if env == 'simple':
            expression = "(beta*x[0]*5 + a*sin(2*(x[0]*5 - b)) + c)*exp(-20*pow(x[0]-0.5, 2))"
        elif env == 'complex':
            if itr % 2 == 0:
                expression = "(beta*x[0]*5 + a*sin(2*(x[0]*5 - b)) + c)*exp(-20*pow(x[0]-0.5, 2))"
            else:
                expression = "-(beta*x[0]*5 + a*sin(2*(x[0]*5 - b)) + c)*exp(-20*pow(x[0]-0.5, 2))"

        u_expre = fe.Expression(expression, degree=3, 
                                beta=beta, a=a, b=b, c=c) 
        u = fe.interpolate(u_expre, domain.function_space)
    else:
        u = fe.interpolate(u, domain.function_space)
    
    # fe.plot(u)
    # plt.title("u" + str(itr+1))
    num_points = 5 # the number of observed points
    coordinates = np.random.uniform(0, 1, (num_points,)) # the coordinates of the observed points

    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates]).T, m=u)
    ## Use equ_solver.forward_solver to solve the forward diffusion problem.
    ## Then using equ_solver.S to get the observed data without noise.
    d_clean = equ_solver.S@equ_solver.forward_solver()
    
    ## Add noises to the clean data
    d = d_clean + noise_level*np.random.normal(0, 1, (len(d_clean),))
    
    return coordinates, d, d_clean, u

if env == "simple":
    n = int(n/2)
## generate n number of data
for itr in range(n):
    coordinates, d, dc, u = generate_data(itr=itr, T=T, num_steps=num_steps, env=env, num_points=num_points)
    coordinates_list.append(coordinates)
    meta_data.append(d)
    clean_data.append(dc)
    u_list.append(u)
    del u
    if itr % 100 == 0:
        print("itr = ", itr)
    
    
# plt.figure()
# min_val, max_val = 0, 0
# for itr in range(n):
#     uu = u_list[itr]
#     if itr == 0:
#         min_val = min(uu.vector()[:])
#         max_val = max(uu.vector()[:])
#     else:
#         min_val = min(min_val, min(uu.vector()[:]))
#         max_val = max(max_val, max(uu.vector()[:]))
#     fe.plot(uu)
#     plt.ylim([min_val, max_val])

## Transform the function lists into vectors to save the data
u_vectors = []
for itr in range(n):
    u_vectors.append(u_list[itr].vector()[:])


if test == False:
    file2 = fe.File(DATA_DIR + env + '_saved_mesh_meta.xml')
    file2 << domain.function_space.mesh()
    
    with open(DATA_DIR + env + '_meta_parameters', "wb") as f:
        pickle.dump(u_vectors, f)
    
    with open(DATA_DIR + env + '_meta_data_x', "wb") as f:
        pickle.dump(coordinates_list, f)
        
    with open(DATA_DIR + env + '_meta_data_dc', "wb") as f:
        pickle.dump(clean_data, f)
        
    with open(DATA_DIR + env + '_meta_data_y', "wb") as f:
        pickle.dump(meta_data, f)
        
    equ_params = [T, num_steps]
    np.save(DATA_DIR + env + "_equation_parameters", equ_params)
elif test == True:
    file2 = fe.File(DATA_DIR + env + '_saved_mesh_meta_test.xml')
    file2 << domain.function_space.mesh()
    with open(DATA_DIR + env + '_meta_parameters_test', "wb") as f:
        pickle.dump(u_vectors, f)
    
    with open(DATA_DIR + env + '_meta_data_x_test', "wb") as f:
        pickle.dump(coordinates_list, f)
    
    with open(DATA_DIR + env + '_meta_data_dc_test', "wb") as f:
        pickle.dump(clean_data, f)
    
    with open(DATA_DIR + env + '_meta_data_y_test', "wb") as f:
        pickle.dump(meta_data, f)
        
    equ_params = [T, num_steps]
    np.save(DATA_DIR + env + "_equation_parameters_test", equ_params)
else:
    raise NotImplementedError("test should be True or False")








