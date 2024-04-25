## import necessary packages
import numpy as np
import fenics as fe
import torch

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import save_expre
from SteadyStateDarcyFlow.common import EquSolver


## Fix the random seeds
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

data_dir = './DATA/'
os.makedirs(data_dir, exist_ok=True)

noise_level = 0.01
## the number of training data
num_training_data = 2000
num_testing_data = 200
## the number of measurement points \{ (x_i, y_i) \}_{i=1}^{num_points}
num = np.arange(1, 100)
num_points = []
for nn in num:
    num_points.append(nn*25)

## domain for solving PDE
equ_nx = 200
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

def generate_model_parameter(domain, branch=1):
    def random_numbers():
        a11 = np.random.uniform(0.1, 0.5)
        a21 = np.random.uniform(0.1, 0.5)
        a31 = np.random.uniform(0.1, 0.5)
        a12 = np.random.uniform(0.1, 0.5)
        a22 = np.random.uniform(0.1, 0.5)
        a32 = np.random.uniform(0.1, 0.5)
        a13 = np.random.uniform(3, 4)
        a23 = np.random.uniform(3, 4)
        a33 = np.random.uniform(3, 4)
        a14 = np.random.uniform(30, 35)
        a24 = np.random.uniform(30, 35)
        a34 = np.random.uniform(30, 35)
        a16 = np.random.uniform(30, 35)
        a26 = np.random.uniform(30, 35)
        a36 = np.random.uniform(30, 35)
        a15 = np.random.uniform(0.15, 0.25)
        a25 = np.random.uniform(0.45, 0.55)
        a35 = np.random.uniform(0.65, 0.75)
        a17 = np.random.uniform(0.65, 0.75)
        a27 = np.random.uniform(0.45, 0.55)
        a37 = np.random.uniform(0.15, 0.25)
        return a11, a21, a31, a12, a22, a32, a13, a23, a33, a14, a24, a34, \
               a15, a25, a35, a16, a26, a36, a17, a27, a37
            
    a = random_numbers()
    a11, a21, a31, a12, a22, a32, a13, a23, a33 = a[:9]
    a14, a24, a34, a15, a25, a35, a16, a26, a36, a17, a27, a37 = a[9:]
    
    f_expre01 = "pow(1-pow((x[0]),2), a11)*pow(1-pow((x[1]),2),a12)*a13*" + \
                "exp(-a14*pow((x[0])-a15,2)-a16*pow((x[1])-a17, 2))"
    f_expre02 = "pow(1-pow((x[0]),2), a21)*pow(1-pow((x[1]),2),a22)*a23*" + \
                "exp(-a24*pow((x[0])-a25,2)-a26*pow((x[1])-a27, 2))"
    f_expre03 = "pow(1-pow((x[0]),2), a31)*pow(1-pow((x[1]),2),a32)*a33*" + \
                "exp(-a34*pow((x[0])-a35,2)-a36*pow((x[1])-a37, 2))"
    
    f_sign = "aa*"
    
    f_expre = f_sign + "(" + f_expre01 + "+" + f_expre02 + "+" + f_expre03 + ")"
    
    if branch == 1:
        aa = 1
    else:
        aa = -1
        
    fun = fe.interpolate(
            fe.Expression(
                f_expre, degree=3, 
                a11=a11, a21=a21, a31=a31, a12=a12, a22=a22, a32=a32,
                a13=a13, a23=a23, a33=a33, a14=a14, a24=a24, a34=a34,
                a15=a15, a25=a25, a35=a35, a16=a16, a26=a26, a36=a36,
                a17=a17, a27=a27, a37=a37, aa = aa
                ), domain.function_space 
            )  
    
    return np.array(fun.vector()[:])  

def generate_measure_points(num_points=100):
    points = []
    for itr in range(num_points):
        xx = np.random.uniform(0, 1)
        yy = np.random.uniform(0, 1)
        points.append((xx, yy))
    
    return np.array(points)


## generate training data

## specify the force term f
f_ = "sin(1*pi*x[0])*sin(1*pi*x[1])"
f = fe.Expression(f_, degree=5)
save_expre(data_dir + 'f_2D.txt', f_)


def generate_datasets(num_data, num_points):
    model_params = []
    dataset_xs = [[] for _ in num_points]
    dataset_ys = [[] for _ in num_points]
    for itr in range(num_data):
        if itr % 2 == 0:
            param = generate_model_parameter(domain, branch=1)
        else:
            param = generate_model_parameter(domain, branch=2)
        
        fun_param = fe.Function(domain.function_space)
        fun_param.vector()[:] = np.array(param)
        
        if itr == 0:
            os.makedirs(data_dir, exist_ok=True)
            np.save(data_dir + 'truth_vec', fun_param.vector()[:])
            file1 = fe.File(data_dir + "truth_fun.xml")
            file1 << fun_param
            file2 = fe.File(data_dir + 'saved_mesh_truth.xml')
            file2 << domain.function_space.mesh()
        
        points = generate_measure_points(num_points=num_points[0])
        equ_solver = EquSolver(domain_equ=domain, m=fun_param, f=f, points=points)
    
        sol = np.array(equ_solver.forward_solver())
        
        for idx, num_point in enumerate(num_points):
            points = generate_measure_points(num_points=num_point)
            equ_solver.update_points(points)
            sol_points = np.array(equ_solver.S@sol).squeeze()
            noise = noise_level*max(sol_points)*np.random.normal(0, 1, len(sol_points))
            noise = noise.squeeze()
            sol_points = sol_points + noise    
            dataset_xs[idx].append(points)
            dataset_ys[idx].append(sol_points)
            
        model_params.append(fun_param.vector()[:])
    
        print("Number: %d/%d" % (itr+1, num_data))
    
    return model_params, dataset_xs, dataset_ys
    

out = generate_datasets(num_training_data, num_points)
train_model_params, train_dataset_xs, train_dataset_ys = out

np.save(data_dir + "train_model_params", train_model_params)
for idx, train_dataset_x in enumerate(train_dataset_xs):
    np.save(data_dir + "train_dataset_x_" + str(num_points[idx]), train_dataset_x)
for idx, train_dataset_y in enumerate(train_dataset_ys):
    np.save(data_dir + "train_dataset_y_" + str(num_points[idx]), train_dataset_y)
    
    
out = generate_datasets(num_testing_data, num_points)
test_model_params, test_dataset_xs, test_dataset_ys = out  

np.save(data_dir + "test_model_params", test_model_params)
for idx, test_dataset_x in enumerate(test_dataset_xs):
    np.save(data_dir + "test_dataset_x_" + str(num_points[idx]), test_dataset_x)
for idx, test_dataset_y in enumerate(test_dataset_ys):
    np.save(data_dir + "test_dataset_y_" + str(num_points[idx]), test_dataset_y)

    


































