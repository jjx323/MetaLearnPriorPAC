## import necessary packages
import numpy as np
import fenics as fe
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D


parser = argparse.ArgumentParser(description="train prediction function f(S;theta)")
parser.add_argument('--env', type=str, default="simple", help='env = "simple" or "complex"')
args = parser.parse_args()

path = "./RESULTS/FNO_results/"
meta_results_dir = "./RESULTS/"
env = args.env

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

idx1 = 2
idx2 = 3

fun_truth = fe.Function(domain.function_space)
fun_FNO = fe.Function(domain.function_space)
fun_mean = fe.Function(domain.function_space)

print(path + env + "_result_" + str(idx1) + ".npy")
tmp = np.array(np.load(path + env + "_result_" + str(idx1) + ".npy"))
if env == "complex":
    nn = 7300
else:
    nn = 7300
learned_mean_f = np.load(
    meta_results_dir + env + str(nn) + "_" + str(equ_nx) + "_meta_prior_mean.npy"
    )

fun_truth.vector()[:] = np.array(tmp[1])
fun_FNO.vector()[:] = np.array(tmp[0])
fun_mean.vector()[:] = np.array(learned_mean_f)

####################################################################
import scienceplots
plt.style.use('science')
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17
# plt.rcParams['axes.linewidth'] = 0.4

if env == "simple":
    plt.figure(figsize=(17,5))
    plt.subplot(1,3,1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(a) True function", pad=15)
    plt.subplot(1,3,2)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig)
    plt.title("(b) Estimated by FNO", pad=15)
    plt.subplot(1,3,3)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig)
    plt.title("(c) Estimated mean function", pad=15)
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2) 
    plt.savefig("DarcyFlowSimple.png", dpi=1000)
    # plt.savefig("fig_simple.pdf")
    plt.close()

if env == "complex":
    plt.figure(figsize=(16,11))
    plt.subplot(2,3,1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(a) True function (Branch 1)", pad=15)
    plt.subplot(2,3,2)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(b) Estimated by FNO (Branch 1)", pad=15)
    plt.subplot(2,3,3)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(c) Estimated mean function", pad=15)
    
    tmp = np.array(np.load(path + "complex_result_" + str(idx2) + ".npy"))
    fun_truth.vector()[:] = np.array(tmp[1])
    fun_FNO.vector()[:] = np.array(tmp[0])
    
    plt.subplot(2,3,4)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(d) True function (Branch 2)", pad=15)
    plt.subplot(2,3,5)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(e) Estimated by FNO (Branch 2)", pad=15)
    plt.subplot(2,3,6)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig, shrink=0.7)
    plt.title("(f) Estimated mean function", pad=15)
    
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0) 
    plt.savefig("DarcyFlowComplex.png", dpi=1000)
    # plt.savefig("fig_complex.pdf")
    plt.close()
































