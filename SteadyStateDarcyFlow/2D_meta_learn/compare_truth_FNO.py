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

tmp = np.array(np.load(path + env + "_result_" + str(idx1) + ".npy"))
if env == "complex":
    nn = 7300#2500
else:
    nn = 7300#6200
learned_mean_f = np.load(
    meta_results_dir + env + str(nn) + "_" + str(equ_nx) + "_meta_prior_mean.npy"
    )

fun_truth.vector()[:] = np.array(tmp[1])
fun_FNO.vector()[:] = np.array(tmp[0])
fun_mean.vector()[:] = np.array(learned_mean_f)

####################################################################
#mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.4

if env == "simple":
    plt.figure(figsize=(17,5))
    plt.subplot(1,3,1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(a) True function")
    plt.subplot(1,3,2)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig)
    plt.title("(b) Estimated by FNO")
    plt.subplot(1,3,3)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig)
    plt.title("(c) Estimated mean function")
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2) 
    plt.savefig(meta_results_dir + "fig_simple.png", dpi=500)
    plt.close()

if env == "complex":
    plt.figure(figsize=(16,11))
    plt.subplot(2,3,1)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(a) True function (Branch 1)")
    plt.subplot(2,3,2)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig)
    plt.title("(b) Estimated by FNO (Branch 1)")
    plt.subplot(2,3,3)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig)
    plt.title("(c) Estimated mean function")
    
    tmp = np.array(np.load(path + "complex_result_" + str(idx2) + ".npy"))
    fun_truth.vector()[:] = np.array(tmp[1])
    fun_FNO.vector()[:] = np.array(tmp[0])
    
    plt.subplot(2,3,4)
    fig = fe.plot(fun_truth)
    plt.colorbar(fig)
    plt.title("(d) True function (Branch 2)")
    plt.subplot(2,3,5)
    fig = fe.plot(fun_FNO)
    plt.colorbar(fig)
    plt.title("(e) Estimated by FNO (Branch 2)")
    plt.subplot(2,3,6)
    fig = fe.plot(fun_mean)
    plt.colorbar(fig)
    plt.title("(f) Estimated mean function")
    
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2) 
    plt.savefig(meta_results_dir + "fig_complex.png", dpi=500)
    plt.close()
































