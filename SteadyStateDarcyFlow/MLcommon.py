#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:30:58 2022

@author: Junxiong Jia
"""

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import dolfin as dl 
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.pardir)

from core.misc import trans2spnumpy, spnumpy2sptorch, trans2sptorch, sptorch2spnumpy, \
    construct_measurement_matrix
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.model import Domain2D
from core.noise import NoiseGaussianIID
from core.optimizer import NewtonCG

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow


###############################################################################
class GaussianFiniteRankTorch(GaussianFiniteRank):
    def __init__(self, domain, domain_=None, mean=None, num_KL=None, 
                 alpha=1.0, beta=1.0, s=2):
        super().__init__(domain=domain, domain_=domain_, mean=mean, num_KL=num_KL,
                         alpha=alpha, beta=beta, s=s)
        self.is_torch = False
        
    def learnable_mean(self):
        if self.is_torch == False:
            self.trans2torch()
        self.mean_vec_torch.requires_grad = True
        
    def learnable_loglam(self):
        if self.is_torch == False:
            self.trans2torch() 
        self.log_lam.requires_grad = True 
        
    def trans2torch(self, device="cpu", dtype=torch.float32):
        if device == "cpu":
            self.trans2torch_cpu(dtype=dtype)
        elif device == "cuda":
            self.trans2torch_cuda(dtype=dtype)
        else:
            raise NotImplementedError("device must be cpu or cuda")
    
    def trans2torch_cpu(self, dtype=torch.float32):
        if self.is_torch == False:
            self.mean_vec_torch = torch.tensor(
                self.mean_vec, dtype=dtype, requires_grad=False
                ) 
            self.lam = torch.tensor(self.lam, dtype=dtype, requires_grad=False)
            self.log_lam = torch.tensor(self.log_lam, dtype=dtype, requires_grad=False)
            self.eigvec = torch.tensor(self.eigvec, dtype=dtype, requires_grad=False)
            self.eigvecT = torch.transpose(self.eigvec, 0, 1)
            self.eigvec_ = torch.tensor(self.eigvec_, dtype=dtype, requires_grad=False)
            self.eigvec_T = torch.transpose(self.eigvec_, 0, 1)
            if type(self.Ms) == np.ndarray:
                self.Ms = torch.tensor(self.Ms, dtype=dtype, requires_grad=False) 
            else:
                self.Ms = torch.tensor(self.Ms.todense(), dtype=dtype, requires_grad=False) 
            self.M_gpu = torch.tensor(self.M.diagonal(), dtype=dtype, requires_grad=False)
            self.M_torch = spnumpy2sptorch(self.M)
            # self.K_torch = spnumpy2sptorch(self.K)
            if type(self.f2sM) == np.ndarray:
                self.f2sM = torch.tensor(self.f2sM, dtype=dtype, requires_grad=False)
            else:
                self.f2sM = torch.tensor(self.f2sM.todense(), dtype=dtype, requires_grad=False)
            if type(self.s2fM) == np.ndarray:
                self.s2fM = torch.tensor(self.s2fM, dtype=dtype, requires_grad=False)
            else:
                self.s2fM = torch.tensor(self.s2fM.todense(), dtype=dtype, requires_grad=False)
            self.is_torch = True
        else:
            print("need not trans2torch")
    
    def trans2torch_cuda(self, dtype=torch.float32):
        if self.is_torch == False:
            self.mean_vec_torch = torch.tensor(
                self.mean_vec, dtype=dtype, requires_grad=False
                ).cuda()
            self.lam = torch.tensor(self.lam, dtype=dtype, requires_grad=False).cuda()
            self.log_lam = torch.tensor(self.log_lam, dtype=dtype, requires_grad=False).cuda()
            self.eigvec = torch.tensor(self.eigvec, dtype=dtype, requires_grad=False).cuda()
            self.eigvecT = torch.transpose(self.eigvec, 0, 1).cuda()
            self.eigvec_ = torch.tensor(self.eigvec_, dtype=dtype, requires_grad=False).cuda()
            self.eigvec_T = torch.transpose(self.eigvec_, 0, 1).cuda()
            if type(self.Ms) == np.ndarray:
                self.Ms = torch.tensor(self.Ms, dtype=dtype, requires_grad=False).cuda() 
            else:
                self.Ms = torch.tensor(self.Ms.todense(), dtype=dtype, requires_grad=False).cuda() 
            # self.M = spnumpy2sptorch(self.M).cuda()
            if type(self.f2sM) == np.ndarray:
                self.f2sM = torch.tensor(self.f2sM, dtype=dtype, requires_grad=False).cuda()
            else:
                self.f2sM = torch.tensor(self.f2sM.todense(), dtype=dtype, requires_grad=False).cuda()
            if type(self.s2fM) == np.ndarray:
                self.s2fM = torch.tensor(self.s2fM, dtype=dtype, requires_grad=False).cuda()
            else:
                self.s2fM = torch.tensor(self.s2fM.todense(), dtype=dtype, requires_grad=False).cuda()
            self.M_gpu = torch.tensor(self.M.diagonal(), dtype=dtype, requires_grad=False).cuda()
            self.is_torch = True
        else:
            print("need not trans2torch")
            
    def trans2numpy(self):
        if self.is_torch == True:
            if self.mean_vec_torch.device.type == "cpu":
                self.mean_vec = np.array(self.mean_vec_torch.detach().numpy())
                self.lam = np.array(np.exp(self.log_lam.detach().numpy()))
                self.log_lam = np.array(self.log_lam.detach().numpy())
                self.eigvec = np.array(self.eigvec) 
                self.eigvec_ = np.array(self.eigvec_) 
                self.Ms = np.array(self.Ms)
                # self.M = sptorch2spnumpy(self.M)
                # self.K = sptorch2spnumpy(self.K)
                self.f2sM = np.array(self.f2sM)
                self.s2fM = np.array(self.s2fM)
            elif self.mean_vec_torch.device.type == "cuda":
                self.mean_vec = np.array(self.mean_vec_torch.cpu().detach().numpy())
                self.lam = np.array(np.exp(self.log_lam.cpu().detach().numpy()))
                self.log_lam = np.array(self.log_lam.cpu().detach().numpy())
                self.eigvec = np.array(self.eigvec.cpu()) 
                self.eigvec_ = np.array(self.eigvec_.cpu()) 
                self.Ms = np.array(self.Ms.cpu())
                # self.M = sptorch2spnumpy(self.M.cpu())
                self.f2sM = np.array(self.f2sM.cpu())
                self.s2fM = np.array(self.s2fM.cpu())
            else:
                raise NotImplementedError("the device must be cpu or cuda")
            self.is_torch = False
        else:
            print("need not trans2numpy")
            
    def generate_sample_zero_mean(self, num_sample=1, device="cpu"):
        assert self.is_eig_available == True

        if self.is_torch == True:
            if num_sample == 1:
                n = torch.normal(0, 1, (len(self.log_lam),)).to(device)
                val = torch.exp(self.log_lam)*n
            else:
                ## not tested 
                n = torch.normal(0, 1, (len(self.log_lam), num_sample)).to(device)
                val = torch.matmul(torch.diag(torch.exp(self.log_lam)), n)
            val = torch.matmul(self.eigvec, val)
            return val
        elif self.is_torch == False:
            if num_sample == 1:
                n = np.random.normal(0, 1, (len(self.log_lam),))
                val = np.exp(self.log_lam)*n
            else:
                ## not tested 
                n = np.random.normal(0, 1, (len(self.log_lam), num_sample))
                val = np.diag(np.exp(self.log_lam))@n
            val = self.eigvec@val
            return np.array(val)
        else:
            raise NotImplementedError("self.is_torch must be True or False")
    
    def generate_sample(self, num_sample=1):
        assert self.is_eig_available == True
        
        if self.is_torch == False:
            if num_sample == 1:
                val = self.mean_vec + self.generate_sample_zero_mean(num_sample=num_sample) 
                if self.mean_vec.shape != val.shape:
                    raise ValueError("mean_vec and val should have same shape")
                val = np.array(val)
            else:
                ## not tested 
                print("Generate sample may has error!")
                temp =self.generate_sample_zero_mean(num_sample=num_sample) 
                val = self.mean_vec.reshape(-1, 1) + temp
                val = np.array(val)
        elif self.is_torch == True:
            device = self.mean_vec_torch.device
            if num_sample == 1:
                val = self.mean_vec_torch + self.generate_sample_zero_mean(num_sample=num_sample, device=device) 
                if self.mean_vec_torch.shape != val.shape:
                    raise ValueError("mean_vec and val should have same shape")
            else:
                ## not tested 
                print("Generate sample may has error!")
                temp =self.generate_sample_zero_mean(num_sample=num_sample, device=device) 
                val = self.mean_vec_torch.reshape(-1, 1) + temp
        else:
            raise NotImplementedError("self.is_torch must be True of False")

        return val
    
    def evaluate_CM_inner_batch(self, u):
        return self.evaluate_CM_norm_batch(u)
    
    def evaluate_CM_norm_batch(self, u):
        if self.is_torch == True:
            if type(u) != type(self.mean_vec_torch):
                u = u.type(self.mean_vec_torch.dtype)
            if u.ndim == 1:
                u = u.reshape(1,-1)
                
            us = torch.matmul(self.f2sM, u.T)
            mean_vec_torch = torch.mv(self.f2sM, self.mean_vec_torch)
            res = us - mean_vec_torch.reshape(-1,1)
            val = torch.matmul(self.Ms, res)
            val = torch.matmul(self.eigvec_T, val)
            lam_n2 = torch.pow(torch.exp(self.log_lam), -2)
            lam_n2 = torch.diag(lam_n2)
            val = torch.matmul(lam_n2, val)
            val = torch.matmul(self.eigvec_, val)
            val = torch.matmul(self.Ms, val) 
            val = torch.matmul(res.T, val)
            val = torch.sum(val.diagonal())
            
            ## need further modification 
            val = val + 100*torch.norm(torch.matmul(self.s2fM, us) - u.T)
        elif self.is_torch == False:
            pass
        else:
            raise NotImplementedError("self.is_torch must be True or False")
        
        return val 
    
    def evaluate_norm(self, u):
        if u.ndim == 1:
            return self.evaluate_CM_inner(u)
        elif u.ndim == 2:
            return self.evaluate_CM_inner_batch(u)
        else:
            raise NotImplementedError("data.ndim should be 1 or 2")
    
    def evaluate_CM_inner(self, u, v=None):
        if v is None:
            v = u
        if self.is_torch == True:
            if type(u) != type(self.mean_vec_torch):
                u = u.type(self.mean_vec_torch.dtype)
            if type(v) != type(self.mean_vec_torch):
                v = v.type(self.mean_vec_torch.dtype)
            
            us = torch.mv(self.f2sM, u)
            vs = torch.mv(self.f2sM, v)
            mean_vec_torch = torch.mv(self.f2sM, self.mean_vec_torch)
            res = vs - mean_vec_torch
            val = torch.mv(self.Ms, res)
            val = torch.mv(self.eigvec_T, val)
            lam_n2 = torch.pow(torch.exp(self.log_lam), -2)
            val = lam_n2*val
            val = torch.mv(self.eigvec_, val) 
            val = torch.mv(self.Ms, val)
            val = torch.sum((us - mean_vec_torch)*val)
            
            ## Projecting to the low dimensional space, the regularization leads 
            ## discontinuous at the boundary. The reasons seem to be the 
            ## smoothness constraints are only added to the grid points of the 
            ## small mesh. 
            val = val + 100*torch.norm(torch.mv(self.s2fM, us) - u)
            val = val + 100*torch.norm(torch.mv(self.s2fM, vs) - v)
        elif self.is_torch == False:
            mean_vec = self.f2sM@self.mean_vec 
            v = self.f2sM@v
            u = self.f2sM@u
            res = v - mean_vec
            val = self.Ms@res
            val = self.eigvec_.T@val
            lam_n2 = np.power(np.exp(self.log_lam), -2)
            val = lam_n2*val
            val = self.eigvec_@val 
            val = self.Ms@val
            val = np.array((u - mean_vec)@val)
        else:
            raise NotImplementedError("self.is_torch must be True or False")
        
        return val


###############################################################################
class GaussianElliptic2Torch(GaussianElliptic2):
    def __init__(self, domain, alpha=1.0, a_fun=fe.Constant(1.0), theta=1.0, 
                 mean_fun=None, tensor=False, boundary='Neumann', 
                 use_LU=True):
        super().__init__(domain=domain, alpha=alpha, a_fun=a_fun, theta=theta, 
                     mean_fun=mean_fun, tensor=tensor, boundary=boundary, 
                     use_LU=use_LU)
        self.is_torch = False
        
    def learnable_mean(self):
        if self.is_torch == False:
            self.trans2torch()
        self.mean_vec_torch.requires_grad = True
        
    def trans2torch(self, device="cpu", dtype=torch.float32):
        if device == "cpu":
            self.trans2torch_cpu(dtype=dtype)
        elif device == "cuda":
            self.trans2torch_cuda(dtype=dtype)
        else:
            raise NotImplementedError("device must be cpu or cuda")
        
    def trans2torch_cpu(self, dtype=torch.float32):
        if self.is_torch == False:
            self.mean_vec_torch = torch.tensor(
                np.array(self.mean_fun.vector()[:]), dtype=dtype, requires_grad=False
                )
            self.K_torch = spnumpy2sptorch(self.K)
            self.KT = np.transpose(self.K)
            self.KT_torch = spnumpy2sptorch(self.KT)
            self.Minv_torch = torch.tensor(
                1.0/self.M.diagonal(), dtype=dtype, requires_grad=False
                )
            self.M_torch = spnumpy2sptorch(self.M)
            self.is_torch = True
        else:
            print("need not trans2torch")
    
    def trans2torch_cuda(self, dtype=torch.float32):
        ## there are critical problems for tranform spare matrix on gpu,
        ## the current implementation can not work correctly. 
        if self.is_torch == False:
            self.mean_vec_torch = torch.tensor(
                np.array(self.mean_fun.vector()[:]), dtype=dtype, requires_grad=False
                ).cuda()
            self.K_torch = spnumpy2sptorch(self.K, device="cuda")
            self.KT = np.transpose(self.K)
            self.KT_torch = spnumpy2sptorch(self.KT, device="cuda")
            self.Minv_torch = torch.tensor(
                1.0/self.M.diagonal(), dtype=dtype, requires_grad=False
                ).cuda()
            self.M_torch = spnumpy2sptorch(self.M, device="cuda")
            self.is_torch = True
        else:
            print("need not trans2torch")
        
    def trans2numpy(self):
        if self.is_torch == True:
            if self.mean_vec_torch.device.type == "cpu":
                self.mean_vec = self.mean_vec_torch.detach().numpy()
                # self.K = sptorch2spnumpy(self.K_torch)
                # self.M = sptorch2spnumpy(self.M_torch)
            elif self.mean_vec_torch.device.type == "cuda":
                self.mean_vec = self.mean_vec_torch.cpu().detach().numpy()
                # self.K = sptorch2spnumpy(self.K_torch.cpu())
                # self.M = sptorch2spnumpy(self.M_torch.cpu())
            else:
                raise NotImplementedError("the device must be cpu or cuda")    
            self.is_torch = False
        else:
            print("need not trans2numpy")
        
    def generate_sample(self, num_sample=1):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = m_{mean} + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        if self.is_torch == True:
            device = self.mean_vec_torch.device
            sample_ = self.generate_sample_zero_mean(num_sample=num_sample)
            sample_ = torch.tensor(sample_, dtype=self.mean_vec_torch.dtype).to(device)
            if num_sample == 1:
                sample = self.mean_vec_torch + sample_
                if self.mean_vec_torch.shape != sample.shape:
                    raise ValueError("sample and mean_vec_torch should have same shape")
            else:
                ## not tested
                print("Generate sample may has error!")
                sample = self.mean_vec_torch.reshape(-1,1) + sample_
        elif self.is_torch == False:
            sample_ = self.generate_sample_zero_mean(num_sample=num_sample)
            if num_sample == 1:
                sample = self.mean_vec + sample_
                if self.mean_vec.shape != sample.shape:
                    raise ValueError("sample and mean_vec_torch should have same shape")
            else:
                ## not tested
                print("Generate sample may has error!")
                sample = self.mean_vec.reshape(-1,1) + sample_

        return sample
    
    def generate_sample_zero_mean(self, num_sample=1):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = 0.0 + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        assert self.K is not None 
        assert self.M_half is not None
        
        n = np.random.normal(0, 1, (self.function_space_dim, num_sample))
        b = self.M_half@n
        self.boundary_vec(b)
        if self.use_LU == False:
            fun_vec = spsl.spsolve(self.K, b)
        elif self.use_LU == True:
            fun_vec = self.luK.solve(b)
        else:
            raise NotImplementedError("use_LU must be True or False")
        return np.array(fun_vec).squeeze()
    

class PriorFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, prior):
        M = spnumpy2sptorch(prior.M)
        K = spnumpy2sptorch(prior.K)
        ctx.save_for_backward(input.cpu(), M, K)
        device = input.device
        dtype = input.dtype
        
        m_vec = np.array(input.cpu().detach())
        ## solve forward equation
        if m_vec.ndim == 1:
            val = prior.K.T@spsl.spsolve(prior.M, prior.K@m_vec)
            output = m_vec@val
            output = torch.tensor(0.5*output, dtype=dtype).to(device)
        elif m_vec.ndim == 2:
            val = prior.K.T@spsl.spsolve(prior.M, prior.K@m_vec.T)
            output = m_vec@val
            output = np.sum(output.diagonal())
            output = torch.tensor(0.5*output, dtype=dtype).to(device)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        device = grad_output.device
        dtype = grad_output.dtype
        input, M, K = ctx.saved_tensors 
        M = sptorch2spnumpy(M)
        K = sptorch2spnumpy(K)
        
        m_vec = np.array(input)
        if m_vec.ndim == 1:
            val = spsl.spsolve(M, K.T@spsl.spsolve(M, K@m_vec))
            val = torch.tensor(val, dtype=dtype).to(device)
        elif m_vec.ndim == 2:
            val = spsl.spsolve(M, K.T@spsl.spsolve(M, K@m_vec.T))
            val = torch.tensor(val.T, dtype=dtype).to(device)
        return grad_output*val, None
        

class HyperPrior(object):
    def __init__(self, measure=None, fun_norm=None):
        self.measure = measure
        self.fun_norm = fun_norm
        if self.fun_norm is None:
            assert hasattr(self.measure, 'evaluate_norm')
        
    def to_torch(self, device='cpu'):
        ## device: string, 'cpu' or 'cuda'
        assert hasattr(self.measure, 'trans2torch')
        self.measure.trans2torch(device=device) 
    
    def evaluate_norm(self, x):
        if self.fun_norm is None:
            assert hasattr(self.measure, 'evaluate_norm')
            val = self.measure.evaluate_norm(x)
        else:
            if self.measure is None:
                val = self.fun_norm(x)
            else:
                val = self.fun_norm(x, self.measure)
        return val 


class HyperPriorAll(object):
    def __init__(self, hyper_params):
        ## hyper_params: lists, [param1, param2, ... paramn]
        ## parami is the instance of the class HyperPrior
        self.params = []
        for param in hyper_params:
            self.params.append(param)
            
    def evaluate_norm(self, xs):
        assert len(xs) == len(self.params)
        val = 0.0
        for idx, x in enumerate(xs):
            val += self.params[idx].evaluate_norm(x)
        return val 
    

class ForwardProcessNN(object):
    def __init__(self, nn_model, coor_transfer=None, normalize_data=None, normalize_param=None, 
                 normalize_data0=None, device='cpu'):
        self.nn_model = nn_model
        self.device = device
        self.normalize_data = normalize_data
        self.normalize_data0 = normalize_data0
        self.normalize_param = normalize_param
        self.coor_transfer = coor_transfer
        
    def __call__(self, dataset, dataset0=None):
        ## dataset: torch.tensor, (batch_size, nx, ny)
        assert dataset.ndim == 3 and type(dataset) == torch.Tensor
        if dataset0 is not None:
            assert dataset0.ndim == 3 and type(dataset0) == torch.Tensor
        lidx, lnx, lny = dataset.shape
        Sy = dataset.to(torch.device(self.device))
        if dataset0 is not None:
            Sy0 = dataset0.to(torch.device(self.device))
        if self.normalize_data is not None:
            Sy = self.normalize_data.encode(Sy)
        if self.normalize_data0 is not None:
            Sy0 = self.normalize_data0.encode(Sy0)
        Sy = Sy.reshape(lidx, lnx, lny, 1)
        if dataset0 is not None:
            Sy0 = Sy0.reshape(lidx, lnx, lny, 1)
        if dataset0 is not None:
            output = self.nn_model(Sy, Sy0).reshape(lidx, lnx, lny)
        else:
            output = self.nn_model(Sy).reshape(lidx, lnx, lny)
        if self.normalize_param is not None:
            output = self.normalize_param.decode(output)
        output = (output.reshape(lidx, -1))[:, self.coor_transfer["d2v"]]
        return output
    

class ForwardProcessPDE(object):
    def __init__(self, noise, equ_params, prior, mesh_transfer, 
                 weight=1.0, L=10, noise_level=0.01, device='cpu'):
        self.noise = noise
        self.prior = prior
        self.device = device
        self.domain = equ_params["domain"]
        self.f = equ_params["f"]
        self.pde_fun = equ_params["pde_fun"]
        self.L = L
        self.s2fM, self.f2sM = mesh_transfer["s2fM"], mesh_transfer["f2sM"]
        self.weight = weight
        self.dataset_x, self.dataset_y = None, None
        self.noise_level = noise_level
        
        points = np.array([[0,0]])
        self.equ_solver = EquSolver(
            domain_equ=self.domain, points=points, m=fe.Constant(0.0), f=self.f
            )
    
    def update_xy(self, dataset_x, dataset_y):
        assert dataset_x.ndim == 3 and dataset_y.ndim == 2
        assert type(dataset_x) == torch.Tensor

        self.dataset_x, self.dataset_y = dataset_x, dataset_y
        
    def update_noise(self, data):
        noise_level_ = self.noise_level*max(abs(data))
        self.noise.dim = len(data)
        self.noise.set_parameters(variance=noise_level_**2)  
        
    def eva_residual_PDE(self, output_nn):
        assert self.dataset_x is not None 
        assert self.dataset_y is not None
        
        nn = self.dataset_x.shape[0]
        panduan = 0
        loss_all = np.zeros(nn)
        
        for idx in range(nn):
            points = self.dataset_x[idx, :].cpu().detach().numpy() 
            ## -------------------------------------------------------------------
            ## Since the data dimension is different for different task, we need 
            ## different loss functions. 
            self.update_noise(self.dataset_y[idx, :])
            self.noise.to_torch(device=self.device)
    
            loss_residual = LossResidual(self.noise)
            
            ## -------------------------------------------------------------------
            ## for each dataset, we need to reconstruct equ_solver since the points are changed
            if panduan == 0:
                self.equ_solver.update_points(points)
                self.equ_solver.update_m()
                pdeasnet = PDEasNet(self.pde_fun, self.equ_solver)
                panduan = 1
            else:
                pdeasnet.equ_solver.update_points(points)
            
            targets = torch.tensor(self.dataset_y[idx], dtype=torch.float32).to(self.device)
            preds = pdeasnet(output_nn[idx, :].cpu())
            preds = preds.to(targets.device)
            val = loss_residual(preds, targets)
            loss_all[idx] = np.array(val.detach().cpu().numpy())
        
        return loss_all
    
    def __call__(self, output_nn):
        assert self.dataset_x is not None 
        assert self.dataset_y is not None
        lnZ = torch.zeros(1).to(self.device)
        nn = self.dataset_x.shape[0]
        panduan = 0
        prior0 = 0.0
        for idx in range(nn):
            points = self.dataset_x[idx, :].cpu().detach().numpy() 
            ## -------------------------------------------------------------------
            ## Since the data dimension is different for different task, we need 
            ## different loss functions. 
            self.update_noise(self.dataset_y[idx, :])
            self.noise.to_torch(device=self.device)
    
            loss_residual = LossResidual(self.noise)
            
            ## -------------------------------------------------------------------
            ## for each dataset, we need to reconstruct equ_solver since the points are changed
            if panduan == 0:
                # equ_solver = EquSolver(
                #     domain_equ=self.domain, points=points, m=fe.Constant(0.0), f=self.f
                #     )
                self.equ_solver.update_points(points)
                self.equ_solver.update_m()
                pdeasnet = PDEasNet(self.pde_fun, self.equ_solver)
                panduan = 1
            else:
                pdeasnet.equ_solver.update_points(points)
            
            loss_res_L = torch.zeros(self.L).to(self.device)
        
            targets = torch.tensor(self.dataset_y[idx], dtype=torch.float32).to(self.device)
            # prior.mean_vec_torch = output_FNO[idx, :]
            self.prior.mean_vec_torch = torch.matmul(self.f2sM, output_nn[idx, :])
            prior0 += 1e2*torch.norm(torch.matmul(self.s2fM, self.prior.mean_vec_torch) \
                                     - output_nn[idx, :])
            
            for ii in range(self.L):
                ul = self.prior.generate_sample()
                ## By experiments, I think the functions provided by CuPy (solving
                ## Ax=b with A is a large sparse matrix) are not efficient compared 
                ## with the cpu version given by SciPy. 
                ul = torch.matmul(self.s2fM, ul).cpu()
                preds = pdeasnet(ul)
                # preds = pdeasnet(ul.cpu())
                preds = preds.to(targets.device)
                val = -loss_residual(preds, targets)
                loss_res_L[ii] = val
            
            ## torch.logsumexp is used to avoid possible instability of computations
            lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                      - torch.log(torch.tensor(self.L, dtype=torch.float32)).to(self.device)
        
        return -self.weight*lnZ + prior0
                      
        
class ForwardPrior(object):
    def __init__(self, prior, forward_nn, rand_idx, dataset, is_log_lam=True, device="cpu", dataset0=None):
        self.forward_nn = forward_nn
        self.rand_idx = rand_idx 
        self.dataset_all = dataset
        self.dataset0_all = dataset0
        self.dataset = dataset[rand_idx, :].to(torch.device(device))
        if self.forward_nn.nn_model.mode == "residual":
            if dataset0 is not None:
                self.dataset0 = dataset0[rand_idx, :].to(torch.device(device))
            else:
                raise ValueError("When forward_nn.mode is residual, dataset0 cannot be None")
        self.device = device
        self.prior = prior
        self.is_log_lam = is_log_lam
        
    def update_rand_idx(self, rand_idx):
        self.rand_idx = rand_idx 
        if self.forward_nn.nn_model.mode == "non-residual":
            self.dataset = self.dataset_all[rand_idx, :].to(torch.device(self.device))
        elif self.forward_nn.nn_model.mode == "residual":
            self.dataset = self.dataset_all[rand_idx, :].to(torch.device(self.device))
            self.dataset0 = self.dataset0_all[rand_idx, :].to(torch.device(self.device))
        else:
            raise ValueError("forward_nn.mode must be residual or non-residual")
        
    def __call__(self, prior_base):
        if self.forward_nn.nn_model.mode == "non-residual":
            val = self.forward_nn(self.dataset) 
        elif self.forward_nn.nn_model.mode == "residual":
            val = self.forward_nn(self.dataset, self.dataset0)
        else:
            raise ValueError("forward_nn.mode must be residual or non-residual")
            
        if self.is_log_lam is True:
            val = self.prior.evaluate_norm([val, prior_base.log_lam])
        elif self.is_log_lam is False:
            val = self.prior.evaluate_norm([val])
        return val


class LossFun(object):
    def __init__(self, forward_process, prior_base, with_hyper_prior=True):
        self.with_hyper_prior = with_hyper_prior
        self.forward_process = forward_process 
        self.prior_base = prior_base
        if self.with_hyper_prior == True:
            assert len(self.forward_process) >= 3
        
    def __call__(self, Sy, dataset_x, dataset_y, rand_idx=None, Sy0=None):
        if Sy0 is None:
            val = self.forward_process["forward_nn"](Sy)
        else:
            if self.forward_process["forward_nn"].nn_model.mode == "residual":
                val = self.forward_process["forward_nn"](Sy, Sy0)
            else:
                raise ValueError("When Sy0 is not None, NN.mode should be residual")
        self.forward_process["forward_pde"].update_xy(dataset_x=dataset_x, dataset_y=dataset_y)
        loss1 = self.forward_process["forward_pde"](val)
        if self.with_hyper_prior == True:
            if rand_idx is None:
                loss2 = self.forward_process["forward_prior"](self.prior_base)
            else:
                self.forward_process["forward_prior"].update_rand_idx(rand_idx)
                loss2 = self.forward_process["forward_prior"](self.prior_base)
            return loss1 + loss2, loss1, loss2  
        return loss1   


class AdaptiveLossFun(object):
    def __init__(self, forward_process):
        self.forward_process = forward_process 

    def __call__(self, Sy, dataset_x, dataset_y, Sy0=None):
        with torch.set_grad_enabled(False):
            if Sy0 is None:
                val = self.forward_process["forward_nn"](Sy)
            else:
                if self.forward_process["forward_nn"].nn_model.mode == "residual":
                    val = self.forward_process["forward_nn"](Sy, Sy0)
                else:
                    raise ValueError("When Sy0 is not None, NN.mode should be residual")
            
            self.forward_process["forward_pde"].update_xy(dataset_x=dataset_x, dataset_y=dataset_y)
            loss = self.forward_process["forward_pde"].eva_residual_PDE(val)
        
        return loss
            
        
##################################################################################
# import time
class PDEFun(torch.autograd.Function):
       
    @staticmethod
    def forward(ctx, input, equ_solver):
        assert input.shape[0] == equ_solver.M.shape[0]
        
        device_input = input.device
        dtype = input.dtype

        m_vec = np.array(input.cpu().detach(), dtype=np.float64)
        ## update the parameter to generate new stiffness matrix 
        equ_solver.update_m(m_vec) 
        nx, ny = equ_solver.domain_equ.nx, equ_solver.domain_equ.nx
        
        if device_input.type == "cpu": 
            K = spnumpy2sptorch(equ_solver.K, dtype=dtype)
            S = torch.tensor(equ_solver.S, dtype=dtype)
            M = spnumpy2sptorch(equ_solver.M, dtype=dtype)
            sol = spsl.spsolve(equ_solver.K, equ_solver.F)
            output = torch.tensor((equ_solver.S@sol).T, dtype=dtype).squeeze()
            sol_forward = torch.tensor(sol, dtype=dtype)
            nx = torch.tensor(nx, dtype=dtype)
            ny = torch.tensor(ny, dtype=dtype)
            ctx.save_for_backward(input, K, S, M, sol_forward, nx, ny)

        elif device_input.type == "cuda":  
            pass
        else:
            raise NotImplementedError("device must be cpu or gpu")
        
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        dtype = grad_output.dtype
        device = grad_output.device
        # init_adjoint_sol = None
        if device.type == "cpu":
            u, K, S, M, sol_forward, nx, ny = ctx.saved_tensors 
            K = sptorch2spnumpy(K)
            S = S.numpy()
            grad_output_numpy = np.array(grad_output.cpu())
            Fs = -S.T@(grad_output_numpy.T)
            # Fs = S.T@(grad_output_numpy.T)
            sol_adjoint = spsl.spsolve(K, Fs)
            # val = torch.tensor(sol, dtype=dtype).squeeze()
            sol_forward = sol_forward.numpy()
            nx, ny = nx.numpy(), ny.numpy()
            u = u.numpy()
            
            ## we need to calculate e^u \grad sol_forward \grad sol_adjoint
            domain = Domain2D(nx=nx, ny=ny, mesh_type='P', mesh_order=1)
            solF = fe.Function(domain.function_space)
            solA = fe.Function(domain.function_space)
            ufun = fe.Function(domain.function_space)
            assert len(ufun.vector()[:]) == len(u)
            assert len(solF.vector()[:]) == len(sol_forward)
            assert len(solA.vector()[:]) == len(sol_adjoint)
            solF.vector()[:] = np.array(sol_forward)
            solA.vector()[:] = np.array(sol_adjoint)
            ufun.vector()[:] = np.array(u)
            v_ = fe.TestFunction(domain.function_space)
            b_ = fe.assemble(
                fe.inner(
                    fe.grad(solA), dl.exp(ufun)*fe.grad(solF)*v_
                    )*fe.dx
                )
            M = sptorch2spnumpy(M)
            val = spsl.spsolve(M, b_[:])
            val = torch.tensor(val, dtype=dtype).squeeze()
            del domain, solF, solA, ufun, v_, b_, M
        elif device.type == "cuda":
            pass
        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return val, None


class PDEasNet(nn.Module):
    def __init__(self, pde_fun, equ_solver):
        super(PDEasNet, self).__init__()
        self.pde_fun = pde_fun
        self.equ_solver = equ_solver

    def update(self, pde_fun, equ_solver):
        self.pde_fun = pde_fun
        self.equ_solver = equ_solver
        
    def forward(self, u):
        output = self.pde_fun(u, self.equ_solver)
        return output


class LossResidual(nn.Module):
    def __init__(self, noise):
        super(LossResidual, self).__init__()
        self.noise = noise
        if self.noise.is_torch == False:
            self.noise.to_torch()

    def update(self, noise):
        self.noise = noise
        if self.noise.is_torch == False:
            self.noise.to_torch()
            
    def cuda(self):
        self.noise.precision = self.noise.precision.cuda()
    
    def to(self, device="cpu"):
        if device == "cuda":
            self.cuda()
        elif device == "cpu":
            pass
        else:
            raise NotImplementedError("device must be cpu or cuda")
        
    def forward(self, predictions, target):
        # self.noise.precision = torch.tensor(self.noise.precision, dtype=predictions.dtype)
        diff = predictions - target
        val = torch.matmul(self.noise.precision, diff)
        loss_val = 0.5*torch.matmul(diff, val)
        return loss_val



## this function calculate M^{-1}S.T d that transfer data to functions
class Dis2Fun(object):
    def __init__(self, domain, points, alpha=0.01):
        self.domain = domain
        u_ = fe.TrialFunction(domain.function_space)
        v_ = fe.TestFunction(domain.function_space)
        A_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.A = trans2spnumpy(A_)
        self.M = trans2spnumpy(M_)
        self.S = construct_measurement_matrix(points, domain.function_space).todense()
        self.alpha = alpha
        self.lu = spsl.splu((self.M + alpha*self.A).tocsc())
        
    def reset_points(self, points):
        self.S = construct_measurement_matrix(points, self.domain.function_space).todense()
    
    def reset_alpha(self, alpha):
        self.alpha = alpha
        self.lu = spsl.splu((self.M + alpha*self.A).tocsc())
    
    def dis2fun(self, d):
        F = ((self.S.T)@d).reshape(-1,1)
        Sy = self.lu.solve(F)
        return np.array(Sy)
    
    def __call__(self, d):
        return self.dis2fun(d)
    
class Dis2FunC(object):
    def __init__(self, domainS, domainF, f, noise_level, dataset_x, dataset_y):
        self.equ_solver = EquSolver(
            domain_equ=domainS, m=fe.Function(domainS.function_space), f=f, 
            points=np.array([[0,0]])
            )
        
        self.noise_level = noise_level
        noise_level_ = self.noise_level*max(abs(dataset_y[0,:]))
        self.noise = NoiseGaussianIID(dim=len(dataset_y[0,:]))
        self.noise.set_parameters(variance=noise_level_**2)
        
        d_noisy = dataset_y[0, :]
        
        self.prior = GaussianFiniteRank(
            domain=domainS, domain_=domainS, alpha=0.1, beta=10, s=2
            )
        self.prior.calculate_eigensystem()
        
        self.model = ModelDarcyFlow(
            d=d_noisy, domain_equ=domainS, prior=self.prior,
            noise=self.noise, equ_solver=self.equ_solver
            )
        
        self.funS = fe.Function(domainS.function_space)
        self.domainF = domainF
        self.newton_cg = None
    
    def reset_points(self, points):
        self.model.update_S(points)
        
    def __call__(self, d, max_iter=2, cg_max=30):
        ## set optimizer NewtonCG
        self.model.update_d(d)
        
        noise_level_ = self.noise_level*max(abs(d))
        if self.newton_cg is None:
            self.model.noise = NoiseGaussianIID(dim=len(d))
        self.model.noise.set_parameters(variance=noise_level_**2)
        
        if self.newton_cg is None:
            self.newton_cg = NewtonCG(model=self.model)
        else:
            self.newton_cg.re_init()
            
        max_iter = max_iter
        
        loss_pre = self.model.loss()[0]
        for itr in range(max_iter):
            self.newton_cg.descent_direction(cg_max=cg_max, method='cg_my')
            # newton_cg.descent_direction(cg_max=30, method='bicgstab')
            # print(newton_cg.hessian_terminate_info)
            self.newton_cg.step(method='armijo', show_step=False)
            if self.newton_cg.converged == False:
                break
            loss = self.model.loss()[0]
            # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-3*loss:
                # print("Iteration stoped at iter = %d" % itr)
                break 
            loss_pre = loss
            
        self.funS.vector()[:] = np.array(self.newton_cg.mk.copy())
        funF = fe.interpolate(self.funS, self.domainF.function_space)
        
        return np.array(funF.vector()[:])
    
class UnitNormalization(object):
    def __init__(self, x, eps=1e-10):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = torch.tensor(eps, dtype=x.dtype)
        
    def encode(self, x):
        x = (x - self.mean)/(self.std + self.eps) 
        return x
    
    def decode(self, x):
        std = self.std + self.eps
        x = (x*std) + self.mean
        return x
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)
        

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, coordinates, mode="non-residual"):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        if mode == "non-residual":
            self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        elif mode == "residual":
            # self.fc0 = nn.Linear(4, self.width) 
            self.fc0 = nn.Linear(3, self.width) 
        else:
            raise ValueError("mode must be non-residual or residual")

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.grid = self.set_grid(coordinates)
        self.mode = mode
        
    def set_grid(self, coordinates):
        gridx = torch.unique(torch.tensor(coordinates[:,0], dtype=torch.float32))
        gridy = torch.unique(torch.tensor(coordinates[:,1], dtype=torch.float32))
        size_x = gridx.shape[0]
        size_y = gridy.shape[0]
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(1, 1, size_x, 1)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(1, size_y, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x, x0=None):
        if self.mode == "residual":
            if x0 is None:
                raise ValueError("model is residual, x0 cannot be None")
        
        batchsize = x.shape[0]
        grid = self.grid.repeat([batchsize, 1, 1, 1]).to(x.device)
        if self.mode == "residual":
            # x = torch.cat((x0, x, grid), dim=-1)
            x = torch.cat((x, grid), dim=-1)
        elif self.mode == "non-residual":
            x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        if self.mode == "non-residual":
            return x
        elif self.mode == "residual":
            # return x0 + x
            return x
        else:
            raise ValueError("mode must be non-residal or residual")
    
    
#####################################################

class SimpleNN(nn.Module):
    def __init__(self, domain, num_hidden_layer=None):
        self.domain = domain
        u_ = fe.TrialFunction(domain.function_space)
        v_ = fe.TestFunction(domain.function_space)
        M_ = fe.assemble(fe.inner(u_, v_)*fe.dx) 
        K_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        M = fe.trans2spnumpy(M_)
        K = fe.trans2spnumpy(K_)
        A = M + 0.1*K
        self.eigval, self.eigvec_ = sl.eigh(self.A.todense(), self.M.todense())
        
        if num_hidden_layer is None:
            num_hidden_layer = len(self.eigval)
        self.num_hidden_layer = num_hidden_layer
        
        # self. = nn.Linear(3, self.width)

def file_process(file_name, contents=None):
    '''
    Delete the old file and create a file with same name
    '''
    # Check if the file exists
    if os.path.exists(file_name):
        # Delete the existing file
        try:
            os.remove(file_name)
            print(f"Deleted existing '{file_name}'.")
        except OSError as e:
            print(f"Error deleting '{file_name}': {e}")

    # Create a new empty file
    try:
        with open(file_name, "w") as f:
            if contents is None:
                pass
            else:
                f.write(contents)
        print(f"Created empty '{file_name}'.")
    except OSError as e:
        print(f"Error creating '{file_name}': {e}")



















