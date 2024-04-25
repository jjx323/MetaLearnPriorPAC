#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:50 2019

@author: jjx323
"""
import numpy as np
from scipy.special import gamma
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import dolfin as dl 
import cupy as cp
import cupyx.scipy.sparse as cpss
import cupyx.scipy.sparse.linalg as cpssl

import sys, os
sys.path.append(os.pardir)
from core.probability import GaussianElliptic2
from core.model import ModelBase
from core.misc import my_project, trans2spnumpy, \
                      construct_measurement_matrix, make_symmetrize

    
###########################################################################
class EquSolver(object):
    def __init__(self, domain_equ, f, m, points):
        self.domain_equ = domain_equ
        self.V_equ = self.domain_equ.function_space
        self.mm = fe.interpolate(m, self.V_equ)
        self.exp_m = fe.Function(self.V_equ)
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        self.f = fe.interpolate(f, self.V_equ)
        self.points = points
        
        self.u_, self.v_ = fe.TrialFunction(self.V_equ), fe.TestFunction(self.V_equ)
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.F_ = fe.assemble(self.f*self.v_*fe.dx)
        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V_equ, fe.Constant('0.0'), boundary)
        self.bc.apply(self.K_)
        self.bc.apply(self.F_)
        
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        self.F = self.F_[:]
        
        self.S = np.array(construct_measurement_matrix(points, self.V_equ).todense())
        
        ## All of the program did not highly rely on FEniCS, 
        ## so the following FEniCS function will be treated only as helpping function
        self.sol_forward = fe.Function(self.V_equ)
        self.sol_adjoint = fe.Function(self.V_equ)
        self.sol_incremental = fe.Function(self.V_equ)
        self.sol_incremental_adjoint = fe.Function(self.V_equ)
        self.Fs = fe.Function(self.V_equ)
        self.m_hat = fe.Function(self.V_equ)
        
        ## All of the solutions will be treated as the solution interact with 
        ## other program
        self.sol_forward_vec = self.sol_forward.vector()[:]
        self.sol_adjoint_vec = self.sol_adjoint.vector()[:]
        self.sol_incremental_vec = self.sol_incremental.vector()[:]
        self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint.vector()[:]
        
        self.is_cuda = False
        self.init_forward_sol, self.init_adjoint_sol = None, None
    
    def update_m(self, m_vec=None):
        if m_vec is None:
            self.mm.vector()[:] = 0.0
        else:
            self.mm.vector()[:] = np.array(m_vec)
        self.exp_m = fe.Function(self.V_equ)
        # self.exp_m.vector()[:] = fe.project(dl.exp(self.mm), self.V_equ).vector()[:]
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.bc.apply(self.K_)
        self.K = trans2spnumpy(self.K_)
        if type(self.F) == cp.ndarray:
            self.K = cpss.csr_matrix(self.K)
        
    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(self.points, self.domain_equ.function_space)
        self.S = np.array(self.S.todense())
        if type(self.F) == cp.ndarray:
            self.S = cp.asarray(self.S)
        
    def get_data(self):
        if type(self.F) == np.ndarray:
            val = self.S@self.sol_forward.vector()[:]
            return np.array(val) 
        elif type(self.F) == cp.ndarray:
            val = self.S@self.sol_forward_vec 
            return val 
    
    def to(self, device="cpu"):
        if device == "cpu":
            self.K, self.F = self.K.get(), self.F.get()
            self.S = self.S.get()
            self.sol_forward_vec = self.sol_forward_vec.get()
            self.sol_adjoint_vec = self.sol_adjoint_vec.get()
            self.sol_incremental_vec = self.sol_incremental_vec.get()
            self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint_vec.get()
            self.is_cuda = False
        elif device == "cuda":
            self.K = cpss.csr_matrix(self.K)
            self.S = cp.asarray(self.S)
            self.F = cp.asarray(self.F)
            self.sol_forward_vec = cp.asarray(self.sol_forward_vec) 
            self.sol_adjoint_vec = cp.asarray(self.sol_adjoint_vec) 
            self.sol_incremental_vec = cp.asarray(self.sol_incremental_vec)
            self.sol_incremental_adjoint_vec = cp.asarray(self.sol_incremental_adjoint_vec)
            self.is_cuda = True 
        else:
            raise NotImplementedError("device must be cpu or cuda")

    def forward_solver(self, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                fe.solve(self.K_, self.sol_forward.vector(), self.F_)
                self.sol_forward_vec = np.array(self.sol_forward.vector()[:])
            elif method == 'numpy':
                self.sol_forward_vec = spsl.spsolve(self.K, self.F)
                # self.sol_forward_vec = spsl.gmres(self.K, self.F, tol=1e-3)[0]
                self.sol_forward_vec = np.array(self.sol_forward_vec)
        elif type(self.F) == cp.ndarray:
            self.sol_forward_vec = cpssl.gmres(self.K, self.F, tol=1e-3)[0]
            # self.sol_forward_vec = cpssl.spsolve(self.K, self.F)
            # self.sol_forward_vec = cpssl.minres(self.K, self.F)[0]
        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_forward_vec
        
    def adjoint_solver(self, vec, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
            
        Fs = -self.S.T@vec
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_adjoint.vector(), self.Fs.vector())
                self.sol_adjoint_vec = np.array(self.sol_adjoint.vector()[:])
            elif method == 'numpy':
                self.sol_adjoint_vec = np.array(spsl.spsolve(self.K, Fs))
        elif type(self.F) == cp.ndarray:
            Fs = cp.asarray(Fs)
            self.sol_adjoint_vec = cpssl.gmres(self.K, Fs, tol=1e-3)[0]
            # self.sol_adjoint_vec = cpssl.spsolve(self.K, Fs)
        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_adjoint_vec
  
    def incremental_forward_solver(self, m_hat, sol_forward=None, method='numpy'):
        if type(sol_forward) == type(None):
            self.sol_forward.vector()[:] = self.sol_forward_vec 
        
        if type(m_hat) == np.ndarray:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                b_ = -fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_forward), fe.grad(self.v_))*fe.dx)
                self.bc.apply(b_)
                fe.solve(self.K_, self.sol_incremental.vector(), b_)
                self.sol_incremental_vec = np.array(self.sol_incremental.vector()[:])
            elif method == 'numpy':
                b_ = fe.inner(self.exp_m*fe.grad(self.sol_forward)*self.u_, fe.grad(self.v_))*fe.dx
                b_ = fe.assemble(b_)   
                b_spnumpy = trans2spnumpy(b_)
                b = b_spnumpy@m_hat
                self.sol_incremental_vec = np.array(spsl.spsolve(self.K, -b))
        elif type(m_hat) == cp.ndarray:
            b_ = fe.inner(self.exp_m*fe.grad(self.sol_forward)*self.u_, fe.grad(self.v_))*fe.dx
            b_ = fe.assemble(b_)   
            b_spnumpy = trans2spnumpy(b_)
            b_spnumpy = cp.asarray(b_spnumpy)
            b = b_spnumpy@m_hat
            self.sol_incremental_vec = cpssl.gmres(self.K, -b, tol=1e-3)[0]
        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_incremental_vec     
        
    def incremental_adjoint_solver(self, vec, m_hat, sol_adjoint=None, simple=False, method='numpy'):
        if type(sol_adjoint) == type(None):
            self.sol_adjoint.vector()[:] = self.sol_adjoint_vec 
        
        Fs = -self.S.T@vec
        Fs = Fs.squeeze()
        if simple == False:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                bl_ = fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_adjoint), fe.grad(self.v_))*fe.dx)
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), -bl_ + self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                bl_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.sol_adjoint)*self.u_, fe.grad(self.v_))*fe.dx)
                bl_spnumpy = trans2spnumpy(bl_)
                if type(m_hat) == cp.ndarray:
                    bl_spnumpy = cp.asarray(bl_spnumpy)
                    bl = bl_spnumpy@m_hat
                    self.sol_incremental_adjoint_vec = cpssl.gmres(self.K, -bl+Fs, tol=1e-3)[0]
                elif type(m_hat) == np.ndarray:
                    bl = bl_spnumpy@m_hat
                    # print(bl.shape, Fs.shape)
                    self.sol_incremental_adjoint_vec = spsl.spsolve(self.K, -bl + Fs)
                else:
                    raise NotImplementedError("device must be cpu or cuda")
        elif simple == True:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                if type(vec) == np.ndarray:
                    val = spsl.spsolve(self.K, Fs)
                    self.sol_incremental_adjoint_vec = np.array(val)
                elif type(vec) == cp.ndarray:
                    self.sol_incremental_adjoint_vec = cpssl.gmres(self.K, Fs, tol=1e-3)[0]
                else:
                    raise NotImplementedError("device must be cpu or cuda")
                
        return self.sol_incremental_adjoint_vec

        
###########################################################################
class ModelDarcyFlow(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
        self.p = fe.Function(self.equ_solver.domain_equ.function_space)
        self.q = fe.Function(self.equ_solver.domain_equ.function_space)
        self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
        self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
        self.u_ = fe.TrialFunction(self.domain_equ.function_space)
        self.v_ = fe.TestFunction(self.domain_equ.function_space)
        self.m_hat = fe.Function(self.domain_equ.function_space)
        self.m = self.equ_solver.mm
        self.loss_residual_now = 0

    def update_m(self, m_vec, update_sol=True):
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
    
    def updata_d(self, d):
        self.d = d
        
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None):
            val = self.noise.precision@vec
        else:
            val = spsl.spsolve(self.noise.covariance, vec)
        return np.array(val)
        
    def loss_residual(self):
        temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()
        temp = (temp - self.noise.mean - self.d)
        if type(self.noise.precision) != type(None): 
            temp = temp@(self.noise.precision)@temp
        else:
            temp = temp@(spsl.spsolve(self.noise.covariance, temp))
        self.loss_residual_now = 0.5*temp
        return self.loss_residual_now
    
    def loss_residual_L2(self):
        temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def eval_grad_residual(self, m_vec):
        self.update_m(m_vec, update_sol=False)
        self.equ_solver.forward_solver()
        vec = np.array(self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d)
        vec = self._time_noise_precision(vec.squeeze()) 
        self.equ_solver.adjoint_solver(vec)
        self.p.vector()[:] = self.equ_solver.sol_forward_vec
        self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
        b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
        return spsl.spsolve(self.equ_solver.M, b_[:])
        
    def eval_hessian_res_vec(self, m_hat_vec):
        # self.m_hat.vector()[:] = m_hat_vec
        self.equ_solver.incremental_forward_solver(m_hat_vec)
        vec = np.array(self.S@self.equ_solver.sol_incremental_vec)
        vec = self._time_noise_precision(vec.squeeze())        
        self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
        self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
        self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
        A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
                                  fe.grad(self.q))*fe.dx)
        A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
                                  fe.grad(self.qq))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
        A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
                                  fe.grad(self.pp))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
        A = A1[:] + A2[:] + A3[:]
        
        return spsl.spsolve(self.equ_solver.M, A)

# ###########################################################################
# class ModelNoPrior(ModelBase):
#     def __init__(self, d, domain_equ, noise, equ_solver, prior=None):
#         super().__init__(d, domain_equ, prior, noise, equ_solver)
#         self.p = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.q = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.u_ = fe.TrialFunction(self.domain_equ.function_space)
#         self.v_ = fe.TestFunction(self.domain_equ.function_space)
#         self.m_hat = fe.Function(self.domain_equ.function_space)
#         self.m = self.equ_solver.mm
#         self.loss_residual_now = 0

#     def update_m(self, m_vec, update_sol=True):
#         self.equ_solver.update_m(m_vec)
#         if update_sol == True:
#             self.equ_solver.forward_solver()
    
#     def updata_d(self, d):
#         self.d = d
        
#     def _time_noise_precision(self, vec):
#         if type(self.noise.precision) != type(None):
#             val = self.noise.precision@vec
#         else:
#             val = spsl.spsolve(self.noise.covariance, vec)
#         return np.array(val)
        
#     def loss_residual(self):
#         temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()
#         temp = (temp - self.noise.mean - self.d)
#         if type(self.noise.precision) != type(None): 
#             temp = temp@(self.noise.precision)@temp
#         else:
#             temp = temp@(spsl.spsolve(self.noise.covariance, temp))
#         self.loss_residual_now = 0.5*temp
#         return self.loss_residual_now
    
#     def loss_residual_L2(self):
#         temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
#         temp = temp@temp
#         return 0.5*temp
    
#     def eval_grad_residual(self, m_vec):
#         self.update_m(m_vec, update_sol=False)
#         self.equ_solver.forward_solver()
#         vec = self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d
#         vec = self._time_noise_precision(vec) 
#         self.equ_solver.adjoint_solver(vec)
#         self.p.vector()[:] = self.equ_solver.sol_forward_vec
#         self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
#         b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
#         return spsl.spsolve(self.equ_solver.M, b_[:])
        
#     def eval_hessian_res_vec(self, m_hat_vec):
#         # self.m_hat.vector()[:] = m_hat_vec
#         self.equ_solver.incremental_forward_solver(m_hat_vec)
#         vec = self.S@self.equ_solver.sol_incremental_vec
#         vec = self._time_noise_precision(vec)        
#         self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
#         self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
#         self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
#         A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
#                                   fe.grad(self.q))*fe.dx)
#         A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
#                                   fe.grad(self.qq))*fe.dx)
#                          # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
#         A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
#                                   fe.grad(self.pp))*fe.dx)
#                          # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
#         A = A1[:] + A2[:] + A3[:]
        
#         return spsl.spsolve(self.equ_solver.M, A)
    
#     def loss(self):
#         loss_res = self.loss_residual()
#         loss_prior = 0.0
#         return loss_res + loss_prior, loss_res, loss_prior

#     def gradient(self, m_vec):
#         grad_res = self.eval_grad_residual(m_vec)
#         grad_prior = 0.0
#         return grad_res + grad_prior, grad_res, grad_prior

#     def hessian(self, m_vec):
#         hessian_res = self.eval_hessian_res_vec(m_vec)
#         hessian_prior = 0.0
#         return hessian_res + hessian_prior


            







