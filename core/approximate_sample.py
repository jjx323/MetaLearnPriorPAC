#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:48:26 2022

"""

import numpy as np
import fenics as fe
import scipy.sparse.linalg as spsl
import scipy.sparse as sps

from core.eigensystem import double_pass
from core.misc import construct_measurement_matrix, smoothing
from core.optimizer import NewtonCG


class LaplaceApproximate(object):
    '''
    Ref: T. Bui-Thanh, O. Ghattas, J. Martin, G. Stadler, 
    A computational framework for infinite-dimensional Bayesian inverse problems
    part I: The linearized case, with application to global seismic inversion,
    SIAM J. Sci. Comput., 2013
    '''
    def __init__(self, model):
        
        assert hasattr(model, "prior") and hasattr(model, "domain_equ")
        assert hasattr(model, "equ_solver") and hasattr(model, "noise")
        assert hasattr(model, "M") and hasattr(model, "S")
        
        self.fun_dim = model.domain_equ.function_space.dim()
        self.prior = model.prior
        self.equ_solver = model.equ_solver
        self.noise = model.noise 
        self.M = sps.csc_matrix(model.M)
        lamped_elements = np.array(np.sum(self.M, axis=1)).flatten()
        self.M_lamped_half = sps.csc_matrix(sps.diags(np.sqrt(lamped_elements)))
        self.Minv_lamped_half = sps.csc_matrix(sps.diags(np.sqrt(1/lamped_elements)))
        self.S = model.S
        
    def set_mean(self, vec):
        self.mean = np.array(vec)
    
    ## linearized_forward_solver is actually the incremental forward solver
    def _linearized_forward_solver(self, m_hat, **kwargs):
        val = self.equ_solver.incremental_forward_solver(m_hat, **kwargs)
        return np.array(val)
    
    ## linearized_adjoint_solver is actually the incremental adjoint solver
    def _linearized_adjoint_solver(self, vec, m_hat, **kwargs):
        val = self.equ_solver.incremental_adjoint_solver(vec, m_hat, **kwargs)
        return np.array(val)
    
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None): 
            vec = self.noise.precision@(vec)
        else:
            vec = spsl.spsolve(self.noise.covariance, vec)
        return np.array(vec)
      
    ## Since the symmetric matrixes are need for computing eigensystem 
    ## F^{*}F = M^{-1} F^T F is not a symmetric matrix, so we multiply M.
    ## The following function actually evaluate M F^{*}F = F^T F
    def eva_Hessian_misfit_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_Hessian_misfit_M)
        return self.linear_ope
    
    def _eva_Hessian_misfit_M(self, vec):
        vec = np.squeeze(vec)
        val = self._linearized_forward_solver(vec)
        val = self._time_noise_precision(self.S@val)
        # print("-------------------------------------------")
        # print(val.shape, vec.shape)
        val = self._linearized_adjoint_solver(val, vec)
        return np.array(self.M@val)
    
    def _eva_prior_var_inv_M(self, vec):
        val = self.prior.K@spsl.spsolve(self.M, self.prior.K@vec)
        return np.array(val)
    
    ## K M^{-1} K as eva_Hessian_misfit_M, we also multiply M
    def eva_prior_var_inv_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_prior_var_inv_M)
        return self.linear_ope
    
    def _eva_prior_var_M(self, vec):
        val = spsl.spsolve(self.prior.K, vec)
        val = spsl.spsolve(self.prior.K, self.M@val)
        return np.array(val)
    
    ## K^{-1} M K^{-1}
    def eva_prior_var_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eva_prior_var_M)
        return self.linear_ope
        
    def calculate_eigensystem(self, num_eigval, method='double_pass', 
                              oversampling_factor=20, cut_val=0.9, **kwargs):
        '''
        Calculate the eigensystem of H_{misfit} v = \lambda \Gamma^{-1} v.
        (\Gamma is the prior covariance operator)
        The related matrixes (H_{misfit} and \Gamma) are not symmetric, 
        however, the standard eigen-system computing algorithm need these matrixes 
        to be symmetric. Hence, we actually compute the following problem:
                M H_{misfit} v = \lambda M \Gamma^{-1} v

        Parameters
        ----------
        num_eigval : int
            calucalte the first num_eigval number of large eigenvalues
        method : str, optional
            double_pass and scipy_eigsh can be choien. The default is 'double_pass'.
        oversampling_factor : int, optional
            To ensure an accurate calculation of the required eigenvalues. The default is 20.
        **kwargs : TYPE
            Depends on which method is employed.

        Returns
        -------
        None.
        
        The computated eignvalue will be in a descending order.
        '''
    
        if method == 'double_pass':
            Hessian_misfit = self._eva_Hessian_misfit_M
            prior_var_inv = self._eva_prior_var_inv_M
            prior_var = self._eva_prior_var_M
            # rs = num_eigval + oversampling_factor
            # omega = np.random.randn(self.M.shape[0], rs)
            self.eigval, self.eigvec = double_pass(
                Hessian_misfit, M=prior_var_inv, Minv=prior_var, r=num_eigval, 
                l=oversampling_factor, n=self.M.shape[0]
                )
            # print(self.eigval)
            index = self.eigval > cut_val
            self.eigval = self.eigval[index]
            self.eigvec = self.eigvec[:, index]
        elif method == 'single_pass':
            raise NotImplementedError
        elif method == 'scipy_eigsh':
            ## The eigsh function in scipy seems much slower than the "double_pass" and 
            ## "single_pass" algorithms implemented in the package. The eigsh function 
            ## is kept as a baseline for testing our implementations. 
            Hessian_misfit = self.eva_Hessian_misfit_M()
            prior_var_inv = self.eva_prior_var_inv_M()
            prior_var = self.eva_prior_var_M()
            self.eigval, self.eigvec = spsl.eigsh(
                Hessian_misfit, M=prior_var_inv, k=num_eigval+oversampling_factor, which='LM', 
                Minv=prior_var, **kwargs
                )  #  optional parameters: maxiter=maxiter, v0=v0 (initial)
            index = self.eigval > cut_val
            if np.sum(index) == num_eigval:
                print("Warring! The eigensystem may be inaccurate!")
            self.eigval = np.flip(self.eigval[index])
            self.eigvec = np.flip(self.eigvec[:, index], axis=1)
        else:
            assert False, "method should be double_pass, scipy_eigsh"
        
        ## In the above, we actually solve the H v1 = \lambda \Gamma^{-1} v1
        ## However, we actually need to solve \Gamma^{1/2} H \Gamma^{1/2} v = \lambda v
        ## Notice that v1 \neq v, we have v = \Gamma^{-1/2} v
        self.eigvec = spsl.spsolve(self.M, self.prior.K@self.eigvec)
        
    def posterior_var_times_vec(self, vec):
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = spsl.spsolve(self.prior.K, self.M@vec)
        val2 = self.eigvec@Dr@self.eigvec.T@self.M@val1
        val = self.M@(val1 - val2)
        val = spsl.spsolve(self.prior.K, val)
        return np.array(val)        
        
    def generate_sample(self):
        assert hasattr(self, "mean") and hasattr(self, "eigval") and hasattr(self, "eigvec")
        n = np.random.normal(0, 1, (self.fun_dim,))
        val1 = self.Minv_lamped_half@n
        pr = 1.0/np.sqrt(self.eigval+1.0) - 1.0
        Pr = sps.csc_matrix(sps.diags(pr))
        val2 = self.eigvec@Pr@self.eigvec.T@self.M@val1
        val = self.M@(val1 + val2)
        val = spsl.spsolve(self.prior.K, val)
        val = self.mean + val
        return np.array(val)
        
    def pointwise_variance_field(self, xx, yy):
        '''
        Calculate the pointwise variance field of the posterior measure. 
        Through a simple calculations, we can find the following formula:
            c_h(xx, yy) = \Phi(xx)^T[K^{-1}MK^{-1]} - K^{-1}MV_r D_r V_r^T M K^{-1}]\Phi(yy),
        which is actually the same as the formula (5.7) in the following paper: 
            A computational framework for infinite-dimensional Bayesian inverse problems
            part I: The linearized case, with application to global seismic inversion,
            SIAM J. Sci. Comput., 2013

        Parameters
        ----------
        xx : list
            [(x_1,y_1), \cdots, (x_N, y_N)]
        yy : list
            [(x_1,y_1), \cdots, (x_M, y_M)]

        Returns
        -------
        None.

        '''
        assert hasattr(self, "eigval") and hasattr(self, "eigvec")
        
        SN = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = SM.T
        
        val = spsl.spsolve(self.prior.K, SM)
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = self.eigvec@Dr@self.eigvec.T@self.M@val
        val = self.M@(val - val1)
        val = spsl.spsolve(self.prior.K, val)
        val = SN@val     
        
        if type(val) == type(self.M):
            val = val.todense()
        
        return np.array(val)












