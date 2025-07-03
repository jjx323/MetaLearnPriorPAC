## import necessary packages
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from collections import OrderedDict
import fenics as fe
import torch
import torch.nn as nn
import torch.optim as optim
# from torch_sparse_solve import solve as torchspsolve

## Add path to the parent directory
import sys, os
sys.path.append(os.pardir)

## Import necessary modules in our programs
from core.misc import trans2spnumpy, spnumpy2sptorch, trans2sptorch, sptorch2spnumpy, \
    construct_measurement_matrix
from core.eigensystem import double_pass
from core.probability import GaussianElliptic2



class Gaussian1DFiniteDifference(object):
    def __init__(self, nx, a_fun=1.0, alpha=1.0, mean_fun=None):
        if mean_fun is None:
            self.mean_fun = np.zeros(nx)
        self.nx = nx
        if type(a_fun) == float or type(a_fun) == int:
            self.a_fun = a_fun*np.ones(nx)
        self.alpha = alpha 
        self.construct_matrix()
        self.is_tensor = False
        
    def construct_matrix(self):
        rows = np.arange(self.nx)
        cols = rows
        self.aI = sps.coo_matrix((self.a_fun, (rows, cols)), shape=[self.nx, self.nx]) 
        self.aI.tocsr()
        t1 = sps.coo_matrix((2*np.ones(self.nx), (rows, cols)), shape=[self.nx, self.nx]) 
        rows = np.arange(self.nx-1)
        cols = np.arange(self.nx-1) + 1
        val = -1*np.ones(self.nx-1)
        t2 = sps.coo_matrix((val, (rows, cols)), shape=[self.nx, self.nx])
        rows = np.arange(self.nx-1) + 1
        cols = np.arange(self.nx-1) 
        t3 = sps.coo_matrix((val, (rows, cols)), shape=[self.nx, self.nx])
        t1.tocsr()
        t2.tocsr()
        t3.tocsr()
        self.Delta = self.alpha*(t1 + t2 + t3)*(self.nx*self.nx)
        
    def trans2learnable(self, learn_mean=False, learn_a=False):
        self.a_fun_learn = torch.tensor(
            np.log(self.a_fun), dtype=torch.float64, requires_grad=learn_a
            )
        self.aI_torch = spnumpy2sptorch(self.aI, dtype=torch.float64)
        self.Delta_torch = spnumpy2sptorch(self.Delta, dtype=torch.float64)
        self.mean_fun_learn = torch.tensor(
            self.mean_fun, dtype=torch.float64, requires_grad=learn_mean
            )
        self.is_tensor = True
        
    def trans2numpy(self):
        if self.is_tensor == True:
            self.mean_fun = np.array(self.mean_fun.cpu().detach().numpy())
            self.a_fun = np.array(torch.exp(self.a_fun_learn).cpu().detach().numpy())
            self.aI = sptorch2spnumpy(self.aI_torch)
            self.Delta = sptorch2spnumpy(self.Delta_torch)                
            self.is_tensor = False
        
    def generate_sample_zero_mean(self, KK):
        n = np.random.normal(0, 1, (self.nx,))
        sample = spsl.spsolve(KK, n)
        return sample
    
    def generate_sample(self):
        if self.is_tensor == False:
            KK = self.aI + self.Delta
            sample_ = self.generate_sample_zero_mean(KK)
            sample = self.mean_fun + sample_
            return sample 
        elif self.is_tensor == True:
            row_idx = torch.arange(self.nx+1)
            row_idx[-1] = self.nx
            col_idx = torch.arange(self.nx)
            val = torch.exp(self.a_fun_learn)
            self.aI_torch = torch.sparse_csr_tensor(
                row_idx, col_idx, val, dtype=torch.float32
                )
            KK = self.aI_torch + self.Delta_torch
            sample_ = self.generate_sample_zero_mean(sptorch2spnumpy(KK))
            sample_ = torch.tensor(sample_, dtype=torch.float32, requires_grad=False)
            sample = self.mean_fun_learn + sample_ 
            return sample
    
    def generate_sample_learn(self):
        return self.generate_sample()
    
    def evaluate_CM_inner(self, u_vec, v_vec=None):
        if v_vec is None:
            v_vec = u_vec
        if self.is_tensor == False:
            temp1 = u_vec - self.mean_fun
            temp2 = v_vec - self.mean_fun
            KK = self.aI + self.Delta 
            out = KK@(KK@temp1)
            out = KK.T@(KK.T@out)
            out = temp2@out 
            return out
        elif self.is_tensor == True:
            temp1 = u_vec - self.mean_fun_learn
            temp2 = v_vec - self.mean_fun_learn
            KK = self.aI_torch + self.Delta_torch
            out = KK@(KK@temp1)
            out = KK@(KK@out)
            out = temp2@out             
            return out


class GaussianElliptic2Learn(GaussianElliptic2):
    def __init__(self, domain, alpha=1.0, a_fun=fe.Constant(1.0), theta=1.0, 
                 mean_fun=None, tensor=False, boundary='Neumann'):
        super().__init__(domain=domain, alpha=alpha, a_fun=a_fun, theta=theta, 
                     mean_fun=mean_fun, tensor=tensor, boundary=boundary, use_LU=False)
        self.log_gamma = np.array(0.0)
        
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        aa1 = fe.Constant(self._alpha)*fe.inner(self._theta*fe.grad(u), fe.grad(v))*fe.dx 
        aa2 = fe.inner(self._a_fun*u, v)*fe.dx
        self.K1_ = fe.assemble(aa1)
        self.K2_ = fe.assemble(aa2)
        self.K1 = trans2spnumpy(self.K1_)
        self.K2 = trans2spnumpy(self.K2_)        
        
    def trans2learnable(self, learn_mean=True, learn_var=True):
        self.mean_vec_learn = torch.tensor(
            self.mean_fun.vector()[:], dtype=torch.float64, requires_grad=learn_mean 
            )
        self.K_torch = spnumpy2sptorch(self.K, dtype=torch.float64)
        self.KT_torch = spnumpy2sptorch(self.K.T, dtype=torch.float64)
        self.K1_torch = spnumpy2sptorch(self.K1, dtype=torch.float64)
        self.K2_torch = spnumpy2sptorch(self.K2, dtype=torch.float64)
        # self.a_fun_learn = torch.tensor(
        #     self._a_fun.vector()[:], dtype=torch.float32, requires_grad=learn_var
        #     )
        self.log_gamma_learn = torch.tensor(0.0, dtype=torch.float64, requires_grad=learn_var)
        self.M_torch = spnumpy2sptorch(self.M, dtype=torch.float64)
        lamped_elements = np.array(np.sum(self.M, axis=1)).flatten()
        self.Minv_torch = torch.tensor(
            1.0/lamped_elements, dtype=torch.float64, requires_grad=False
            )
        self.M_lamped_half_torch = spnumpy2sptorch(self.M_lamped_half, dtype=torch.float64)
        
        self.K1dense = self.K1_torch.to_dense()
        self.K2dense = self.K2_torch.to_dense()
    
    def trans2numpy(self):
        self.mean_vec = np.array(self.mean_vec_learn.cpu().detach().numpy())
        self.log_gamma = self.log_gamma_learn.cpu().detach().numpy()
        KK = self.K1_torch + torch.exp(self.log_gamma_learn)*self.K2_torch
        self.K = sptorch2spnumpy(KK, dtype=np.float64)

    def set_log_gamma(self, new_log_gamma):
        self.log_gamma_learn = torch.tensor(
            new_log_gamma, dtype=torch.float64, requires_grad=True
            )
        
    def generate_sample_zero_mean(self, gamma=1.0, method='numpy'):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = 0.0 + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        assert self.K is not None 
        assert self.M_lamped_half is not None
        
        KK = self.K1dense + gamma*self.K2dense
        # KK = gamma*self.K1dense + self.K2dense
        n = np.random.normal(0, 1, (self.function_space_dim,))
        n = torch.tensor(n, dtype=torch.float64)
        if self.bc == "Dirichlet":
            n[self.index_boundary] = 0.0
        b = self.M_lamped_half_torch@n
        fun_vec = torch.linalg.solve(KK, b)
        return fun_vec
        
    def generate_sample_learn(self, method='numpy'):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = m_{mean} + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        method == 'FEniCS' or 'numpy'
        '''
        
        gamma = torch.exp(self.log_gamma_learn)
        sample_ = self.generate_sample_zero_mean(gamma=gamma, method=method)
        sample = self.mean_vec_learn + sample_
        return sample
    
    def evaluate_CM_inner(self, u_vec, v_vec=None):
        if v_vec is None:
            v_vec = u_vec
        temp1 = u_vec - self.mean_vec_learn
        temp2 = v_vec - self.mean_vec_learn
        KK = self.K1dense + torch.exp(self.log_gamma_learn)*self.K2dense
        # KK = torch.exp(self.log_gamma_learn)*self.K1dense + self.K2dense
        out = self.Minv_torch*torch.mv(KK, temp1)
        # out = torch.mv(self.K_torch, temp1)
        out = torch.mv(KK.T, out)
        out = torch.sum(temp2*out)
            
        return out
        

class GaussianFiniteRank(object):
    '''
    [1] F. J. Pinski, G. Simpson, A. M. Stuart, H. Weber, 
    Algorithms for Kullback-Leibler approximation of probability measures in 
    infinite dimensions, SIAM J. Sci. Comput., 2015.
    
    [2] T. Bau-Thanh, Q. P. Nguyen, FEM-based discretization-invariant MCMC methods
    for PDE-constrained Bayesian inverse problems, Inverse Problems & Imaging, 2016
    
    Base Gaussian \mathcal{C}_0 = [\alpha (\beta I - \Delta)]^{-s}
    Typical example s = 2
    \mathcal{C}_0 v = \lambda^2 v
    
    Due to the calculations of eigendecompositions, this method may be not suitable
    for large-scale problems. A possible idea: projecting large-scale problems 
    to rough grid and calculate the eigendecompositions. 
    
    domain: the original fine grid
    domain_: the approximate rough grid used to evaluate the eigendecomposition
    !!! Only P1 elements can be employed!!!
    '''
    def __init__(self, domain, domain_=None, mean=None, num_gamma=50, num_KL=None, 
                 alpha=1.0, beta=1.0, s=2, boundary="NeumannZero"):
        self.num_gamma = num_gamma ## first num_gamma eigenvalues may be updated
        if domain_ is None:
            domain_ = domain
        self.domain = domain
        self.domain_ = domain_
        assert boundary == "NeumannZero" or boundary == "DirichletZero"
        self.boundary = boundary
        if boundary == "DirichletZero":
            def boundary_fun(x, on_boundary):
                return on_boundary 
            self.bc = fe.DirichletBC(
                self.domain.function_space, fe.Constant("0.0"), boundary_fun
                )
            self.bc_ = fe.DirichletBC(
                self.domain_.function_space, fe.Constant("0.0"), boundary_fun
                ) 
                
        u_, v_ = fe.TrialFunction(domain.function_space), fe.TestFunction(domain.function_space) 
        M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        if boundary == "DirichletZero":
            M0_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
            self.bc.apply(M0_)
            self.M0 = trans2spnumpy(M0_)
        self.M = trans2spnumpy(M_)
        self.dim_full = self.M.shape[0]
        u_, v_ = fe.TrialFunction(domain_.function_space), fe.TestFunction(domain_.function_space)
        Ms_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        if boundary == "DirichletZero":
            Ms0_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
            self.bc_.apply(Ms0_)
            self.Ms0 = trans2spnumpy(Ms0_)
        self.Ms = trans2spnumpy(Ms_)
        Delta_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        if boundary == "DirichletZero":
            Delta0_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
            self.bc_.apply(Delta0_)
            self.Delta0 = trans2spnumpy(Delta0_) 
        self.Delta = trans2spnumpy(Delta_) 
        self.dim = self.Ms.shape[0]
        if num_KL is None: num_KL = self.dim 
        self.num_KL = num_KL
        self.s = s
        self.K_org = alpha*(self.Delta + beta*self.Ms)
        if boundary == "DirichletZero":
            self.K_org0 = alpha*(self.Delta0 + beta*self.Ms0)    
        if mean is None:
            self.mean_vec = np.zeros(self.dim_full)
        else:
            self.mean_vec = fe.interpolate(mean, self.domain.function_space).vector()[:]
        
        ## help function
        self.fun = fe.Function(self.domain.function_space)
        self.fun_ = fe.Function(self.domain_.function_space)
        self.is_eig_available = False
        self.is_tensor = False
        
        ## construct interpolation matrix
        coor = self.domain_.mesh.coordinates()
        # v2d = fe.vertex_to_dof_map(self.domain_.function_space)
        d2v = fe.dof_to_vertex_map(self.domain_.function_space)
        ## full to small matrix
        self.f2sM = construct_measurement_matrix(coor[d2v], self.domain.function_space)
        
        coor = self.domain.mesh.coordinates()
        # v2d = fe.vertex_to_dof_map(self.domain.function_space)
        d2v = fe.dof_to_vertex_map(self.domain.function_space)
        ## small to full matrix
        self.s2fM = construct_measurement_matrix(coor[d2v], self.domain_.function_space)

    def _K_org_inv_x(self, x):
        return np.array(spsl.spsolve(self.K_org, x)) 
    
    def _K_org_x(self, x):
        return np.array(self.K_org@x)
    
    def _M_x(self, x):
        return np.array(self.Ms@x)
    
    def _Minv_x(self, x):
        return np.array(spsl.spsolve(self.Ms, x))
    
    def _K_org_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._K_org_x)
        return linear_op
    
    def _K_org_inv_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._K_org_x) 
        return linear_op 
        
    def _M_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._M_x)
        return linear_op 
    
    def _Minv_x_op(self):
        linear_op = spsl.LinearOperator((self.dim, self.dim), matvec=self._Minv_x)
        return linear_op
    
    def calculate_eigensystem(self, l=20):
        ## calculat the eigensystem of I-\Delta, i.e., M^{-1}K_org V = sigma V
        ## Since the eigensystem calculator usually need the input to be symmetric
        ## matrix, we solve K_org V = sigma M V instead. 
        ## If self.num_gamma not reach self.dim, set l=20 to make sure an explicit
        ## calculations of the eigensystem. 
        
        assert self.num_KL <= self.dim

        if self.boundary == "NeumannZero": 
            self.sigma, self.eigvec_ = sl.eigh(self.K_org.todense(), self.Ms.todense())
            self.lam = np.power(self.sigma, -self.s/2)
            self.log_gamma = np.log(self.lam[:self.num_gamma])
            self.lam_res = self.lam[self.num_gamma:]
        elif self.boundary == "DirichletZero":
            temp0 = sl.inv(self.Ms0.todense())
            temp = temp0@self.K_org0.todense()
            self.sigma, self.eigvec_ = sl.eig(temp)
            self.sigma = np.real(self.sigma)[:-2]
            nn = np.sqrt((self.eigvec_.T@(self.Ms0@self.eigvec_)).diagonal())
            self.eigvec_ = (self.eigvec_/nn)[:,:-2]
            self.num_gamma = self.num_gamma - 2
            self.lam = np.power(self.sigma, -self.s/2)
            self.log_gamma = np.log(self.lam[:self.num_gamma])
            self.lam_res = self.lam[self.num_gamma:]

        self.eigvec = self.s2fM@self.eigvec_
        
        self.num_KL = len(self.lam)
        self.is_eig_available = True
        
    def update_mean_fun(self, mean_vec):
        if self.is_tensor == True:
            if type(mean_vec) == torch.Tensor:
                self.mean_vec = mean_vec
            else:
                self.mean_vec = torch.tensor(mean_vec, dtype=torch.float32)
        if self.is_tensor == False:
            if type(mean_vec) == torch.Tensor:
                self.mean_vec = np.array(mean_vec.cpu().detach().numpy())
            elif type(mean_vec) == np.ndarray:
                self.mean_vec = mean_vec
        
    def generate_sample_zero_mean(self):
        assert self.is_eig_available == True
        if self.is_tensor == True:
            n = torch.normal(0, 1, (self.num_KL,))
            if self.num_gamma < self.num_KL:
                val = torch.cat([torch.exp(self.log_gamma_learn)*n[:self.num_gamma], 
                                self.lam_res*n[self.num_gamma:]])
            else:
                val = torch.exp(self.log_gamma_learn)*n
            val = torch.mv(self.eigvec, val)
            return val
        else:
            n = np.random.normal(0, 1, (len(self.lam),))
            if self.num_gamma < self.num_KL:
                val = np.concatenate((np.exp(self.log_gamma)*n[:self.num_gamma], 
                                     self.lam_res*n[self.num_gamma:]))
            else:
                val = np.exp(self.log_gamma)*n
            val = self.eigvec@val
            return np.array(val)
    
    def generate_sample(self):
        assert self.is_eig_available == True
        assert self.is_tensor == False
        
        val = self.mean_vec + self.generate_sample_zero_mean() 
        return np.array(val)
        
    def generate_sample_learn(self):
        assert self.is_eig_available == True
        assert self.is_tensor == True
        
        val = self.mean_vec_learn + self.generate_sample_zero_mean()
        return val
    
    def reset_lam(self):
        assert self.is_eig_available == True
        ## gamma is the first num_gamma eigenvalue of self.lam 
        assert len(self.log_gamma) == self.num_gamma 
        if self.is_tensor == True:
            self.lam[:self.num_gamma] = torch.exp(self.log_gamma)
        else:
            self.lam[:self.num_gamma] = np.array(np.exp(self.log_gamma))
        
    def trans2learnable(self, mean=True, log_gamma=False):
        ## trans mean or log_gamma into torch.tensor that are optimized by meta-learning
        if mean == True:
            self.mean_vec_learn = torch.tensor(self.mean_vec, dtype=torch.float32, requires_grad=True)
        else:
            self.mean_vec_learn = torch.tensor(self.mean_vec, dtype=torch.float32, requires_grad=False)
        
        if log_gamma == True:
            self.log_gamma_learn = torch.tensor(self.log_gamma, dtype=torch.float32, \
                                          requires_grad=True)
        else:
            self.log_gamma_learn = torch.tensor(self.log_gamma, dtype=torch.float32, \
                                          requires_grad=False)
        
        self.lam_res = torch.tensor(self.lam_res, dtype=torch.float32, requires_grad=False)
        self.lam = torch.tensor(self.lam, dtype=torch.float32, requires_grad=False)
        self.eigvec = torch.tensor(self.eigvec, dtype=torch.float32, requires_grad=False)
        self.eigvec_ = torch.tensor(self.eigvec_, dtype=torch.float32, requires_grad=False)
        if type(self.Ms) == np.ndarray:
            self.Ms = torch.tensor(self.Ms, dtype=torch.float32, requires_grad=False) 
        else:
            self.Ms = torch.tensor(self.Ms.todense(), dtype=torch.float32, requires_grad=False) 
        self.M = spnumpy2sptorch(self.M)
        if type(self.f2sM) == np.ndarray:
            self.f2sM = torch.tensor(self.f2sM, dtype=torch.float32, requires_grad=False)
        else:
            self.f2sM = torch.tensor(self.f2sM.todense(), dtype=torch.float32, requires_grad=False)
        if type(self.s2fM) == np.ndarray:
            self.s2fM = torch.tensor(self.s2fM, dtype=torch.float32, requires_grad=False)
        else:
            self.s2fM = torch.tensor(self.s2fM.todense(), dtype=torch.float32, requires_grad=False)
        if self.boundary == "DirichletZero":
            if type(self.Ms0) == np.ndarray:
                self.Ms0 = torch.tensor(self.Ms0, dtype=torch.float32, requires_grad=False)
            else:
                self.Ms0 = torch.tensor(self.Ms0.todense(), dtype=torch.float32, requires_grad=False)
            self.M0 = spnumpy2sptorch(self.M0)
                
        self.is_tensor = True
        
    def trans2numpy(self):
        if self.is_tensor == True:
            self.mean_vec = np.array(self.mean_vec_learn.cpu().detach().numpy())
            self.log_gamma = np.array(self.log_gamma_learn.cpu().detach().numpy())
            self.lam_res = np.array(self.lam_res.cpu().detach().numpy())
            self.lam = np.array(self.lam.cpu().detach().numpy())
            self.eigvec = np.array(self.eigvec) 
            self.eigvec_ = np.array(self.eigvec_) 
            self.Ms = np.array(self.Ms)
            self.M = sptorch2spnumpy(self.M)
            self.f2sM = np.array(self.f2sM)
            self.s2fM = np.array(self.s2fM)
            if self.boundary == "DirichletZero":
                self.Ms0 = np.array(self.Ms0)
                self.M0 = sptorch2spnumpy(self.M)
                
            self.is_tensor = False
    
    def evaluate_CM_inner(self, u, v=None):
        if v is None:
            v = u
        if self.is_tensor == True:
            assert type(u) == type(v) == torch.Tensor
            v = torch.mv(self.f2sM, v)
            mean_vec_learn = torch.mv(self.f2sM, self.mean_vec_learn)
            res = v - mean_vec_learn
            if self.boundary == "DirichletZero":
                val = torch.mv(self.Ms0, res)
            elif self.boundary == "NeumannZero":
                val = torch.mv(self.Ms, res)
            val = torch.mv(self.eigvec_.T, val)
            if self.num_gamma < self.num_KL:
                Lam = torch.cat([
                    torch.pow(torch.exp(self.log_gamma_learn), -2),
                    torch.pow(self.lam_res, -2)
                    ])
            else:
                Lam = torch.pow(torch.exp(self.log_gamma_learn), -2)
            val = Lam*val
            val = torch.mv(self.eigvec_, val) 
            if self.boundary == "DirichletZero":
                val = torch.mv(self.Ms0.T, val)
            elif self.boundary == "NeumannZero":
                val = torch.mv(self.Ms.T, val)
            val = torch.sum((torch.mv(self.f2sM, u) - mean_vec_learn)*val)
            
            return val
        else:
            mean_vec = self.f2sM@self.mean_vec 
            v = self.f2sM@v
            u = self.f2sM@u
            res = v - mean_vec
            if self.boundary == "DirichletZero":
                val = self.Ms0@res
            elif self.boundary == "NeumannZero":
                val = self.Ms@res
            val = self.eigvec_.T@val
            if self.num_gamma < self.num_KL:
                Lam = np.concatenate([
                    np.power(np.exp(self.log_gamma), -2),
                    np.power(self.lam_res, -2)
                    ])
            else:
                Lam = np.power(np.exp(self.log_gamma), -2)
            self.Lam = Lam
            val = Lam*val
            val = self.eigvec_@val 
            if self.boundary == "DirichletZero":
                val = (self.Ms0.T)@val
            elif self.boundary == "NeumannZero":
                val = (self.Ms.T)@val
            val = (u - mean_vec)@val
            
            return val
            

    def evaluate_grad(self, u_vec):
        assert self.is_tensor == False 
        assert type(u_vec) is np.ndarray
        
        u_vec = self.f2sM@u_vec
        if self.boundary == "DirichletZero":
            val = self.eigvec_.T@self.Ms0@u_vec
        elif self.boundary == "NeumannZero":
            val = self.eigvec_.T@self.Ms@u_vec
        if self.num_gamma < self.num_KL:
            Lam = np.concatenate([
                np.power(np.exp(self.log_gamma), -2),
                np.power(self.lam_res, -2)
                ])
        else:
            Lam = np.power(np.exp(self.log_gamma), -2)
        val = Lam*val
        val = self.eigvec_@val
        val = self.s2fM@val
        
        return np.array(val) 
    
    def evaluate_hessian_vec(self, u_vec):
        return self.evaluate_grad(u_vec)
    
    def precondition(self, m_vec):
        m_vec = self.f2sM@m_vec
        val = self.eigvec_.T@np.array(m_vec)
        if self.num_gamma < self.num_KL:
            Lam = np.concatenate([
                np.power(np.exp(self.log_gamma), 2),
                np.power(self.lam_res, 2)
                ])
        else:
            Lam = np.power(np.exp(self.log_gamma), 2) 
        val = Lam*val
        val = self.eigvec_@val
        val = self.s2fM@val
        
        return np.array(val)
    
    def precondition_inv(self, m_vec):
        m_vec = np.array(self.f2sM@m_vec)
        if self.boundary == "DirichletZero":
            val = self.Ms0@m_vec 
        elif self.boundary == "NeumannZero":
            val = self.Ms@m_vec
        val = self.eigvec_.T@val
        if self.num_gamma < self.num_KL:
            Lam = np.concatenate([
                np.power(np.exp(self.log_gamma), -2),
                np.power(self.lam_res, -2)
                ])
        else:
            Lam = np.power(np.exp(self.log_gamma), -2)
        val = Lam*val
        val = self.eigvec_@val
        if self.boundary == "DirichletZero":
            val = self.Ms0@val
        elif self.boundary == "NeumannZero":
            val = self.Ms@val
        val = self.s2fM@val
        
        return np.array(val)
    
    
### ---------------------------------------------------------------------------

class PDEFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, equ_solver):
        M = spnumpy2sptorch(equ_solver.M)
        A = spnumpy2sptorch(equ_solver.A)
        S = torch.tensor(equ_solver.S.todense(), dtype=torch.float32)
        bc_idx = torch.tensor(equ_solver.bc_idx, dtype=torch.int64)
        dt = torch.tensor(equ_solver.dt, dtype=torch.float32)
        num_steps = torch.tensor(equ_solver.num_steps, dtype=torch.int64)
        ctx.save_for_backward(input, M, A, S, bc_idx, dt, num_steps)
        
        m_vec = np.array(input.detach(), dtype=np.float32)
        ## solve forward equation
        v_n = m_vec[:].copy()
        
        for itr in range(equ_solver.num_steps):
            rhs = equ_solver.M@v_n
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[equ_solver.bc_idx] = 0.0 
            v = spsl.spsolve(equ_solver.M + equ_solver.dt*equ_solver.A, rhs)
            v_n = v.copy()   
        
        output = torch.tensor(equ_solver.S@v_n, dtype=torch.float32)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, M, A, S, bc_idx, dt, num_steps = ctx.saved_tensors 
        M = sptorch2spnumpy(M)
        A = sptorch2spnumpy(A)
        S = S.numpy()
        bc_idx = np.int64(bc_idx) 
        dt = dt.numpy()
        
        v_n = -spsl.spsolve(M, (S.T)@np.array(grad_output))
        
        for itr in range(num_steps):
            rhs = M@v_n 
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[bc_idx] = 0.0 
            v = spsl.spsolve(M + dt*A, rhs)
            v_n = v.copy()    

        val = torch.tensor(-v_n, dtype=torch.float32)
        return val, None


class PDEFunBatched(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, equ_solver):
        ## input shape (num_batched, vector_dim)
        assert input.shape[1] == equ_solver.M.shape[1]
        M = spnumpy2sptorch(equ_solver.M)
        A = spnumpy2sptorch(equ_solver.A)
        S = torch.tensor(equ_solver.S.todense(), dtype=torch.float32)
        bc_idx = torch.tensor(equ_solver.bc_idx, dtype=torch.int64)
        dt = torch.tensor(equ_solver.dt, dtype=torch.float32)
        num_steps = torch.tensor(equ_solver.num_steps, dtype=torch.int64)
        ctx.save_for_backward(input, M, A, S, bc_idx, dt, num_steps)
        
        m_vec = np.array(input.detach(), dtype=np.float32)
        ## solve forward equation
        v_n = m_vec.T.copy()
        
        for itr in range(equ_solver.num_steps):
            rhs = equ_solver.M@v_n
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[equ_solver.bc_idx, :] = 0.0 
            v = spsl.spsolve(equ_solver.M + equ_solver.dt*equ_solver.A, rhs)
            v_n = v.copy()   
        
        output = torch.tensor((equ_solver.S@v_n).T, dtype=torch.float32)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, M, A, S, bc_idx, dt, num_steps = ctx.saved_tensors 
        M = sptorch2spnumpy(M)
        A = sptorch2spnumpy(A)
        S = S.numpy()
        bc_idx = np.int64(bc_idx) 
        dt = dt.numpy()
        
        grad_output_numpy = np.array(grad_output)
        v_n = -spsl.spsolve(M, (S.T)@(grad_output_numpy.T)) 
        
        for itr in range(num_steps):
            rhs = M@v_n 
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[bc_idx, :] = 0.0 
            v = spsl.spsolve(M + dt*A, rhs)
            v_n = v.copy()    

        val = torch.tensor((-v_n).T, dtype=torch.float32)
        return val, None


class PDEasNet(nn.Module):
    def __init__(self, pde_fun, equ_solver):
        super(PDEasNet, self).__init__()
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
            self.noise.to_tensor()
        
    def forward(self, predictions, target):
        # self.noise.precision = torch.tensor(self.noise.precision, dtype=predictions.dtype)
        diff = predictions - target
        val = torch.matmul(self.noise.precision, diff)
        loss_val = 0.5*torch.matmul(diff, val)
        return loss_val


class LossResidualBatched(nn.Module):
    def __init__(self, noise):
        super(LossResidual, self).__init__()
        self.noise = noise
        if self.noise.is_torch == False:
            self.noise.to_tensor()
        
    def forward(self, predictions, target):
        # self.noise.precision = torch.tensor(self.noise.precision, dtype=predictions.dtype)
        diff = predictions - target
        val = torch.matmul(self.noise.precision, diff.T)
        loss_val = 0.5*torch.matmul(diff, val)
        loss_val = torch.sum(torch.diag(loss_val))
        return loss_val


class PriorFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, prior):
        M = spnumpy2sptorch(prior.M)
        K = spnumpy2sptorch(prior.K)
        ctx.save_for_backward(input, M, K)
        
        m_vec = np.array(input.detach(), dtype=np.float32)
        ## solve forward equation
        val = prior.K.T@spsl.spsolve(prior.M, prior.K@m_vec)
        output = m_vec@val
        output = torch.tensor(output, dtype=torch.float32)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, M, K = ctx.saved_tensors 
        M = sptorch2spnumpy(M)
        K = sptorch2spnumpy(K)
        
        m_vec = np.array(input, dtype=np.float32)
        val = spsl.spsolve(M, K.T@spsl.spsolve(M, K@m_vec))
        val = torch.tensor(val, dtype=torch.float32)
        return grad_output*val, None


class PriorFunFR(torch.autograd.Function):
    ## need add some comments
    @staticmethod 
    def forward(ctx, input, prior):
        ctx.save_for_backward(input, prior.mean, prior.M, prior.log_gamma, prior.lam_res, \
                              torch.tensor(prior.num_KL, dtype=torch.int64), \
                              torch.tensor(prior.num_gamma, dtype=torch.int64), \
                              prior.eigvec)
        res = input - prior.mean
        val = torch.mv(prior.M, res)
        val = torch.mv(prior.eigvec.T, val)
        if prior.num_gamma < prior.num_KL:
            val = torch.cat([
                torch.pow(torch.exp(prior.log_gamma), -2)*val[:prior.num_gamma], 
                torch.pow(prior.lam_res, -2)*val[prior.num_gamma:]
                ])
        else:
            val = torch.pow(torch.exp(prior.log_gamma), -2)*val
        val = torch.mv(prior.eigvec, val)
        val = torch.mv(prior.M, val)
        val = torch.sum((input - prior.mean)*val)
        output = 0.5*val
        
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, mean, MM, log_gamma, lam_res, num_KL, num_gamma, eigvec = ctx.saved_tensors
        # num_KL = torch.tensor(num_KL, dtype=torch.int64)
        # num_gamma = torch.tensor(num_gamma, dtype=torch.int64)
        val = torch.mv(MM, input)
        val = torch.mv(eigvec.T, val)
        if num_gamma < num_KL:
            val = torch.cat([
                torch.pow(torch.exp(log_gamma), -2)*val[:num_gamma], 
                torch.pow(lam_res, -2)*val[num_gamma:]
                ])
        else:
            val = torch.pow(torch.exp(log_gamma), -2)*val 
        val = torch.mv(eigvec, val)
        return grad_output*val, None
 

class LossPrior(nn.Module):
    def __init__(self, prior_fun, prior):
        super(LossPrior, self).__init__()
        self.prior = prior 
        self.prior_fun = prior_fun
    
    def forward(self, predictions, target=None):
        if target is None:
            loss_val = self.prior_fun(predictions, self.prior)
        else:
            assert predictions.shape == target.shape, "predictions should have same shape as target"
            loss_val = self.prior_fun(predictions - target, self.prior)
        
        return loss_val 



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    