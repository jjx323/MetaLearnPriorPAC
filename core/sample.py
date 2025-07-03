#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:56:03 2022

@author: Junxiong Jia
"""

import numpy as np
import os
from scipy.special import logsumexp

###################################################################
class pCN(object):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems, 
    Hankbook of Uncertainty Quantification, 2017
    '''
    def __init__(self, prior, phi, beta=0.01, save_num=np.int64(1e4), path=None):
        
        assert hasattr(prior, "generate_sample") and hasattr(prior, "mean_vec")
        assert hasattr(prior, "generate_sample_zero_mean")
        
        self.prior = prior
        self.phi = phi
        self.beta = beta
        self.dt = (2*(2 - beta**2 - 2*np.sqrt(1-beta**2)))/(beta**2)
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.acc_rate = 0.0
        self.chain = []
        
    def generate_chain(self, length_total=1e5, callback=None, uk=None, index=None, beta=None):
        
        # assert type(length_total) == type(1.0) or type(length_total) == type(np.array(1.0)) or type(length_total) == type(1)
        if beta is not None:
            self.beta = beta
            self.dt = (2*(2 - self.beta**2 - 2*np.sqrt(1-self.beta**2)))/(self.beta**2)

        if uk is None:
            uk = self.prior.generate_sample()
        else:
            uk = uk
        self.chain.append(uk.copy())
        ac_rate = 0
        ac_num = 0
        
        def aa(u_new, phi_old):
            #return min(1, np.exp(self.phi(u_old)-self.phi(u_new)))
            #print(self.phi(u_old)-self.phi(u_new))
            phi_new = self.phi(u_new)
            assert phi_new <= 1e20, "The algorithm cannot work when phi > 1e20"
            panduan = np.exp(min(0.0, phi_old-phi_new))
            return panduan, phi_new
        
        si = 0
        if index == None: index = 0
        phi_old = 1e20   # a large enough number 
        i = 1
        m0 = self.prior.mean_vec
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            xik = self.prior.generate_sample_zero_mean()
            ## ref: Algorithms for Kullback-Leibler approximation of probability 
            ## measures in infinite dimensions, SIAM J. SCI. COMPUT, 2015
            vk = m0 + a*(uk - m0) + b*xik
            t, phi_new = aa(vk, phi_old)
            r = np.random.uniform(0, 1)
            if t >= r:
                self.chain.append(vk.copy())
                uk = vk.copy()
                ac_num += 1
                phi_old = phi_new
            else: 
                self.chain.append(uk.copy())
            ac_rate = ac_num/i 
            self.acc_rate = ac_rate
            i += 1
            
            if self.path is not None:
                si += 1
                if np.int64(si) == np.int64(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + str(np.int64(index)), self.chain)
                    # del chain
                    self.chain = []
                    index += 1
        
            if callback is not None:
                callback([uk, i, ac_rate, ac_num])

        if self.path is None:
            return [self.chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int64(index)]
        

class SMC_para(object):
    '''
    Ref: M. Dashti, A. M. Stuart, The Bayesian Approach to Inverse Problems,
    Handbook of Uncertainty Quantification, Springer, 2017 [Section 5.3]
    '''
    def __init__(self, comm, model, num_particles):
        self.comm = comm
        assert num_particles >= comm.size and num_particles % self.comm.size==0
        self.num_per_worker = int(self.N // self.comm.size)
        self.N = num_particles
        self.model = model
        self.prior = model.prior
        self.num_dofs = model.num_dofs
        self.potential=model.loss_residual

        # self.num_particles = num_particles
        # self.potential_funs = []

    def prepare(self, particles=None):
        self.particles_all = particles
        self.idxs = np.arange(self.num_per_worker) + self.comm.rank * self.num_per_worker
        if particles is None:
            self.particles_local = self.prior.generate_sample(self.num_per_worker)
        else:
            self.particles_local = particles[self.idxs]
        self.ESS = None
        self.sum_h = 0 # the loss fun = sum_h*loss_res, thus sum_h is needed in all kernels
        self.average_acc_rates = None # to adjust beta_pCN, average acc rates are needed in all kernels

        # self.weights = np.ones(self.num_particles) * (1 / self.num_particles)
        # particles = []
        # for idx in range(self.num_particles):
        #     particles.append(self.model.prior.generate_sample())
        # self.particles = np.array(particles)

    def gather_samples(self):                                # After smc.prepare(), each core keeps its particles. 
        tem = self.comm.gather(self.particles_local, root=0) # This function lets core 0 collect all particles.
        if self.comm.rank==0:
            self.particles_all = np.concatenate(tem, axis=0)
    
    def resample(self):
        self.phis_local = []
        for x in self.particles_local:
            self.phis_local.append(self.potential(x))        
        self.gather_phis()
        if self.comm.rank == 0:
            tmp = -self.phis_all * self.h
            self.weights = np.exp(tmp - logsumexp(tmp))
            idx_resample = np.random.choice(self.N, self.N, True, self.weights)
            self.particles_all = self.particles_all[idx_resample, :]
        self.particles_all = self.comm.bcast(self.particles_all, root=0)
        self.particles_local = self.particles_all[self.idxs, :]

    def gather_phis(self):
        tem = self.comm.gather(self.phis_local, root=0)
        if self.comm.rank==0:
            self.phis_all = np.array(tem).ravel()
    
    def search_h(self, num_particles, h0 = 0.1, ESS_threshold_ratio = 0.6,):
        if self.comm.rank==0:
            particles=self.particles_all[np.arange(num_particles),:]
            h = h0
            phis=[]
            for x in particles:
                phis.append(self.potential(x))
            phis=np.array(phis)
            while True:
                tmp = -phis * h
                weights = np.exp(tmp - logsumexp(tmp))
                ESS = 1 / (weights @ weights)
                if ESS > ESS_threshold_ratio * num_particles:
                    break
                h /= 2 
            self.ESS = ESS
            self.h = h
            self.sum_h+= h
        self.sum_h = self.comm.bcast(self.sum_h, root=0)

    def transition(self, sampler, beta=0.8, length_total=1000):
        self.acc_rates = np.zeros(self.num_per_worker)
        for idx in range(self.num_per_worker): 
            sampler.generate_chain(uk=self.particles_local[idx], beta=beta, length_total=length_total)
            self.acc_rates[idx] = sampler.acc_rate
            tmp = np.array(sampler.chain[-1]).squeeze()
            self.particles_local[idx] = tmp.copy()
        
    def gather_acc_rates(self):
        tem = self.comm.gather(self.acc_rates, root=0)
        if self.comm.rank==0:
            acc_rates_all = np.array(tem).ravel()
            self.average_acc_rates = np.mean(acc_rates_all)
        self.average_acc_rates = self.comm.bcast(self.average_acc_rates, root=0)


class SMC:
    '''
    Ref: M. Dashti, A. M. Stuart, The Bayesian Approach to Inverse Problems,
    Handbook of Uncertainty Quantification, Springer, 2017 [Section 5.3]
    '''
    def __init__(self, model, num_particles):
        self.model = model
        self.num_particles = num_particles
        self.potential_funs = []

    def prepare(self):
        self.weights = np.ones(self.num_particles) * (1 / self.num_particles)
        particles = []
        for idx in range(self.num_particles):
            particles.append(self.model.prior.generate_sample())
        self.particles = np.array(particles)

    def transition(self, sampler, len_chain, info_acc_rate=True, **kwargs):
        if not hasattr(sampler, 'generate_chain') or not hasattr(sampler, 'acc_rate') or not hasattr(sampler, 'chain'):
            raise AttributeError("sampler must have 'generate_chain', 'acc_rate', and 'chain' attributes")

        if not isinstance(len_chain, np.int64) or len_chain <= 0:
            raise ValueError("len_chain must be a positive integer with type numpy.int64")

        if not isinstance(info_acc_rate, bool):
            raise ValueError("info_acc_rate must be a bool")

        self.acc_rates = np.zeros(self.num_particles)
        for idx in range(self.num_particles):
            sampler.generate_chain(length_total=len_chain, uk=self.particles[idx, :], **kwargs)
            if info_acc_rate == True:
                print("acc_rate = %.5f" % sampler.acc_rate)
            self.acc_rates[idx] = sampler.acc_rate
            tmp = np.array(sampler.chain[-1]).squeeze()
            if np.any(np.isnan(tmp)):
                raise ValueError("particles should not contain np.nan, there must be problems when sampling!")
            # if np.any(np.isnan(tmp)):
            #     tmp = self.particles_local[idx, :]
            #     print("sampler failed to sampling!")
            self.particles[idx, :] = tmp.copy()

    def resampling(self, h):
        tmp = -h*self.potential_funs
        self.weights = np.exp(tmp - logsumexp(tmp))
        ## resampling (without comm.gather all of the particles to root 0, it seems complex
        ## for doing the resampling procedure.)
        idx_resample = np.random.choice(len(self.weights), len(self.weights), True, self.weights)
        self.particles = self.particles[idx_resample, :]
        self.weights = np.ones(self.num_particles) * (1 / self.num_particles)

    def eval_potential_funs(self, potential_fun):
        tmp = np.zeros(self.num_particles, dtype=np.float64)
        self.potential_funs = []
        for idx in range(self.num_particles):
            self.potential_funs.append(potential_fun(self.particles[idx, :]))
        self.potential_funs = np.array(self.potential_funs)
        # print("potential_funs_shape = ", self.potential_funs.shape)

    def search_h(self, h):
        while 1:
            h_phis = -h*self.potential_funs
            weights = np.exp(h_phis - logsumexp(h_phis))
            ess = 1/(weights@weights)
            if ess > 0.6*self.num_particles:
                break 
            else:
                h = h/2
        return h 
        





