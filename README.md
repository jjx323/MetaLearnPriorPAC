# Code for Learning Prediction Functions of the Prior Measure

## Overview
1. Directory **core** contains the main functions and classes that are useful for implementing the algorithms. Specifically,
- **probability.py**: This file contains classes of GaussianElliptic2[The Gaussian measure implemented by finite element methods based on solving elliptic differential equations used for generating samples and also contains the functionality of evaluate the gradient and Hessian operators];
GaussianFiniteRank[The Gaussian measure implemented by finite element methods and eigensystem decomposition].
- **noise.py**: This file contains the class NoiseGaussianIID.
- **model.py**: This file contains the class Domain and two classes Domain2D and Domain1D inherit from the parent class Domain; contains the parent class ModelBase of the model classes employed in specific examples, the class ModelBase incorporates the components of the domain, prior, equation solver, and noise. 
- **linear_eq_solver.py**: contains the function cg_my which is our implementation of the conjugate gradient algorithm for solving linear equations. 
- **eigensystem.py**: This file contains the function double_pass which is our implementation of an algorithm for calculating eigensystem. 
- **approximate_sample.py**: This file contains the class LaplaceApproximate, which can be used to compute the Laplace approximation of the posterior measures. 
- **optimizer.py**: This file contains the class OptimBase[incorporate an implementation of armijo_line_search can be employed for each optimizer]; the class GradientDescent[an implementation of the gradient descent algorithm]; the class NewtonCG[an implementation of the Newton conjugate gradient algorithm]. 
- **sample.py**: This file contains the class pCN, which is a type of discrete invariant Markov chain Monte Carlo sampling algorithm. 
- **Plotting.py**: This file contains some functions that can draw functions generated by FEniCS. 
- **misc.py**: This file contains functions of trans2spnumpy, trans2sptorch, spnumpy2sptorch, sptorch2spnumpy, and sptensor2cude, which will be useful for transferring sparse matrixes to different forms required for doing calculations in numpy, pytorch, and FEniCS. This file also contains the function construct_measurement_matrix, which will be used for generating a sparse matrix S. The matrix S times a function generated by FEniCS to get the values at the measurement points.

2. Directory **BackwardDiffusion** contains the main functions and classes for the backward diffusion problem.
- **common.py**: This file contains the class EquSolver and the class ModelBackwardDiffusion. The class EquSolver contains the implementations of solvers of forward, adjoint, incremental forward, and incremental adjoint equations. These equations are necessary for implementing gradient and Newton-type optimization algorithms. The class ModelBackwardDiffusion contains the function of calculating the loss, the gradient, and the Hessian operator.
- **meta_common.py**: This file contains the classes Gaussian1DFiniteDifference, GaussianElliptic2Learn, GaussianFiniteRank, PDEFun, PDEFunBatched, PDEasNet, LossResidual, LossResidualBatched, PriorFun, PriorFunFR, LossPrior. The functionalities of these classes are obvious from their names. In these functions, we rewrite some functions in the file common.py to make them take advantage of the autograd ability in PyTorch.

  The subdirectory **1D_meta_learning** contains the main files for our numerical results.
- **NN_library.py**: This file contains the class FNO1D and some aulixary functions. In this class, we implement the Fourier neural operator for 1D functions.
- **generate_meta_data.py**: Python scripts that can generate the learning data with parameter settings according to Section 4 of the paper https://arxiv.org/abs/2310.12436
- **meta_learn_mean.py**: Python scripts for learning a prior measure $\mathcal{N}(f(\theta), \mathcal{C}_0)$, where the mean function $f(\theta)$ is independent of the data. 
- **meta_learn_FNO.py**: Python scripts for learning a prior measure $\mathcal{N}(f(S; \theta), \mathcal{C}_0)$, where the mean function $f(S; \theta)$ depends on the data. In the program, the function $f(S;\theta)$ is implemented as a Fourier neural operator.  
- **MAPSimpleCompare.py**: Compare the relative errors of maximum a posteriori estimates obtained by the optimization algorithm under the simple environment setting. 
- **MAPComplexCompare.py**: Compare the relative errors of maximum a posteriori estimates obtained by the optimization algorithm under the complex environment setting.

  3. Directory **SteadyStateDarcyFlow** contains the main functions and classes for the Darcy flow problem.
- **common.py**: This file contains the class EquSolver and the class ModelDarcyFlow. The class EquSolver contains the implementations of solvers of forward, adjoint, incremental forward, and incremental adjoint equations. These equations are necessary for implementing gradient and Newton-type optimization algorithms. The class ModelDarcyFlow contains the function of calculating the loss, the gradient, and the Hessian operator.
- **meta_common.py**: This file contains the classes GaussianFiniteRankTorch, GaussianElliptic2Torch, PriorFun, HyperPrior, HyerPriorAll, ForwardProcessNN, ForwardProcessPDE, ForwardPrior, LossFun, PDEFun, PDEasNet, LossResidual, Dis2Fun, LpLoss, FNO2d. The functionalities of these classes are obvious from their names. In these functions, we rewrite some functions in the file common.py to make them take advantage of the autograd ability in PyTorch.
  
