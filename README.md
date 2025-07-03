# Code for Learning Prediction Functions of the Prior Measure

## Overview

The program relies on FEniCS (Version 2019.1.0), PyTorch (Version 1.12.1+cu116), NumPy (Version 1.23.3), and SciPy (Version 1.13.0). You may need to install FEniCS in a conda environment and then install PyTorch using pip. Installing PyTorch via conda may cause conflicts, but installing it via pip has not caused any issues for us.

### 1. Directory **core**

Contains the main functions and classes for implementing the algorithms. Specifically:

- **probability.py**:
  - `GaussianElliptic2`: A Gaussian measure implemented using finite element methods based on solving elliptic differential equations. It supports sample generation and evaluates gradient and Hessian operators.
  - `GaussianFiniteRank`: A Gaussian measure implemented using finite element methods and eigensystem decomposition.
- **noise.py**: Contains the class `NoiseGaussianIID`.
- **model.py**:
  - Contains the parent class `Domain` and its subclasses `Domain2D` and `Domain1D`.
  - Contains the parent class `ModelBase`, which is used for specific examples. `ModelBase` incorporates domain, prior, equation solver, and noise components.
- **linear_eq_solver.py**: Contains the function `cg_my`, an implementation of the conjugate gradient algorithm for solving linear equations.
- **eigensystem.py**: Contains the function `double_pass`, an implementation of an eigensystem calculation algorithm.
- **approximate_sample.py**: Contains the class `LaplaceApproximate` for computing Laplace approximations of posterior measures.
- **optimizer.py**:
  - `OptimBase`: Includes an implementation of `armijo_line_search` for optimizers.
  - `GradientDescent`: An implementation of the gradient descent algorithm.
  - `NewtonCG`: An implementation of the Newton conjugate gradient algorithm.
- **sample.py**:
  - `pCN`: A discrete invariant Markov chain Monte Carlo sampling algorithm.
  - `SMC`: An implementation of the sequential Monte Carlo sampling algorithm.
- **Plotting.py**: Contains functions for visualizing FEniCS-generated functions.
- **misc.py**:
  - Functions for converting sparse matrices between NumPy, PyTorch, and FEniCS formats: `trans2spnumpy`, `trans2sptorch`, `spnumpy2sptorch`, `sptorch2spnumpy`, `sptensor2cude`.
  - `construct_measurement_matrix`: Generates a sparse matrix **S** that, when multiplied by a FEniCS-generated function, extracts values at measurement points.

### 2. Directory **BackwardDiffusion**

Contains the main functions and classes for the backward diffusion problem.

- **common.py**:
  - `EquSolver`: Implements solvers for forward, adjoint, incremental forward, and incremental adjoint equations, which are required for gradient and Newton-type optimization algorithms.
  - `ModelBackwardDiffusion`: Computes the loss, gradient, and Hessian operator.
- **meta_common.py**:  
  Contains classes rewritten to leverage PyTorch's autograd capability:
  - `Gaussian1DFiniteDifference`, `GaussianElliptic2Learn`, `GaussianFiniteRank`
  - `PDEFun`, `PDEFunBatched`, `PDEasNet`
  - `LossResidual`, `LossResidualBatched`
  - `PriorFun`, `PriorFunFR`, `LossPrior`

#### Subdirectory **1D_meta_learning**

Contains files for numerical results:

- **NN_library.py**: Implements the 1D Fourier neural operator (`FNO1D`) and auxiliary functions.
- **generate_meta_data.py**: Generates training data.
- **meta_learn_mean.py**: Learns a prior measure $\mathcal{N}(f_m(\theta_1), \mathcal{C}_0(\theta_2)$, where the mean $f_m(\theta_1)$ is data-independent.
- **meta_learn_FNO.py**: Learns a prior measure $\mathcal{N}(f_m(S; \theta_1), \mathcal{C}_0(\theta_2)$, where the mean $f_m(S; \theta_1)$ is data-dependent and implemented as a Fourier neural operator.
- **MAPSimpleCompare.py**: Compares relative errors of maximum a posteriori (MAP) estimates under simple settings.
- **MAPComplexCompare.py**: Compares relative errors of MAP estimates under complex settings.
- **var_simple_analysis.py**: Computes the variance field under simple settings.
- **var_complex_analysis.py**: Computes the variance field under complex settings.

### 3. Directory **SteadyStateDarcyFlow**

Contains the main functions and classes for the Darcy flow problem.

- **common.py**:
  - `EquSolver`: Implements solvers for forward, adjoint, incremental forward, and incremental adjoint equations.
  - `ModelDarcyFlow`: Computes the loss, gradient, and Hessian operator.
- **MLcommon.py**:  
  Contains PyTorch-optimized versions of functions:
  - `GaussianFiniteRankTorch`, `GaussianElliptic2Torch`
  - `PriorFun`, `HyperPrior`, `HyerPriorAll`
  - `ForwardProcessNN`, `ForwardProcessPDE`, `ForwardPrior`
  - `LossFun`, `PDEFun`, `PDEasNet`, `LossResidual`
  - `Dis2Fun`, `LpLoss`, `FNO2d`

#### Subdirectory **2D_meta_learning**

Contains files for numerical results:

- **generate_meta_data.py**: Generates training data.
- **meta_learn_mean.py**: Learns a prior measure $\mathcal{N}(f_m(\theta_1), \mathcal{C}_0(\theta_2))$ with a data-independent mean.
- **meta_learn_mean_FNO.py**: Learns a prior measure $\mathcal{N}(f_m(S; \theta_1), \mathcal{C}_0(\theta_2))$ with a data-dependent mean (Fourier neural operator).
- **run_map.py**: Compares MAP estimate errors under simple and complex settings.
- **compare_truth_FNO.py**: Generates comparison plots for different methods.
- **run_smc_mix.py**: Runs the mixture Gaussian sequential Monte Carlo (MGSMC) algorithm on test datasets.
- **smc_mix_analysis.py**: Analyzes MGSMC results to compute posterior means and credible intervals.
- **test_map.py**: Tests a positive-valued sample in the complex environment with `max_iter_num = 100000` to evaluate performance differences between unlearned and learned data-independent Bayesian models.

## Workflows

### The Backward Diffusion Problem

Run the following commands sequentially to generate the training and testing datasets:

```bash
python generate_meta_data.py --env "simple"  
python generate_meta_data.py --env "complex"  
python generate_meta_data.py --test_true --env "simple"  
python generate_meta_data.py --test_true --env "complex"  
```

Run the following commands sequentially to learn the mean function and the FNO:

```bash
python meta_learn_mean.py --env "simple"  
python meta_learn_mean.py --env "complex"  
python meta_learn_FNO.py --env "simple"  
python meta_learn_FNO.py --env "complex"  
```

Run the following commands sequentially to obtain the maximum a posteriori estimates:

```bash
python MAPSimpleCompare.py  
python MAPComplexCompare.py  
python var_simple_analysis.py  
python var_complex_analysis.py  
python meta_learn_mean_test_L.py --env "simple"  
python meta_learn_FNO_test_hidden_dim.py --env "complex"  
```

After executing all commands, a directory named `RESULTS-PAPER-MAP` will be created, containing two files:

- `errors_simple.txt`
- `errors_complex.txt`

Additionally, the folder `RESULTS-PAPER-MAP` will contain figures similar to those shown in the paper.

**Alternative**: You can run all the above commands at once by executing:

```bash
./run_1D.sh  
```

---

### The Darcy Flow Problem

Run the following commands sequentially to generate the training and testing datasets:

```bash
python generate_meta_data.py  
```

Run the following commands sequentially to learn the mean function and the FNO:

```bash
python meta_learn_mean.py --env "simple"  
python meta_learn_mean.py --env "complex"  
python meta_learn_mean_FNO.py --env "simple"  
python meta_learn_mean_FNO.py --env "complex"  
```

Run the following commands sequentially to obtain the maximum a posteriori estimates:

```bash
python prepare_test.py --env "simple"  
python prepare_test.py --env "complex"  
python run_map.py --env "simple"  
python run_map.py --env "complex"  
python compare_truth_FNO.py --env "simple"  
python compare_truth_FNO.py --env "complex"  
python run_smc_mix.py --env "simple"  
python run_smc_mix.py --env "complex"  
python smc_mix_analysis.py --env "simple"  
python smc_mix_analysis.py --env "complex"  
python test_map.py  
python analysis_test_map.py
```

After executing all commands, the following directories will be created:

- `RESULTS`
- `RESULTS-PAPER-MAP`
- `RESULT-PAPER-SMC-MIX`

These directories will contain:

- Two error files: `simple_errors.txt` and `complex_errors.txt`
- Figures similar to those shown in the paper

**Alternative**: You can run all the above commands at once by executing:

```bash
./run_2D.sh  
```