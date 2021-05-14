import fenics as fe
import numpy as np
import ufl
 

# ---------------------------------------------------------------- 
# History function
def get_history(history, psi_plus):
    return fe.conditional(fe.gt(history, psi_plus), history, psi_plus)
    

# ---------------------------------------------------------------- 
# Degradation functions
def g_d(d):
    degrad = (1 - d)**2;
    return degrad 


def g_d_prime(d, degrad_func):
    d = fe.variable(d)
    degrad = degrad_func(d)
    degrad_prime = fe.diff(degrad, d)
    return degrad_prime


# ---------------------------------------------------------------- 
# Linear elasticity
def strain(grad_u):
    return 0.5*(grad_u + grad_u.T)


def psi_linear_elasticity(epsilon, lamda, mu):
    return lamda / 2 * fe.tr(epsilon)**2 + mu * fe.inner(epsilon, epsilon)


# ----------------------------------------------------------------
# Model A: Essentially no decomposition
def psi_plus_linear_elasticity_model_A(epsilon, lamda, mu):
    return psi_linear_elasticity(epsilon, lamda, mu)


def psi_minus_linear_elasticity_model_A(epsilon, lamda, mu):
    return 0.


# ----------------------------------------------------------------
# Model B: Amor paper https://doi.org/10.1016/j.jmps.2009.04.011
# TODO(Tianju): Check if bulk_mod is correct under plane strain assumption
def psi_plus_linear_elasticity_model_B(epsilon, lamda, mu):
    dim = 2
    bulk_mod = lamda + 2. * mu / dim
    tr_epsilon_plus = ufl.Max(fe.tr(epsilon), 0)
    return bulk_mod / 2. * tr_epsilon_plus**2 + mu * fe.inner(fe.dev(epsilon), fe.dev(epsilon))


def psi_minus_linear_elasticity_model_B(epsilon, lamda, mu):
    dim = 2
    bulk_mod = lamda + 2. * mu / dim
    tr_epsilon_minus = ufl.Min(fe.tr(epsilon), 0)
    return bulk_mod / 2. * tr_epsilon_minus**2


# ----------------------------------------------------------------
# Model C: Miehe paper https://doi.org/10.1002/nme.2861
# Eigenvalue decomposition for 2x2 matrix
# See https://yutsumura.com/express-the-eigenvalues-of-a-2-by-2-matrix-in-terms-of-the-trace-and-determinant/

# Remarks(Tianju): The ufl functions Max and Min do not seem to behave as expected
# For example, the following line of code works
# tr_epsilon_plus = (fe.tr(epsilon) + np.absolute(fe.tr(epsilon))) / 2
# However, the following line of code does not work (Newton solver never converges)
# tr_epsilon_plus = ufl.Max(fe.tr(epsilon), 0)

# Remarks(Tianju): If Newton solver fails to converge, consider using a non-zero initial guess for the displacement field
# For example, use Model A to solve for one step and then switch back to Model C
# The reason for the failure is not clear.
# It may be because of the singular nature of Model C that causes trouble for UFL to take derivatives at the kink.
def psi_plus_linear_elasticity_model_C(epsilon, lamda, mu):
    sqrt_delta = fe.conditional(fe.gt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon), 0), fe.sqrt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon)), 0)
    eigen_value_1 = (fe.tr(epsilon) + sqrt_delta) / 2
    eigen_value_2 = (fe.tr(epsilon) - sqrt_delta) / 2
    tr_epsilon_plus = fe.conditional(fe.gt(fe.tr(epsilon), 0.), fe.tr(epsilon), 0.)
    eigen_value_1_plus = fe.conditional(fe.gt(eigen_value_1, 0.), eigen_value_1, 0.)
    eigen_value_2_plus = fe.conditional(fe.gt(eigen_value_2, 0.), eigen_value_2, 0.)
    return lamda / 2 * tr_epsilon_plus**2 + mu * (eigen_value_1_plus**2 + eigen_value_2_plus**2)


def psi_minus_linear_elasticity_model_C(epsilon, lamda, mu):
    sqrt_delta = fe.conditional(fe.gt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon), 0), fe.sqrt(fe.tr(epsilon)**2 - 4 * fe.det(epsilon)), 0)
    eigen_value_1 = (fe.tr(epsilon) + sqrt_delta) / 2
    eigen_value_2 = (fe.tr(epsilon) - sqrt_delta) / 2
    tr_epsilon_minus = fe.conditional(fe.lt(fe.tr(epsilon), 0.), fe.tr(epsilon), 0.)
    eigen_value_1_minus = fe.conditional(fe.lt(eigen_value_1, 0.), eigen_value_1, 0.)
    eigen_value_2_minus = fe.conditional(fe.lt(eigen_value_2, 0.), eigen_value_2, 0.)
    return lamda / 2 * tr_epsilon_minus**2 + mu * (eigen_value_1_minus**2 + eigen_value_2_minus**2)


# TODO(Tianju): Collapse the three functions into one
# ---------------------------------------------------------------- 
# Cauchy stress
def cauchy_stress_plus(epsilon, psi_plus):
    epsilon = fe.variable(epsilon)
    energy_plus = psi_plus(epsilon)
    sigma_plus = fe.diff(energy_plus, epsilon)
    return sigma_plus

    
def cauchy_stress_minus(epsilon, psi_minus):
    epsilon = fe.variable(epsilon)
    energy_minus = psi_minus(epsilon)
    sigma_minus = fe.diff(energy_minus, epsilon)
    return sigma_minus


def cauchy_stress(epsilon, psi):
    epsilon = fe.variable(epsilon)
    energy = psi(epsilon)
    sigma = fe.diff(energy, epsilon)
    return sigma
