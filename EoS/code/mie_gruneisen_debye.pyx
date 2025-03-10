#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# mie_gruneisen_debye.pyx

import numpy as np
cimport numpy as np
from scipy.optimize import brentq
import warnings
from scipy.integrate import quad, IntegrationWarning
import os

# Import C math functions for performance
from libc.math cimport exp, log, pow, isnan, round

os.makedirs('binary_file/', exist_ok=True)

# --- Module-level constants ---
cdef double k_B      = 1.380649e-23   # Boltzmann constant (J/K)
cdef double R        = 8.314          # J/K/mol

# --- Top-level integrand functions ---

def _integrand_debye_1(x):
    """Integrand for I1(z) = ∫₀ᶻ x³/(eˣ - 1) dx."""
    if x == 0.0:
        return 0.0
    return (x**3) / (exp(x) - 1)

def _integrand_debye_2(x):
    """Integrand for I2(z) = ∫₀ᶻ x⁴ eˣ/(eˣ - 1)² dx."""
    if x == 0.0:
        return 0.0
    return (x**4 * exp(x)) / ((exp(x) - 1)**2)

# --- Debye Integrals ---
cpdef double debye_integral_1(double z):
    """
    I1(z) = ∫₀ᶻ x³/(eˣ - 1) dx.
    """
    cdef double val, err
    cdef int limit = 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, err = quad(_integrand_debye_1, 0.0, z, limit=limit)
    return val

cpdef double debye_integral_2(double z):
    """
    I2(z) = ∫₀ᶻ x⁴ eˣ/(eˣ - 1)² dx.
    """
    cdef double val, err
    cdef int limit = 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, err = quad(_integrand_debye_2, 0.0, z, limit=limit)
    return val

# --- Material Functions ---
cpdef double Theta_D(double v, double Theta_D0, double gamma0, double q, double v0):
    """
    Debye temperature as a function of specific volume v.
    Theta_D(v)=Theta_D0 * exp[(gamma0/q)*(1 - (v/v0)^q)]
    """
    return Theta_D0 * exp((gamma0 / q) * (1 - pow(v / v0, q)))

cpdef double gamma_val(double v, double gamma0, double q, double v0):
    """
    Grüneisen parameter: gamma(v)=gamma0*(v/v0)^q.
    """
    return gamma0 * pow(v / v0, q)

cpdef double e_th(double v, double T, double Theta_D0, double gamma0, double q, double v0):
    """
    Thermal energy per unit mass:
      e_th = 9 n k_B T (T/Theta_D)^3 I1(Theta_D/T)
    """
    cdef double theta_d = Theta_D(v, Theta_D0, gamma0, q, v0)
    cdef double z = theta_d / T
    cdef double I1_val = debye_integral_1(z)
    return 9.0 * n * k_B * T * pow(T/theta_d, 3) * I1_val

cpdef double delta_e_th(double v, double T, double Theta_D0, double gamma0, double q, double v0):
    """
    Thermal energy difference relative to the reference temperature T0.
    """
    return e_th(v, T, Theta_D0, gamma0, q, v0) - e_th(v, T0, Theta_D0, gamma0, q, v0)

cpdef double s(double v, double T, double Theta_D0, double gamma0, double q, double v0):
    """
    Entropy per unit mass:
      s = 9 n k_B (T/Theta_D)^3 I1(Theta_D/T) - 3 n k_B ln(1 - exp(-Theta_D/T))
    """
    cdef double theta_d = Theta_D(v, Theta_D0, gamma0, q, v0)
    cdef double I1_val = debye_integral_1(theta_d / T)
    return 9.0 * n * k_B * pow(T/theta_d, 3) * I1_val - 3.0 * n * k_B * log(1 - exp(-theta_d/T))

cpdef double c_V(double v, double T, double Theta_D0, double gamma0, double q, double v0):
    """
    Specific heat at constant volume (per unit mass):
      c_V = 9 n k_B (T/Theta_D)^3 I2(Theta_D/T)
    """
    cdef double theta_d = Theta_D(v, Theta_D0, gamma0, q, v0)
    cdef double I2_val = debye_integral_2(theta_d / T)
    return 9.0 * n * k_B * pow(T/theta_d, 3) * I2_val

# --- Cold Curve Pressure ---
cpdef double P0(double v, double v0, double K0, double K0_prime):
    """
    Cold-curve pressure using the BM3 equation of state.
    """
    cdef double x = v0 / v
    cdef double first  = pow(x, 7.0/3.0) - pow(x, 5.0/3.0)
    cdef double second = 1.0 - 0.75 * (4.0 - K0_prime) * (pow(x, 2.0/3.0) - 1.0)
    return 1.5 * K0 * first * second

cpdef double pressure(double v, double T, double Theta_D0, double gamma0, double q,
                        double v0, double K0, double K0_prime):
    """
    Total pressure per unit mass:
      P = P0(v) + (gamma(v)/v)*delta_e_th(v,T)
    """
    cdef double gam = gamma_val(v, gamma0, q, v0)
    return P0(v, v0, K0, K0_prime) + (gam / v) * delta_e_th(v, T, Theta_D0, gamma0, q, v0)

cpdef double bulk_modulus(double v, double T, double Theta_D0, double gamma0, double q,
                            double v0, double K0, double K0_prime, double delta=1e-6):
    """
    Isothermal bulk modulus per unit mass:
      B_T = - v (dP/dv)_T,
    where the derivative is computed by finite differences.
    """
    cdef double P_plus  = pressure(v + delta, T, Theta_D0, gamma0, q, v0, K0, K0_prime)
    cdef double P_minus = pressure(v - delta, T, Theta_D0, gamma0, q, v0, K0, K0_prime)
    cdef double dPdv = (P_plus - P_minus) / (2.0 * delta)
    return -v * dPdv

cpdef double thermal_expansion(double v, double T, double Theta_D0, double gamma0, double q,
                                 double v0, double K0, double K0_prime):
    """
    Thermal expansion coefficient (per unit mass):
      α = [gamma(v) c_V] / [B_T v]
    """
    cdef double BT = bulk_modulus(v, T, Theta_D0, gamma0, q, v0, K0, K0_prime)
    return gamma_val(v, gamma0, q, v0) * c_V(v, T, Theta_D0, gamma0, q, v0) / (BT * v)

cpdef double c_P(double v, double T, double Theta_D0, double gamma0, double q,
                 double v0, double K0, double K0_prime):
    """
    Specific heat at constant pressure (per unit mass):
      c_P = c_V + T v α² B_T
    """
    cdef double cV        = c_V(v, T, Theta_D0, gamma0, q, v0)
    cdef double BT        = bulk_modulus(v, T, Theta_D0, gamma0, q, v0, K0, K0_prime)
    cdef double alpha_val = thermal_expansion(v, T, Theta_D0, gamma0, q, v0, K0, K0_prime)
    return cV + T * v * alpha_val * alpha_val * BT

# --- Volume Inversion ---
def volume_from_pressure(double P_target, double T, double Theta_D0, double gamma0, double q,
                           double v0, double K0, double K0_prime, double guess,
                           double v_min_factor=0.5, double v_max_factor=2.0):
    """
    Inverts the pressure function to solve for the volume per unit mass given a target pressure and temperature.
    
    Parameters:
      P_target     : target pressure value
      T            : temperature at which the pressure is evaluated
      Theta_D0, gamma0, q, v0, K0, K0_prime: material parameters used in the pressure model.
      v_min_factor : factor to multiply v0 to get the lower bound of the search interval.
      v_max_factor : factor to multiply v0 to get the upper bound of the search interval.
      
    Returns:
      v_solution   : volume per unit mass that yields pressure P_target at temperature T.
                     nan if no real solution
    """
    # Define a Python function for the root finder.
    def f(double v):
        return pressure(v, T, Theta_D0, gamma0, q, v0, K0, K0_prime) - P_target

    cdef double v_min = v_min_factor * guess
    cdef double v_max = v_max_factor * guess

    if f(v_min) * f(v_max) > 0:
        #print('Root not bracketed: try adjusting v_min_factor and v_max_factor.')
        return np.nan
     
    v_solution = brentq(f, v_min, v_max)
    return v_solution

cpdef bint is_not_nan(double value):
    """Returns True if value is NOT NaN"""
    return not isnan(value)

# Function to copy values from one 2D memoryview to another
cpdef void copy_memoryview(double[:, :] src, double[:, :] dest):
    """Copies values from one 2D memoryview to another of the same shape."""
    cdef Py_ssize_t i, j, rows = src.shape[0], cols = src.shape[1]

    for i in range(rows):
        for j in range(cols):
            dest[i, j] = src[i, j]  # Assign value

# initialize required eos parameters 

cdef double n, Theta_D0, gamma0, q, v0, K0, K0_prime, T0
cdef double n_coeff, V0, M 

cdef int P_len = 201
cdef int T_len = 251
cdef int x_len = 21
cdef double[:] P_grid = np.linspace(0.0, 10000.0, P_len) * 1e9
cdef double[:] T_grid = np.linspace(100.0, 50100.0, T_len)
cdef double[:] x_grid = np.linspace(0.0, 1.0, x_len) # x = 1.0 -> all Fe-16Si, 16% Si by mass; x = 0.0 -> all liquid Fe, 0% Si by mass

cdef double[:,:,:] cV_grid = np.zeros((x_len, P_len, T_len))
cdef double[:,:,:] cP_grid = np.zeros((x_len, P_len, T_len))
cdef double[:,:,:] s_grid = np.zeros((x_len, P_len, T_len))
cdef double[:,:,:] eth_grid = np.zeros((x_len, P_len, T_len))
cdef double[:,:,:] alpha_grid = np.zeros((x_len, P_len, T_len))
cdef double[:,:,:] rho_grid = np.zeros((x_len, P_len, T_len))

cdef double[:,:] cV_Fel_grid = np.zeros((P_len, T_len))
cdef double[:,:] cP_Fel_grid = np.zeros((P_len, T_len))
cdef double[:,:] s_Fel_grid = np.zeros((P_len, T_len))
cdef double[:,:] eth_Fel_grid = np.zeros((P_len, T_len))
cdef double[:,:] alpha_Fel_grid = np.zeros((P_len, T_len))
cdef double[:,:] rho_Fel_grid = np.zeros((P_len, T_len))

cdef double[:,:] cV_Fea_grid = np.zeros((P_len, T_len))
cdef double[:,:] cP_Fea_grid = np.zeros((P_len, T_len))
cdef double[:,:] s_Fea_grid = np.zeros((P_len, T_len))
cdef double[:,:] eth_Fea_grid = np.zeros((P_len, T_len))
cdef double[:,:] alpha_Fea_grid = np.zeros((P_len, T_len))
cdef double[:,:] rho_Fea_grid = np.zeros((P_len, T_len))

cdef Py_ssize_t i, j, k

print('Compute eos tables of endmember minerals')

print('Check if Fe-16Si tables (Fischer) exist')

mineral = 'Fe16Si'
DATAFILE = 'binary_file/' + mineral + '.npz'

if os.path.exists(DATAFILE):
    print('Fe-16Si table exists')
    data_alloy = np.load(DATAFILE)
    copy_memoryview(data_alloy['cV_PT_grid_J_K_kg'], cV_Fea_grid) 
    copy_memoryview(data_alloy['cP_PT_grid_J_K_kg'], cP_Fea_grid) 
    copy_memoryview(data_alloy['s_PT_grid_J_K_kg'], s_Fea_grid) 
    copy_memoryview(data_alloy['eth_PT_grid_J_kg'], eth_Fea_grid)
    copy_memoryview(data_alloy['alpha_PT_grid__K'], alpha_Fea_grid)  
    copy_memoryview(data_alloy['rho_PT_grid_kg_m3'], rho_Fea_grid)
else:
    print('Fe-16Si table does not exist. Computing EoS tables')
    # parameters for Fe16Si (Fischer et al.)
    n        = 0.1257e26      # atoms/kg
    n_coeff  = n * k_B / R
    Theta_D0 = 417.0   # Debye temperature (K)
    gamma0   = 1.8     # Grüneisen parameter
    q        = 1.0     # Exponent
    V0       = 6.799e-6  # Reference volume (m³/mol)
    M        = 0.04825
    v0       = V0 / M
    K0       = 206.5e9
    K0_prime = 4.0
    T0       = 300.0

    guess = v0

    for i in range(P_len):
        for j in range(T_len):
            volume = volume_from_pressure(P_grid[i], T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime, guess,
                             v_min_factor=0.001, v_max_factor=10.0)
            if is_not_nan(volume):
                cV_Fea_grid[i][j] = c_V(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                cP_Fea_grid[i][j] = c_P(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                s_Fea_grid[i][j] = s(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                eth_Fea_grid[i][j] = e_th(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                alpha_Fea_grid[i][j] = thermal_expansion(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                rho_Fea_grid[i][j] = 1.0 / volume
            else:
                volume = v0
                cV_Fea_grid[i][j] = c_V(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                cP_Fea_grid[i][j] = c_P(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                s_Fea_grid[i][j] = s(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                eth_Fea_grid[i][j] = e_th(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                alpha_Fea_grid[i][j] = thermal_expansion(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                rho_Fea_grid[i][j] = 1.0 / volume
        if i % 10 == 0:
            print('Density at P =', int(P_grid[i]/1e9), 'GPa and T =', int(T_grid[100]), 'K is', round(rho_Fea_grid[i][100]*100.0)/100.0, 'kg/m^3')

    data_alloy = dict(
        P_grid_Pa = P_grid,
        T_grid_K = T_grid,
        cV_PT_grid_J_K_kg = cV_Fea_grid,
        cP_PT_grid_J_K_kg = cP_Fea_grid,
        s_PT_grid_J_K_kg = s_Fea_grid,
        eth_PT_grid_J_kg = eth_Fea_grid,
        alpha_PT_grid__K = alpha_Fea_grid,
        rho_PT_grid_kg_m3 = rho_Fea_grid
        )
    np.savez(DATAFILE , **data_alloy) 


###################################################################################################################

print('Check if liquid Fe table (Dorogokupets 2017) exists')

mineral = 'liquidFe'
DATAFILE = 'binary_file/' + mineral + '.npz'

if os.path.exists(DATAFILE):
    print('Liquid Fe table exists')
    data_liq = np.load(DATAFILE)

    copy_memoryview(data_liq['cV_PT_grid_J_K_kg'], cV_Fel_grid) 
    copy_memoryview(data_liq['cP_PT_grid_J_K_kg'], cP_Fel_grid) 
    copy_memoryview(data_liq['s_PT_grid_J_K_kg'], s_Fel_grid) 
    copy_memoryview(data_liq['eth_PT_grid_J_kg'], eth_Fel_grid)
    copy_memoryview(data_liq['alpha_PT_grid__K'], alpha_Fel_grid)  
    copy_memoryview(data_liq['rho_PT_grid_kg_m3'], rho_Fel_grid)
else:
    print('Liquid Fe table does not exist. Computing EoS tables')

    # parameters for liquid Fe (2017)
    K0       = 83.7e9
    K0_prime = 5.97
    v0       = 1.0 / 7037.8
    Theta_D0 = 263.0
    gamma0   = 2.033
    q        = 1.168
    T0       = 1181.0
    n        = 1.078e25

    guess = v0

    for i in range(P_len):
        for j in range(T_len):
            volume = volume_from_pressure(P_grid[i], T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime, guess,
                             v_min_factor=0.001, v_max_factor=10.0)
            if is_not_nan(volume):
                cV_Fel_grid[i][j] = c_V(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                cP_Fel_grid[i][j] = c_P(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                s_Fel_grid[i][j] = s(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                eth_Fel_grid[i][j] = e_th(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                alpha_Fel_grid[i][j] = thermal_expansion(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                rho_Fel_grid[i][j] = 1.0 / volume
            else:
                volume = v0
                cV_Fel_grid[i][j] = c_V(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                cP_Fel_grid[i][j] = c_P(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                s_Fel_grid[i][j] = s(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                eth_Fel_grid[i][j] = e_th(volume, T_grid[j], Theta_D0, gamma0, q, v0)
                alpha_Fel_grid[i][j] = thermal_expansion(volume, T_grid[j], Theta_D0, gamma0, q, v0, K0, K0_prime)
                rho_Fel_grid[i][j] = 1.0 / volume
        if i % 10 == 0:
            #print(int(P_grid[i]/1e9), rho_grid[i][50], rho_grid[i][150], rho_grid[i][250], rho_grid[i][350])
            print('Density at P =', int(P_grid[i]/1e9), 'GPa and T =', T_grid[100], 'K is', round(rho_Fel_grid[i][100]*100.0)/100.0, 'kg/m^3')

    data_liq = dict(
        P_grid_Pa = P_grid,
        T_grid_K = T_grid,
        cV_PT_grid_J_K_kg = cV_Fel_grid,
        cP_PT_grid_J_K_kg = cP_Fel_grid,
        s_PT_grid_J_K_kg = s_Fel_grid,
        eth_PT_grid_J_kg = eth_Fel_grid,
        alpha_PT_grid__K = alpha_Fel_grid,
        rho_PT_grid_kg_m3 = rho_Fel_grid
        )
    np.savez(DATAFILE , **data_liq) 

###################################################################################################################

print('Compute EoS tables for liquid Fe + Fe-16Si, assuming linear averaging mixing rules')

cdef double v_mix, v_alloy, v_liquid

mineral = 'Fe_Si_mix'
DATAFILE = 'binary_file/' + mineral + '.npz'

cdef double[:] x_Si_grid = x_grid.copy()

for i in range(x_len):
    x_Si_grid[i] = x_grid[i] * 0.16
    for j in range(P_len):
        for k in range(T_len):
            v_alloy = 1.0 / rho_Fea_grid[j][k]
            v_liquid = 1.0 / rho_Fel_grid[j][k]
            v_mix = x_grid[i] * v_alloy + (1.0 - x_grid[i]) * v_liquid
            cV_grid[i][j][k] = cV_Fea_grid[j][k] * x_grid[i] + cV_Fel_grid[j][k] * (1.0 - x_grid[i])
            cP_grid[i][j][k] = cP_Fea_grid[j][k] * x_grid[i] + cP_Fel_grid[j][k] * (1.0 - x_grid[i])
            s_grid[i][j][k] = s_Fea_grid[j][k] * x_grid[i] + s_Fel_grid[j][k] * (1.0 - x_grid[i])
            eth_grid[i][j][k] = eth_Fea_grid[j][k] * x_grid[i] + eth_Fel_grid[j][k] * (1.0 - x_grid[i])
            alpha_grid[i][j][k] = (alpha_Fea_grid[j][k] * x_grid[i] * v_alloy + alpha_Fel_grid[j][k] * (1.0 - x_grid[i]) * v_liquid) / v_mix
            rho_grid[i][j][k] = 1.0 / v_mix
            if i % 3 == 0 and j % 25 == 0 and k % 25 == 0:
                print(i, j, k, rho_grid[i][j][k], cV_grid[i][j][k])
data_mix = dict(
    x_grid = x_Si_grid,
    P_grid_Pa = P_grid,
    T_grid_K = T_grid,
    cV_PT_grid_J_K_kg = cV_grid,
    cP_PT_grid_J_K_kg = cP_grid,
    s_PT_grid_J_K_kg = s_grid,
    eth_PT_grid_J_kg = eth_grid,
    alpha_PT_grid__K = alpha_grid,
    rho_PT_grid_kg_m3 = rho_grid
    )

np.savez(DATAFILE , **data_mix) 
             


