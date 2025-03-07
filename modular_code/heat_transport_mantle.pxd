# planet_structure.pyx
# cython: boundscheck=False, wraparound=False

"""
Module: planet_structure
-------------------------
This module contains a Cython class for calculating the initial structure
of a rocky planet. It encapsulates the input parameters and provides methods
to compute and refine the profiles (e.g. radius, pressure, density, temperature).
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, pow, pi
from scipy.optimize import fsolve  # if needed for nonlinear solves
from scipy import interpolate
from scipy.signal import savgol_filter
cimport cython
import eos_tables

# Optionally, you could declare a DEdouble[:]
cdef class PlanetStructure:
    """
    Class: PlanetStructure
    ----------------------
    Encapsulates the initial structure calculation of a rocky planet.
    
    Attributes:
        mass                    Total planet mass (kg)
        core_fraction           Core mass fraction (dimensionless)
        equilibrium_temperature Equilibrium (surface) temperature (K)
        pressure_surface        Surface pressure (Pa); default is 1e5 Pa.
    """
    cdef double mass
    cdef double core_fraction
    cdef double equilibrium_temperature
    cdef double pressure_surface

    def __init__(self, double mass, double core_fraction,
                 double equilibrium_temperature, double pressure_surface=1e5):
        """
        Initialize the planet with the given parameters.
        """
        self.mass = mass
        self.core_fraction = core_fraction
        self.equilibrium_temperature = equilibrium_temperature
        self.pressure_surface = pressure_surface

    cpdef dict calculate_initial_profile(self):
        """
        Calculate the initial radial, pressure, density, and temperature profiles.
        (This simplified version uses linear gradients as placeholders.)

        Returns:
            A dictionary with the following keys:
                'radius'      : 1D NumPy array of radii (m)
                'pressure'    : 1D NumPy array of pressures (Pa)
                'density'     : 1D NumPy array of densities (kg/m^3)
                'temperature' : 1D NumPy array of temperatures (K)
        """
        cdef int n_zones = 100  # Number of radial zones (adjust as needed)
        cdef double[:] radius = np.linspace(1e3, 6.4e6, n_zones)
        cdef double[:] pressure = np.empty(n_zones, dtype=np.float64)
        cdef double[:] density = np.empty(n_zones, dtype=np.float64)
        cdef double[:] temperature = np.empty(n_zones, dtype=np.float64)
        cdef int i

        # For a real model, these initial guesses would be computed from scaling laws.
        # Here we set a dummy central value and then interpolate linearly.
        cdef double P_center = 1e11  # Pa; a placeholder central pressure
        cdef double rho_center = 1e4  # kg/m^3; placeholder central density
        cdef double T_center = 5000.0 # K; placeholder central temperature

        # Linearly decrease to surface conditions
        pressure[0] = P_center
        density[0] = rho_center
        temperature[0] = T_center

        for i in range(1, n_zones):
            pressure[i] = pressure[0] - (pressure[0] - self.pressure_surface) * i / (n_zones - 1)
            density[i] = density[0] - (density[0] - 3000.0) * i / (n_zones - 1)
            temperature[i] = temperature[0] - (temperature[0] - self.equilibrium_temperature) * i / (n_zones - 1)

        # Optionally refine the structure using a Henyey solver:
        self._apply_henyey_solver(&radius[0], &pressure[0], n_zones)

        if DEBUG:
            print("Initial profile calculated.")
        return {"radius": radius,
                "pressure": pressure,
                "density": density,
                "temperature": temperature}

    cdef void _apply_henyey_solver(self, double* radius, double* pressure, int n_zones):
        """
        Applies a Henyey relaxation method to refine the structure so that
        the boundary condition at the surface is satisfied.
        
        In this dummy implementation, we simply enforce the surface pressure
        on the outermost zones.
        
        Parameters:
            radius  : pointer to the radius array (modified in place)
            pressure: pointer to the pressure array (modified in place)
            n_zones : number of radial zones
        """
        cdef int i
        # For example, enforce the surface boundary condition on the last 10% of zones.
        for i in range(n_zones * 9 // 10, n_zones):
            pressure[i] = self.pressure_surface
            # Optionally adjust radius if needed (dummy here)
            if DEBUG:
                print("Zone %d: Enforcing surface pressure." % i)

    def refine_structure(self):
        """
        A public method to run the initial structure calculation and (if necessary)
        perform additional refinements.
        
        Returns:
            A dictionary containing the refined profiles.
        """
        profile = self.calculate_initial_profile()
        # Further refinement routines (e.g., iterative matching of an adiabat) can be added here.
        if DEBUG:
            print("Structure refinement complete.")
        return profile