# planet.pxd
# Declarations for the Planet class

import numpy as np
cimport numpy as np
    
cdef class Planet:
    cdef public double M_pl, CMF, Teq, t_end
    cdef public double x_c, T_an_c, P_c, T_c, rho_c, dqdy_c, P_surf
    cdef public int m_z, c_z, zone

    cdef public double[:] radius, pressure, rho, gravity, mass, h, x_melt
    cdef public double[:] entropy, temperature, T_cell, melt_frac, dsdr_array
    cdef public double[:] dTdP, dPdr, alpha, cP, kappa
    cdef public double[:] logr, logp, logrho, dqdy
    cdef public double[:] Qrad
    cdef public double[:] Area, r_cell, p_cell, s_cell
    cdef public double[:] old_pressure, old_radius

    cdef public double[:] T_Fe_melt, P_Fe_melt

    cdef public double inner_core_radius


    #scalar
    cdef double ppv_rheology

    #1D array
    cdef double[:] dsdr, viscosity, convection, l_mlt, eddy_k