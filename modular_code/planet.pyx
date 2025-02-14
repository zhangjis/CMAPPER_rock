# planet.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

cimport planet  # so that other modules using pxd can see the declarations

cdef class Planet:
    
    def __cinit__(self, double M_pl, double CMF, double Teq, double t_end,
                       double x_c, double P_c, double T_c,  
                       double P_surf, int m_z, int c_z):
        self.M_pl = M_pl # planet mass
        self.CMF = CMF # core mass fraction
        self.Teq = Teq # planet equilibrium temperature
        self.t_end = t_end # evolutionary time 
        self.c_z = c_z # shell number in the core
        self.m_z = m_z # shell number in the mantle
        self.zone = c_z + m_z # total shell number
        self.x_c = x_c # Si by mass fraction in core
        self.P_c = P_c # central pressure
        self.T_c = T_c # central temperature
        self.P_surf = P_surf # surface pressure, default to 1 bar.
        self.T_an_c = 6000.0 # core potential temperature (proxy for entropy). 0 as a placeholder 

        self.inner_core_radius = 0.0 # the core starts off fully liquid 
        
        # quantities without an initial value or do not depend on an initial value, use 1.0 as placeholder.
        self.rho_c = 1.0
        self.dqdy_c = 1.0

        # Allocate arrays for the entire planet (core+mantle).
        #np.empty(self.zone, dtype=np.float64)
        self.radius      = np.zeros(self.zone)
        self.pressure    = np.zeros(self.zone)
        self.rho         = np.zeros(self.zone)
        self.gravity     = np.zeros(self.zone)
        self.entropy     = np.zeros(self.zone)
        self.temperature = np.zeros(self.zone)
        self.T_cell      = np.zeros(self.zone)
        self.cP          = np.zeros(self.zone)
        self.mass        = np.zeros(self.zone)  # enclosed mass
        self.h           = np.zeros(self.zone)  # dm
        self.x_melt      = np.zeros(self.zone)  # is this also melt_frac?
        self.melt_frac   = np.zeros(self.zone)
        self.alpha       = np.zeros(self.zone)
        self.dTdP        = np.zeros(self.zone)  # adiabatic temperature gradient
        self.dPdr        = np.zeros(self.zone)  # hydrostatic equilibrium
        self.dqdy        = np.zeros(self.zone)  # dlog(rho)/dlog(P)
        self.logr        = np.zeros(self.zone)
        self.logp        = np.zeros(self.zone)
        self.logrho      = np.zeros(self.zone)
        self.Qrad        = np.zeros(self.zone)
        self.r_cell      = np.zeros(self.zone)
        self.p_cell      = np.zeros(self.zone)
        self.s_cell      = np.zeros(self.zone)
        self.Area        = np.zeros(self.zone)
        self.old_pressure= np.zeros(self.zone)
        self.old_radius  = np.zeros(self.zone)
        self.T_Fe_melt   = np.zeros(self.zone)
        self.P_Fe_melt   = np.zeros(self.zone)













  