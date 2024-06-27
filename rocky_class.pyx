#!python
#cython: boundscheck=False

cimport numpy as np
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import fsolve
from pynbody.analysis.interpolate import interpolate3d
from pynbody.analysis.interpolate import interpolate2d
from libc cimport math
cimport cython
import time

###### Overall structure of the code. 
# line 25-25    : load input file
# line 27-161   : load EoS tables
# line 163-222  : define constants
# line 224-529  : Routines for initializing thermal and structure profiles: line 
# line 531-769  : Routines for Henyey solver: line (its purpose is to find solutions to the planet structure equations that satisfy the boundary conditions at both planet center and surface)
# line 771-1623 : Routines for heat transport in the core and the mantle
# line 1625-1682: Initialize thermal and structural profiles
# line 1684-1742: while loop for updating thermal profiles using heat transport routines and structural profiles using Henyey solver

load_file=np.loadtxt('input.txt') # read in input files.

print('Read EoS tables')
T_liq=np.loadtxt('mixture_table/map_en_pv_ppv/T_liq_Py_1500GPa.txt')
rho_liq=np.loadtxt('mixture_table/map_en_pv_ppv/rho_liq_Py_1500GPa.txt')
CP_liq=np.loadtxt('mixture_table/map_en_pv_ppv/CP_liq_Py_1500GPa.txt')
alpha_liq=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_liq_Py_1500GPa.txt')
dTdP_liq=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_liq_Py_1500GPa.txt')
dqdy_liq=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_liq_Py_1500GPa.txt')
y_grid=np.loadtxt('mixture_table/map_en_pv_ppv/y.txt')
P_solidus_liquidus=np.loadtxt('mixture_table/map_en_pv_ppv/solid_P.txt')
S_liq_array=P_solidus_liquidus[:,2][:1500].copy()
S_sol_array=P_solidus_liquidus[:,1][:1500].copy()

P_grid_pv=np.loadtxt('mixture_table/map_en_pv_ppv/P_pv.txt')
P_grid_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/P_ppv.txt')
P_grid_en=np.loadtxt('mixture_table/map_en_pv_ppv/P_en.txt')
P_grid=P_solidus_liquidus[:,0][:1500].copy()

rho_Fel=np.loadtxt('EoS/paper2/rho_Fel.txt')
alpha_Fel=np.loadtxt('EoS/paper2/alpha_Fel.txt')
dqdy_Fel=np.loadtxt('EoS/paper2/dqdy_Fel.txt')
T_Fel=np.loadtxt('EoS/paper2/T_Fel.txt')
P_Fel=np.loadtxt('EoS/paper2/P_Fel.txt')
rho_Fea=np.loadtxt('EoS/paper2/rho_Fe16Si.txt')
alpha_Fea=np.loadtxt('EoS/paper2/alpha_Fe16Si.txt')
dqdy_Fea=np.loadtxt('EoS/paper2/dqdy_Fe16Si.txt')
T_Fea=np.loadtxt('EoS/paper2/T_Fe16Si.txt')
P_Fea=np.loadtxt('EoS/paper2/P_Fe16Si.txt')

Tlg,Plg=np.meshgrid(T_Fel,P_Fel,sparse=True)
Tag,Pag=np.meshgrid(T_Fea,P_Fea,sparse=True)
f_rho_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,rho_Fel)
f_rho_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,rho_Fea)
f_alpha_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,alpha_Fel)
f_alpha_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,alpha_Fea)
f_dqdy_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,dqdy_Fel)
f_dqdy_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,dqdy_Fea)

loaded_T=np.loadtxt('EoS/paper2/Fe_adiabat.txt')
load_original_T=loaded_T.reshape(loaded_T.shape[0],loaded_T.shape[1]//989,989)#141
x_core_grid=np.loadtxt('EoS/paper2/Fe_adiabat_xgrid.txt')
Tref_core_grid=np.loadtxt('EoS/paper2/Fe_adiabat_Tgrid.txt')
pre_adiabat_pressure=np.loadtxt('EoS/paper2/Fe_adiabat_Pgrid.txt')
f_adiabat=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, pre_adiabat_pressure), load_original_T)

loaded_dTdT0=np.loadtxt('EoS/paper2/Fe_dTdT0.txt')
load_original_dTdT0=loaded_dTdT0.reshape(loaded_dTdT0.shape[0],loaded_dTdT0.shape[1]//989,989)
loaded_dT0dP=np.loadtxt('EoS/paper2/Fe_dT0dP.txt')
load_original_dT0dP=loaded_dT0dP.reshape(loaded_dT0dP.shape[0],loaded_dT0dP.shape[1]//826,826)
Tgrid_core_grid=np.loadtxt('EoS/paper2/Fe_adiabat_P_Tgridgrid.txt')
f_dT0dP=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, Tgrid_core_grid), load_original_dT0dP)

T_sol_pv=np.loadtxt('mixture_table/map_en_pv_ppv/T_sol_pv_Py.txt')
rho_sol_pv=np.loadtxt('mixture_table/map_en_pv_ppv/rho_sol_pv_Py.txt')
alpha_sol_pv=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_sol_pv_Py.txt')
dTdP_sol_pv=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_sol_pv_Py.txt')
dqdy_sol_pv=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_sol_pv_Py.txt')

T_sol_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/T_sol_ppv_Py.txt')
rho_sol_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/rho_sol_ppv_Py.txt')
alpha_sol_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_sol_ppv_Py.txt')
dTdP_sol_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_sol_ppv_Py.txt')
dqdy_sol_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_sol_ppv_Py.txt')

T_sol_en=np.loadtxt('mixture_table/map_en_pv_ppv/T_sol_en_Py.txt')
rho_sol_en=np.loadtxt('mixture_table/map_en_pv_ppv/rho_sol_en_Py.txt')
alpha_sol_en=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_sol_en_Py.txt')
dTdP_sol_en=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_sol_en_Py.txt')
dqdy_sol_en=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_sol_en_Py.txt')

T_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/T_mix_pv_Py.txt')
rho_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/rho_mix_pv_Py.txt')
alpha_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_mix_pv_Py.txt')
dTdP_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_mix_pv_Py.txt')
dqdy_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_mix_pv_Py.txt')
CP_mix_pv=np.loadtxt('mixture_table/map_en_pv_ppv/CP_mix_pv_Py.txt')

T_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/T_mix_ppv_Py.txt')
rho_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/rho_mix_ppv_Py.txt')
alpha_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_mix_ppv_Py.txt')
dTdP_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_mix_ppv_Py.txt')
dqdy_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_mix_ppv_Py.txt')
CP_mix_ppv=np.loadtxt('mixture_table/map_en_pv_ppv/CP_mix_ppv_Py.txt')

T_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/T_mix_en_Py.txt')
rho_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/rho_mix_en_Py.txt')
alpha_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/alpha_mix_en_Py.txt')
dTdP_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/dTdP_mix_en_Py.txt')
dqdy_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/dqdy_mix_en_Py.txt')
CP_mix_en=np.loadtxt('mixture_table/map_en_pv_ppv/CP_mix_en_Py.txt')

S_liq_P=interpolate.interp1d(P_grid,S_liq_array)
S_sol_P=interpolate.interp1d(P_grid,S_sol_array)
T_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,T_liq)
rho_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,rho_liq)
CP_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,CP_liq)
alpha_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,alpha_liq)
dTdP_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,dTdP_liq)
dqdy_Py_liq=interpolate.RectBivariateSpline(P_grid,y_grid,dqdy_liq)

T_Py_sol_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,T_sol_pv)
rho_Py_sol_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,rho_sol_pv)
alpha_Py_sol_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,alpha_sol_pv)
dTdP_Py_sol_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,dTdP_sol_pv)
dqdy_Py_sol_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,dqdy_sol_pv)

T_Py_sol_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,T_sol_en)
rho_Py_sol_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,rho_sol_en)
alpha_Py_sol_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,alpha_sol_en)
dTdP_Py_sol_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,dTdP_sol_en)
dqdy_Py_sol_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,dqdy_sol_en)

T_Py_sol_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,T_sol_ppv)
rho_Py_sol_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,rho_sol_ppv)
alpha_Py_sol_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,alpha_sol_ppv)
dTdP_Py_sol_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dTdP_sol_ppv)
dqdy_Py_sol_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dqdy_sol_ppv)

T_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,T_mix_en)
rho_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,rho_mix_en)
CP_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,CP_mix_en)
alpha_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,alpha_mix_en)
dTdP_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,dTdP_mix_en)
dqdy_Py_mix_en=interpolate.RectBivariateSpline(P_grid_en,y_grid,dqdy_mix_en)

T_Py_mix_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,T_mix_ppv)
rho_Py_mix_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,rho_mix_ppv)
alpha_Py_mix_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,alpha_mix_ppv)
dTdP_Py_mix_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dTdP_mix_ppv)
dqdy_Py_mix_ppv=interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dqdy_mix_ppv)

T_Py_mix_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,T_mix_pv)
rho_Py_mix_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,rho_mix_pv)
alpha_Py_mix_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,alpha_mix_pv)
dTdP_Py_mix_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,dTdP_mix_pv)
dqdy_Py_mix_pv=interpolate.RectBivariateSpline(P_grid_pv,y_grid,dqdy_mix_pv)

# all variables are in SI units, unless otherwise noted. 
cdef double M_pl=load_file[0]*5.972e24 # planet mass in kg
cdef double CMF=load_file[1] # core mass fraction
cdef double t_end=load_file[2]*86400.0*365.0*1e9 # end time for the simulation in second
cdef dict qrad={} # radiogenic heating in the mantle. Relative to Earth's current day value in W/kg.
qrad['K']=load_file[3]  # potassium 40
qrad['Th']=load_file[4] # Thorium
qrad['U8']=load_file[5] # Uranium 238
qrad['U5']=load_file[6] # Uranium 235

cdef double x_c=0.105 # concentration of light elements in the core by mass.
cdef double Teq=255.0 # equilibrium temperature in K. 
cdef double Q_rad_c_0=0.0 # Current day core radiogenic heating in W/kg.
cdef double P_surf=1e5 # Surface pressure in Pa.

cdef int c_z=350 # zones in the core
cdef int m_z=350 # zones in the mantle 
cdef int zone=c_z+m_z # total number of zones in the planet
cdef double P_c=1000e9 # initial guess of the central pressure in Pa. Subsequent update in the code is the actual central pressure in Pa.
cdef double T_c=10500.0 # Central temperature in K
cdef double T_an_c_i=7000.0 # initial guess of the entropy temperature of the core in K. 

cdef double MMF=1.0-CMF # mantle mass fraction

cdef double d_Pc=1.0 # Adjustment in central pressure to find the actual central pressure using Runge-Kutta method. 
cdef double dsdr_c=-1e-6 # initial entropy gradient (an arbitrary choice)
cdef double rtol=10.0 # initialize relative tolerance for numerical techniques

cdef Py_ssize_t iteration, i

cdef double initial=1.0 # 1.0 for true and 0.0 for false
cdef double t=0.0 # time in second

cdef double G = 6.674e-11 # gravitational constant
cdef double sigma = 5.670373e-08 # Stefan boltzmann constant
cdef double k_b = 1.38e-23 # boltzmann constant
cdef double N_A = 6.022e23 # mol^-1
cdef double L_pv = 7.322e5 # latent heat of magma
cdef double L0 = 2.44e-8 # lorentz number
cdef double mu_0 = 4.0*math.pi*1e-7 # vacuum permeability 
cdef double L_sigma=0.0 # thickness of magma ocean in m 

cdef double mf_l=0.16 
cdef double S_max=5100.0#4739.0#5597.5#5384   # max of specific entropy in the EoS table for silicate. J/K/kg
cdef double S_min=100.0 # min of specific entropy in the EoS table for silicate. J/K/kg

cdef double C_P_liquidFe = 840.0 # specific heat of liquid Fe. J/K/kg
cdef double C_P_Fe = 840.0 # specific heat of solid Fe
cdef double CP_m_s = 1265.0 # specific heat of mantle silicate

cdef double k_l=10.0 # thermal conductivity of magma. W/m/K
cdef double k_ppv=10.0 # thermal conductivity of Mg-postperovskite
cdef double k_pv=10.0 # thermal conductivity of Mg-perovskite
cdef double k_en=4.0 # thermal conductivity of enstatite 

cdef double[:] k_array=np.zeros(zone)
cdef double[:] phase=np.zeros(zone)
for i in range(zone):
    k_array[i]=k_l
    phase[i]=0.0

cdef class c_initial_profile:
    cdef double M_pl
    cdef int c_z
    cdef int m_z
    cdef double CMF
    cdef double MMF
    cdef double P_c
    cdef double P_surf
    cdef double T_c
    cdef double x_c
    cdef double dsdr_c
    cdef double d_Pc
    cdef double rtol
    cdef double T_an_c
    
    def __cinit__(self, double M_pl, int c_z, int m_z, double CMF, double MMF, double P_c, 
        double P_surf, double T_c, double x_c, double dsdr_c, double d_Pc, double rtol, double T_an_c):
        self.M_pl = M_pl
        self.c_z = c_z
        self.m_z = m_z
        self.CMF = CMF
        self.MMF = MMF
        self.P_c = P_c # initial guess of the central pressure
        self.P_surf = P_surf
        self.x_c = x_c 
        self.T_c = T_c
        self.dsdr_c = dsdr_c
        self.d_Pc = d_Pc
        self.rtol = rtol
        self.T_an_c= T_an_c
        
    cpdef double dlnrdm(self, double r, double p, double density): # mass conservation equation
        return 1.0/(4.0*math.pi*r**3.0*density)
    
    cpdef double dlnPdm(self, double r, double p, double M): # hydrostatic equilibrium
        return -G*M/(4.0*math.pi*r**4.0*p)
    
    cpdef double f_r0(self, double h, double rho_c): # get the radius at the first zone above the center of the planet
        return (3.0*h/(4.0*math.pi*rho_c))**(1.0/3.0)
    
    cpdef double f_p0(self, double h, double rho_c): # get the pressure at the first zone above the center of the planet
        return self.P_c-3.0*G/(8.0*math.pi)*(4.0*math.pi*rho_c/3.0)**(4.0/3.0)*(h)**(2.0/3.0)

    cpdef double f_Tan_c(self, double x): # f_s_c and return_s_c together find the entropy temperature corresponding to the input central temperature
        cdef double x_alloy=self.x_c/mf_l
        return f_adiabat([x_alloy,x,self.P_c])[0]-self.T_c

    cpdef double return_Tan_c(self,double guess):
        return fsolve(self.f_Tan_c,x0=guess)[0]

    cpdef double mass_core(self):
        return self.CMF*self.M_pl

    cpdef double mass_mantle(self):
        return self.MMF*self.M_pl

    cpdef tuple f_M_dm(self):
        # return the mass profile of the planet
        cdef Py_ssize_t i
        cdef int zone = self.c_z+self.m_z
        cdef double c_m = self.mass_core()
        cdef double m_m = self.mass_mantle()
        cdef double h_c = c_m/self.c_z
        cdef double h_m = m_m/self.m_z
        cdef double[:] mass=np.zeros(zone)
        cdef double[:] h=np.zeros(zone)
        mass[0]=h_c
        h[0]=h_c
        for i in range(1,zone):
            if i<self.c_z:
                mass[i]=mass[i-1]+h_c
                h[i]=h_c
            elif i>=self.c_z and i<self.c_z+self.m_z:
                mass[i]=mass[i-1]+h_m
                h[i]=h_m          
        return mass, h

    cpdef double rho_mix(self, double x, double rho_l, double rho_s):
        # return the average density of mixtures using volume additive rules.
        return (x/rho_l+(1.0-x)/rho_s)**(-1.0)
    
    cpdef double alpha_mix(self,double x,double alpha_l,double alpha_s,double rho,double rho_l,double rho_s):
        # return the average thermal expansion coefficient of mixtures using volume additive rules.
        return x*rho/rho_l*alpha_l+(1.0-x)*rho/rho_s*alpha_s
    
    cpdef double dqdy_mix(self,double x, double rho_tot, double rho_l, double rho_s, double dqdy_l, double dqdy_s, double pressure):
        # return the dlog(rho)/dlog(P) of mixtures
        if pressure==0.0:
            pressure=10.0**(-6.0)
        cdef double drhodP_l, drhodP_s, value1, value2
        drhodP_l=dqdy_l*rho_l/pressure
        drhodP_s=dqdy_s*rho_s/pressure
        value1=x*rho_tot**(2.0)*rho_l**(-2.0)*drhodP_l
        value2=(1.0-x)*rho_tot**(2.0)*rho_s**(-2.0)*drhodP_s
        return (value1+value2)*pressure/rho_tot

    def y_T_liq(self,guess,P,T):
        # find the y-value (a measure of entropy) at a particular pressure and temperature level.
        def func(x):
            func=T_Py_liq(P,x)[0][0]-T#P_vinet(x,K0,K0_prime,rho0)-P
            return func
        return fsolve(func,x0=guess)[0]

    cpdef dict RK4(self):
        # integrate the strutural equations from the planet center to the planet surface using 4th order Runge Kutta method
        # returns all thermophysical/structural quantities at the top boundaries of each zone
        # returns temperature at the center of each zone as well
        cdef int zone=self.c_z+self.m_z
        cdef Py_ssize_t i
        cdef double[:] radius=np.zeros(zone)
        cdef double[:] logr=np.zeros(zone)
        cdef double[:] pressure=np.zeros(zone)
        cdef double[:] logp=np.zeros(zone)
        cdef double[:] rho=np.zeros(zone)
        cdef double[:] logrho=np.zeros(zone)
        cdef double[:] gravity=np.zeros(zone)
        cdef double[:] temperature=np.zeros(zone)
        cdef double[:] s_array=np.zeros(zone)
        cdef double[:] dTdP=np.zeros(zone)
        cdef double[:] alpha=np.zeros(zone)
        cdef double[:] dqdy=np.zeros(zone)
        cdef double[:] cP=np.zeros(zone)
        cdef double[:] T_cell=np.zeros(zone)
        cdef double[:] mass=np.zeros(zone)
        cdef double[:] h=np.zeros(zone)
        cdef double[:] melt_frac=np.zeros(zone)
        cdef double[:] phase=np.zeros(zone)

        mass, h=self.f_M_dm()
        pressure[zone-1]=1e9 # random value to initialize the condition for the while loop. 
        
        cdef double x_alloy=self.x_c/mf_l
        cdef double rho_l, rho_a, dqdy_l, dqdy_a, alpha_l ,alpha_a
        cdef double rho_c, dqdy_c, r0, p0, rho0
        cdef double k1r, k1p, k2r, k2p, k3r, k3p, k4r, k4p
        cdef double rho_v
        cdef double s_sol_val, s_liq_val, y_value, y, s_new
        
        cdef Py_ssize_t iteration=0
        while abs(pressure[zone-1]-self.P_surf)/self.P_surf>self.rtol:
            self.P_c=self.P_c-self.d_Pc*(pressure[zone-1]-self.P_surf)
            self.T_an_c=self.return_Tan_c(self.T_an_c)
            #self.T_c=f_adiabat([x_alloy,self.T_an_c,self.P_c])[0]
            rho_l=f_rho_Fel(self.P_c,self.T_an_c)[0][0]
            rho_a=f_rho_Fea(self.P_c,self.T_an_c)[0][0]
            rho_c=self.rho_mix(x_alloy,rho_a,rho_l)
            dqdy_l=f_dqdy_Fel(self.P_c,self.T_an_c)[0][0]
            dqdy_a=f_dqdy_Fea(self.P_c,self.T_an_c)[0][0]
            dqdy_c=self.dqdy_mix(x_alloy,rho_c,rho_a,rho_l,dqdy_a,dqdy_l,self.P_c)

            r0=self.f_r0(mass[0],rho_c)
            p0=self.f_p0(mass[0],rho_c)
            rho_l=f_rho_Fel(p0,self.T_an_c)[0][0]
            rho_a=f_rho_Fea(p0,self.T_an_c)[0][0]
            rho0=self.rho_mix(x_alloy,rho_a,rho_l)
            temperature[0]=self.T_c
            radius[0]=r0
            logr[0]=np.log(r0)
            pressure[0]=p0
            logp[0]=np.log(p0)
            rho_l=f_rho_Fel(p0,self.T_an_c)[0][0]
            rho_a=f_rho_Fea(p0,self.T_an_c)[0][0]
            rho0=self.rho_mix(x_alloy,rho_a,rho_l)
            rho[0]=rho0
            logrho[0]=np.log(rho0)
            gravity[0]=G*mass[0]/r0**2.0
            dqdy_l=f_dqdy_Fel(p0,self.T_an_c)[0][0]
            dqdy_a=f_dqdy_Fea(p0,self.T_an_c)[0][0]
            dqdy[0]=self.dqdy_mix(x_alloy,rho0,rho_a,rho_l,dqdy_a,dqdy_l,p0)
            alpha_l=f_alpha_Fel(p0,self.T_an_c)[0]
            alpha_a=f_alpha_Fea(p0,self.T_an_c)[0]
            alpha[0]=self.alpha_mix(x_alloy,alpha_a,alpha_l,rho0,rho_a,rho_l)
            cP[0]=C_P_liquidFe

            for i in range(1, int(zone)):
                if i<=self.c_z:
                    k1r=self.dlnrdm(radius[i-1]             , pressure[i-1]             , rho[i-1])
                    k1p=self.dlnPdm(radius[i-1]             , pressure[i-1]             , mass[i-1])    
                    rho_l=f_rho_Fel(pressure[i-1]+h[i]/2.0*k1p,self.T_an_c)[0][0]
                    rho_a=f_rho_Fea(pressure[i-1]+h[i]/2.0*k1p,self.T_an_c)[0][0]
                    rho_v=self.rho_mix(x_alloy,rho_a,rho_l)
                    k2r=self.dlnrdm(radius[i-1]+h[i]/2.0*k1r, pressure[i-1]+h[i]/2.0*k1p, rho_v)
                    k2p=self.dlnPdm(radius[i-1]+h[i]/2.0*k1r, pressure[i-1]+h[i]/2.0*k1p, mass[i-1]+h[i]/2.0)
                    rho_l=f_rho_Fel(pressure[i-1]+h[i]/2.0*k2p,self.T_an_c)[0][0]
                    rho_a=f_rho_Fea(pressure[i-1]+h[i]/2.0*k2p,self.T_an_c)[0][0]
                    rho_v=self.rho_mix(x_alloy,rho_a,rho_l)
                    k3r=self.dlnrdm(radius[i-1]+h[i]/2.0*k2r, pressure[i-1]+h[i]/2.0*k2p, rho_v)
                    k3p=self.dlnPdm(radius[i-1]+h[i]/2.0*k2r, pressure[i-1]+h[i]/2.0*k2p, mass[i-1]+h[i]/2.0)
                    rho_l=f_rho_Fel(pressure[i-1]+h[i]*k3p,self.T_an_c)[0][0]
                    rho_a=f_rho_Fea(pressure[i-1]+h[i]*k3p,self.T_an_c)[0][0]
                    rho_v=self.rho_mix(x_alloy,rho_a,rho_l)
                    k4r=self.dlnrdm(radius[i-1]+h[i]*k3r    , pressure[i-1]+h[i]*k3p    , rho_v)
                    k4p=self.dlnPdm(radius[i-1]+h[i]*k3r    , pressure[i-1]+h[i]*k3p    , mass[i-1]+h[i])
                    logr[i]=logr[i-1]+h[i]/6.0*(k1r+2.0*k2r+2.0*k3r+k4r)
                    logp[i]=logp[i-1]+h[i]/6.0*(k1p+2.0*k2p+2.0*k3p+k4p)
                    radius[i]=np.exp(logr[i])
                    pressure[i]=np.exp(logp[i])
                    gravity[i]=G*mass[i]/radius[i]**2.0
                    rho_l=f_rho_Fel(pressure[i],self.T_an_c)[0][0]
                    rho_a=f_rho_Fea(pressure[i],self.T_an_c)[0][0]
                    rho[i]=self.rho_mix(x_alloy,rho_a,rho_l)
                    logrho[i]=np.log(rho[i])

                    alpha_l=f_alpha_Fel(pressure[i],self.T_an_c)[0][0]
                    alpha_a=f_alpha_Fea(pressure[i],self.T_an_c)[0][0]
                    alpha[i]=self.alpha_mix(x_alloy,alpha_a,alpha_l,rho[i],rho_a,rho_l)
                    cP[i]=C_P_liquidFe
                    temperature[i]=f_adiabat([x_alloy,self.T_an_c,pressure[i]])[0]
                    dqdy_l=f_dqdy_Fel(pressure[i],self.T_an_c)[0][0]
                    dqdy_a=f_dqdy_Fea(pressure[i],self.T_an_c)[0][0]
                    dqdy[i]=self.dqdy_mix(x_alloy,rho[i],rho_a,rho_l,dqdy_a,dqdy_l,pressure[i])
                else: 
                    if i==self.c_z+1:
                        s_sol_val=S_sol_P(pressure[i-1]).tolist()
                        s_liq_val=S_liq_P(pressure[i-1]).tolist()
                        y_value=self.y_T_liq(0.5,pressure[i-1],temperature[i-1])
                        s_array[i-1]=y_value*(S_max-s_liq_val)+s_liq_val
                    s_sol_val=S_sol_P(pressure[i-1]).tolist()
                    s_liq_val=S_liq_P(pressure[i-1]).tolist()
                    s_new=s_array[i-1]
                    if s_new>=s_liq_val:
                        y=(s_new-s_liq_val)/(S_max-s_liq_val)
                        rho_v=rho_Py_liq(pressure[i-1],y)[0][0]
                    k1r=self.dlnrdm(radius[i-1]             , pressure[i-1]             , rho_v)
                    k1p=self.dlnPdm(radius[i-1]             , pressure[i-1]             , mass[i-1])

                    s_sol_val=S_sol_P(pressure[i-1]+h[i]/2.0*k1p).tolist()
                    s_liq_val=S_liq_P(pressure[i-1]+h[i]/2.0*k1p).tolist()
                    s_new=s_array[i-1]+dsdr_c*h[i]/2.0*k1r
                    if s_new>=s_liq_val:
                        y=(s_new-s_liq_val)/(S_max-s_liq_val)
                        rho_v=rho_Py_liq(pressure[i-1]+h[i]/2.0*k1p,y)[0][0]
                    k2r=self.dlnrdm(radius[i-1]+h[i]/2.0*k1r, pressure[i-1]+h[i]/2.0*k1p, rho_v)
                    k2p=self.dlnPdm(radius[i-1]+h[i]/2.0*k1r, pressure[i-1]+h[i]/2.0*k1p, mass[i-1]+h[i]/2.0)

                    s_sol_val=S_sol_P(pressure[i-1]+h[i]/2.0*k2p).tolist()
                    s_liq_val=S_liq_P(pressure[i-1]+h[i]/2.0*k2p).tolist()
                    s_new=s_array[i-1]+dsdr_c*h[i]/2.0*k2r
                    if s_new>=s_liq_val:
                        y=(s_new-s_liq_val)/(S_max-s_liq_val)
                        rho_v=rho_Py_liq(pressure[i-1]+h[i]/2.0*k2p,y)[0][0]
                    k3r=self.dlnrdm(radius[i-1]+h[i]/2.0*k2r, pressure[i-1]+h[i]/2.0*k2p, rho_v)
                    k3p=self.dlnPdm(radius[i-1]+h[i]/2.0*k2r, pressure[i-1]+h[i]/2.0*k2p, mass[i-1]+h[i]/2.0)

                    s_sol_val=S_sol_P(pressure[i-1]+h[i]*k3p).tolist()
                    s_liq_val=S_liq_P(pressure[i-1]+h[i]*k3p).tolist()
                    s_new=s_array[i-1]+dsdr_c*h[i]*k3r
                    if s_new>=s_liq_val:
                        y=(s_new-s_liq_val)/(S_max-s_liq_val)
                        rho_v=rho_Py_liq(pressure[i-1]+h[i]*k3p,y)[0][0]
                    k4r=self.dlnrdm(radius[i-1]+h[i]*k3r    , pressure[i-1]+h[i]*k3p    , rho_v)
                    k4p=self.dlnPdm(radius[i-1]+h[i]*k3r    , pressure[i-1]+h[i]*k3p    , mass[i-1]+h[i])

                    logr[i]=logr[i-1]+h[i]/6.0*(k1r+2.0*k2r+2.0*k3r+k4r)
                    logp[i]=logp[i-1]+h[i]/6.0*(k1p+2.0*k2p+2.0*k3p+k4p)
                    radius[i]=np.exp(logr[i])
                    pressure[i]=np.exp(logp[i])
                    gravity[i]=G*mass[i]/radius[i]**2.0
                    s_sol_val=S_sol_P(pressure[i]).tolist()
                    s_liq_val=S_liq_P(pressure[i]).tolist()
                    s_array[i]=s_array[i-1]+dsdr_c*(radius[i]-radius[i-1])
                    if s_array[i]>=s_liq_val:
                        phase[i]=0.0
                        y=(s_array[i]-s_liq_val)/(S_max-s_liq_val)
                        rho[i]=rho_Py_liq(pressure[i],y)[0][0]
                        alpha[i]=alpha_Py_liq(pressure[i],y)[0][0]
                        temperature[i]=T_Py_liq(pressure[i],y)[0][0] 
                        dqdy[i]=dqdy_Py_liq(pressure[i],y)[0][0] 
                        cP[i]=CP_Py_liq(pressure[i],y)[0][0]
                        dTdP[i]=dTdP_Py_liq(pressure[i],y)[0][0]
                        melt_frac[i]=1.0
                    logrho[i]=np.log(rho[i])

            if iteration%100==0:
                print(iteration,pressure[zone-1]/1e9)
            iteration=iteration+1
        
        for i in range(zone):
            if i==0:
                T_cell[i]=temperature[i]
            else:
                T_cell[i]=(temperature[i-1]+temperature[i])/2.0

        cdef dict results={}
        results['mass']=mass.copy()
        results['radius']=radius.copy()
        results['logr']=logr.copy()
        results['pressure']=pressure.copy()
        results['logp']=logp.copy()
        results['rho']=rho.copy()
        results['logrho']=logrho.copy()
        results['gravity']=gravity.copy()
        results['s_array']=s_array.copy()
        results['temperature']=temperature.copy()
        results['T_cell']=T_cell.copy()
        results['alpha']=alpha.copy()
        results['cP']=cP.copy()
        results['dqdy']=dqdy.copy()
        results['dTdP']=dTdP.copy()
        results['P_c']=self.P_c
        results['rho_c']=rho_c
        results['dqdy_c']=dqdy_c
        results['dm']=h
        results['T_an_c']=self.T_an_c
        
        return results

cdef class c_henyey:
    # Henyey code to find solution to the structural equations given a reasonable guess (either from RK4 or planet structure at the previous timestep)
    cdef double[:] mass
    cdef double[:] radius
    cdef double[:] logr
    cdef double[:] pressure
    cdef double[:] logp
    cdef double[:] rho
    cdef double[:] logrho
    cdef double[:] gravity
    cdef double[:] s_array
    cdef double[:] dqdy
    cdef double P_c, rho_c, dqdy_c, P_s, rtol
    cdef double T_an_c, x_c
    cdef int c_z, m_z

    def __cinit__(self, double[:] mass, double[:] radius, double[:] logr, double[:] pressure, double[:] logp, double[:] rho, double[:] logrho, double[:] gravity, double[:] s_array, double[:] dqdy, 
        double P_c, double rho_c, double dqdy_c, double P_s, double rtol, double T_an_c, int c_z, int m_z, double x_c):
        self.mass = mass
        self.radius = radius
        self.logr = logr
        self.pressure = pressure
        self.logp = logp
        self.rho = rho
        self.logrho = logrho
        self.gravity = gravity
        self.s_array = s_array
        self.dqdy = dqdy
        self.P_c = P_c
        self.rho_c = rho_c
        self.dqdy_c = dqdy_c
        self.P_s = P_s
        self.rtol = rtol
        self.T_an_c = T_an_c
        self.x_c = x_c
        self.c_z = c_z
        self.m_z = m_z

    cpdef double rho_mix(self, double x, double rho_l, double rho_s):
        return (x/rho_l+(1.0-x)/rho_s)**(-1.0)
    
    cpdef double alpha_mix(self,double x,double alpha_l,double alpha_s,double rho,double rho_l,double rho_s):
        return x*rho/rho_l*alpha_l+(1.0-x)*rho/rho_s*alpha_s
    
    cpdef double dqdy_mix(self,double x, double rho_tot, double rho_l, double rho_s, double dqdy_l, double dqdy_s, double pressure):
        if pressure==0.0:
            pressure=10.0**(-6.0)
        cdef double drhodP_l, drhodP_s, value1, value2
        drhodP_l=dqdy_l*rho_l/pressure
        drhodP_s=dqdy_s*rho_s/pressure
        value1=x*rho_tot**(2.0)*rho_l**(-2.0)*drhodP_l
        value2=(1.0-x)*rho_tot**(2.0)*rho_s**(-2.0)*drhodP_s
        return (value1+value2)*pressure/rho_tot
        
    cpdef dict henyey_m(self, double dsdr_c, double initial):
        cdef int zone=len(self.radius)
        cdef Py_ssize_t i
        cdef Py_ssize_t iteration=0
        cdef double x_alloy=self.x_c/mf_l
        cdef double[:] d_p=np.ones(zone)
        cdef double[:] d_r=np.ones(zone)
        cdef double[:] old_pressure=self.pressure.copy()
        cdef double[:] old_radius=self.radius.copy()
        cdef double[:] A_r=np.zeros(zone)
        cdef double[:] A_p=np.zeros(zone)
        cdef double[:] a=np.zeros(zone)
        cdef double[:] b=np.zeros(zone) 
        cdef double[:] c=np.zeros(zone)
        cdef double[:] d=np.zeros(zone)
        cdef double[:] A=np.zeros(zone)
        cdef double[:] B=np.zeros(zone) 
        cdef double[:] C=np.zeros(zone)
        cdef double[:] D=np.zeros(zone)
        cdef double[:] alp=np.zeros(zone)
        cdef double[:] gam=np.zeros(zone)
        cdef double[:] delta_y=np.zeros(zone)
        cdef double[:] delta_x=np.zeros(zone)
        cdef double[:] r_cell=np.zeros(zone)
        cdef double[:] p_cell=np.zeros(zone)
        cdef double[:] Area=np.zeros(zone)

        cdef double delta_P_center, logP_c, logrho_c, alp_0, gam_0
        cdef double v_bd, v_ABCD, v_P, v_r
        cdef double s_sol_val, s_liq_val, y, rho_l, rho_a, dqdy_l, dqdy_a
        
        while np.max(np.abs(d_p))>rtol or np.max(np.abs(d_r))>rtol:
            old_pressure=self.pressure.copy()
            old_radius=self.radius.copy()
            
            self.pressure[-1]=self.P_s
            self.logp[-1]=math.log(self.pressure[-1])

            s_sol_val=S_sol_P(self.pressure[-1]).tolist()
            s_liq_val=S_liq_P(self.pressure[-1]).tolist()
            if self.s_array[-1]<s_sol_val:
                y=(self.s_array[-1]-S_min)/(s_sol_val-S_min)
                self.rho[-1]=rho_Py_sol_en(self.pressure[-1],y)[0][0] 
                self.dqdy[-1]=dqdy_Py_sol_en(self.pressure[-1],y)[0][0] 
            elif self.s_array[-1]>s_liq_val:
                y=(self.s_array[-1]-s_liq_val)/(S_max-s_liq_val)
                self.rho[-1]=rho_Py_liq(self.pressure[-1],y)[0][0]
                self.dqdy[-1]=dqdy_Py_liq(self.pressure[-1],y)[0][0]
            else:
                y=(self.s_array[-1]-s_sol_val)/(s_liq_val-s_sol_val)
                self.rho[-1]=rho_Py_mix_en(self.pressure[-1],y)[0][0] 
                self.dqdy[-1]=dqdy_Py_mix_en(self.pressure[-1],y)[0][0] 
            self.logrho[-1]=math.log(self.rho[-1])
            
            logP_c=math.log(self.P_c)
            logrho_c=math.log(self.rho_c)
            
            A_r=np.zeros(zone)
            A_p=np.zeros(zone)
            for i in range(zone):
                if i==0:
                    A_r[i]=self.logr[i]-1.0/3.0*(math.log(3.0*self.mass[i]/(4.0*np.pi))-logrho_c)
                    A_p[i]=self.logp[i]-logP_c+G/2.0*math.pow((4.0*math.pi*math.pow(self.mass[i],2.0/3.0)),1.0/3.0)*math.exp(4.0*logrho_c/3.0-logP_c)
                else:
                    A_r[i]=self.logr[i]-self.logr[i-1]-1.0/(4.0*np.pi)*(self.mass[i]-self.mass[i-1])*math.exp(-0.5*(self.logrho[i]+self.logrho[i-1])-1.5*(self.logr[i]+self.logr[i-1]))
                    A_p[i]=self.logp[i]-self.logp[i-1]+G/(8.0*np.pi)*(math.pow(self.mass[i],2.0)-math.pow(self.mass[i-1],2.0))*math.exp(-0.5*(self.logp[i]+self.logp[i-1])-2.0*(self.logr[i]+self.logr[i-1]))
                    
            a=np.zeros(zone); b=np.zeros(zone) 
            c=np.zeros(zone); d=np.zeros(zone) # lower case is for pressure
            A=np.zeros(zone); B=np.zeros(zone) 
            C=np.zeros(zone); D=np.zeros(zone) # Upper case is for radius
            
            for i in range(zone):
                if i==0:
                    v_bd=(G/(8.0*math.pi))*(math.pow(self.mass[i],2.0))*math.exp(-2.0*self.logr[i])*math.exp(-0.5*(self.logp[i]+logP_c))*(-0.5)
                    a[i]=(G/(8.0*math.pi))*(math.pow(self.mass[i],2.0))*math.exp(-0.5*(self.logp[i]+logP_c))*math.exp(-2.0*self.logr[i])*(-2.0)
                    c[i]=a[i]     
                    b[i]=-1.0+v_bd
                    d[i]=1.0+v_bd
                    v_ABCD=(1.0/(4.0*math.pi))*self.mass[i]*math.exp(-0.5*(self.logrho[i]+logrho_c))*math.exp(-1.5*self.logr[i])
                    A[i]=-1.0-v_ABCD*(-1.5)
                    B[i]=-v_ABCD*(-0.5)*self.dqdy_c
                    C[i]=1.0-v_ABCD*(-1.5)
                    D[i]=-v_ABCD*(-0.5)*self.dqdy[i]
                else:
                    v_P=A_p[i]-self.logp[i]+self.logp[i-1]
                    v_r=A_r[i]-self.logr[i]+self.logr[i-1]
                    a[i]=-2.0*v_P
                    b[i]=-1.0-v_P/2.0
                    c[i]=-2.0*v_P
                    d[i]=1.0-v_P/2.0
                    A[i]=-1.0-1.5*v_r
                    B[i]=-0.5*v_r*self.dqdy[i-1]
                    C[i]=1.0-1.5*v_r
                    D[i]=-0.5*v_r*self.dqdy[i]
            
            alp=np.zeros(zone)
            gam=np.zeros(zone)
            alp_0=0.0
            gam_0=0.0
            for i in range(zone):
                if i==0:
                    alp[i]=(d[i]*(B[i]-A[i]*alp_0)-D[i]*(b[i]-a[i]*alp_0))/(c[i]*(B[i]-A[i]*alp_0)-C[i]*(b[i]-a[i]*alp_0))
                    gam[i]=((B[i]-A[i]*alp_0)*(A_p[i]-a[i]*gam_0)-(b[i]-a[i]*alp_0)*(A_r[i]-A[i]*gam_0))/(c[i]*(B[i]-A[i]*alp_0)-C[i]*(b[i]-a[i]*alp_0))
                else:
                    alp[i]=(d[i]*(B[i]-A[i]*alp[i-1])-D[i]*(b[i]-a[i]*alp[i-1]))/(c[i]*(B[i]-A[i]*alp[i-1])-C[i]*(b[i]-a[i]*alp[i-1]))
                    gam[i]=((B[i]-A[i]*alp[i-1])*(A_p[i]-a[i]*gam[i-1])-(b[i]-a[i]*alp[i-1])*(A_r[i]-A[i]*gam[i-1]))/(c[i]*(B[i]-A[i]*alp[i-1])-C[i]*(b[i]-a[i]*alp[i-1]))
            
            delta_y=np.zeros(zone)
            delta_x=np.zeros(zone)
            delta_y[-1]=0.0
            delta_x[-1]=-gam[-1]
            for i in range(zone-1, -1, -1):
                if i>0-1 and i<zone-1:
                    delta_y[i]=-(A_r[i+1]-A[i+1]*gam[i]+C[i+1]*delta_x[i+1]+D[i+1]*delta_y[i+1])/(B[i+1]-A[i+1]*alp[i])
                    delta_x[i]=-gam[i]-alp[i]*delta_y[i]
            delta_P_center=(A_r[0]+C[0]*delta_x[0]+D[0]*delta_y[0])/B[0]
            self.P_c=math.exp(logP_c+delta_P_center)
            for i in range(zone):
                self.radius[i]=math.exp(self.logr[i]+delta_x[i])
                self.pressure[i]=math.exp(self.logp[i]+delta_y[i])
                self.logr[i]=math.log(self.radius[i])
                self.logp[i]=math.log(self.pressure[i])
                self.gravity[i]=(G*self.mass[i]/self.radius[i]**2.0)
                if i<=self.c_z:
                    rho_l=f_rho_Fel(self.pressure[i],self.T_an_c)[0][0]
                    rho_a=f_rho_Fea(self.pressure[i],self.T_an_c)[0][0]
                    self.rho[i]=self.rho_mix(x_alloy,rho_a,rho_l)
                    dqdy_l=f_dqdy_Fel(self.pressure[i],self.T_an_c)[0][0]
                    dqdy_a=f_dqdy_Fea(self.pressure[i],self.T_an_c)[0][0]
                    self.dqdy[i]=self.dqdy_mix(x_alloy,self.rho[i],rho_a,rho_l,dqdy_a,dqdy_l,self.pressure[i])
                else:
                    s_sol_val=S_sol_P(self.pressure[i]).tolist()
                    s_liq_val=S_liq_P(self.pressure[i]).tolist()
                    y=(self.s_array[i]-s_liq_val)/(S_max-s_liq_val)
                    self.rho[i]=rho_Py_liq(self.pressure[i],y)[0][0]
                    self.dqdy[i]=dqdy_Py_liq(self.pressure[i],y)[0][0]
                self.logrho[i]=math.log(self.rho[i])
                d_p[i]=(self.pressure[i]-old_pressure[i])/old_pressure[i]
                d_r[i]=(self.radius[i]-old_radius[i])/old_radius[i]
            rho_l=f_rho_Fel(self.P_c,self.T_an_c)[0][0]
            rho_a=f_rho_Fea(self.P_c,self.T_an_c)[0][0]
            self.rho_c=self.rho_mix(x_alloy,rho_a,rho_l)
            dqdy_l=f_dqdy_Fel(self.P_c,self.T_an_c)[0][0]
            dqdy_a=f_dqdy_Fea(self.P_c,self.T_an_c)[0][0]
            self.dqdy_c=self.dqdy_mix(x_alloy,self.rho_c,rho_a,rho_l,dqdy_a,dqdy_l,self.P_c)

            iteration=iteration+1
        
        for i in range(zone):
            if i==0:
                r_cell[i]=self.radius[i]/2.0
                p_cell[i]=(self.pressure[i]+self.P_c)/2.0
            else:
                r_cell[i]=(self.radius[i]+self.radius[i-1])/2.0
                p_cell[i]=(self.pressure[i]+self.pressure[i-1])/2.0 
            Area[i]=4.0*np.pi*self.radius[i]**2.0
   
        cdef dict results={}
        results['radius']=self.radius.copy()
        results['logr']=self.logr.copy()
        results['pressure']=self.pressure.copy()
        results['logp']=self.logp.copy()
        results['rho']=self.rho.copy()
        results['logrho']=self.logrho.copy()
        results['gravity']=self.gravity.copy()
        results['dqdy']=self.dqdy.copy()
        results['r_cell']=r_cell.copy()
        results['p_cell']=p_cell.copy()
        results['Area']=Area.copy()
        results['P_c']=self.P_c
        results['rho_c']=self.rho_c
        results['dqdy_c']=self.dqdy_c
        
        cdef double[:] s_cell=np.zeros(zone)
        if initial==1.0:
            s_cell=np.zeros(zone)
            for i in range(zone):
                if i==self.c_z:
                    s_cell[i]=self.s_array[i]-dsdr_c*self.radius[i]/2.0
                else:
                    s_cell[i]=(self.s_array[i-1]+self.s_array[i])/2.0    
            results['s_cell']=s_cell.copy()

        return results

cdef class heat_transport:
    # heat transport in the core and mantle
    cdef double[:] mass 
    cdef double[:] h
    cdef double[:] x_m
    cdef double[:] radius
    cdef double[:] r_cell 
    cdef double[:] pressure 
    cdef double[:] p_cell
    cdef double[:] rho
    cdef double[:] g
    cdef double[:] dqdy 
    cdef double[:] Area        
    cdef double[:] s_array
    cdef double[:] s_cell 
    cdef double[:] T
    cdef double[:] T_cell
    cdef double[:] phase 
    cdef double[:] alpha 
    cdef double[:] cP
    cdef double[:] k_array 
    cdef double[:] dTdP
    cdef double M_pl
    cdef double Teq
    cdef dict qrad
    cdef double t
    cdef double dt
    cdef double epsilon
    cdef double tauK
    cdef double tauTh
    cdef double tauU8
    cdef double tauU5
    cdef double qKE
    cdef double qThE
    cdef double qU8E
    cdef double qU5E
    cdef double[:] phase_c
    cdef double P_c, rho_c, dqdy_c
    cdef double Q_rad_c_0
    cdef int c_z, m_z
    cdef double CMF
    cdef double Mic, Pic, Ric, alpha_ic, rho_ic, Tic, x_core, min_pre_adia_T, T_cmb
    cdef int solid_index
    cdef double l_alpha
    cdef double l_beta

    def __cinit__(self, double[:] mass, double[:] h, double[:] radius, double[:] r_cell, double[:] pressure, double[:] p_cell, double[:] rho, double[:] g, double[:] dqdy, double[:] Area, 
                 double[:] s_array, double[:] s_cell, double[:] T, double[:] T_cell, double[:] x_m, double[:] phase, double[:] phase_c, double[:] alpha, double[:] cP, double[:] k_array, double[:] dTdP, 
                 double Mic, double Pic, double Ric, double alpha_ic, double rho_ic, double Tic, int solid_index, double x_core, double min_pre_adia_T, double T_cmb,
                 double P_c, double rho_c, double dqdy_c ,double Q_rad_c_0, double Teq, dict qrad, double t, double dt, int c_z, int m_z, double M_pl, double CMF):
        self.mass = mass
        self.h = h
        self.radius = radius
        self.r_cell = r_cell
        self.pressure = pressure
        self.P_c = P_c
        self.p_cell = p_cell
        self.rho = rho
        self.rho_c = rho_c
        self.g = g
        self.dqdy = dqdy
        self.dqdy_c = dqdy_c
        self.Area = Area
        self.s_array = s_array
        self.s_cell = s_cell
        self.T = T
        self.T_cell = T_cell
        self.x_m = x_m
        self.phase = phase # list of phases, 'liquid MgSiO3', 'ppv', 'pv', 'en', 'melt MgSiO3' 
        self.phase_c = phase_c
        self.alpha = alpha
        self.cP = cP
        self.k_array = k_array
        self.dTdP = dTdP
        
        self.Mic = Mic
        self.Pic = Pic
        self.Ric = Ric
        self.Tic = Tic
        self.alpha_ic = alpha_ic
        self.rho_ic = rho_ic
        self.solid_index = solid_index
        self.x_core = self.x_core
        self.min_pre_adia_T = min_pre_adia_T # same as T_an_c
        self.T_cmb = T_cmb
        
        self.M_pl = M_pl
        self.Teq = Teq
        self.CMF = CMF
        self.Q_rad_c_0 = Q_rad_c_0
        
        self.qrad = qrad # a dictionary of qK, qTh, qU8, qU5. Values are ratios to Earth concentration at current day
        self.t = t
        self.dt = dt
        
        self.c_z = c_z
        self.m_z = m_z
        
        # mixing length parameter for the mantle
        self.l_alpha = 0.82
        self.l_beta = 1.0
        
        # tau is in Gyr
        self.tauK = 1.25 
        self.tauTh = 14.0 
        self.tauU8 = 4.47
        self.tauU5 = 0.704
        # q is in W/kg for current Earth concentration
        self.qKE = 8.69e-13
        self.qThE = 2.24e-12
        self.qU8E = 1.97e-12
        self.qU5E = 8.48e-14
        
    cpdef double[:] penta_solver(self,double[:] a, double[:] b, double[:] c, double[:] d, double[:] e, double[:] y, int zone): # a,e->n-2; b,d->n-1; c->n; y->n or f; zone->n
        cdef Py_ssize_t i
        cdef double[:] alpha=np.zeros(zone) #1,2,3,...,n-1; n-1
        cdef double[:] beta=np.zeros(zone) #1,2,3,...,n-2; n-2
        cdef double[:] z=np.zeros(zone) #1,2,3,...,n; n
        cdef double[:] gamma=np.zeros(zone) #2,3,4,...,n; n-1
        cdef double[:] mu=np.zeros(zone) #1,2,3,...,n; n

        #i=0
        mu[0]=c[0]
        gamma[0]=0.0 # book keeping
        alpha[0]=d[0]/mu[0]
        beta[0]=e[0]/mu[0]
        z[0]=y[0]/mu[0]

        #i=1
        gamma[1]=b[1]
        mu[1]=c[1]-alpha[0]*gamma[1]
        alpha[1]=(d[1]-beta[0]*gamma[1])/mu[1]
        beta[1]=e[1]/mu[1]
        z[1]=(y[1]-z[0]*gamma[1])/mu[1]

        #i 2<->n-3
        for i in range(2, zone-2):
            gamma[i]=b[i]-alpha[i-2]*a[i]
            mu[i]=c[i]-beta[i-2]*a[i]-alpha[i-1]*gamma[i]
            alpha[i]=(d[i]-beta[i-1]*gamma[i])/mu[i]
            beta[i]=e[i]/mu[i]
            z[i]=(y[i]-z[i-2]*a[i]-z[i-1]*gamma[i])/mu[i]

        #i=n-2
        i=zone-2
        gamma[i]=b[i]-alpha[i-2]*a[i]
        mu[i]=c[i]-beta[i-2]*a[i]-alpha[i-1]*gamma[i]
        alpha[i]=(d[i]-beta[i-1]*gamma[i])/mu[i]
        beta[i]=0.0
        z[i]=(y[i]-z[i-2]*a[i]-z[i-1]*gamma[i])/mu[i]

        #i=n-1
        i=zone-1
        gamma[i]=b[i]-alpha[i-2]*a[i]
        mu[i]=c[i]-beta[i-2]*a[i]-alpha[i-1]*gamma[i]
        alpha[i]=0.0
        beta[i]=0.0
        z[i]=(y[i]-z[i-2]*a[i]-z[i-1]*gamma[i])/mu[i]

        # solving for x
        cdef double[:] solution=np.zeros(zone)
        solution[zone-1]=z[zone-1]
        solution[zone-2]=z[zone-2]-alpha[zone-2]*solution[zone-1]
        for i in range(zone-3,-1,-1):
            solution[i]=z[i]-alpha[i]*solution[i+1]-beta[i]*solution[i+2]

        return solution
    
    cpdef int find_nearest(self, double[:] array, double value):
        cdef int idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
    
    cpdef double f_qrad(self):
        # return radiogenic heating in W/kg
        cdef double oneGyr = 1e9*365.0*86400.0
        cdef double v_log2 = math.log(2.0)
        cdef double a = self.qrad['K']*self.qKE*math.exp(v_log2*(4.5-self.t/oneGyr)/self.tauK)
        cdef double b = self.qrad['Th']*self.qThE*math.exp(v_log2*(4.5-self.t/oneGyr)/self.tauTh)
        cdef double c = self.qrad['U8']*self.qU8E*math.exp(v_log2*(4.5-self.t/oneGyr)/self.tauU8)
        cdef double d = self.qrad['U5']*self.qU5E*math.exp(v_log2*(4.5-self.t/oneGyr)/self.tauU5)
        return a+b+c+d 
    
    cpdef double f_Qrad(self, double m):
        # return radoiogenic heating in W
        return self.f_qrad()*m

    cpdef double[:] f_viscosity(self):
        cdef double A=1.67
        cdef double B=7.4e-17
        cdef double n=3.5
        cdef double E=5.0e+5
        cdef double V=1.0e-5
        cdef double R=8.314
        cdef double epsilon=1.0e-15
        cdef double y=0.0
        cdef double z=0.0
        cdef double[:] value=np.ones(len(self.radius))
        cdef double[:] eta_s=np.zeros(len(self.radius))
        cdef double[:] eta_m=np.zeros(len(self.radius))
        cdef double eta_m_0=100.0
        cdef Py_ssize_t i
        cdef double eta0, p_decay
        for i in range(self.c_z+1, self.c_z+self.m_z):
            if self.pressure[i]<125e9:
                if self.pressure[i]<23.0*10.0**9.0:
                    B=3.5e-15; n=3.0; E=4.3e+5
                eta_s[i]=0.5*(1.0/B**(1.0/n))*math.exp((E+self.pressure[i]*V)/(n*R*self.T[i]))*epsilon**((1.0-n)/n)/self.rho[i]
            else:
                eta0=1.9e21; E=1.62e5; p_decay=1610e9
                V=1.4e-6*math.exp(-self.pressure[i]/p_decay)
                eta_s[i]=eta0*math.exp((E+self.pressure[i]*V)/(R*self.T[i])-E/(R*1600.0))/self.rho[i]
            eta_m[i]=eta_m_0/self.rho[i]
           
            y=(self.x_m[i]-0.4)/0.15
            z=0.5*(1.0+math.tanh(y))
            value[i]=10.0**(z*math.log10(eta_m[i])+(1.0-z)*math.log10(eta_s[i]))
        return value  

    cpdef double f_viscosity_single(self,double Pres,double Temp):
        cdef double A=1.67
        cdef double B=7.4e-17
        cdef double n=3.5
        cdef double E=5.0e+5
        cdef double V=1.0e-5
        cdef double R=8.314
        cdef double epsilon=1.0e-15
        cdef double y=0.0
        cdef double z=0.0
        cdef double eta_m_0=100.0
        cdef double eta0, p_decay
        if Pres<125e9:
            if Pres<23e9:
                B=3.5e-15; n=3.0; E=4.3e+5
            eta_s=0.5*(1.0/B**(1.0/n))*math.exp((E+Pres*V)/(n*R*Temp))*epsilon**((1.0-n)/n)/self.rho[self.c_z]
        else:
            eta0=1.9e21; E=1.62e5; p_decay=1610e9
            V=1.4e-6*math.exp(-Pres/p_decay)
            eta_s=eta0*math.exp((E+Pres*V)/(R*Temp)-E/(R*1600.0))/self.rho[self.c_z]
        eta_m=eta_m_0/self.rho[self.c_z]
        y=(self.x_m[self.c_z]-0.4)/0.15
        z=0.5*(1.0+math.tanh(y))
        return 10.0**(z*math.log10(eta_m)+(1.0-z)*math.log10(eta_s))
       
    cpdef double[:] f_dsdr(self):
        # return the entropy gradient at the top boundaries of each zone
        cdef Py_ssize_t i
        cdef double[:] dsdr=np.zeros(len(self.radius))
        for i in range(self.c_z,self.c_z+self.m_z-1):
            dsdr[i]=(self.s_cell[i]-self.s_cell[i+1])/(self.r_cell[i]-self.r_cell[i+1])
        for i in range(self.c_z+self.m_z,len(self.radius)-1):
            dsdr[i]=(self.s_cell[i]-self.s_cell[i+1])/(self.r_cell[i]-self.r_cell[i+1])
        return dsdr
    
    cpdef double[:] f_convection(self,double[:] dsdr):
        # determine whether the individual zones are convecting using the Schwarzschild criterion
        # yes convection if ds/dr<0
        cdef Py_ssize_t i
        cdef double[:] convection=np.zeros(len(self.radius))
        for i in range(len(self.radius)):
            if dsdr[i]<0.0:
                convection[i]=1.0
            else:
                convection[i]=0.0
        return convection
        
    cpdef double[:] f_l(self,double[:] convection):
        # calculate the mixing length parameter (F. Wagner). 
        # give user options to use different l_alpha and l_beta in the future
        cdef Py_ssize_t i
        cdef double[:] l_mlt=np.zeros(len(self.radius))
        for i in range(self.c_z,len(self.radius)):
            if convection[i]==1.0:
                if i>self.c_z+self.m_z:
                    l_mlt[i]=self.pressure[i]/(self.rho[i]*self.g[i])
                else:
                    if (self.radius[i]-self.radius[self.c_z-1])<=(self.radius[self.c_z+self.m_z-1]-self.radius[self.c_z-1])/2.0*self.l_beta:
                        l_mlt[i]=self.l_alpha*(self.radius[i]-self.radius[self.c_z-1])/self.l_beta
                    else:
                        l_mlt[i]=self.l_alpha*(self.radius[self.c_z+self.m_z-1]-self.radius[i])/(2.0-self.l_beta)
        return l_mlt
        
    cpdef double[:] f_eddy_T_low_nu(self,double[:] l,double[:] dsdr):
        # eddy diffusivity with low viscosity. convective velocity is limited by the terminal velocity of falling parcel.
        cdef Py_ssize_t i
        cdef double[:] value=np.zeros(len(self.radius))
        for i in range(self.c_z,len(self.radius)):
            value[i]=(self.alpha[i]*self.g[i]*l[i]**4.0*self.T[i]/(16.0*self.cP[i])*(-dsdr[i]))**0.5
        return value#(np.asarray(self.alpha)*np.asarray(self.g)*np.asarray(l)**4.0*np.asarray(self.T)/(16.0*np.asarray(self.cP))*(-np.asarray(dsdr)))**0.5
 
    cpdef double[:] f_eddy_T_high_nu(self,double[:] l,double[:] v,double[:] dsdr):
        # eddy diffusivity with high viscosity. convective velocity is limited by viscous drag force. 
        cdef Py_ssize_t i
        cdef double[:] value=np.zeros(len(self.radius))
        for i in range(self.c_z,len(self.radius)):
            value[i]=(self.alpha[i]*self.g[i]*l[i]**4.0*self.T[i]/(16.0*self.cP[i])*(-dsdr[i]))**0.5
        return np.asarray(self.alpha)*np.asarray(self.g)*np.asarray(l)**4.0*np.asarray(self.T)/(18.0*np.asarray(v)*np.asarray(self.cP))*(-np.asarray(dsdr))

    cpdef double[:] f_kappa(self):
        # thermal diffusivity
        cdef Py_ssize_t i
        cdef double[:] value=np.zeros(len(self.radius))
        for i in range(self.c_z,len(self.radius)):
            value[i]=self.k_array[i]/(self.rho[i]*self.cP[i])
        return value#np.asarray(self.k_array)/(np.asarray(self.rho)*np.asarray(self.cP))

    cpdef double[:] f_eddy_k(self,double[:] eddy_low,double[:] eddy_high,double[:] v):
        # eddy diffusivity
        cdef double[:] eddy_k=np.zeros(len(self.radius))
        for i in range(self.c_z,self.c_z+self.m_z):
            if eddy_low[i]/v[i]>9.0/8.0:
                eddy_k[i]=eddy_low[i]
            else:
                eddy_k[i]=eddy_high[i]
        return eddy_k
    
    cpdef double[:] f_dPdr(self):
        # returns dP/dr
        cdef Py_ssize_t i
        cdef double[:] value=np.zeros(len(self.radius))
        for i in range(len(self.radius)):
            value[i]=-G*self.mass[i]*self.rho[i]/self.radius[i]**2.0
        return value#-G*np.asarray(self.mass)*np.asarray(self.rho)/np.asarray(self.radius)**2.0
    
    # f_matrix completes the matrix to be inverted in penta_solver (a backward euler method)
    # We use a finite volume method to discretize the zones

    cpdef tuple f_matrix(self,double[:] eddy_k,double[:] kappa,double[:] dPdr):
        cdef double[:,:] mat = np.zeros((5,len(self.radius)))
        cdef double[:] yy = np.zeros(len(self.radius))
        cdef Py_ssize_t i
        cdef double vm1, vm3
        for i in range(self.c_z+1, self.c_z+self.m_z-1):
            vm1=(self.Area[i-1]/(self.h[i]*self.T_cell[i])*self.rho[i-1]*self.T[i-1]*
                 (kappa[i-1]+eddy_k[i-1])/(self.r_cell[i-1]-self.r_cell[i]))
            vm3=(self.Area[i]/(self.h[i]*self.T_cell[i])*self.rho[i]*self.T[i]*
                 (kappa[i]+eddy_k[i])/(self.r_cell[i]-self.r_cell[i+1]))
            mat[1][i]=vm1
            mat[2][i]=1.0/self.dt-vm1-vm3
            mat[3][i]=vm3
            yy[i]=(self.s_cell[i]/self.dt
                   -self.Area[i-1]/(self.h[i]*self.T_cell[i])*self.k_array[i]*self.dTdP[i-1]*dPdr[i-1]
                   +self.Area[i]/(self.h[i]*self.T_cell[i])*self.k_array[i]*self.dTdP[i]*dPdr[i]
                   +self.f_qrad()/self.T_cell[i])

        cdef double Fcmb=-self.k_array[self.c_z]*(self.T[self.c_z-1]-self.T[self.c_z])/(self.radius[self.c_z-1]-self.radius[self.c_z])
        cdef double Fsurf=sigma*(self.T[self.c_z+self.m_z-1]**4.0-self.Teq**4.0)
        cdef double v_b=4.0/self.cP[self.c_z+self.m_z-1]
        cdef double v_a=v_b*sigma*self.Teq**4.0 

        i=self.c_z+self.m_z-1
        mat[1][i]=(self.Area[i-1]/(self.h[i]*self.T_cell[i])*self.rho[i-1]*self.T[i-1]
                *(kappa[i-1]+eddy_k[i-1])/(self.r_cell[i-1]-self.r_cell[i]))
        mat[2][i]=1.0/self.dt-mat[1][i]+self.Area[i]/(self.h[i]*self.T_cell[i])*(v_b*Fsurf+v_a)
        yy[i]=(self.s_cell[i]/self.dt 
           -self.Area[i-1]/(self.h[i]*self.T_cell[i])*self.k_array[i]*self.dTdP[i-1]*dPdr[i-1]
           -self.Area[i]*(Fsurf-(v_b*Fsurf+v_a)*self.s_cell[i])/(self.h[i]*self.T_cell[i])
           +self.f_qrad()/self.T_cell[i]) 

        i = self.c_z
        mat[3][i]= (self.Area[i]/(self.h[i]*self.T_cell[i])*self.rho[i]*self.T[i]
                *(kappa[i]+eddy_k[i])/(self.r_cell[i]-self.r_cell[i+1]))
        mat[2][i]=1.0/self.dt-mat[3][i]
        yy[i]=(self.s_cell[i]/self.dt+self.Area[i]/(self.h[i]*self.T_cell[i])*self.k_array[i]*self.dTdP[i]*dPdr[i]
            +self.Area[i-1]*Fcmb/self.h[i]/self.T_cell[i]+self.f_qrad()/self.T_cell[i])

        return mat, yy, Fcmb, Fsurf
    
    cpdef tuple f_y(self,double[:] s_liq,double[:] s_liq_c,double[:] s_sol,double[:] s_sol_c):
        cdef double[:] y=np.zeros(len(self.radius))
        cdef double[:] y_c=np.zeros(len(self.radius))
        cdef Py_ssize_t i
        for i in range(self.c_z,self.c_z+self.m_z):
            if self.s_cell[i]>=s_liq_c[i]:
                y_c[i]=(self.s_cell[i]-s_liq_c[i])/(S_max-s_liq_c[i])
                self.phase_c[i]=0.0
            elif self.s_cell[i]<=s_sol_c[i]:
                y_c[i]=(self.s_cell[i]-S_min)/(s_sol_c[i]-S_min)
                self.phase_c[i]=2.0
            else:
                y_c[i]=(self.s_cell[i]-s_sol_c[i])/(s_liq_c[i]-s_sol_c[i])
                self.phase_c[i]=1.0
                
            if self.s_array[i]>=s_liq[i]:
                y[i]=(self.s_array[i]-s_liq[i])/(S_max-s_liq[i]) 
                self.phase[i]=0.0
                self.x_m[i]=1.0
            elif self.s_array[i]<=s_sol[i]:
                y[i]=(self.s_array[i]-S_min)/(s_sol[i]-S_min)
                self.phase[i]=2.0
                self.x_m[i]=0.0
            else:
                y[i]=(self.s_array[i]-s_sol[i])/(s_liq[i]-s_sol[i])
                self.phase[i]=1.0
                self.x_m[i]=y[i]
        return y, y_c, self.phase, self.phase_c,self.x_m
    
    cpdef tuple f_update(self,double[:] y_array,double[:] y_c_array):
        cdef Py_ssize_t i

        T_c_l=np.zeros(len(self.radius))
        T_c_pv=np.zeros(len(self.radius))
        T_c_ppv=np.zeros(len(self.radius))
        T_c_en=np.zeros(len(self.radius))
        T_c_pv_m=np.zeros(len(self.radius))
        T_c_ppv_m=np.zeros(len(self.radius))
        T_c_en_m=np.zeros(len(self.radius))
        
        T_l=np.zeros(len(self.radius))
        T_pv=np.zeros(len(self.radius))
        T_ppv=np.zeros(len(self.radius))
        T_en=np.zeros(len(self.radius))
        T_pv_m=np.zeros(len(self.radius))
        T_ppv_m=np.zeros(len(self.radius))
        T_en_m=np.zeros(len(self.radius))
        
        dqdy_l=np.zeros(len(self.radius))
        dqdy_pv=np.zeros(len(self.radius))
        dqdy_ppv=np.zeros(len(self.radius))
        dqdy_en=np.zeros(len(self.radius))
        dqdy_pv_m=np.zeros(len(self.radius))
        dqdy_ppv_m=np.zeros(len(self.radius))
        dqdy_en_m=np.zeros(len(self.radius))
        
        rho_l=np.zeros(len(self.radius))
        rho_pv=np.zeros(len(self.radius))
        rho_ppv=np.zeros(len(self.radius))
        rho_en=np.zeros(len(self.radius))
        rho_pv_m=np.zeros(len(self.radius))
        rho_ppv_m=np.zeros(len(self.radius))
        rho_en_m=np.zeros(len(self.radius))
        
        cP_l=np.zeros(len(self.radius))
        cP_pv=np.zeros(len(self.radius))
        cP_ppv=np.zeros(len(self.radius))
        cP_en=np.zeros(len(self.radius))
        cP_pv_m=np.zeros(len(self.radius))
        cP_ppv_m=np.zeros(len(self.radius))
        cP_en_m=np.zeros(len(self.radius))
        
        alpha_l=np.zeros(len(self.radius))
        alpha_pv=np.zeros(len(self.radius))
        alpha_ppv=np.zeros(len(self.radius))
        alpha_en=np.zeros(len(self.radius))
        alpha_pv_m=np.zeros(len(self.radius))
        alpha_ppv_m=np.zeros(len(self.radius))
        alpha_en_m=np.zeros(len(self.radius))
        
        dTdP_l=np.zeros(len(self.radius))
        dTdP_pv=np.zeros(len(self.radius))
        dTdP_ppv=np.zeros(len(self.radius))
        dTdP_en=np.zeros(len(self.radius))
        dTdP_pv_m=np.zeros(len(self.radius))
        dTdP_ppv_m=np.zeros(len(self.radius))
        dTdP_en_m=np.zeros(len(self.radius))

        p_cell_np=np.asarray(self.p_cell)
        pressure_np=np.asarray(self.pressure)
        y_c_np=np.asarray(y_c_array)
        y_np=np.asarray(y_array)
        
        T_c_l[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, T_liq)
        T_c_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, T_sol_pv)
        T_c_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, T_sol_ppv)
        T_c_en[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, T_sol_en)
        T_c_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, T_mix_pv)
        T_c_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, T_mix_ppv)
        T_c_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(p_cell_np[self.c_z:self.c_z+self.m_z],y_c_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, T_mix_en)
        
        T_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, T_liq)
        T_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, T_sol_pv)
        T_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, T_sol_ppv)
        T_en[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, T_sol_en)
        T_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, T_mix_pv)
        T_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, T_mix_ppv)
        T_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, T_mix_en)

        dqdy_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, dqdy_liq)
        dqdy_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, dqdy_sol_pv)
        dqdy_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, dqdy_sol_ppv)
        dqdy_en[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, dqdy_sol_en)
        dqdy_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, dqdy_mix_pv)
        dqdy_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, dqdy_mix_ppv)
        dqdy_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, dqdy_mix_en)

        rho_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, rho_liq)
        rho_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, rho_sol_pv)
        rho_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, rho_sol_ppv)
        rho_en[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, rho_sol_en)
        rho_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, rho_mix_pv)
        rho_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, rho_mix_ppv)
        rho_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, rho_mix_en)

        cP_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, CP_liq)
        cP_pv[self.c_z:self.c_z+self.m_z]=CP_m_s*np.ones(self.m_z)
        cP_ppv[self.c_z:self.c_z+self.m_z]=CP_m_s*np.ones(self.m_z)
        cP_en[self.c_z:self.c_z+self.m_z]=CP_m_s*np.ones(self.m_z)
        cP_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, CP_mix_pv)
        cP_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, CP_mix_ppv)
        cP_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, CP_mix_en)

        alpha_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, alpha_liq)
        alpha_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, alpha_sol_pv)
        alpha_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, alpha_sol_ppv)
        alpha_en[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, alpha_sol_en)
        alpha_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, alpha_mix_pv)
        alpha_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, alpha_mix_ppv)
        alpha_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, alpha_mix_en)

        dTdP_l[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid, y_grid, dTdP_liq)
        dTdP_pv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, dTdP_sol_pv)
        dTdP_ppv[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, dTdP_sol_ppv)
        dTdP_en[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, dTdP_sol_en)
        dTdP_pv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_pv, y_grid, dTdP_mix_pv)
        dTdP_ppv_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_ppv, y_grid, dTdP_mix_ppv)
        dTdP_en_m[self.c_z:self.c_z+self.m_z]=interpolate2d(pressure_np[self.c_z:self.c_z+self.m_z],y_np[self.c_z:self.c_z+self.m_z], P_grid_en, y_grid, dTdP_mix_en)
        
        cdef double[:] logrho=np.zeros(len(self.radius))

        for i in range(self.c_z,self.c_z+self.m_z):
            if self.phase_c[i]==0.0:
                self.T_cell[i]=T_c_l[i]
            elif self.phase_c[i]==2.0:
                if self.p_cell[i]>=125.0e9:
                    self.T_cell[i]=T_c_ppv[i]
                elif self.p_cell[i]<23.0e9:
                    self.T_cell[i]=T_c_en[i]
                else:
                    self.T_cell[i]=T_c_pv[i]
            else:
                if self.p_cell[i]>=125.0e9:
                    self.T_cell[i]=T_c_ppv_m[i]
                elif self.p_cell[i]<23.0e9:
                    self.T_cell[i]=T_c_en_m[i]
                else:
                    self.T_cell[i]=T_c_pv_m[i]
            if self.phase[i]==0.0:
                self.T[i]=T_l[i]
                self.alpha[i]=alpha_l[i]
                self.cP[i]=cP_l[i]
                self.rho[i]=rho_l[i]
                self.dTdP[i]=dTdP_l[i]
                self.dqdy[i]=dqdy_l[i]
                self.k_array[i]=k_l
                logrho[i]=math.log(self.rho[i])
            elif self.phase[i]==2.0:
                if self.pressure[i]>=125.0e9:
                    self.T[i]=T_ppv[i]
                    self.alpha[i]=alpha_ppv[i]
                    self.cP[i]=cP_ppv[i]
                    self.rho[i]=rho_ppv[i]
                    self.dTdP[i]=dTdP_ppv[i]
                    self.dqdy[i]=dqdy_ppv[i]
                    self.k_array[i]=k_ppv
                    logrho[i]=math.log(self.rho[i])
                elif self.pressure[i]<23.0e9:
                    self.T[i]=T_en[i]
                    self.alpha[i]=alpha_en[i]
                    self.cP[i]=cP_en[i]
                    self.rho[i]=rho_en[i]
                    self.dTdP[i]=dTdP_en[i]
                    self.dqdy[i]=dqdy_en[i]
                    self.k_array[i]=k_en
                    logrho[i]=math.log(self.rho[i])
                else:
                    self.T[i]=T_pv[i]
                    self.alpha[i]=alpha_pv[i]
                    self.cP[i]=cP_pv[i]
                    self.rho[i]=rho_pv[i]
                    self.dTdP[i]=dTdP_pv[i]
                    self.dqdy[i]=dqdy_pv[i]
                    self.k_array[i]=k_pv
                    logrho[i]=math.log(self.rho[i])
            else:
                if self.pressure[i]>=125.0e9:
                    self.T[i]=T_ppv_m[i]
                    self.alpha[i]=alpha_ppv_m[i]
                    self.cP[i]=cP_ppv_m[i]
                    self.rho[i]=rho_ppv_m[i]
                    self.dTdP[i]=dTdP_ppv_m[i]
                    self.dqdy[i]=dqdy_ppv_m[i]
                    self.k_array[i]=k_ppv
                    logrho[i]=math.log(self.rho[i])
                elif self.pressure[i]<23.0e9:
                    self.T[i]=T_en_m[i]
                    self.alpha[i]=alpha_en_m[i]
                    self.cP[i]=cP_en_m[i]
                    self.rho[i]=rho_en_m[i]
                    self.dTdP[i]=dTdP_en_m[i]
                    self.dqdy[i]=dqdy_en_m[i]
                    self.k_array[i]=k_en
                    logrho[i]=math.log(self.rho[i])
                else:
                    self.T[i]=T_pv_m[i]
                    self.alpha[i]=alpha_pv_m[i]
                    self.cP[i]=cP_pv_m[i]
                    self.rho[i]=rho_pv_m[i]
                    self.dTdP[i]=dTdP_pv_m[i]
                    self.dqdy[i]=dqdy_pv_m[i]
                    self.k_array[i]=k_pv
                    logrho[i]=math.log(self.rho[i])

        return self.T_cell,self.T,self.alpha,self.cP,self.rho,self.dTdP,self.dqdy,logrho,self.k_array
    """
    cpdef tuple mat_core_cond(self, double[:] old_P):
        cdef double[:] aa=np.zeros(self.c_z)
        cdef double[:] bb=np.zeros(self.c_z)
        cdef double[:] cc=np.zeros(self.c_z)
        cdef double[:] dd=np.zeros(self.c_z)
        cdef double[:] ee=np.zeros(self.c_z)
        cdef double[:] ff=np.zeros(self.c_z)
        cdef Py_ssize_t i
        cdef double[:] solution_T=np.zeros(self.c_z)
        cdef double Q_ICB
        i=0
        bb[i]=0.0
        dd[i]=-(self.k_array[i]*self.Area[i])/(self.h[i]*self.cP[i])/(self.r_cell[i+1]-self.r_cell[i])
        cc[i]=1.0/self.dt-dd[i]
        ff[i]=self.T_cell[i]/self.dt+self.alpha[i]*self.T[i]/self.rho[i]/self.cP[i]*(self.pressure[i]-old_P[i])/self.dt
        i=self.c_z-1
        bb[i]=-(self.k_array[i-1]*self.Area[i-1])/(self.h[i]*self.cP[i-1])/(self.r_cell[i]-self.r_cell[i-1])
        cc[i]=1.0/self.dt-bb[i]
        dd[i]=0.0
        ff[i]=self.T_cell[i]/self.dt+self.alpha[i]*self.T[i]/self.rho[i]/self.cP[i]*(self.pressure[i]-old_P[i])/self.dt
        for i in range(1,self.c_z-1):
            bb[i]=-(self.k_array[i-1]*self.Area[i-1])/(self.h[i]*self.cP[i-1])/(self.r_cell[i]-self.r_cell[i-1])
            dd[i]=-(self.k_array[i]*self.Area[i])/(self.h[i]*self.cP[i])/(self.r_cell[i+1]-self.r_cell[i])
            cc[i]=1.0/self.dt-dd[i]-bb[i]
            ff[i]=self.T_cell[i]/self.dt+self.alpha[i]*self.T[i]/self.rho[i]/self.cP[i]*(self.pressure[i]-old_P[i])/self.dt
        solution_T=self.penta_solver(aa[:self.c_z],bb[:self.c_z],cc[:self.c_z],dd[:self.c_z],ee[:self.c_z],ff[:self.c_z],self.c_z)
        if self.Mic>0.0:
            Q_ICB=(-self.k_array[self.solid_index]*(solution_T[self.solid_index]-solution_T[self.solid_index+1])
                   /(self.r_cell[self.solid_index]-self.r_cell[self.solid_index+1])*self.Area[self.solid_index])
        else:
            Q_ICB=0.0
        return solution_T,Q_ICB

    cpdef tuple f_core_adia(self,double[:] old_P,double Q_ICB,double Fcmb):
        cdef Py_ssize_t i
        cdef double Q_th=-Fcmb*self.Area[self.c_z-1]
        cdef int pre_adia_p_idx=self.find_nearest(np.asarray(pre_adia_pressure),self.pressure[self.c_z-1])
        pre_adia_p=np.zeros(pre_adia_p_idx+1)
        for i in range(len(pre_adia_p)-1):
            pre_adia_p[i]=pre_adia_pressure[i]
        pre_adia_p[pre_adia_p_idx]=self.pressure[self.c_z-1]
        pre_adia_p=pre_adia_p[::-1]
        
        cdef double x_alloy=self.x_core/mf_l  ## mf_l related to EoS
        cdef int x_idx=self.find_nearest(x_core_grid,x_alloy)
        cdef int Tref_idx=self.find_nearest(Tref_core_grid,self.min_pre_adia_T)
        
        dTdT0_cmb=interpolate3d(np.ones(1)*x_alloy,np.ones(1)*self.min_pre_adia_T,np.ones(1)*self.pressure[self.c_z-1],x_core_grid[x_idx:x_idx+2],Tref_core_grid[Tref_idx:Tref_idx+2],pre_adia_pressure,load_original_dTdT0[x_idx:x_idx+2,Tref_idx:Tref_idx+2])[0]
        dT0dPcmb=f_dT0dP([x_alloy,self.min_pre_adia_T,self.T[self.c_z-1]])[0]
        delta_Pcmb=self.pressure[self.c_z-1]-old_P[self.c_z-1]
        dTdT0_array=interpolate3d(np.ones(self.c_z)*x_alloy,np.ones(self.c_z)*self.min_pre_adia_T,self.pressure[:self.c_z],x_core_grid[x_idx:x_idx+2],Tref_core_grid[Tref_idx:Tref_idx+2],pre_adia_pressure,load_original_dTdT0[x_idx:x_idx+2,Tref_idx:Tref_idx+2])
        for i in range(self.c_z):
            outer_adiabat_value=self.h[i]*C_P_Fe*dTdT0_array[i]
            outer_adiabat_array[i]=outer_adiabat_value/dTdT0_cmb
            outer_adiabat_Pcmb_array[i]=outer_adiabat_value*dT0dPcmb
        outer_adiabat=np.sum(outer_adiabat_array[self.solid_index:])
        outer_adiabat_Pcmb=np.sum(outer_adiabat_Pcmb_array[self.solid_index:])*delta_Pcmb
        # Latent heat when there's ongoing inner core solidification
        #if initial_phase[0]==ph_Fe_sol and initial_phase[core_outer_index]==ph_Fe_liq:
        #if Mic>0.0 and Mic<=self.M_pl*self.CMF:
        #    dTdT0_ic=interpolate3d(np.ones(1)*x_alloy,np.ones(1)*self.min_pre_adia_T,np.ones(1)*self.Pic,x_core_grid[x_idx:x_idx+2],Tref_core_grid[Tref_idx:Tref_idx+2],pre_adia_pressure,load_original_dTdT0[x_idx:x_idx+2,Tref_idx:Tref_idx+2])[0]
        #    dmicdPic=-4.0*math.pi/G*self.Ric**4.0/self.Mic
        #    dPicdTic=1.0/(self.alpha_ic/self.rho_ic*self.Tic/C_P_Fe)
        #    dTicdTcmb=dTdT0_ic/dTdT0_cmb
        #    dmicdTcmb=dmicdPic*dPicdTic*dTicdTcmb
        #else:
        #    dmicdTcmb=0.0
        dmicdTcmb=0.0
        outer_adiabat=outer_adiabat-dmicdTcmb*L_Fe
        
        Q_rad_c=self.Q_rad_c_0*math.exp(-self.t/86400.0/365.0/1e9/1.2)
        Q=Q_th+Q_rad_c+Q_ICB
        delta_Tcmb=(self.dt*Q-outer_adiabat_Pcmb)/outer_adiabat
        self.T_cmb=self.T_cmb+delta_Tcmb
        # Look for the reference temperature, min_pre_adia_T, for the core adiabat. 
        pre_adia_T=[self.T_cmb]
        for i in range(0,len(pre_adia_p)-1):
            rho_liquid=f_rho_Fel(pre_adia_T[i],pre_adia_p[i])[0]
            rho_alloy=f_rho_Fea(pre_adia_T[i],pre_adia_p[i])[0]
            alpha_liquid=f_alpha_Fel(pre_adia_T[i],pre_adia_p[i])[0]
            alpha_alloy=f_alpha_Fea(pre_adia_T[i],pre_adia_p[i])[0]
            pre_adia_rho=rho_mix(x_alloy,rho_alloy,rho_liquid)
            pre_adia_alpha=alpha_mix(x_alloy,alpha_alloy,alpha_liquid,pre_adia_rho,rho_alloy,rho_liquid)
            delta_T=f_dTdP(pre_adia_alpha,pre_adia_rho,C_P_Fe,pre_adia_T[i])*(pre_adia_p[i+1]-pre_adia_p[i])
            pre_adia_T.append(pre_adia_T[i]+delta_T)
        self.min_pre_adia_T=pre_adia_T[len(pre_adia_T)-1] 
        adiabat_array=interpolate3d(np.ones(self.c_z)*x_alloy,np.ones(self.c_z)*self.min_pre_adia_T,self.pressure[:self.c_z], x_core_grid[x_idx:x_idx+2], Tref_core_grid[Tref_idx:Tref_idx+2], pre_adia_pressure, load_original_T[x_idx:x_idx+2,Tref_idx:Tref_idx+2])
        self.T[:self.c_z]=adiabat_array.copy()
        
        rho_liquid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fel, T_Fel, rho_Fel)
        rho_solid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fes, T_Fes, rho_Fes)
        rho_alloy_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fea, T_Fea, rho_Fea)
        alpha_liquid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fel, T_Fel, alpha_Fel)
        alpha_solid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fes, T_Fes, alpha_Fes)
        alpha_alloy_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fea, T_Fea, alpha_Fea)
        dqdy_liquid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fel, T_Fel, dqdy_Fel)
        dqdy_solid_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fes, T_Fes, dqdy_Fes)
        dqdy_alloy_array=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fea, T_Fea, dqdy_Fea)  
        rho_liquid_array_a=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fel, T_Fel, rho_Fel)
        rho_alloy_array_a=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fea, T_Fea, rho_Fea)
        alpha_liquid_array_a=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fel, T_Fel, alpha_Fel)
        alpha_alloy_array_a=interpolate2d(self.pressure[:self.c_z], self.T[:self.c_z], P_Fea, T_Fea, alpha_Fea)
        
        logrho=np.zeros(self.c_z)
        for i in range(self.c_z):
            if i==0:
                rho_liquid=f_rho_Fel(self.T[0],self.P_c)[0]
                rho_alloy=f_rho_Fea(self.T[0],self.P_c)[0]
                self.rho_c=rho_mix(x_alloy,rho_alloy,rho_liquid)
                dqdy_liquid=f_dqdy_Fel(self.T[0],self.P_c)[0]
                dqdy_alloy=f_dqdy_Fea(self.T[0],self.P_c)[0]
                self.dqdy_c=dqdy_mix(x_alloy,self.rho_c,rho_alloy,rho_liquid,dqdy_alloy,dqdy_liquid,self.P_c)
            
            rho_liquid=rho_liquid_array[i]
            rho_alloy=rho_alloy_array[i]
            alpha_liquid=alpha_liquid_array[i]
            alpha_alloy=alpha_alloy_array[i]
            dqdy_liquid=dqdy_liquid_array[i]
            dqdy_alloy=dqdy_alloy_array[i]
            self.rho[i]=rho_mix(x_alloy,rho_alloy,rho_liquid)
            self.dqdy[i]=dqdy_mix(x_alloy,self.rho[i],rho_alloy,rho_liquid,dqdy_alloy,dqdy_liquid,self.pressure[i])
            self.alpha[i]=alpha_mix(x_alloy,alpha_alloy,alpha_liquid,self.rho[i],rho_alloy,rho_liquid)
            logrho[i]=np.log(self.rho[i])
        
        return self.T, self.rho, self.alpha, self.dqdy, logrho, self.rho_c, self.dqdy_c
    """
    cpdef double f_dt(self,double dt_thres):
        # adjust dt in each timestep. dt is controlled by the maximum change in entropy in the planet
        cdef double ds_thres=2.5e-4
        if dt_thres<ds_thres:
            if dt_thres<0.95*ds_thres:
                self.dt=self.dt*1.01
            else:
                self.dt=self.dt+30.0
        else:
            if dt_thres>1.05*ds_thres:
                self.dt=self.dt*0.75
            else:
                self.dt=self.dt*0.95 
        if self.dt<30.0:
            self.dt=30.0
        if self.dt>86400.0*365.0*1e6:
            self.dt=86400.0*365.0*1e6
        return self.dt
    
    cpdef dict main(self):
        cdef Py_ssize_t i
        cdef double[:] old_s=(self.s_array).copy()
        cdef double[:] old_scell=(self.s_cell).copy()
        cdef double[:] old_T=(self.T).copy()
        cdef double[:] old_Tcell=(self.T_cell).copy()
        cdef double[:] old_P=(self.pressure).copy()
        cdef double[:] s_liq=np.zeros(len(self.radius))
        cdef double[:] s_liq_c=np.zeros(len(self.radius))
        cdef double[:] s_sol=np.zeros(len(self.radius))
        cdef double[:] s_sol_c=np.zeros(len(self.radius))
        for i in range(self.c_z, self.c_z+self.m_z):
            s_liq[i]=S_liq_P(self.pressure[i]).tolist()
            s_liq_c[i]=S_liq_P(self.p_cell[i]).tolist()
            s_sol[i]=S_sol_P(self.pressure[i]).tolist()
            s_sol_c[i]=S_sol_P(self.p_cell[i]).tolist()
        
        cdef double[:] v=self.f_viscosity()
        cdef double[:] dsdr=self.f_dsdr()
        cdef double[:] convection=self.f_convection(dsdr)
        cdef double[:] l_mlt=self.f_l(convection)
        cdef double[:] eddy_low=self.f_eddy_T_low_nu(l_mlt,dsdr)
        cdef double[:] eddy_high=self.f_eddy_T_high_nu(l_mlt,v,dsdr)
        cdef double[:] eddy_k=self.f_eddy_k(eddy_low,eddy_high,v)
        cdef double[:] dPdr=self.f_dPdr()
        cdef double[:] kappa=self.f_kappa()
        
        cdef double[:,:] mat=np.zeros((5,len(self.radius)))
        cdef double[:] yy=np.zeros(len(self.radius))
        cdef double Fcmb, Fsurf
        mat, yy, Fcmb, Fsurf=self.f_matrix(eddy_k,kappa,dPdr)
        self.s_cell[self.c_z:self.c_z+self.m_z]=self.penta_solver(mat[0][self.c_z:self.c_z+self.m_z],
                                                               mat[1][self.c_z:self.c_z+self.m_z],
                                                               mat[2][self.c_z:self.c_z+self.m_z],
                                                               mat[3][self.c_z:self.c_z+self.m_z],
                                                               mat[4][self.c_z:self.c_z+self.m_z],
                                                               yy[self.c_z:self.c_z+self.m_z],
                                                               self.m_z)
        
        cdef double[:] y_array=np.zeros(len(self.radius))
        cdef double[:] y_c_array=np.zeros(len(self.radius))
        cdef double[:] logrho=np.zeros(len(self.radius))
        for i in range(self.c_z,self.c_z+self.m_z-1):
            self.s_array[i]=(self.s_cell[i]+self.s_cell[i+1])/2.0
        self.s_array[self.c_z+self.m_z-1]=self.s_cell[self.c_z+self.m_z-1]
        y_array,y_c_array,self.phase,self.phase_c,self.x_m=self.f_y(s_liq,s_liq_c,s_sol,s_sol_c)

        self.T_cell,self.T,self.alpha,self.cP,self.rho,self.dTdP,self.dqdy,logrho,self.k_array=self.f_update(y_array,y_c_array)
        
        # change the stepsize in time
        cdef double dt_thres=np.max(np.abs((np.asarray(old_scell[self.c_z:self.c_z+self.m_z])-np.asarray(self.s_cell[self.c_z:self.c_z+self.m_z]))/np.asarray(old_scell[self.c_z:self.c_z+self.m_z])))
        cdef double dt=self.f_dt(dt_thres)
        
        cdef double[:] F_conv=np.zeros(len(dsdr))
        cdef double[:] v_conv=np.zeros(len(dsdr))
        cdef double[:] Rem=np.zeros(len(dsdr))
        for i in range(self.c_z+1,zone):
            F_conv[i]=-self.rho[i]*self.T[i]*eddy_k[i]*dsdr[i]
            v_conv[i]=eddy_k[i]/(l_mlt[i]+1e-10)
            Rem[i]=mu_0*v_conv[i]*L_sigma*5e4  

        # save results
        cdef dict results={}
        results['average s_m']=sum(self.s_array[self.c_z:self.c_z+self.m_z])/len(self.s_array[self.c_z:self.c_z+self.m_z])
        results['average T_m']=sum(self.T[self.c_z:self.c_z+self.m_z])/len(self.T[self.c_z:self.c_z+self.m_z])
        results['average T_c']=sum(self.T[:self.c_z])/len(self.T[:self.c_z])
        results['Fcmb']=Fcmb
        results['Fsurf']=Fsurf
        results['Fcond_cmb']=self.k_array[self.c_z-1]*self.alpha[self.c_z-1]*self.g[self.c_z-1]*self.T[self.c_z-1]/self.cP[self.c_z-1]
        results['t']=self.t
        results['dt']=self.dt
        results['dt_thres']=dt_thres
        results['s_array']=self.s_array.copy()
        results['s_cell']=self.s_cell.copy()
        results['T_array']=self.T.copy()
        results['T_cell']=self.T_cell.copy()
        results['cP']=self.cP.copy()
        results['rho']=self.rho.copy()
        results['logrho']=logrho.copy()
        results['dTdP']=self.dTdP.copy()
        results['dqdy']=self.dqdy.copy()
        results['alpha']=self.alpha.copy()
        results['phase']=self.phase.copy()
        results['phase_c']=self.phase_c.copy() # phase at the cell center
        results['rho_c']=self.rho_c
        results['dqdy_c']=self.dqdy_c
        results['k']=self.k_array.copy()
        results['x_m']=self.x_m.copy()
        results['old_s']=old_s.copy()
        results['old_scell']=old_scell.copy()
        results['old_T']=old_T.copy()
        results['old_Tcell']=old_Tcell.copy()
        results['T surf']=self.T[-1]
        results['eddy_k']=eddy_k.copy()
        results['dsdr']=dsdr.copy()
        results['l_mlt']=l_mlt.copy()
        results['Fconv']=F_conv.copy()
        results['vconv']=v_conv.copy()
        results['Rem']=Rem.copy()
        results['viscosity']=v.copy()
        
        return results
            
# timestep where we save thermal and structure profiles (in years)
save_t=[1.0]
for i in range(1,1000):
    if save_t[i-1]<10000.0:
        save_t.append(save_t[i-1]+20.0)
    elif save_t[i-1]<1e8:
        save_t.append(save_t[i-1]+int(save_t[i-1]/20.0))
    else:
        save_t.append(save_t[i-1]+int(save_t[i-1]/50.0))


# initialize the structural profile using 4th order Runge Kutta method
cdef c_initial_profile initial_profile=c_initial_profile(M_pl, c_z, m_z, CMF, MMF, 
                                    P_c, P_surf, T_c, x_c, 
                                    dsdr_c, d_Pc, rtol, T_an_c_i)
cdef dict ri=initial_profile.RK4() 

# improve the solution by RK4 using a henyey code
rtol=1e-3
cdef c_henyey henyey_obj=c_henyey(ri['mass'], ri['radius'], ri['logr'], ri['pressure'], ri['logp'], 
                    ri['rho'], ri['logrho'], ri['gravity'], ri['s_array'], ri['dqdy'], 
                    ri['P_c'], ri['rho_c'], ri['dqdy_c'], P_surf, rtol,
                    ri['T_an_c'], c_z, m_z, x_c)
cdef dict rh=henyey_obj.henyey_m(dsdr_c,initial) # Using henyey code to relax the solution to RK4 such that the solution satisfies the boundary condition at both planet center and surface

cdef dict rt={}
rt['t']=0.0
rt['dt']=1.0
rt['s_array']=ri['s_array'].copy()
rt['s_cell']=rh['s_cell'].copy()
rt['T_array']=ri['temperature'].copy()
rt['T_cell']=ri['T_cell'].copy()
rt['alpha']=ri['alpha'].copy()
rt['cP']=ri['cP'].copy()
rt['dTdP']=ri['dTdP'].copy()

rt['phase']=phase.copy()
rt['phase_c']=np.zeros(len(ri['mass']))
rt['k']=k_array.copy()
rt['x_m']=np.ones(len(ri['mass']))

rt['rho_c']=rh['rho_c']
rt['dqdy_c']=rh['dqdy_c']

average_s=[]
average_Tm=[]
average_Tc=[]
T_surf=[]
t_array=[]
dt_array=[]
Rpl=[]
Pc=[]
Fcmb=[]
Fsurf=[]
Rc=[]
Pcmb=[]
Tcmb=[]
Fcond=[]

cdef heat_transport transport_obj

#while iteration<1000:
#while rt['T_array'][-1]>255.02:
while t<t_end:
    transport_obj=heat_transport(ri['mass'], ri['dm'], rh['radius'], rh['r_cell'], rh['pressure'], rh['p_cell'], 
                                rh['rho'], rh['gravity'], rh['dqdy'], rh['Area'], 
                                rt['s_array'], rt['s_cell'], rt['T_array'], rt['T_cell'],
                                rt['x_m'], rt['phase'], rt['phase_c'],
                                rt['alpha'], rt['cP'], rt['k'], rt['dTdP'], 
                                0.0,0.0,0.0,0.0,0.0,0.0,0, 0.105, ri['T_an_c'], rt['T_array'][c_z], #ri['T_an_c']
                                rh['P_c'], rt['rho_c'], rt['dqdy_c'],
                                Q_rad_c_0, Teq, qrad, t, rt['dt'], c_z, m_z, M_pl, CMF)
    rt=transport_obj.main()
    henyey_obj=c_henyey(ri['mass'], rh['radius'], rh['logr'], rh['pressure'], rh['logp'], 
                            rt['rho'], rt['logrho'], rh['gravity'], rt['s_array'], rt['dqdy'], 
                            rh['P_c'], rt['rho_c'], rt['dqdy_c'], P_surf, rtol,
                            ri['T_an_c'], c_z, m_z, x_c) # ri['T_an_c'] needs to be updated

    rh=henyey_obj.henyey_m(dsdr_c,initial=0.0)
    
    average_s.append(rt['average s_m'])
    average_Tm.append(rt['average T_m'])
    average_Tc.append(rt['average T_c'])
    t_array.append(rt['t'])
    dt_array.append(rt['dt'])
    Pc.append(rh['pressure'][0])
    Rpl.append(rh['radius'][-1])
    T_surf.append(rt['T_array'][-1])
    Fcmb.append(rt['Fcmb'])
    Fsurf.append(rt['Fsurf'])
    Rc.append(rh['radius'][c_z])
    Pcmb.append(rh['pressure'][c_z])
    Tcmb.append(rt['T_array'][c_z])
    Fcond.append(rt['Fcond_cmb'])
    
    t=t+rt['dt']
    iteration=iteration+1
    if iteration%5==0:
        if t/86400.0/365.0<1e3:
            t_val=t/86400.0/365.0
            print('time:%2.2fyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa' %(t_val,rt['Fcmb'],rt['Fsurf'],0.0,rt['T_array'][-1],rh['pressure'][0]/1e9,rh['pressure'][c_z]/1e9))
        elif t/86400.0/365.0>=1e3 and t/86400.0/365.0<1e6:
            t_val=t/86400.0/365.0/1e3
            print('time:%2.2fkyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa' %(t_val,rt['Fcmb'],rt['Fsurf'],0.0,rt['T_array'][-1],rh['pressure'][0]/1e9,rh['pressure'][c_z]/1e9))
        elif t/86400.0/365.0>=1e6 and t/86400.0/365.0<1e9:
            t_val=t/86400.0/365.0/1e6
            print('time:%2.2fMyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa' %(t_val,rt['Fcmb'],rt['Fsurf'],0.0,rt['T_array'][-1],rh['pressure'][0]/1e9,rh['pressure'][c_z]/1e9))
        else:
            t_val=t/86400.0/365.0/1e9
            print('time:%2.2fGyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa' %(t_val,rt['Fcmb'],rt['Fsurf'],0.0,rt['T_array'][-1],rh['pressure'][0]/1e9,rh['pressure'][c_z]/1e9))

    for ind in range(len(save_t)):
        if t<save_t[ind]*86400.0*365.0+rt['dt']and t>save_t[ind]*86400.0*365.0-rt['dt']:
            np.savetxt('results/profile/structure_'+str(int(save_t[ind]))+'.txt',np.transpose([rh['radius'],rh['pressure'],rt['T_array'],rt['rho'],rh['gravity'], ri['mass']]))
            np.savetxt('results/profile/property_'+str(int(save_t[ind]))+'.txt',np.transpose([rt['alpha'], rt['cP'],rh['gravity'],rt['Fconv'],rt['vconv'],rt['Rem'],rt['viscosity']]))
            np.savetxt('results/evolution_temp.txt',np.transpose([t_array,dt_array,average_s,average_Tm,average_Tc,T_surf,Tcmb,Fsurf,Fcmb,Fcond,Rpl,Rc,Pc,Pcmb]))

np.savetxt('results/evolution.txt',np.transpose([t_array,dt_array,average_s,average_Tm,average_Tc,T_surf,Tcmb,Fsurf,Fcmb,Fcond,Rpl,Rc,Pc,Pcmb]))

# Give ppl an option to plot the thermal profiles vs t, like MESA 
# EoS: input parameters for BM3, Vinet, Keane etcs. Thermal correction = True or False.  

## save: solid/liquid core mass, R solid core/total core, T_cmb, P_cmb, rho_cmb, T_c, P_c, rho_c
## Rossby number? Dipole moment and B_surf.
