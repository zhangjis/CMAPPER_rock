#!python
#cython: boundscheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import fsolve
from libc cimport math
cimport cython
import time

### TODO @Jisheng this needs updating
###### Overall structure of the code.
# line 25-25    : load input file
# line 27-161   : load EoS tables
# line 163-222  : define constants
# line 224-529  : Routines for initializing thermal and structure profiles: line
# line 531-769  : Routines for Henyey solver: line (its purpose is to find solutions to the planet structure equations that satisfy the boundary conditions at both planet center and surface)
# line 771-1623 : Routines for heat transport in the core and the mantle
# line 1625-1682: Initialize thermal and structural profiles
# line 1684-1742: while loop for updating thermal profiles using heat transport routines and structural profiles using Henyey solver

load_file=np.loadtxt('input.txt')
results_foldername='results_Mpl'+str(load_file[0])+'_CMF'+str(load_file[1])+'_time'+str(load_file[2])+'_Qrad'+str(load_file[3])+'_'+str(load_file[4])+'_'+str(load_file[5])+'_'+str(load_file[6])+'_Teq'+str(load_file[8])

print('Read EoS tables')
T_liq=np.loadtxt('EoS/mantle/T_liq_Py_1500GPa.txt')
rho_liq=np.loadtxt('EoS/mantle/rho_liq_Py_1500GPa.txt')
CP_liq=np.loadtxt('EoS/mantle/CP_liq_Py_1500GPa.txt')
alpha_liq=np.loadtxt('EoS/mantle/alpha_liq_Py_1500GPa.txt')
dTdP_liq=np.loadtxt('EoS/mantle/dTdP_liq_Py_1500GPa.txt')
dqdy_liq=np.loadtxt('EoS/mantle/dqdy_liq_Py_1500GPa.txt')
y_grid=np.loadtxt('EoS/mantle/y.txt')
P_solidus_liquidus=np.loadtxt('EoS/mantle/solid_P.txt')
S_liq_array=P_solidus_liquidus[:,2][:1500].copy()
S_sol_array=P_solidus_liquidus[:,1][:1500].copy()

P_grid_pv=np.loadtxt('EoS/mantle/P_pv.txt')
P_grid_ppv=np.loadtxt('EoS/mantle/P_ppv.txt')
P_grid_en=np.loadtxt('EoS/mantle/P_en.txt')
P_grid=P_solidus_liquidus[:,0][:1500].copy()

if load_file[0]<1.25:
    rho_Fel=np.loadtxt('EoS/Fe_core/rho_Fel_60GPa.txt')
    alpha_Fel=np.loadtxt('EoS/Fe_core/alpha_Fel_60GPa.txt')
    dqdy_Fel=np.loadtxt('EoS/Fe_core/dqdy_Fel_60GPa.txt')
    T_Fel=np.loadtxt('EoS/Fe_core/T_Fel_60GPa.txt')
    P_Fel=np.loadtxt('EoS/Fe_core/P_Fel_60GPa.txt')
    rho_Fea=np.loadtxt('EoS/Fe_core/rho_Fe16Si_60GPa.txt')
    alpha_Fea=np.loadtxt('EoS/Fe_core/alpha_Fe16Si_60GPa.txt')
    dqdy_Fea=np.loadtxt('EoS/Fe_core/dqdy_Fe16Si_60GPa.txt')
    T_Fea=np.loadtxt('EoS/Fe_core/T_Fe16Si_60GPa.txt')
    P_Fea=np.loadtxt('EoS/Fe_core/P_Fe16Si_60GPa.txt')
    loaded_T=np.loadtxt('EoS/Fe_core/Fe_adiabat_60GPa.txt')
    load_original_T=loaded_T.reshape(loaded_T.shape[0],loaded_T.shape[1]//995,995)#141
    x_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_xgrid_60GPa.txt')
    Tref_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_Tgrid_60GPa.txt')
    pre_adiabat_pressure=np.loadtxt('EoS/Fe_core/Fe_adiabat_Pgrid_60GPa.txt')
    f_adiabat=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, pre_adiabat_pressure), load_original_T)

    loaded_dTdT0=np.loadtxt('EoS/Fe_core/Fe_dTdT0_60GPa.txt')
    load_original_dTdT0=loaded_dTdT0.reshape(loaded_dTdT0.shape[0],loaded_dTdT0.shape[1]//995,995)
    loaded_dT0dP=np.loadtxt('EoS/Fe_core/Fe_dT0dP_60GPa.txt')
    load_original_dT0dP=loaded_dT0dP.reshape(loaded_dT0dP.shape[0],loaded_dT0dP.shape[1]//951,951)
    Tgrid_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_P_Tgridgrid_60GPa.txt')
    f_dT0dP=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, Tgrid_core_grid), load_original_dT0dP)

else:
    rho_Fel=np.loadtxt('EoS/Fe_core/rho_Fel.txt')
    alpha_Fel=np.loadtxt('EoS/Fe_core/alpha_Fel.txt')
    dqdy_Fel=np.loadtxt('EoS/Fe_core/dqdy_Fel.txt')
    T_Fel=np.loadtxt('EoS/Fe_core/T_Fel.txt')
    P_Fel=np.loadtxt('EoS/Fe_core/P_Fel.txt')
    rho_Fea=np.loadtxt('EoS/Fe_core/rho_Fe16Si.txt')
    alpha_Fea=np.loadtxt('EoS/Fe_core/alpha_Fe16Si.txt')
    dqdy_Fea=np.loadtxt('EoS/Fe_core/dqdy_Fe16Si.txt')
    T_Fea=np.loadtxt('EoS/Fe_core/T_Fe16Si.txt')
    P_Fea=np.loadtxt('EoS/Fe_core/P_Fe16Si.txt')
    loaded_T=np.loadtxt('EoS/Fe_core/Fe_adiabat.txt')
    load_original_T=loaded_T.reshape(loaded_T.shape[0],loaded_T.shape[1]//989,989)#141
    x_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_xgrid.txt')
    Tref_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_Tgrid.txt')
    pre_adiabat_pressure=np.loadtxt('EoS/Fe_core/Fe_adiabat_Pgrid.txt')
    f_adiabat=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, pre_adiabat_pressure), load_original_T)

    loaded_dTdT0=np.loadtxt('EoS/Fe_core/Fe_dTdT0.txt')
    load_original_dTdT0=loaded_dTdT0.reshape(loaded_dTdT0.shape[0],loaded_dTdT0.shape[1]//989,989)
    loaded_dT0dP=np.loadtxt('EoS/Fe_core/Fe_dT0dP.txt')
    load_original_dT0dP=loaded_dT0dP.reshape(loaded_dT0dP.shape[0],loaded_dT0dP.shape[1]//826,826)
    Tgrid_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_P_Tgridgrid.txt')
    f_dT0dP=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, Tgrid_core_grid), load_original_dT0dP)


Tlg,Plg=np.meshgrid(T_Fel,P_Fel,sparse=True)
Tag,Pag=np.meshgrid(T_Fea,P_Fea,sparse=True)
f_rho_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,rho_Fel)
f_rho_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,rho_Fea)
f_alpha_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,alpha_Fel)
f_alpha_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,alpha_Fea)
f_dqdy_Fel=interpolate.RectBivariateSpline(P_Fel,T_Fel,dqdy_Fel)
f_dqdy_Fea=interpolate.RectBivariateSpline(P_Fea,T_Fea,dqdy_Fea)



T_sol_pv=np.loadtxt('EoS/mantle/T_sol_pv_Py.txt')
rho_sol_pv=np.loadtxt('EoS/mantle/rho_sol_pv_Py.txt')
alpha_sol_pv=np.loadtxt('EoS/mantle/alpha_sol_pv_Py.txt')
dTdP_sol_pv=np.loadtxt('EoS/mantle/dTdP_sol_pv_Py.txt')
dqdy_sol_pv=np.loadtxt('EoS/mantle/dqdy_sol_pv_Py.txt')

T_sol_ppv=np.loadtxt('EoS/mantle/T_sol_ppv_Py.txt')
rho_sol_ppv=np.loadtxt('EoS/mantle/rho_sol_ppv_Py.txt')
alpha_sol_ppv=np.loadtxt('EoS/mantle/alpha_sol_ppv_Py.txt')
dTdP_sol_ppv=np.loadtxt('EoS/mantle/dTdP_sol_ppv_Py.txt')
dqdy_sol_ppv=np.loadtxt('EoS/mantle/dqdy_sol_ppv_Py.txt')

T_sol_en=np.loadtxt('EoS/mantle/T_sol_en_Py.txt')
rho_sol_en=np.loadtxt('EoS/mantle/rho_sol_en_Py.txt')
alpha_sol_en=np.loadtxt('EoS/mantle/alpha_sol_en_Py.txt')
dTdP_sol_en=np.loadtxt('EoS/mantle/dTdP_sol_en_Py.txt')
dqdy_sol_en=np.loadtxt('EoS/mantle/dqdy_sol_en_Py.txt')

T_mix_pv=np.loadtxt('EoS/mantle/T_mix_pv_Py.txt')
rho_mix_pv=np.loadtxt('EoS/mantle/rho_mix_pv_Py.txt')
alpha_mix_pv=np.loadtxt('EoS/mantle/alpha_mix_pv_Py.txt')
dTdP_mix_pv=np.loadtxt('EoS/mantle/dTdP_mix_pv_Py.txt')
dqdy_mix_pv=np.loadtxt('EoS/mantle/dqdy_mix_pv_Py.txt')
CP_mix_pv=np.loadtxt('EoS/mantle/CP_mix_pv_Py.txt')

T_mix_ppv=np.loadtxt('EoS/mantle/T_mix_ppv_Py.txt')
rho_mix_ppv=np.loadtxt('EoS/mantle/rho_mix_ppv_Py.txt')
alpha_mix_ppv=np.loadtxt('EoS/mantle/alpha_mix_ppv_Py.txt')
dTdP_mix_ppv=np.loadtxt('EoS/mantle/dTdP_mix_ppv_Py.txt')
dqdy_mix_ppv=np.loadtxt('EoS/mantle/dqdy_mix_ppv_Py.txt')
CP_mix_ppv=np.loadtxt('EoS/mantle/CP_mix_ppv_Py.txt')

T_mix_en=np.loadtxt('EoS/mantle/T_mix_en_Py.txt')
rho_mix_en=np.loadtxt('EoS/mantle/rho_mix_en_Py.txt')
alpha_mix_en=np.loadtxt('EoS/mantle/alpha_mix_en_Py.txt')
dTdP_mix_en=np.loadtxt('EoS/mantle/dTdP_mix_en_Py.txt')
dqdy_mix_en=np.loadtxt('EoS/mantle/dqdy_mix_en_Py.txt')
CP_mix_en=np.loadtxt('EoS/mantle/CP_mix_en_Py.txt')

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

#import a grid of Pc and Tc values. Using interpolated values as initial guesses for Pc and Tc
load_Pc=np.loadtxt('EoS/Guess_initial/Pc.txt')
load_Tc=np.loadtxt('EoS/Guess_initial/Tc.txt')
load_Mplgrid=np.loadtxt('EoS/Guess_initial/Mpl_grid.txt')
load_CMFgrid=np.loadtxt('EoS/Guess_initial/CMF_grid.txt')
f_Pc_i=interpolate.RectBivariateSpline(load_Mplgrid,load_CMFgrid,load_Pc)
f_Tc_i=interpolate.RectBivariateSpline(load_Mplgrid,load_CMFgrid,load_Tc)


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
cdef double Teq=load_file[8] # equilibrium temperature in K.
cdef double Q_rad_c_0=0.0 # Current day core radiogenic heating in W/kg.
cdef double P_surf=1e5 # Surface pressure in Pa.

cdef int zone=int((load_file[0]-1.0)*80.0+600.0) # total number of zones in the planet
cdef int c_z=int(load_file[1]*zone) # zones in the core
cdef int m_z=zone-c_z # zones in the mantle
cdef double P_c=f_Pc_i(load_file[0],load_file[1]*100.0)[0][0]#1000e9 # initial guess of the central pressure in Pa. Subsequent update in the code is the actual central pressure in Pa.
cdef double T_c=f_Tc_i(load_file[0],load_file[1]*100.0)[0][0]#10500.0 # Central temperature in K
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
    if i<c_z:
        phase[i]=1.0
    else:
        phase[i]=4.0

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
                        if s_array[i-1]>5075.0:
                            s_array[i-1]=5075.0
                            y_value=(s_array[i-1]-s_liq_val)/(S_max-s_liq_val)

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
                        phase[i]=4.0
                        y=(s_array[i]-s_liq_val)/(S_max-s_liq_val)
                        rho[i]=rho_Py_liq(pressure[i],y)[0][0]
                        alpha[i]=alpha_Py_liq(pressure[i],y)[0][0]
                        temperature[i]=T_Py_liq(pressure[i],y)[0][0]
                        dqdy[i]=dqdy_Py_liq(pressure[i],y)[0][0]
                        cP[i]=CP_Py_liq(pressure[i],y)[0][0]
                        dTdP[i]=dTdP_Py_liq(pressure[i],y)[0][0]
                        melt_frac[i]=1.0
                    logrho[i]=np.log(rho[i])

            if iteration%10==0:
                print('Iteration:%d Surface pressure:%2.2f bar' %(iteration,pressure[zone-1]/1e5))
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
print('Final check of surface pressure:%2.2f bar'%(rh['pressure'][-1]/1e5))

cdef double[:] PdV=np.zeros(zone);
cdef double[:] dEG=np.zeros(zone);
cdef double[:] dw=np.zeros(zone)
cdef double[:] new_V=np.zeros(zone);
cdef double[:] new_EG=np.zeros(zone);
cdef double[:] new_v_top=np.zeros(zone)
cdef double[:] kappa=np.zeros(zone)

cdef double[:] melt_frac=np.ones(zone)
cdef double[:] dsdr_array=np.ones(zone)*dsdr_c

cdef double[:] mass_cell=np.zeros(zone)
cdef double[:] x_cell=np.ones(zone)

cdef double[:] dPdr=np.zeros(zone)
cdef double[:] dxdr=np.zeros(zone)


for i in range(zone):
    if i==0:
        new_V[i]=4.0/3.0*math.pi*rh['radius'][i]**3.0
        mass_cell[i]=ri['mass'][0]/2.0
    else:
        new_V[i]=4.0/3.0*math.pi*rh['radius'][i]**3.0-4.0/3.0*math.pi*rh['radius'][i-1]**3.0
        mass_cell[i]=(ri['mass'][i-1]+ri['mass'][i])/2.0
    new_v_top[i]=4.0/3.0*math.pi*rh['radius'][i]**3.0

for i in range(zone):
    new_EG[i]=-ri['dm'][i]*ri['mass'][i]/rh['r_cell'][i]
    dPdr[i]=-G*ri['mass'][i]*rh['rho'][i]/rh['radius'][i]**2.0
    kappa[i]=10.0/(rh['rho'][i]*ri['cP'][i])

np.savetxt(results_foldername+'/profile/t0/henyey0.txt',np.transpose([rh['radius'],rh['pressure'],rh['r_cell'],rh['p_cell'],
    rh['rho'], rh['gravity'], rh['dqdy'], phase, rh['Area'],PdV,dEG,dw,new_V,new_EG,new_v_top,
                                       rh['radius'],rh['pressure'],rh['r_cell'],rh['p_cell'],rh['rho'], rh['gravity'],
                                       new_V,new_EG,new_v_top,rh['Area']]))
np.savetxt(results_foldername+'/profile/t0/structure0.txt',np.transpose([ri['temperature'],ri['T_cell'],melt_frac,ri['s_array'],rh['s_cell'],
                                         dsdr_array,ri['s_array'],rh['s_cell'],ri['mass'],mass_cell,x_cell]))
np.savetxt(results_foldername+'/profile/t0/property0.txt',np.transpose([ri['alpha'],ri['cP'],kappa,ri['dTdP'],dPdr,dxdr]))
np.savetxt(results_foldername+'/profile/t0/previous0.txt',np.transpose([0.0, 1.0, rh['P_c'], 0.0, ri['temperature'][c_z-1], ri['T_an_c'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, c_z, m_z]))


cpdef double T_simon(double P, double x):
    cdef double T0=6500.0*(P/340.0)**0.515
    cdef double x0=((1.0-x)/55.845)/((1.0-x)/55.845+x/28.0855) #molar fraction of Fe with S as impurity
    cdef double T=T0/(1.0-np.log(x0))
    return T

cdef double x_init=0.105
cdef double[:] T_Fe_melt=np.zeros(c_z)
cdef double[:] x_melt=np.zeros(c_z)

for i in range(c_z):
    if i==c_z-1:
        x=0.1519
    else:
        x=x_init*(M_pl*CMF-ri['mass'][0])/(M_pl*CMF-ri['mass'][i])
        if x>0.1519:
            x=0.1519
    x_melt[i]=x
    if load_file[0]==1.0:
        T_Fe_melt[i]=T_simon(rh['pressure'][i]/1e9+2.0,x)
    else:
        T_Fe_melt[i]=T_simon(rh['pressure'][i]/1e9,x)

cdef double[:] T_melt_P=rh['pressure'].copy()
if load_file[0]==1.0:
    for i in range(zone):
        T_melt_P[i]=T_melt_P[i]+2e9
T_melt_P[0]=2.0*T_melt_P[1]-T_melt_P[2]
T_Fe_melt[0]=2.0*T_Fe_melt[1]-T_Fe_melt[2]

np.savetxt(results_foldername+'/profile/t0/Fe_melt.txt',np.transpose([T_melt_P[:c_z],T_Fe_melt]),header='pressure, core melting temperature')

#### pre-tabulate surface flux table F(s)

cpdef double f_oc(double r1, double r2, double rx, double v1, double v2):
    cdef double value=v1-(r1-rx)*(v2-v1)/(r2-r1)
    return value

cpdef double f_viscosity(double T, double P, double density, double phase, double x, double rho_m, double rho_s, double width):
    cdef double A=1.67
    cdef double B=7.4e-17
    cdef double n=3.5
    cdef double E=5.0e+5
    cdef double V=1.0e-5
    cdef double R=8.314
    cdef double epsilon=1.0e-15
    cdef double y=0.0
    cdef double z=0.0
    if P<23.0*10.0**9.0:
        B=3.5e-15
        n=3.0
        E=4.3e+5
    cdef double eta_s=0.5*(1.0/B**(1.0/n))*np.exp((E+P*V)/(n*R*T))*epsilon**((1.0-n)/n)/density
    cdef double eta0=0.0
    cdef double p_decay=0.0
    if P>=125e9:
        eta0=1.05e34#1.9e21#
        E=7.8e5#1.62e5#
        p_decay=1100e9#1610e9#
        V=1.7e-6*np.exp(-P/p_decay)#1.4e-6*np.exp(-P/p_decay)#
        eta_s=eta0*np.exp((E+P*V)/(R*T)-E/(R*1600.0))/density
    cdef double eta_m=100.0/density
    cdef double value1=0.0
    cdef double value2=0.0
    cdef double value=0.0
    y=(x-0.4)/width
    z=0.5*(1.0+math.tanh(y))
    value=10.0**(z*math.log10(eta_m)+(1.0-z)*math.log10(eta_s))
    return value

cdef double Racr=660.0 # critical rayleigh number

cdef double[:] s=ri['s_array'],
cdef double[:] s_cell=rh['s_cell']
cdef double[:] P=rh['pressure']
cdef double[:] P_cell=rh['p_cell']
cdef double[:] R=rh['radius']
cdef double[:] R_cell=rh['r_cell']
cdef double[:] g=rh['gravity']

cdef double[:] g_cell=g.copy()
cdef double[:] s_sol=np.ones(len(g))
cdef double[:] s_sol_cell=np.ones(len(g))
cdef double[:] s_liq=np.ones(len(g))
cdef double[:] s_liq_cell=np.ones(len(g))

for i in range(len(g)):
    if i==0:
        g_cell[i]=g[i]/2.0
    else:
        g_cell[i]=(g[i-1]+g[i])/2.0
    if i>len(g)-10:
        s_sol[i]=S_sol_P(P[i]).tolist()
        s_sol_cell[i]=S_sol_P(P_cell[i]).tolist()
        s_liq[i]=S_liq_P(P[i]).tolist()
        s_liq_cell[i]=S_liq_P(P_cell[i]).tolist()

#cdef double[:] s_grid=np.linspace(rh['s_cell'][-1]+50.0,2800.0,8800)
cdef double[:] s_grid=np.linspace(5080.0,2800.0,8801)

cdef double[:] Fsurf_grid=np.zeros(len(s_grid))
cdef double[:] delta_BL_grid=np.zeros(len(s_grid))
cdef double[:] Tsurf_grid=np.zeros(len(s_grid))
cdef double[:] vissurf_grid=np.zeros(len(s_grid))

rtol=1e-3

#### this is only for the first grid point in s. the rest grid points in s will use delta_BL and T_s obtained from the previous grid point. 
# start with delta_BL=1cm
cdef double delta_BL=0.01
# provide an T_s as an initial guess for f = lambda x: x**4+k_l/sigma/old_delta_BL*x-Teq**4.0-k_l/sigma/old_delta_BL*T_BL
cdef double T_s=ri['temperature'][-1]-500.0
cdef double break_flag=0.0 # flag for breaking the for loop/while loop

cdef double rerr=1.0
cdef double old_T_s, old_delta_BL, R_BL 
cdef double g_BL, P_BL, sliq_BL, ssol_BL
cdef double y_BL, x_BL, T_BL, rho_BL, cP_BL, alpha_BL, nu_BL
cdef int i_r
cdef double smoothing_width=0.25
for i in range(0, len(s_grid)):
    rerr=1.0
    iteration=0
    while rerr>rtol:
        old_T_s=T_s
        old_delta_BL=delta_BL
        R_BL=R[-1]-old_delta_BL

        # using values at boundary[-2] and cell_center[-1] 
        # extrapolate to R=R[-1]-old_delta_BL ->(R_BL)
        g_BL=f_oc(R_cell[-1],R[-2],R_BL,g_cell[-1],g[-2])
        P_BL=f_oc(R_cell[-1],R[-2],R_BL,P_cell[-1],P[-2])
        sliq_BL=S_liq_P(P_BL).tolist()
        ssol_BL=S_sol_P(P_BL).tolist()

        # find properties at R_BL
        if s_grid[i]>=sliq_BL:
            y_BL=(s_grid[i]-sliq_BL)/(S_max-sliq_BL)
            x_BL=1.0
            T_BL=T_Py_liq(P_BL,y_BL)[0][0]
            rho_BL=rho_Py_liq(P_BL,y_BL)[0][0]
            cP_BL=CP_Py_liq(P_BL,y_BL)[0][0]
            alpha_BL=alpha_Py_liq(P_BL,y_BL)[0][0]
        elif s_grid[i]<=ssol_BL:
            print('should not happen')
        else:
            y_BL=(s_grid[i]-ssol_BL)/(sliq_BL-ssol_BL)
            x_BL=y_BL
            T_BL=T_Py_mix_en(P_BL,y_BL)[0][0]
            rho_BL=rho_Py_mix_en(P_BL,y_BL)[0][0]
            cP_BL=CP_Py_mix_en(P_BL,y_BL)[0][0]
            alpha_BL=alpha_Py_mix_en(P_BL,y_BL)[0][0]
        nu_BL=f_viscosity(T_BL, P_BL, rho_BL, 0.0, x_BL, 0.0,0.0,smoothing_width)

        # solving for new T_surface using new entropy and old delta_BL
        f = lambda x: x**4+k_en/sigma/old_delta_BL*x-Teq**4.0-k_en/sigma/old_delta_BL*T_BL
        T_s=fsolve(f,old_T_s)[0]

        # update delta_BL using new delta_T_BL=T_BL-T_s
        delta_T_BL=T_BL-T_s
        delta_BL=(Racr*nu_BL*(k_en/(rho_BL*cP_BL))/(alpha_BL*g_BL*delta_T_BL))**(1.0/3.0)
        
        rerr=abs(delta_BL-old_delta_BL)/old_delta_BL
        
        if iteration>20:
            if abs(delta_BL-delta_BL_grid[i-1])<abs(old_delta_BL-delta_BL_grid[i-1]):
                break 
            else:
                delta_BL=old_delta_BL
                T_s=old_T_s
                break
        if delta_BL>R[-1]-R_cell[-1]:
            i_r=i+1
            break_flag=1.0
            break    

        iteration=iteration+1
    Fsurf_grid[i]=k_en*delta_T_BL/delta_BL
    delta_BL_grid[i]=delta_BL
    Tsurf_grid[i]=T_s
    vissurf_grid[i]=nu_BL

    if break_flag==1.0:
        break

from scipy.interpolate import UnivariateSpline
dFds=np.zeros(len(s_grid))
s_array=np.linspace(5080.0,2800.0,8801)
Fsurf_array=np.zeros(len(s_array))
for i in range(len(s_array)):
    Fsurf_array[i]=Fsurf_grid[i]


F_of_s=UnivariateSpline(s_array[::-1], Fsurf_array[::-1])
f_dFds=F_of_s.derivative()
for i in range(len(dFds)):
    dFds[i]=f_dFds(s_grid[i]).tolist()

np.savetxt(results_foldername+'/profile/t0/Fsurf.txt',np.transpose([s_grid[:i_r],Fsurf_grid[:i_r],delta_BL_grid[:i_r],Tsurf_grid[:i_r], dFds[:i_r], vissurf_grid[:i_r]]),header='surface entropy, surface flux, surface boundary layer thickness, surface temperature, dFsurf/ds')
