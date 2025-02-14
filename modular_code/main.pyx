# main.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc cimport math

from planet cimport Planet
from structure cimport RK4, henyey_solver
import eos_tables

from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import fsolve

####### Read in values from input.txt

load_file=np.loadtxt('input.txt')
results_foldername='results_Mpl'+str(load_file[0])+'_CMF'+str(load_file[1])+'_time'+str(load_file[2])+'_Qrad'+str(load_file[3])+'_'+str(load_file[4])+'_'+str(load_file[5])+'_'+str(load_file[6])+'_Teq'+str(load_file[8])+'_Qradc'+str(load_file[9])+'_eta'+str(load_file[7])+'_mzmulti'+str(load_file[10])

####### Creat local references for eos interpolators in eos.py

cdef object f_adiabat
cdef object f_dT0dP
cdef object f_rho_Fel
cdef object f_rho_Fea
cdef object f_rho_Fes
cdef object f_alpha_Fel
cdef object f_alpha_Fea
cdef object f_alpha_Fes
cdef object f_dqdy_Fel
cdef object f_dqdy_Fea
cdef object f_dqdy_Fes
cdef object dTdT0_cmb_interp
cdef object T_interp
P_core_grid = eos_tables.P_core_grid

if load_file[0]<1.25:
    f_adiabat = eos_tables.f_adiabat60
    f_dT0dP = eos_tables.f_dT0dP60
    f_rho_Fel = eos_tables.f_rho_Fel60
    f_rho_Fea = eos_tables.f_rho_Fea60
    f_rho_Fes = eos_tables.f_rho_Fes60
    f_alpha_Fel = eos_tables.f_alpha_Fel60
    f_alpha_Fea = eos_tables.f_alpha_Fea60
    f_alpha_Fes = eos_tables.f_alpha_Fes60
    f_dqdy_Fel = eos_tables.f_dqdy_Fel60
    f_dqdy_Fea = eos_tables.f_dqdy_Fea60
    f_dqdy_Fes = eos_tables.f_dqdy_Fes60
    dTdT0_cmb_interp = eos_tables.dTdT0_cmb_interp60
    T_interp = eos_tables.T_interp60
else:
    f_adiabat = eos_tables.f_adiabat
    f_dT0dP = eos_tables.f_dT0dP
    f_rho_Fel = eos_tables.f_rho_Fel
    f_rho_Fea = eos_tables.f_rho_Fea
    f_rho_Fes = eos_tables.f_rho_Fes
    f_alpha_Fel = eos_tables.f_alpha_Fel
    f_alpha_Fea = eos_tables.f_alpha_Fea
    f_alpha_Fes = eos_tables.f_alpha_Fes
    f_dqdy_Fel = eos_tables.f_dqdy_Fel
    f_dqdy_Fea = eos_tables.f_dqdy_Fea
    f_dqdy_Fes = eos_tables.f_dqdy_Fes
    dTdT0_cmb_interp = eos_tables.dTdT0_cmb_interp
    T_interp = eos_tables.T_interp

cdef object S_liq_P=eos_tables.S_liq_P
cdef object S_sol_P=eos_tables.S_sol_P
cdef object T_Py_liq=eos_tables.T_Py_liq
cdef object rho_Py_liq=eos_tables.rho_Py_liq
cdef object CP_Py_liq=eos_tables.CP_Py_liq
cdef object alpha_Py_liq=eos_tables.alpha_Py_liq
cdef object dTdP_Py_liq=eos_tables.dTdP_Py_liq
cdef object dqdy_Py_liq=eos_tables.dqdy_Py_liq

cdef object T_Py_sol_pv=eos_tables.T_Py_sol_pv
cdef object rho_Py_sol_pv=eos_tables.rho_Py_sol_pv
cdef object alpha_Py_sol_pv=eos_tables.alpha_Py_sol_pv
cdef object dTdP_Py_sol_pv=eos_tables.dTdP_Py_sol_pv
cdef object dqdy_Py_sol_pv=eos_tables.dqdy_Py_sol_pv

cdef object T_Py_sol_en=eos_tables.T_Py_sol_en
cdef object rho_Py_sol_en=eos_tables.rho_Py_sol_en
cdef object alpha_Py_sol_en=eos_tables.alpha_Py_sol_en
cdef object dTdP_Py_sol_en=eos_tables.dTdP_Py_sol_en
cdef object dqdy_Py_sol_en=eos_tables.dqdy_Py_sol_en

cdef object T_Py_sol_ppv=eos_tables.T_Py_sol_ppv
cdef object rho_Py_sol_ppv=eos_tables.rho_Py_sol_ppv
cdef object alpha_Py_sol_ppv=eos_tables.alpha_Py_sol_ppv
cdef object dTdP_Py_sol_ppv=eos_tables.dTdP_Py_sol_ppv
cdef object dqdy_Py_sol_ppv=eos_tables.dqdy_Py_sol_ppv

cdef object T_Py_mix_en=eos_tables.T_Py_mix_en
cdef object rho_Py_mix_en=eos_tables.rho_Py_mix_en
cdef object CP_Py_mix_en=eos_tables.CP_Py_mix_en
cdef object alpha_Py_mix_en=eos_tables.alpha_Py_mix_en
cdef object dTdP_Py_mix_en=eos_tables.dTdP_Py_mix_en
cdef object dqdy_Py_mix_en=eos_tables.dqdy_Py_mix_en

cdef object T_Py_mix_ppv=eos_tables.T_Py_mix_ppv
cdef object rho_Py_mix_ppv=eos_tables.rho_Py_mix_ppv
cdef object alpha_Py_mix_ppv=eos_tables.alpha_Py_mix_ppv
cdef object dTdP_Py_mix_ppv=eos_tables.dTdP_Py_mix_ppv
cdef object dqdy_Py_mix_ppv=eos_tables.dqdy_Py_mix_ppv

cdef object T_Py_mix_pv=eos_tables.T_Py_mix_pv
cdef object rho_Py_mix_pv=eos_tables.rho_Py_mix_pv
cdef object alpha_Py_mix_pv=eos_tables.alpha_Py_mix_pv
cdef object dTdP_Py_mix_pv=eos_tables.dTdP_Py_mix_pv
cdef object dqdy_Py_mix_pv=eos_tables.dqdy_Py_mix_pv

cdef object T_interp_2d_liq = eos_tables.T_interp_2d_liq
cdef object T_interp_2d_sol_pv = eos_tables.T_interp_2d_sol_pv
cdef object T_interp_2d_sol_ppv = eos_tables.T_interp_2d_sol_ppv
cdef object T_interp_2d_sol_en = eos_tables.T_interp_2d_sol_en
cdef object T_interp_2d_mix_pv = eos_tables.T_interp_2d_mix_pv
cdef object T_interp_2d_mix_ppv = eos_tables.T_interp_2d_mix_ppv
cdef object T_interp_2d_mix_en = eos_tables.T_interp_2d_mix_en
cdef object T_interp_2d_dy_liq = eos_tables.T_interp_2d_dy_liq
cdef object T_interp_2d_dy_sol_pv = eos_tables.T_interp_2d_dy_sol_pv
cdef object T_interp_2d_dy_sol_ppv = eos_tables.T_interp_2d_dy_sol_ppv
cdef object T_interp_2d_dy_sol_en = eos_tables.T_interp_2d_dy_sol_en
cdef object T_interp_2d_dy_mix_pv = eos_tables.T_interp_2d_dy_mix_pv
cdef object T_interp_2d_dy_mix_ppv = eos_tables.T_interp_2d_dy_mix_ppv
cdef object T_interp_2d_dy_mix_en = eos_tables.T_interp_2d_dy_mix_en
cdef object T_interp_2d_o_liq = eos_tables.T_interp_2d_o_liq
cdef object T_interp_2d_o_sol_pv = eos_tables.T_interp_2d_o_sol_pv
cdef object T_interp_2d_o_sol_ppv = eos_tables.T_interp_2d_o_sol_ppv
cdef object T_interp_2d_o_sol_en = eos_tables.T_interp_2d_o_sol_en
cdef object T_interp_2d_o_mix_pv = eos_tables.T_interp_2d_o_mix_pv
cdef object T_interp_2d_o_mix_ppv = eos_tables.T_interp_2d_o_mix_ppv
cdef object T_interp_2d_o_mix_en = eos_tables.T_interp_2d_o_mix_en
cdef object T_interp_2d__liq = eos_tables.T_interp_2d__liq
cdef object T_interp_2d__mix_pv = eos_tables.T_interp_2d__mix_pv
cdef object T_interp_2d__mix_ppv = eos_tables.T_interp_2d__mix_ppv
cdef object T_interp_2d__mix_en = eos_tables.T_interp_2d__mix_en
cdef object T_interp_2d_pha_liq = eos_tables.T_interp_2d_pha_liq
cdef object T_interp_2d_pha_sol_pv = eos_tables.T_interp_2d_pha_sol_pv
cdef object T_interp_2d_pha_sol_ppv = eos_tables.T_interp_2d_pha_sol_ppv
cdef object T_interp_2d_pha_sol_en = eos_tables.T_interp_2d_pha_sol_en
cdef object T_interp_2d_pha_mix_pv = eos_tables.T_interp_2d_pha_mix_pv
cdef object T_interp_2d_pha_mix_ppv = eos_tables.T_interp_2d_pha_mix_ppv
cdef object T_interp_2d_pha_mix_en = eos_tables.T_interp_2d_pha_mix_en
cdef object T_interp_2d_dP_liq = eos_tables.T_interp_2d_dP_liq
cdef object T_interp_2d_dP_sol_pv = eos_tables.T_interp_2d_dP_sol_pv
cdef object T_interp_2d_dP_sol_ppv = eos_tables.T_interp_2d_dP_sol_ppv
cdef object T_interp_2d_dP_sol_en = eos_tables.T_interp_2d_dP_sol_en
cdef object T_interp_2d_dP_mix_pv = eos_tables.T_interp_2d_dP_mix_pv
cdef object T_interp_2d_dP_mix_ppv = eos_tables.T_interp_2d_dP_mix_ppv
cdef object T_interp_2d_dP_mix_en = eos_tables.T_interp_2d_dP_mix_en
cdef object interp_2d_rho_Fel = eos_tables.interp_2d_rho_Fel
cdef object interp_2d_rho_Fes = eos_tables.interp_2d_rho_Fes
cdef object interp_2d_rho_Fea = eos_tables.interp_2d_rho_Fea
cdef object interp_2d_alpha_Fel = eos_tables.interp_2d_alpha_Fel
cdef object interp_2d_alpha_Fes = eos_tables.interp_2d_alpha_Fes
cdef object interp_2d_alpha_Fea = eos_tables.interp_2d_alpha_Fea
cdef object interp_2d_dqdy_Fel = eos_tables.interp_2d_dqdy_Fel
cdef object interp_2d_dqdy_Fes = eos_tables.interp_2d_dqdy_Fes
cdef object interp_2d_dqdy_Fea = eos_tables.interp_2d_dqdy_Fea
cdef object interp_2d_rho_Fel_a = eos_tables.interp_2d_rho_Fel_a
cdef object interp_2d_rho_Fea_a = eos_tables.interp_2d_rho_Fea_a
cdef object interp_2d_alpha_Fel_a = eos_tables.interp_2d_alpha_Fel_a
cdef object interp_2d_alpha_Fea_a = eos_tables.interp_2d_alpha_Fea_a

cdef Py_ssize_t iteration, i

#import a grid of Pc and Tc values. Using interpolated values as initial guesses for Pc and Tc
load_Pc = np.loadtxt('../EoS/Guess_initial/Pc.txt')
load_Tc = np.loadtxt('../EoS/Guess_initial/Tc.txt')
load_Mplgrid = np.loadtxt('../EoS/Guess_initial/Mpl_grid.txt')
load_CMFgrid = np.loadtxt('../EoS/Guess_initial/CMF_grid.txt')
f_Pc = interpolate.RectBivariateSpline(load_Mplgrid,load_CMFgrid,load_Pc)
f_Tc = interpolate.RectBivariateSpline(load_Mplgrid,load_CMFgrid,load_Tc)

cdef double M_pl=load_file[0]*5.972e24 # planet mass in kg
cdef double CMF=load_file[1] # core mass fraction
cdef double t_end=load_file[2]*86400.0*365.0*1e9 # evolutionary time for the simulation in second
cdef dict qrad={} # radiogenic heating in the mantle. Relative to Earth's current day value in W/kg.
qrad['K']=load_file[3]  # potassium 40
qrad['Th']=load_file[4] # Thorium
qrad['U8']=load_file[5] # Uranium 238
qrad['U5']=load_file[6] # Uranium 235

cdef double x_c=0.105 # concentration of light elements in the core by mass.
cdef double Teq=load_file[8] # equilibrium temperature in K.
cdef double Qrad_c0=0.0 # Current day core radiogenic heating in W/kg.
cdef double P_surf=1e5 # Surface pressure in Pa.

cdef int c_z=int(100*load_file[0]+(load_file[1]-0.1)*250) # zones in the core
cdef int m_z=int((200+10*load_file[0])*load_file[10]) # zones in the mantle
cdef int zone=int(c_z+m_z) # total number of zones in the planet

cdef double P_c=f_Pc(load_file[0],load_file[1]*100.0)[0][0]+20e9 # initial guess of the central pressure in Pa. Subsequent update in the code is the actual central pressure in Pa.
cdef double T_c=f_Tc(load_file[0],load_file[1]*100.0)[0][0]+500.0 # Central temperature in K

#### variable needed for structure.RK4
cdef double rtol_RK4 = 10.0
cdef double rtol_henyey = 1e-6
cdef double d_Pc = 1.0
cdef double c_array_start=2.0
cdef double ms_array_0=0.0
cdef double dsdr_c = -1e-6
cdef double initial = 1.0
if load_file[0]>=3.0:
    ms_array_0=-1.0-(load_file[0]-3.0)*0.1
else:
    ms_array_0=-1.0-(load_file[0]-3.0)*0.5
cdef double m_array_start=ms_array_0+(load_file[1]-0.1)*1.0#ms_array[int((load_file[1]+1e-10-0.1)*10.0)]


###### initialize a planet object. attributes include both input and output parameters. 
print('Initializing the planet')
cdef Planet planet = Planet(M_pl, CMF, Teq, t_end, x_c, P_c, T_c, P_surf, m_z, c_z)
print('Solving for interior structure')
print('4th order Runge-kutta (shooting technique)')
RK4(planet,m_array_start,c_array_start,d_Pc,rtol_RK4)
print('Henyey relaxation scheme to refine the solution')
henyey_solver(planet,dsdr_c,initial,rtol_henyey)
print('Tabulate surface flux as a function of entropy')

















