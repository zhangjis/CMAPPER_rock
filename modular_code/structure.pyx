# integration.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc cimport math
cimport cython
from scipy.optimize import fsolve

import eos_tables
from planet cimport Planet

cdef double G = 6.67430e-11
cdef double pi = 3.141592653589793
cdef double mf_l = 0.16
cdef double dsdr_c=-1e-6 # initial entropy gradient (an arbitrary choice)
cdef double S_max=5100.0#4739.0#5597.5#5384   # max of specific entropy in the EoS table for silicate. J/K/kg
cdef double S_min=100.0 # min of specific entropy in the EoS table for silicate. J/K/kg
cdef double C_P_liquidFe=840.0

####### Read in values from input.txt

load_file=np.loadtxt('input.txt')

###### create local references of eos interpolators in eos_tables. 

cdef object f_adiabat
cdef object f_dT0dP
cdef object f_rho_Fel
cdef object f_rho_Fea
cdef object f_alpha_Fel
cdef object f_alpha_Fea
cdef object f_dqdy_Fel
cdef object f_dqdy_Fea

if load_file[0] < 1.25:
    f_adiabat = eos_tables.f_adiabat60
    f_dT0dP = eos_tables.f_dT0dP60
    f_rho_Fel = eos_tables.f_rho_Fel60_structure
    f_rho_Fea = eos_tables.f_rho_Fea60_structure
    f_alpha_Fel = eos_tables.f_alpha_Fel60_structure
    f_alpha_Fea = eos_tables.f_alpha_Fea60_structure
    f_dqdy_Fel = eos_tables.f_dqdy_Fel60_structure
    f_dqdy_Fea = eos_tables.f_dqdy_Fea60_structure
else:
    f_adiabat = eos_tables.f_adiabat
    f_dT0dP = eos_tables.f_dT0dP
    f_rho_Fel = eos_tables.f_rho_Fel_structure
    f_rho_Fea = eos_tables.f_rho_Fea_structure
    f_alpha_Fel = eos_tables.f_alpha_Fel_structure
    f_alpha_Fea = eos_tables.f_alpha_Fea_structure
    f_dqdy_Fel = eos_tables.f_dqdy_Fel_structure
    f_dqdy_Fea = eos_tables.f_dqdy_Fea_structure

cdef object S_liq_P=eos_tables.S_liq_P#interpolate.interp1d(P_grid,S_liq_array)
cdef object S_sol_P=eos_tables.S_sol_P#interpolate.interp1d(P_grid,S_sol_array)
cdef object T_Py_liq=eos_tables.T_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,T_liq)
cdef object rho_Py_liq=eos_tables.rho_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,rho_liq)
cdef object CP_Py_liq=eos_tables.CP_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,CP_liq)
cdef object alpha_Py_liq=eos_tables.alpha_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,alpha_liq)
cdef object dTdP_Py_liq=eos_tables.dTdP_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,dTdP_liq)
cdef object dqdy_Py_liq=eos_tables.dqdy_Py_liq#interpolate.RectBivariateSpline(P_grid,y_grid,dqdy_liq)

cdef object T_Py_sol_pv=eos_tables.T_Py_sol_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,T_sol_pv)
cdef object rho_Py_sol_pv=eos_tables.rho_Py_sol_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,rho_sol_pv)
cdef object alpha_Py_sol_pv=eos_tables.alpha_Py_sol_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,alpha_sol_pv)
cdef object dTdP_Py_sol_pv=eos_tables.dTdP_Py_sol_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,dTdP_sol_pv)
cdef object dqdy_Py_sol_pv=eos_tables.dqdy_Py_sol_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,dqdy_sol_pv)

cdef object T_Py_sol_en=eos_tables.T_Py_sol_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,T_sol_en)
cdef object rho_Py_sol_en=eos_tables.rho_Py_sol_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,rho_sol_en)
cdef object alpha_Py_sol_en=eos_tables.alpha_Py_sol_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,alpha_sol_en)
cdef object dTdP_Py_sol_en=eos_tables.dTdP_Py_sol_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,dTdP_sol_en)
cdef object dqdy_Py_sol_en=eos_tables.dqdy_Py_sol_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,dqdy_sol_en)

cdef object T_Py_sol_ppv=eos_tables.T_Py_sol_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,T_sol_ppv)
cdef object rho_Py_sol_ppv=eos_tables.rho_Py_sol_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,rho_sol_ppv)
cdef object alpha_Py_sol_ppv=eos_tables.alpha_Py_sol_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,alpha_sol_ppv)
cdef object dTdP_Py_sol_ppv=eos_tables.dTdP_Py_sol_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dTdP_sol_ppv)
cdef object dqdy_Py_sol_ppv=eos_tables.dqdy_Py_sol_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dqdy_sol_ppv)

cdef object T_Py_mix_en=eos_tables.T_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,T_mix_en)
cdef object rho_Py_mix_en=eos_tables.rho_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,rho_mix_en)
cdef object CP_Py_mix_en=eos_tables.CP_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,CP_mix_en)
cdef object alpha_Py_mix_en=eos_tables.alpha_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,alpha_mix_en)
cdef object dTdP_Py_mix_en=eos_tables.dTdP_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,dTdP_mix_en)
cdef object dqdy_Py_mix_en=eos_tables.dqdy_Py_mix_en#interpolate.RectBivariateSpline(P_grid_en,y_grid,dqdy_mix_en)

cdef object T_Py_mix_ppv=eos_tables.T_Py_mix_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,T_mix_ppv)
cdef object rho_Py_mix_ppv=eos_tables.rho_Py_mix_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,rho_mix_ppv)
cdef object alpha_Py_mix_ppv=eos_tables.alpha_Py_mix_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,alpha_mix_ppv)
cdef object dTdP_Py_mix_ppv=eos_tables.dTdP_Py_mix_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dTdP_mix_ppv)
cdef object dqdy_Py_mix_ppv=eos_tables.dqdy_Py_mix_ppv#interpolate.RectBivariateSpline(P_grid_ppv,y_grid,dqdy_mix_ppv)

cdef object T_Py_mix_pv=eos_tables.T_Py_mix_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,T_mix_pv)
cdef object rho_Py_mix_pv=eos_tables.rho_Py_mix_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,rho_mix_pv)
cdef object alpha_Py_mix_pv=eos_tables.alpha_Py_mix_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,alpha_mix_pv)
cdef object dTdP_Py_mix_pv=eos_tables.dTdP_Py_mix_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,dTdP_mix_pv)
cdef object dqdy_Py_mix_pv=eos_tables.dqdy_Py_mix_pv#interpolate.RectBivariateSpline(P_grid_pv,y_grid,dqdy_mix_pv)

cpdef double dlnrdm(double r, double p, double density): # mass conservation equation
    return 1.0/(4.0*math.pi*r**3.0*density)

cpdef double dlnPdm(double r, double p, double M): # hydrostatic equilibrium
    return -G*M/(4.0*math.pi*r**4.0*p)

cpdef double f_r0(double h, double rho_c): # get the radius at the first zone above the center of the planet
    return (3.0*h/(4.0*math.pi*rho_c))**(1.0/3.0)

cpdef double f_p0(double P_c, double h, double rho_c): # get the pressure at the first zone above the center of the planet
    return P_c-3.0*G/(8.0*math.pi)*(4.0*math.pi*rho_c/3.0)**(4.0/3.0)*(h)**(2.0/3.0)

cpdef double f_Tan_c(double x, double x_c, double P_c, double T_c): # f_s_c and return_s_c together find the entropy temperature corresponding to the input central temperature
    cdef double x_alloy=x_c/mf_l
    return f_adiabat([x_alloy,x,P_c])[0]-T_c

cpdef double return_Tan_c(double guess, double x_c, double P_c, double T_c):
    return fsolve(f_Tan_c,x0=guess,args=(x_c, P_c, T_c))[0]

cpdef double f_T_Py(double x,double P,double T):
    return T_Py_liq(P,x)[0][0]-T

cpdef double return_T_Py(double guess, double P, double T):
    return fsolve(f_T_Py,x0=guess,args=(P,T))[0] 

cpdef double rho_mix(double x, double rho_l, double rho_s):
    # return the average density of mixtures using volume additive rules.
    return (x/rho_l+(1.0-x)/rho_s)**(-1.0)

cpdef double alpha_mix(double x,double alpha_l,double alpha_s,double rho,double rho_l,double rho_s):
    # return the average thermal expansion coefficient of mixtures using volume additive rules.
    return x*rho/rho_l*alpha_l+(1.0-x)*rho/rho_s*alpha_s

cpdef double dqdy_mix(double x, double rho_tot, double rho_l, double rho_s, double dqdy_l, double dqdy_s, double pressure):
    # return the dlog(rho)/dlog(P) of mixtures
    if pressure==0.0:
        pressure=10.0**(-6.0)
    cdef double drhodP_l, drhodP_s, value1, value2
    drhodP_l=dqdy_l*rho_l/pressure
    drhodP_s=dqdy_s*rho_s/pressure
    value1=x*rho_tot**(2.0)*rho_l**(-2.0)*drhodP_l
    value2=(1.0-x)*rho_tot**(2.0)*rho_s**(-2.0)*drhodP_s
    return (value1+value2)*pressure/rho_tot

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void f_M_dm(Planet planet, double m_array_start, double c_array_start):
    # return the mass profile of the planet (1D), mass for enclosed mass and h for mass of shells
    cdef Py_ssize_t i
    cdef int zone = planet.c_z + planet.m_z
    cdef double c_m = planet.M_pl * planet.CMF
    cdef double m_m = planet.M_pl - c_m
    cdef double h_c = c_m / planet.c_z
    cdef double h_m = m_m / planet.m_z
    #cdef double[:] mass = np.zeros(zone)
    #cdef double[:] h = np.zeros(zone)
    
    cdef int len_top=int(planet.m_z*0.5)
    cdef int len_bot=planet.m_z-len_top
    cdef double[:] top_array=np.linspace(4.0,m_array_start,len_top)
    cdef double[:] bot_array=np.linspace(m_array_start,4.0,len_bot)
    cdef double[:] mantle_tanh=np.zeros(planet.m_z)
    cdef double[:] mantle_norm=np.zeros(planet.m_z)
    cdef double mantle_tanh_sum=0.0
    for i in range(planet.m_z):
        if i<len_bot:
            mantle_tanh[i]=0.5*(1.0+math.tanh(bot_array[i]))
        else:
            mantle_tanh[i]=0.5*(1.0+math.tanh(top_array[i-len_bot]))
        mantle_tanh_sum=mantle_tanh_sum+mantle_tanh[i]
    for i in range(planet.m_z):
        mantle_norm[i]=mantle_tanh[i]/mantle_tanh_sum*m_m
    
    cdef double[:] c_array=np.linspace(c_array_start,4.0,planet.c_z)
    cdef double[:] c_tanh=np.zeros(planet.c_z)
    cdef double[:] c_norm=np.zeros(planet.c_z)
    cdef double c_tanh_sum=0.0
    for i in range(planet.c_z):
        c_tanh[i]=0.5*(1.0+math.tanh(c_array[i]))
        c_tanh_sum=c_tanh_sum+c_tanh[i]
    for i in range(planet.c_z):
        c_norm[i]=c_tanh[i]/c_tanh_sum*c_m


    if planet.M_pl<1.5*5.972e24 and planet.CMF>0.5:
        planet.mass[0]=h_c
        planet.h[0]=h_c  
    else:
        planet.mass[0]=c_norm[0]
        planet.h[0]=c_norm[0]    
    for i in range(1,zone):
        if i<planet.c_z:
            if planet.M_pl<1.5*5.972e24 and planet.CMF>0.5:
                planet.mass[i]=planet.mass[i-1]+h_c
                planet.h[i]=h_c
            else:
                planet.mass[i]=planet.mass[i-1]+c_norm[i]
                planet.h[i]=c_norm[i]
        elif i>=planet.c_z and i<planet.c_z+planet.m_z:
            if planet.M_pl<1.5*5.972e24 and planet.CMF>0.5:
                planet.mass[i]=planet.mass[i-1]+h_m
                planet.h[i]=h_m
            else:
                planet.mass[i]=planet.mass[i-1]+mantle_norm[i-planet.c_z]
                planet.h[i]=mantle_norm[i-planet.c_z]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void RK4(Planet planet,
    double m_array_start,
    double c_array_start,
    double d_Pc,
    double rtol
    ):
    """
    Integrates the structure equations from the center to the surface
    over a mass grid. 

    It will reiterative a few times until the surface boundary condition satifies rtol.
    
    Returns:
      (r, P, rho, g, T, S, cp, alpha, T_cell)
    """
    cdef int zone = planet.c_z + planet.m_z 

    f_M_dm(planet, m_array_start, c_array_start) # update planet.mass and planet.h
    planet.pressure[zone-1]=1e9 # arbitrary value to initialize the condition for the while loop.

    cdef double x_alloy=planet.x_c/mf_l
    cdef double rho_l, rho_a, dqdy_l, dqdy_a, alpha_l ,alpha_a
    cdef double r0, p0, rho0
    cdef double k1r, k1p, k2r, k2p, k3r, k3p, k4r, k4p
    cdef double rho_v
    cdef double s_sol_val, s_liq_val, y_value, y, s_new

    cdef Py_ssize_t iteration=0
    cdef Py_ssize_t i

    cdef double T_an_c_guess=6000.0
    while abs(planet.pressure[zone-1]-planet.P_surf)/planet.P_surf>rtol:
        planet.P_c=planet.P_c-d_Pc*(planet.pressure[zone-1]-planet.P_surf)
        planet.T_an_c=return_Tan_c(T_an_c_guess, planet.x_c, planet.P_c, planet.T_c)
        rho_l=f_rho_Fel(planet.P_c,planet.T_an_c)[0][0]
        rho_a=f_rho_Fea(planet.P_c,planet.T_an_c)[0][0]
        planet.rho_c=rho_mix(x_alloy,rho_a,rho_l)
        dqdy_l=f_dqdy_Fel(planet.P_c,planet.T_an_c)[0][0]
        dqdy_a=f_dqdy_Fea(planet.P_c,planet.T_an_c)[0][0]
        planet.dqdy_c=dqdy_mix(x_alloy,planet.rho_c,rho_a,rho_l,dqdy_a,dqdy_l,planet.P_c)
        
        r0=f_r0(planet.mass[0],planet.rho_c)
        p0=f_p0(planet.P_c,planet.mass[0],planet.rho_c)
        rho_l=f_rho_Fel(p0,planet.T_an_c)[0][0]
        rho_a=f_rho_Fea(p0,planet.T_an_c)[0][0]
        rho0=rho_mix(x_alloy,rho_a,rho_l)
        planet.temperature[0]=planet.T_c
        planet.radius[0]=r0
        planet.logr[0]=math.log(r0)
        planet.pressure[0]=p0
        planet.logp[0]=math.log(p0)
        rho_l=f_rho_Fel(p0,planet.T_an_c)[0][0]
        rho_a=f_rho_Fea(p0,planet.T_an_c)[0][0]
        rho0=rho_mix(x_alloy,rho_a,rho_l)
        planet.rho[0]=rho0
        planet.logrho[0]=math.log(rho0)
        planet.gravity[0]=G*planet.mass[0]/r0**2.0
        dqdy_l=f_dqdy_Fel(p0,planet.T_an_c)[0][0]
        dqdy_a=f_dqdy_Fea(p0,planet.T_an_c)[0][0]
        planet.dqdy[0]=dqdy_mix(x_alloy,rho0,rho_a,rho_l,dqdy_a,dqdy_l,p0)
        alpha_l=f_alpha_Fel(p0,planet.T_an_c)[0][0]
        alpha_a=f_alpha_Fea(p0,planet.T_an_c)[0][0]
        planet.alpha[0]=alpha_mix(x_alloy,alpha_a,alpha_l,rho0,rho_a,rho_l)
        planet.cP[0]=C_P_liquidFe

        for i in range(1, int(zone)):
            if i<=planet.c_z:
                k1r=dlnrdm(planet.radius[i-1]             , planet.pressure[i-1]             , planet.rho[i-1])
                k1p=dlnPdm(planet.radius[i-1]             , planet.pressure[i-1]             , planet.mass[i-1])
                rho_l=f_rho_Fel(planet.pressure[i-1]+planet.h[i]/2.0*k1p,planet.T_an_c)[0][0]
                rho_a=f_rho_Fea(planet.pressure[i-1]+planet.h[i]/2.0*k1p,planet.T_an_c)[0][0]
                rho_v=rho_mix(x_alloy,rho_a,rho_l)
                k2r=dlnrdm(planet.radius[i-1]+planet.h[i]/2.0*k1r, planet.pressure[i-1]+planet.h[i]/2.0*k1p, rho_v)
                k2p=dlnPdm(planet.radius[i-1]+planet.h[i]/2.0*k1r, planet.pressure[i-1]+planet.h[i]/2.0*k1p, planet.mass[i-1]+planet.h[i]/2.0)
                rho_l=f_rho_Fel(planet.pressure[i-1]+planet.h[i]/2.0*k2p,planet.T_an_c)[0][0]
                rho_a=f_rho_Fea(planet.pressure[i-1]+planet.h[i]/2.0*k2p,planet.T_an_c)[0][0]
                rho_v=rho_mix(x_alloy,rho_a,rho_l)
                k3r=dlnrdm(planet.radius[i-1]+planet.h[i]/2.0*k2r, planet.pressure[i-1]+planet.h[i]/2.0*k2p, rho_v)
                k3p=dlnPdm(planet.radius[i-1]+planet.h[i]/2.0*k2r, planet.pressure[i-1]+planet.h[i]/2.0*k2p, planet.mass[i-1]+planet.h[i]/2.0)
                rho_l=f_rho_Fel(planet.pressure[i-1]+planet.h[i]*k3p,planet.T_an_c)[0][0]
                rho_a=f_rho_Fea(planet.pressure[i-1]+planet.h[i]*k3p,planet.T_an_c)[0][0]
                rho_v=rho_mix(x_alloy,rho_a,rho_l)
                k4r=dlnrdm(planet.radius[i-1]+planet.h[i]*k3r    , planet.pressure[i-1]+planet.h[i]*k3p    , rho_v)
                k4p=dlnPdm(planet.radius[i-1]+planet.h[i]*k3r    , planet.pressure[i-1]+planet.h[i]*k3p    , planet.mass[i-1]+planet.h[i])
                planet.logr[i]=planet.logr[i-1]+planet.h[i]/6.0*(k1r+2.0*k2r+2.0*k3r+k4r)
                planet.logp[i]=planet.logp[i-1]+planet.h[i]/6.0*(k1p+2.0*k2p+2.0*k3p+k4p)
                planet.radius[i]=math.exp(planet.logr[i])
                planet.pressure[i]=math.exp(planet.logp[i])
                planet.gravity[i]=G*planet.mass[i]/planet.radius[i]**2.0
                rho_l=f_rho_Fel(planet.pressure[i],planet.T_an_c)[0][0]
                rho_a=f_rho_Fea(planet.pressure[i],planet.T_an_c)[0][0]
                planet.rho[i]=rho_mix(x_alloy,rho_a,rho_l)
                planet.logrho[i]=math.log(planet.rho[i])

                alpha_l=f_alpha_Fel(planet.pressure[i],planet.T_an_c)[0][0]
                alpha_a=f_alpha_Fea(planet.pressure[i],planet.T_an_c)[0][0]
                planet.alpha[i]=alpha_mix(x_alloy,alpha_a,alpha_l,planet.rho[i],rho_a,rho_l)
                planet.cP[i]=C_P_liquidFe
                planet.temperature[i]=f_adiabat([x_alloy,planet.T_an_c,planet.pressure[i]])[0]
                dqdy_l=f_dqdy_Fel(planet.pressure[i],planet.T_an_c)[0][0]
                dqdy_a=f_dqdy_Fea(planet.pressure[i],planet.T_an_c)[0][0]
                planet.dqdy[i]=dqdy_mix(x_alloy,planet.rho[i],rho_a,rho_l,dqdy_a,dqdy_l,planet.pressure[i])
            else:
                if i==planet.c_z+1:
                    s_sol_val=S_sol_P(planet.pressure[i-1]).tolist()
                    s_liq_val=S_liq_P(planet.pressure[i-1]).tolist()
                    y_value=return_T_Py(0.5,planet.pressure[i-1],planet.temperature[i-1])
                    planet.entropy[i-1]=y_value*(S_max-s_liq_val)+s_liq_val
                    if planet.entropy[i-1]>5050.0:
                        planet.entropy[i-1]=5050.0
                        y_value=(planet.entropy[i-1]-s_liq_val)/(S_max-s_liq_val)
                s_sol_val=S_sol_P(planet.pressure[i-1]).tolist()
                s_liq_val=S_liq_P(planet.pressure[i-1]).tolist()
                s_new=planet.entropy[i-1]
                if s_new>=s_liq_val:
                    y=(s_new-s_liq_val)/(S_max-s_liq_val)
                    rho_v=rho_Py_liq(planet.pressure[i-1],y)[0][0]
                k1r=dlnrdm(planet.radius[i-1]             , planet.pressure[i-1]             , rho_v)
                k1p=dlnPdm(planet.radius[i-1]             , planet.pressure[i-1]             , planet.mass[i-1])

                s_sol_val=S_sol_P(planet.pressure[i-1]+planet.h[i]/2.0*k1p).tolist()
                s_liq_val=S_liq_P(planet.pressure[i-1]+planet.h[i]/2.0*k1p).tolist()
                s_new=planet.entropy[i-1]+dsdr_c*planet.h[i]/2.0*k1r
                if s_new>=s_liq_val:
                    y=(s_new-s_liq_val)/(S_max-s_liq_val)
                    rho_v=rho_Py_liq(planet.pressure[i-1]+planet.h[i]/2.0*k1p,y)[0][0]
                k2r=dlnrdm(planet.radius[i-1]+planet.h[i]/2.0*k1r, planet.pressure[i-1]+planet.h[i]/2.0*k1p, rho_v)
                k2p=dlnPdm(planet.radius[i-1]+planet.h[i]/2.0*k1r, planet.pressure[i-1]+planet.h[i]/2.0*k1p, planet.mass[i-1]+planet.h[i]/2.0)

                s_sol_val=S_sol_P(planet.pressure[i-1]+planet.h[i]/2.0*k2p).tolist()
                s_liq_val=S_liq_P(planet.pressure[i-1]+planet.h[i]/2.0*k2p).tolist()
                s_new=planet.entropy[i-1]+dsdr_c*planet.h[i]/2.0*k2r
                if s_new>=s_liq_val:
                    y=(s_new-s_liq_val)/(S_max-s_liq_val)
                    rho_v=rho_Py_liq(planet.pressure[i-1]+planet.h[i]/2.0*k2p,y)[0][0]
                k3r=dlnrdm(planet.radius[i-1]+planet.h[i]/2.0*k2r, planet.pressure[i-1]+planet.h[i]/2.0*k2p, rho_v)
                k3p=dlnPdm(planet.radius[i-1]+planet.h[i]/2.0*k2r, planet.pressure[i-1]+planet.h[i]/2.0*k2p, planet.mass[i-1]+planet.h[i]/2.0)

                s_sol_val=S_sol_P(planet.pressure[i-1]+planet.h[i]*k3p).tolist()
                s_liq_val=S_liq_P(planet.pressure[i-1]+planet.h[i]*k3p).tolist()
                s_new=planet.entropy[i-1]+dsdr_c*planet.h[i]*k3r
                if s_new>=s_liq_val:
                    y=(s_new-s_liq_val)/(S_max-s_liq_val)
                    rho_v=rho_Py_liq(planet.pressure[i-1]+planet.h[i]*k3p,y)[0][0]
                k4r=dlnrdm(planet.radius[i-1]+planet.h[i]*k3r    , planet.pressure[i-1]+planet.h[i]*k3p    , rho_v)
                k4p=dlnPdm(planet.radius[i-1]+planet.h[i]*k3r    , planet.pressure[i-1]+planet.h[i]*k3p    , planet.mass[i-1]+planet.h[i])

                planet.logr[i]=planet.logr[i-1]+planet.h[i]/6.0*(k1r+2.0*k2r+2.0*k3r+k4r)
                planet.logp[i]=planet.logp[i-1]+planet.h[i]/6.0*(k1p+2.0*k2p+2.0*k3p+k4p)
                planet.radius[i]=math.exp(planet.logr[i])
                planet.pressure[i]=math.exp(planet.logp[i])
                planet.gravity[i]=G*planet.mass[i]/planet.radius[i]**2.0
                s_sol_val=S_sol_P(planet.pressure[i]).tolist()
                s_liq_val=S_liq_P(planet.pressure[i]).tolist()
                planet.entropy[i]=planet.entropy[i-1]+dsdr_c*(planet.radius[i]-planet.radius[i-1])
                if planet.entropy[i]>=s_liq_val:
                    y=(planet.entropy[i]-s_liq_val)/(S_max-s_liq_val)
                    planet.rho[i]=rho_Py_liq(planet.pressure[i],y)[0][0]
                    planet.alpha[i]=alpha_Py_liq(planet.pressure[i],y)[0][0]
                    planet.temperature[i]=T_Py_liq(planet.pressure[i],y)[0][0]
                    planet.dqdy[i]=dqdy_Py_liq(planet.pressure[i],y)[0][0]
                    planet.cP[i]=CP_Py_liq(planet.pressure[i],y)[0][0]
                    planet.dTdP[i]=dTdP_Py_liq(planet.pressure[i],y)[0][0]
                    planet.melt_frac[i]=1.0
                planet.logrho[i]=math.log(planet.rho[i])

        if iteration%10==0:
            print('Iteration:%d Pressure at planet surface:%2.2f bar and center:%2.2f GPa' %(iteration,planet.pressure[zone-1]/1e5,planet.pressure[0]/1e9))
        iteration=iteration+1

    for i in range(zone):
        if i==0:
            planet.T_cell[i]=planet.temperature[i]
        else:
            planet.T_cell[i]=(planet.temperature[i-1]+planet.temperature[i])/2.0
    

cpdef double T_simon(double P, double x):
    cdef double T0=6500.0*(P/340.0)**0.515
    cdef double x0=((1.0-x)/55.845)/((1.0-x)/55.845+x/28.0855) #molar fraction of Fe with S as impurity
    cdef double T=T0/(1.0-np.log(x0))
    return T

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void henyey_solver(Planet planet, double dsdr_c, double initial, double rtol):
    """
    Parameters:
      planet      - An object with the following attributes:
                    mass, radius, logr, pressure, logp, rho, logrho, gravity,
                    entropy, dqdy, P_c, rho_c, dqdy_c, P_s, rtol, T_an_c, x_c,
                    c_z, m_z, and also methods: rho_mix, dqdy_mix.
      dsdr_c      - A parameter for the entropy gradient.
      initial     - A flag (1.0 indicates initial calculation of s_cell).
      
    Returns a dictionary of updated structure arrays and parameters.
    """
    cdef int zone = planet.m_z+planet.c_z
    planet.old_pressure = planet.pressure.copy()
    planet.old_radius = planet.radius.copy()
    cdef Py_ssize_t i
    cdef Py_ssize_t iteration = 0
    cdef double x_alloy = planet.x_c / mf_l  # mf_l should be defined elsewhere
    cdef double[:] d_p = np.ones(zone, dtype=np.float64)
    cdef double[:] d_r = np.ones(zone, dtype=np.float64)
    cdef double[:] A_r = np.zeros(zone, dtype=np.float64)
    cdef double[:] A_p = np.zeros(zone, dtype=np.float64)
    cdef double[:] a = np.zeros(zone, dtype=np.float64)
    cdef double[:] b = np.zeros(zone, dtype=np.float64)
    cdef double[:] c = np.zeros(zone, dtype=np.float64)
    cdef double[:] d = np.zeros(zone, dtype=np.float64)
    cdef double[:] A = np.zeros(zone, dtype=np.float64)
    cdef double[:] B = np.zeros(zone, dtype=np.float64)
    cdef double[:] C = np.zeros(zone, dtype=np.float64)
    cdef double[:] D = np.zeros(zone, dtype=np.float64)
    cdef double[:] alp = np.zeros(zone, dtype=np.float64)
    cdef double[:] gam = np.zeros(zone, dtype=np.float64)
    cdef double[:] delta_y = np.zeros(zone, dtype=np.float64)
    cdef double[:] delta_x = np.zeros(zone, dtype=np.float64)

    cdef double delta_P_center, logP_c, logrho_c, alp_0, gam_0
    cdef double v_bd, v_ABCD, v_P, v_r
    cdef double s_sol_val, s_liq_val, y, rho_l, rho_a, dqdy_l, dqdy_a

    cdef double x_frac
    cdef double max_impurities_concentration = 0.1519

    # Iterative loop until convergence in both pressure and radius changes.
    while np.max(np.abs(d_p)) > rtol or np.max(np.abs(d_r)) > rtol:
        planet.old_pressure = planet.pressure.copy()
        planet.old_radius = planet.radius.copy()

        planet.pressure[zone-1] = planet.P_surf
        planet.logp[zone-1] = math.log(planet.pressure[zone-1])

        s_sol_val = S_sol_P(planet.pressure[zone-1]).tolist()
        s_liq_val = S_liq_P(planet.pressure[zone-1]).tolist()
        if planet.entropy[zone-1] < s_sol_val:
            y = (planet.entropy[zone-1] - S_min) / (s_sol_val - S_min)
            planet.rho[zone-1] = rho_Py_sol_en(planet.pressure[zone-1], y)[0][0]
            planet.dqdy[zone-1] = dqdy_Py_sol_en(planet.pressure[zone-1], y)[0][0]
        elif planet.entropy[zone-1] > s_liq_val:
            y = (planet.entropy[zone-1] - s_liq_val) / (S_max - s_liq_val)
            planet.rho[zone-1] = rho_Py_liq(planet.pressure[zone-1], y)[0][0]
            planet.dqdy[zone-1] = dqdy_Py_liq(planet.pressure[zone-1], y)[0][0]
        else:
            y = (planet.entropy[zone-1] - s_sol_val) / (s_liq_val - s_sol_val)
            planet.rho[zone-1] = rho_Py_mix_en(planet.pressure[zone-1], y)[0][0]
            planet.dqdy[zone-1] = dqdy_Py_mix_en(planet.pressure[zone-1], y)[0][0]
        planet.logrho[zone-1] = math.log(planet.rho[zone-1])

        logP_c = math.log(planet.P_c)
        logrho_c = math.log(planet.rho_c)

        # Reinitialize A_r and A_p for this iteration.
        A_r = np.zeros(zone, dtype=np.float64)
        A_p = np.zeros(zone, dtype=np.float64)
        for i in range(zone):
            if i == 0:
                A_r[i] = planet.logr[i] - 1.0/3.0 * (math.log(3.0 * planet.mass[i] / (4.0 * math.pi)) - logrho_c)
                A_p[i] = planet.logp[i] - logP_c + G/2.0 * math.pow((4.0 * math.pi * math.pow(planet.mass[i], 2.0/3.0)), 1.0/3.0) * math.exp(4.0 * logrho_c/3.0 - logP_c)
            else:
                A_r[i] = planet.logr[i] - planet.logr[i-1] - 1.0/(4.0*math.pi) * (planet.mass[i]-planet.mass[i-1]) * math.exp(-0.5*(planet.logrho[i]+planet.logrho[i-1]) - 1.5*(planet.logr[i]+planet.logr[i-1]))
                A_p[i] = planet.logp[i] - planet.logp[i-1] + G/(8.0*math.pi) * (math.pow(planet.mass[i],2.0) - math.pow(planet.mass[i-1],2.0)) * math.exp(-0.5*(planet.logp[i]+planet.logp[i-1]) - 2.0*(planet.logr[i]+planet.logr[i-1]))

        a = np.zeros(zone, dtype=np.float64)
        b = np.zeros(zone, dtype=np.float64)
        c = np.zeros(zone, dtype=np.float64)
        d = np.zeros(zone, dtype=np.float64)
        A = np.zeros(zone, dtype=np.float64)
        B = np.zeros(zone, dtype=np.float64)
        C = np.zeros(zone, dtype=np.float64)
        D = np.zeros(zone, dtype=np.float64)

        for i in range(zone):
            if i == 0:
                v_bd = (G/(8.0*math.pi)) * math.pow(planet.mass[i], 2.0) * math.exp(-2.0 * planet.logr[i]) * math.exp(-0.5*(planet.logp[i] + logP_c)) * (-0.5)
                a[i] = (G/(8.0*math.pi)) * math.pow(planet.mass[i], 2.0) * math.exp(-0.5*(planet.logp[i] + logP_c)) * math.exp(-2.0 * planet.logr[i]) * (-2.0)
                c[i] = a[i]
                b[i] = -1.0 + v_bd
                d[i] = 1.0 + v_bd
                v_ABCD = (1.0/(4.0*math.pi)) * planet.mass[i] * math.exp(-0.5*(planet.logrho[i] + logrho_c)) * math.exp(-1.5 * planet.logr[i])
                A[i] = -1.0 - v_ABCD * (-1.5)
                B[i] = -v_ABCD * (-0.5) * planet.dqdy_c
                C[i] = 1.0 - v_ABCD * (-1.5)
                D[i] = -v_ABCD * (-0.5) * planet.dqdy[i]
            else:
                v_P = A_p[i] - planet.logp[i] + planet.logp[i-1]
                v_r = A_r[i] - planet.logr[i] + planet.logr[i-1]
                a[i] = -2.0 * v_P
                b[i] = -1.0 - v_P/2.0
                c[i] = -2.0 * v_P
                d[i] = 1.0 - v_P/2.0
                A[i] = -1.0 - 1.5 * v_r
                B[i] = -0.5 * v_r * planet.dqdy[i-1]
                C[i] = 1.0 - 1.5 * v_r
                D[i] = -0.5 * v_r * planet.dqdy[i]
        alp = np.zeros(zone, dtype=np.float64)
        gam = np.zeros(zone, dtype=np.float64)
        alp_0 = 0.0
        gam_0 = 0.0
        for i in range(zone):
            if i == 0:
                alp[i] = (d[i]*(B[i]-A[i]*alp_0) - D[i]*(b[i]-a[i]*alp_0)) / (c[i]*(B[i]-A[i]*alp_0) - C[i]*(b[i]-a[i]*alp_0))
                gam[i] = ((B[i]-A[i]*alp_0)*(A_p[i]-a[i]*gam_0) - (b[i]-a[i]*alp_0)*(A_r[i]-A[i]*gam_0)) / (c[i]*(B[i]-A[i]*alp_0) - C[i]*(b[i]-a[i]*alp_0))
            else:
                alp[i] = (d[i]*(B[i]-A[i]*alp[i-1]) - D[i]*(b[i]-a[i]*alp[i-1])) / (c[i]*(B[i]-A[i]*alp[i-1]) - C[i]*(b[i]-a[i]*alp[i-1]))
                gam[i] = ((B[i]-A[i]*alp[i-1])*(A_p[i]-a[i]*gam[i-1]) - (b[i]-a[i]*alp[i-1])*(A_r[i]-A[i]*gam[i-1])) / (c[i]*(B[i]-A[i]*alp[i-1]) - C[i]*(b[i]-a[i]*alp[i-1]))
        delta_y = np.zeros(zone, dtype=np.float64)
        delta_x = np.zeros(zone, dtype=np.float64)
        delta_y[zone-1] = 0.0
        delta_x[zone-1] = -gam[zone-1]
        for i in range(zone-1, -1, -1):
            if i >= 0 and i < zone-1:
                delta_y[i] = -(A_r[i+1] - A[i+1]*gam[i] + C[i+1]*delta_x[i+1] + D[i+1]*delta_y[i+1]) / (B[i+1] - A[i+1]*alp[i])
                delta_x[i] = -gam[i] - alp[i] * delta_y[i]
        delta_P_center = (A_r[0] + C[0]*delta_x[0] + D[0]*delta_y[0]) / B[0]
        planet.P_c = math.exp(logP_c + delta_P_center)
        for i in range(zone):
            planet.radius[i] = math.exp(planet.logr[i] + delta_x[i])
            planet.pressure[i] = math.exp(planet.logp[i] + delta_y[i])
            planet.logr[i] = math.log(planet.radius[i])
            planet.logp[i] = math.log(planet.pressure[i])
            planet.gravity[i] = (G * planet.mass[i] / math.pow(planet.radius[i], 2.0))
            if i <= planet.c_z:
                rho_l = f_rho_Fel(planet.pressure[i], planet.T_an_c)[0][0]
                rho_a = f_rho_Fea(planet.pressure[i], planet.T_an_c)[0][0]
                planet.rho[i] = rho_mix(x_alloy, rho_a, rho_l)
                dqdy_l = f_dqdy_Fel(planet.pressure[i], planet.T_an_c)[0][0]
                dqdy_a = f_dqdy_Fea(planet.pressure[i], planet.T_an_c)[0][0]
                planet.dqdy[i] = dqdy_mix(x_alloy, planet.rho[i], rho_a, rho_l, dqdy_a, dqdy_l, planet.pressure[i])
            else:
                s_sol_val = S_sol_P(planet.pressure[i]).tolist()
                s_liq_val = S_liq_P(planet.pressure[i]).tolist()
                y = (planet.entropy[i] - s_liq_val) / (S_max - s_liq_val)
                planet.rho[i] = rho_Py_liq(planet.pressure[i], y)[0][0]
                planet.dqdy[i] = dqdy_Py_liq(planet.pressure[i], y)[0][0]
            planet.logrho[i] = math.log(planet.rho[i])
            d_p[i] = (planet.pressure[i] - planet.old_pressure[i]) / planet.old_pressure[i]
            d_r[i] = (planet.radius[i] - planet.old_radius[i]) / planet.old_radius[i]
        rho_l = f_rho_Fel(planet.P_c, planet.T_an_c)[0][0]
        rho_a = f_rho_Fea(planet.P_c, planet.T_an_c)[0][0]
        planet.rho_c = rho_mix(x_alloy, rho_a, rho_l)
        dqdy_l = f_dqdy_Fel(planet.P_c, planet.T_an_c)[0][0]
        dqdy_a = f_dqdy_Fea(planet.P_c, planet.T_an_c)[0][0]
        planet.dqdy_c = dqdy_mix(x_alloy, planet.rho_c, rho_a, rho_l, dqdy_a, dqdy_l, planet.P_c)
        print('Iteration:%d Pressure at planet surface:%2.2f bar and center:%2.2f GPa' %(iteration,planet.pressure[zone-1]/1e5,planet.pressure[0]/1e9))
        iteration = iteration + 1

    # Compute cell-averaged values.
    for i in range(zone):
        if i == 0:
            planet.r_cell[i] = planet.radius[i] / 2.0
            planet.p_cell[i] = (planet.pressure[i] + planet.P_c) / 2.0
        else:
            planet.r_cell[i] = (planet.radius[i] + planet.radius[i-1]) / 2.0
            planet.p_cell[i] = (planet.pressure[i] + planet.pressure[i-1]) / 2.0
        planet.Area[i] = 4.0 * math.pi * math.pow(planet.radius[i], 2.0)

    if initial == 1.0:
        for i in range(zone):
            if i == planet.c_z:
                planet.s_cell[i] = planet.entropy[i] - dsdr_c * planet.radius[i] / 2.0
            else:
                planet.s_cell[i] = (planet.entropy[i-1] + planet.entropy[i]) / 2.0
            
    
            if i < planet.c_z:
                x_frac = planet.x_c*(planet.M_pl*planet.CMF-planet.mass[0])/(planet.M_pl*planet.CMF-planet.mass[i])
                if x_frac > max_impurities_concentration:
                    x_frac = max_impurities_concentration
                if load_file[0]==1.0:
                    planet.T_Fe_melt[i] = T_simon(planet.pressure[i]/1e9+12.0,x_frac)
                    planet.P_Fe_melt[i] = planet.pressure[i] + 12e9
                else:
                    planet.T_Fe_melt[i] = T_simon(planet.pressure[i]/1e9,x_frac)
                    planet.P_Fe_melt[i] = planet.pressure[i]
        planet.P_Fe_melt[0]=2.0*planet.P_Fe_melt[1]-planet.P_Fe_melt[2]
        planet.T_Fe_melt[0]=2.0*planet.T_Fe_melt[1]-planet.T_Fe_melt[2]

    #still nedd to tabulate table

