#!python
# cython: boundscheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from libc cimport math
cimport cython
import eos_tables

# # suppress certain warnings
# import warnings
# warnings.filterwarnings('ignore', 'The iteration is not making good progress')

cpdef double sigma_silicate(double P,double T):
    cdef double E_ion=131000.0
    cdef double V_ion=0.437e-6
    cdef double sigma0_ion=1.0811e9
    cdef double E_el=108.6*1000.0
    cdef double V_el=0.0611e-6
    cdef double sigma0_el=1.754e9
    cdef double sigma_el=sigma0_el/T*math.exp(-(E_el+P*V_el)/(8.3145*T))
    cdef double sigma_ion=sigma0_ion/T*math.exp(-(E_ion+P*V_ion)/(8.3145*T))
    return sigma_el+sigma_ion

cpdef double rho_mix(double x, double rho_l, double rho_s):
    cdef double value=(x/rho_l+(1.0-x)/rho_s)**(-1.0)
    return value
cpdef double alpha_mix(double x,double alpha_l,double alpha_s,double rho,double rho_l,double rho_s):
    return x*rho/rho_l*alpha_l+(1.0-x)*rho/rho_s*alpha_s
cpdef double dqdy_mix(double x, double rho_tot, double rho_l, double rho_s, double dqdy_l, double dqdy_s, double pressure):
    if pressure==0.0:
        pressure=10.0**(-6.0)
    cdef double drhodP_l=dqdy_l*rho_l/pressure
    cdef double drhodP_s=dqdy_s*rho_s/pressure
    cdef double value1=x*rho_tot**(2.0)*rho_l**(-2.0)*drhodP_l
    cdef double value2=(1.0-x)*rho_tot**(2.0)*rho_s**(-2.0)*drhodP_s
    cdef double value=(value1+value2)*pressure/rho_tot
    return value
cpdef int find_nearest(double[:] array, double value):
    cdef int idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

cpdef double linear_intersect(double a0,double a1,double A0,double A1,double b0,double b1,double B0,double B1):
    cdef double up=(B0*A1-B1*A0)/(A1-A0)-(b0*a1-b1*a0)/(a1-a0)
    cdef double down=(b1-b0)/(a1-a0)-(B1-B0)/(A1-A0)
    cdef double value=up/down
    return value
cpdef double f_ic(double p0, double p1, double pic, double v0, double v1):
    cdef double value=v1+(v0-v1)/(p0-p1)*(pic-p1)
    return value

cpdef double f_oc(double r1, double r2, double rx, double v1, double v2):
    cdef double value=v1-(r1-rx)*(v2-v1)/(r2-r1)
    return value

cpdef double func(x):
    cdef double value=f_interp(x)-f_interp_Tsimon(x)
    return value

cpdef double T_liquidpv(double P):#P->GPa
    cdef double T_simon=1831.0*(1.0+P/4.6)**0.33
    return T_simon
cpdef double T_simon(double P, double x):
    cdef double T0=6500.0*(P/340.0)**0.515
    cdef double x0=((1.0-x)/55.845)/((1.0-x)/55.845+x/28.0855) #molar fraction of Fe with S as impurity
    cdef double T=T0/(1.0-math.log(x0))
    return T
cpdef double Sti_liq(double P):
    return 5400.0*(P/140.0)**0.48
cpdef double Sti_sol(double P):
    return 5400.0*(P/140.0)**0.48/(1.0-math.log(0.79))

cpdef double Fiq_liq(double P):
    return 2022.0+54.21*P-0.34*P**2.0+9.0747e-4*P**3.0
cpdef double Fiq_sol(double P):
    return 1621.0+38.415*P-0.1958*P**2.0+3.8369e-4*P**3.0

cpdef double And_liq(double P):
    return 1940.0*(P/29.0+1.0)**(1.0/1.9)
cpdef double And_sol(double P):
    return 2045.0*(P/92.0+1.0)**(1.0/1.3)

cpdef double T_sol_fiq(double P):
    cdef double value=0.0
    cdef double value1=Fiq_sol(P)
    cdef double value2=Sti_sol(P)
    if P<150:
        if value1<value2:
            value=value2
        else:
            value=value1
    else:
        value=value2
    return value

cpdef double T_liq_fiq(double P):
    cdef double value=0.0
    cdef double value1=Fiq_liq(P)
    cdef double value2=Sti_liq(P)
    if P<150:
        if value1<value2:
            value=value2
        else:
            value=value1
    else:
        value=value2
    return value

cpdef double f_dTdP(double alpha, double rho, double C_P, double T):
    cdef double value=alpha*T/(rho*C_P)
    return value

cpdef double f_viscosity(double T, double P, double density, double phase, double x, double rho_m, double rho_s, double width, double ppv_eta_model):
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
        if ppv_eta_model==1.0:
            eta0=1.05e34#1.9e21#
            E=7.8e5#1.62e5#
            p_decay=1100e9#1610e9#
            V=1.7e-6*np.exp(-P/p_decay)#1.4e-6*np.exp(-P/p_decay)#
            eta_s=eta0*np.exp((E+P*V)/(R*T)-E/(R*1600.0))/density
        elif ppv_eta_model==2.0:
            eta0=1.9e21#
            E=1.62e5#
            p_decay=1610e9#
            V=1.4e-6*np.exp(-P/p_decay)#
            eta_s=eta0*np.exp((E+P*V)/(R*T)-E/(R*1600.0))/density
    cdef double eta_m=100.0/density
    cdef double value1=0.0
    cdef double value2=0.0
    cdef double value=0.0
    y=(x-0.4)/width
    z=0.5*(1.0+math.tanh(y))
    value=10.0**(z*math.log10(eta_m)+(1.0-z)*math.log10(eta_s))
    return value

cpdef double f_eddy_T_high_nu(double T,double C_P,double alpha,double g,double l,double v,double dsdr):
    cdef double value=alpha*g*l**4.0*T/(18.0*v*C_P)*(-dsdr)
    return value
cpdef double f_eddy_T_low_nu(double T,double C_P,double alpha,double g,double l,double v,double dsdr):
    cdef double value=(alpha*g*l**4.0*T/(16.0*C_P)*(-dsdr))**0.5
    return value

cpdef double[:] penta_solver(double[:] a, double[:] b, double[:] c, double[:] d, double[:] e, double[:] y):
    """
    Solve the pentadiagonal system:
       A[i]*x[i-2] + B[i]*x[i-1] + C[i]*x[i] + D[i]*x[i+1] + E[i]*x[i+2] = F[i],  for i = 0,...,n-1.
    
    The inputs A, B, C, D, E, F are assumed to be memoryviews over contiguous double arrays.
    (The boundary assumptions are that A[0] = A[1] = 0 and E[n-2] = E[n-1] = 0.)
    
    All temporary arrays are allocated as memoryviews.
    The returned object is a NumPy array (backed by a memoryview) containing the solution.
    """
    cdef Py_ssize_t zone = y.shape[0]
    cdef Py_ssize_t i

    cdef double[:] alpha=np.empty(zone, dtype=np.double) #1,2,3,...,n-1; n-1
    cdef double[:] beta=np.empty(zone, dtype=np.double) #1,2,3,...,n-2; n-2
    cdef double[:] z=np.empty(zone, dtype=np.double) #1,2,3,...,n; n
    cdef double[:] gamma=np.empty(zone, dtype=np.double) #2,3,4,...,n; n-1
    cdef double[:] mu=np.empty(zone, dtype=np.double) #1,2,3,...,n; n
    cdef double[:] solution=np.empty(zone, dtype=np.double)
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
    solution[zone-1]=z[zone-1]
    solution[zone-2]=z[zone-2]-alpha[zone-2]*solution[zone-1]
    for i in range(zone-3,-1,-1):
        solution[i]=z[i]-alpha[i]*solution[i+1]-beta[i]*solution[i+2]

    return solution

cdef Py_ssize_t i,j,x_idx,Tref_idx

load_file=np.loadtxt('input.txt')
results_foldername='results_Mpl'+str(load_file[0])+'_CMF'+str(load_file[1])+'_time'+str(load_file[2])+'_Qrad'+str(load_file[3])+'_'+str(load_file[4])+'_'+str(load_file[5])+'_'+str(load_file[6])+'_Teq'+str(load_file[8])+'_Qradc'+str(load_file[9])+'_eta'+str(load_file[7])+'_mzmulti'+str(load_file[10])

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


print('Read structure and property profiles')
mass_profile=np.loadtxt(results_foldername+'/profile/t0/structure0.txt')
P_profile=np.loadtxt(results_foldername+'/profile/t0/henyey0.txt')
cdef double[:] mass=mass_profile[:,8]
cdef double[:] melt_pressure=P_profile[:,1]
cdef double[:] melt_pressure_cell=P_profile[:,3]
#cdef double[:] mass_cell=mass_profile[:,9]
cdef double[:] h=mass_profile[:,11]

Fe_melt_file=np.loadtxt(results_foldername+'/profile/t0/Fe_melt.txt')
cdef double[:] melt_pressure_Fe=Fe_melt_file[:,0]
cdef double[:] T_Fe_melt=Fe_melt_file[:,1]
cdef double[:] melt_pressure_Fe_GPa=np.zeros(len(melt_pressure_Fe))
for i in range(len(melt_pressure_Fe)):
    melt_pressure_Fe_GPa[i]=melt_pressure_Fe[i]/1e9
f_interp_Tsimon=interpolate.interp1d(melt_pressure_Fe_GPa,T_Fe_melt)

initial_profile=np.loadtxt(results_foldername+'/profile/t0/structure0.txt')
initial_property=np.loadtxt(results_foldername+'/profile/t0/property0.txt')
initial_henyey=np.loadtxt(results_foldername+'/profile/t0/henyey0.txt')
previous=np.loadtxt(results_foldername+'/profile/t0/previous0.txt')
Fsurf_table=np.loadtxt(results_foldername+'/profile/t0/Fsurf.txt')

sgrid=Fsurf_table[:,0]
Fsurf_sgrid=Fsurf_table[:,1]
delta_UBL_sgrid=Fsurf_table[:,2]
Tsurf_sgrid=Fsurf_table[:,3]
dFds_sgrid=Fsurf_table[:,4]
vissurf_sgrid=Fsurf_table[:,5]

cdef double Ts_FsTable_min=np.min(Tsurf_sgrid)

f_Fsurf_s=interpolate.interp1d(sgrid,Fsurf_sgrid)
f_deltaUBL_s=interpolate.interp1d(sgrid,delta_UBL_sgrid)
f_Tsurf_s=interpolate.interp1d(sgrid,Tsurf_sgrid)
f_dFds_s=interpolate.interp1d(sgrid,dFds_sgrid)
f_vissurf_s=interpolate.interp1d(sgrid,vissurf_sgrid)

cdef double t=0.0#previous[0]
cdef double dt=5e7#previous[1]
cdef double P_center=previous[2]
cdef double delta_P_center=previous[3]
cdef double T_cmb=previous[4]
cdef int end_ite=100
cdef double end_time=load_file[2]*86400.0*365.0*1e9
cdef double P_surf=1e5 # Surface pressure in Pa.
cdef double ppv_eta_model=load_file[7]
cdef double max_impurities_concentration=0.1519

cdef double dt_thres, ds_thres
cdef double ds_thres_xl=1e-2
cdef double ds_thres_ma=0.0#1e-2
cdef double ds_thres_dr=1.0#1e-3
cdef double ds_thres_MO=2.0#2e-2
cdef double ds_thres_cmb=3.0
cdef double ds_thres_Fs=4.0#2e-2
rdiff=np.zeros(5)

cdef double[:] initial_radius=initial_henyey[:,0]
cdef double[:] radius_cell=initial_henyey[:,2]
cdef double[:] initial_pressure=initial_henyey[:,1]
cdef double[:] pressure_cell=initial_henyey[:,3]
cdef double[:] initial_density=initial_henyey[:,4]
cdef double[:] initial_gravity=initial_henyey[:,5]
cdef double[:] initial_dqdy=initial_henyey[:,6]
cdef double[:] initial_phase=initial_henyey[:,7]
cdef double[:] old_phase=initial_phase.copy()

pressure_np=initial_henyey[:,1]
pressure_cell_np=initial_henyey[:,3]

cdef double[:] initial_temperature=initial_profile[:,0]
cdef double[:] temperature_cell=initial_profile[:,1]
cdef double[:] melt_frac=initial_profile[:,2]
cdef double[:] initial_S=initial_profile[:,3]
cdef double[:] S_cell=initial_profile[:,4]
cdef double[:] dsdr=initial_profile[:,5]
cdef double[:] old_S=initial_profile[:,6]
cdef double[:] old_Scell=initial_profile[:,7]
cdef double[:] x_cell=initial_profile[:,10]

cdef double Fsurf
Fsurf=f_Fsurf_s(S_cell[-1]).tolist()
cdef double dFds

cdef double[:] alpha=initial_property[:,0]
cdef double[:] CP=initial_property[:,1]
cdef double[:] kappa=initial_property[:,2]
cdef double[:] dTdP=initial_property[:,3]
cdef double[:] dPdr=initial_property[:,4]
cdef double[:] dxdr=initial_property[:,5]

cdef double[:] old_pressure=initial_henyey[:,16]
cdef double[:] old_pressure_cell=initial_henyey[:,18]
cdef double[:] old_radius=initial_henyey[:,15]
cdef double[:] old_radius_cell=initial_henyey[:,17]
cdef double[:] old_density=initial_henyey[:,19]
cdef double[:] old_gravity=initial_henyey[:,20]

cdef double[:] PdV=initial_henyey[:,9]
cdef double[:] dEG=initial_henyey[:,10]
cdef double[:] dw=initial_henyey[:,11]
cdef double[:] new_V=initial_henyey[:,12]
cdef double[:] old_V=initial_henyey[:,21]
cdef double[:] new_EG=initial_henyey[:,13]
cdef double[:] old_EG=initial_henyey[:,22]
cdef double[:] new_v_top=initial_henyey[:,14]
cdef double[:] old_v_top=initial_henyey[:,23]
cdef double inner_EG, outer_EG

cdef double[:] Area=initial_henyey[:,8]
cdef double[:] old_Area=initial_henyey[:,24]

cdef double[:] old_T=initial_temperature.copy()
cdef double[:] old_T_cell=temperature_cell.copy()
cdef double[:] old_kappa=kappa.copy()
cdef double[:] old_dTdP=dTdP.copy()
cdef double[:] old_dPdr=dPdr.copy()

cdef double ph_Fe_sol=0.0
cdef double ph_Fe_liq=1.0
cdef double ph_pv_sol=2.0
cdef double ph_pv_liq=4.0
cdef double ph_pv_mix=5.0

cdef double sigma=5.670373e-08
cdef double L_pv=7.322*10.0**5.0
cdef double L_Fe=1.2*10.0**6.0
cdef double L0=2.44e-8 # lorentz number
cdef double mu_0=4.0*math.pi*1e-7
cdef double G=6.674e-11
cdef double CP_s=1265.0
cdef double C_P_Fe=840.0
cdef double k_b=1.38e-23 #J/K
cdef double N_A=6.022e+23 # mol^-1
cdef double T0_Feliquid=1181.0
cdef double T0_Fesolid=300.0
cdef double m_H=1.67e-27 # kg
cdef double M_E=5.972e24

cdef double k_pv=10.0
cdef double k_Fe=40.0
cdef double k_ol=10.0#4.0

cdef double tau_m_Th=14.0
cdef double Q_radm_Th=2.24*10.0**(-12.0)*load_file[3]
cdef double tau_m_K=1.25
cdef double Q_radm_K=8.69*10.0**(-13.0)*load_file[4]
cdef double tau_m_U8=4.47
cdef double Q_radm_U8=1.97*10.0**(-12.0)*load_file[5]
cdef double tau_m_U5=0.704
cdef double Q_radm_U5=8.48*10.0**(-14.0)*load_file[6]

cdef double Fs=239.575#287043.75
cdef double Teq=load_file[8]
cdef double CMF=load_file[1]
cdef double x_init=0.105
cdef int mantle_zone=previous[-1]
cdef int core_zone=previous[-2]
cdef int zone=mantle_zone+core_zone
cdef int core_outer_index=core_zone-1
cdef double S_width_s=5.0
cdef double S_width_m=1.0
cdef double Re_width=0.3
cdef double mf_l=0.16
cdef double l_alpha=0.82#0.431#0.82#0.431#0.82
cdef double l_beta=1.0#0.6734#1.0#0.6734#1.0

cdef double M_pl=load_file[0]*5.972e24
cdef double mantle_mass=M_pl*(1.0-CMF)
cdef double h_mantle=M_pl*(1.0-CMF)/mantle_zone
cdef double h_core=M_pl*CMF/core_zone

cdef double Q_rad_c_0=load_file[9]*1.2e12/(math.exp(-4.5/1.2))/(M_E*0.326)*(M_pl*CMF)

cdef double v_b=4.0/CP[zone-1]
cdef double v_a=v_b*sigma*Teq**4.0

cdef double rho_liquid,rho_alloy,alpha_liquid,alpha_alloy,dqdy_liquid,dqdy_alloy,x_alloy

cdef double[:] rho_solid_array=np.zeros(core_outer_index+1)
cdef double[:] dqdy_solid_array=np.zeros(core_outer_index+1)
cdef double[:] alpha_solid_array=np.zeros(core_outer_index+1)
cdef double[:] rho_liquid_array_a=np.zeros(core_outer_index+1)
cdef double[:] rho_alloy_array_a=np.zeros(core_outer_index+1)
cdef double[:] alpha_liquid_array_a=np.zeros(core_outer_index+1)
cdef double[:] alpha_alloy_array_a=np.zeros(core_outer_index+1)
cdef double[:] rho_liquid_array=np.zeros(core_outer_index+1)
cdef double[:] rho_alloy_array=np.zeros(core_outer_index+1)
cdef double[:] alpha_liquid_array=np.zeros(core_outer_index+1)
cdef double[:] alpha_alloy_array=np.zeros(core_outer_index+1)
cdef double[:] dqdy_liquid_array=np.zeros(core_outer_index+1)
cdef double[:] dqdy_alloy_array=np.zeros(core_outer_index+1)

cdef double[:] S_liquidus=np.zeros(zone)
cdef double[:] S_solidus=np.zeros(zone)
cdef double[:] S_solidus_cell=np.zeros(zone)
cdef double[:] S_liquidus_cell=np.zeros(zone)
for i in range(core_outer_index+1,zone):
    S_liquidus[i]=S_liq_P(melt_pressure[i])
    S_liquidus_cell[i]=S_liq_P(melt_pressure_cell[i])
    S_solidus[i]=S_sol_P(melt_pressure[i])
    S_solidus_cell[i]=S_sol_P(melt_pressure_cell[i])
    
cdef double rho_m_v, rho_s_v, alpha_s_v, alpha_m_v, CP_m_v, dqdy_m_v, dqdy_s_v,density_value
cdef double[:] rho_m_array=np.zeros(zone)
cdef double[:] rho_s_array=np.zeros(zone)

t_array=[]
dt_array=[]
average_Tm=[]
Fcmb_array=[]
Fsurf_array=[]
average_S=[]
average_x=[]
melt_frac_min=[]
S_min_array=[]
Qrad_array=[]
Rp=[]
P_center_array=[]
P_cmb_array=[]
Rc=[]
Fcond_cmb=[]
average_Tc=[]
Tcmb_array=[]
Tsurf_array=[]
Ric_array=[]
Mic_array=[]
D_magma=[]
Tic_array=[]
Pic_array=[]
x_core_array=[]
Buoy_T=[]
Buoy_x=[]
Q_ICB_array=[]
T_Fe_en=[]
delta_r_array=[]
Tmbase_array=[]
vmbase_array=[]
rhombase_array=[]
cpmbase_array=[]
alphambase_array=[]
gmbase_array=[]
T_ra_array=[]
P_ra_array=[]
x_ra_array=[]
P_mbase_array=[]
S_liq_mbase_array=[]
S_sol_mbase_array=[]
S_mbase_array=[]
T_liq_mbase_array=[]
T_sol_mbase_array=[]
delta_r_s_array=[]
Qrad_c_array=[]
vconv_core=[]

T_center_array=[]

S_s_array=[]
T_s_array=[]
r_c_m1_array=[]

dMic_list=[]
Ra_T_list=[]
delta_r_list=[]
Tcmb_list=[]

delta_T_ra_s_raw_array=[]

Fconv1_m1=[]
Fcond2_m1=[]
Fcond1_m1=[]
T_m1_array=[]
T_m2_array=[]
T_c_m1_array=[]
T_c_m2_array=[]
S_m1_array=[]
S_c_m1_array=[]
S_m2_array=[]
S_c_m2_array=[]
A_m1_array=[]
A_m2_array=[]
vis_surf_array=[]
v_m2_array=[]
P_m2_array=[]
cdef double vis_surf_value

L_sigma_array=[]
D_MO_dynamo_array=[]
MO_dynamo_bot_array=[]
MO_dynamo_top_array=[]

core_dipole_m=[]
solid_index_arr=[]

Urey_array=[]
Qsurf_array=[]
Qcmb_array=[]
L_Fe_array=[]


cdef double[:] viscosity=np.zeros(zone)
cdef double[:] convection=np.zeros(zone)
cdef double[:] l_mlt=np.zeros(zone)
cdef double[:] eddy_T_high_nu=np.zeros(zone)
cdef double[:] eddy_T_low_nu=np.zeros(zone)
cdef double[:] eddy_x=np.zeros(zone)
cdef double[:] Re=np.zeros(zone)
cdef double[:] eddy_k=np.zeros(zone)
cdef double[:] v_conv=np.zeros(zone)

cdef double[:] new_S=np.zeros(zone)
cdef double[:] new_S_cell=np.zeros(zone)
cdef double[:] new_T_cell=np.zeros(zone)
cdef double[:] new_T=np.zeros(zone)
cdef double[:] new_x=np.zeros(zone)
cdef double[:] new_alpha=np.zeros(zone)
cdef double[:] new_density=np.zeros(zone)
cdef double[:] new_CP=np.zeros(zone)
cdef double[:] new_dTdP=np.zeros(zone)
cdef double[:] new_dPdr=np.zeros(zone)
cdef double[:] new_kappa=np.zeros(zone)
cdef double[:] new_phase=np.zeros(zone)
cdef double[:] new_dqdy=np.zeros(zone)
cdef double[:] new_x_cell=np.zeros(zone)
cdef double[:] new_rho_m=np.zeros(zone)
cdef double[:] new_rho_s=np.zeros(zone)

new_T_np=np.zeros(zone)
y_array=np.zeros(zone)

rho_m_array_c=np.zeros(zone)
rho_pv_array=np.zeros(zone)
rho_en_array=np.zeros(zone)
new_T_cell_l=np.zeros(zone)
new_T_cell_pv=np.zeros(zone)
new_T_cell_ppv=np.zeros(zone)
new_T_cell_en=np.zeros(zone)
new_T_cell_pv_mix=np.zeros(zone)
new_T_cell_ppv_mix=np.zeros(zone)
new_T_cell_en_mix=np.zeros(zone)
new_T_l=np.zeros(zone)
new_T_pv=np.zeros(zone)
new_T_ppv=np.zeros(zone)
new_T_en=np.zeros(zone)
new_T_pv_mix=np.zeros(zone)
new_T_ppv_mix=np.zeros(zone)
new_T_en_mix=np.zeros(zone)
new_dqdy_l=np.zeros(zone)
new_dqdy_pv=np.zeros(zone)
new_dqdy_ppv=np.zeros(zone)
new_dqdy_en=np.zeros(zone)
new_dqdy_pv_mix=np.zeros(zone)
new_dqdy_ppv_mix=np.zeros(zone)
new_dqdy_en_mix=np.zeros(zone)
new_density_l=np.zeros(zone)
new_density_pv=np.zeros(zone)
new_density_ppv=np.zeros(zone)
new_density_en=np.zeros(zone)
new_density_pv_mix=np.zeros(zone)
new_density_ppv_mix=np.zeros(zone)
new_density_en_mix=np.zeros(zone)
new_alpha_l=np.zeros(zone)
new_alpha_pv=np.zeros(zone)
new_alpha_ppv=np.zeros(zone)
new_alpha_en=np.zeros(zone)
new_alpha_pv_mix=np.zeros(zone)
new_alpha_ppv_mix=np.zeros(zone)
new_alpha_en_mix=np.zeros(zone)
new_CP_l=np.zeros(zone)
new_CP_pv=np.zeros(zone)
new_CP_ppv=np.zeros(zone)
new_CP_en=np.zeros(zone)
new_CP_pv_mix=np.zeros(zone)
new_CP_ppv_mix=np.zeros(zone)
new_CP_en_mix=np.zeros(zone)
new_dTdP_l=np.zeros(zone)
new_dTdP_pv=np.zeros(zone)
new_dTdP_ppv=np.zeros(zone)
new_dTdP_en=np.zeros(zone)
new_dTdP_pv_mix=np.zeros(zone)
new_dTdP_ppv_mix=np.zeros(zone)
new_dTdP_en_mix=np.zeros(zone)

new_T_old=np.zeros(zone)
new_dqdy_old=np.zeros(zone)
new_density_old=np.zeros(zone)
new_alpha_old=np.zeros(zone)
new_CP_old=np.zeros(zone)
new_dTdP_old=np.zeros(zone)
new_T_cell_old=np.zeros(zone)
rho_m_array_old=np.zeros(zone)
rho_s_array_old=np.zeros(zone)

cdef double[:] k_array=np.zeros(zone)
for i in range(zone):
    if i<core_outer_index+1:
        k_array[i]=k_Fe
    else:
        if initial_pressure[i]<23.0*10.0**9.0:
            k_array[i]=k_ol
        else:
            k_array[i]=k_pv

cdef double Fcmb, Q_rad_m, v1,v2,v3,y,Y_s,Z_s,Y_m,Z_m,v4,v5,v6,v7,Z_Re,Y_Re
cdef double ds_top
cdef double S_max=5100.0#5384.0#5100.0#4739.0
cdef double S_min=100.0

cdef double Q_th,Q,Q_rad_c,delta_Tcmb, Q_ICB
cdef double outer_adiabat
cdef double gamma_mix, debye_T_mix, debye_int_mix, debye_int_0_mix
cdef double x, rho_tot, dqdy_tot, KT_value

cdef double[:] Tc_array=np.zeros(core_outer_index+1)
cdef double[:] outer_adiabat_array=np.zeros(core_outer_index+1)
cdef double[:] dTdT0_array=np.zeros(core_outer_index+1)
cdef double[:] adiabat_array=np.zeros(core_outer_index+1)
cdef double dTdT0_ic

cdef double min_pre_adia_T=previous[5]
cdef double Ric=previous[6]
cdef double rho_ic=previous[7]
cdef double Mic=previous[8]
cdef double Tic=previous[9]
cdef double alpha_ic=previous[10]
cdef double Pic=previous[11]
cdef int solid_index=previous[12]
cdef double x_core=0.105#previous[]

cdef double old_Tic, old_Pic
cdef double a0,b0,a1,b1,A0,B0,A1,B1
cdef double[:] pressure_GPa_record=np.zeros(zone)
cdef double[:] pressure_record=np.zeros(zone)
pressure_record=initial_pressure.copy()
cdef double[:] x_record=np.ones(zone)*x_init
cdef double[:] phase_change=np.zeros(zone)
cdef double[:] alpha_Fe_l=alpha.copy()
cdef double[:] rho_Fe_l=initial_density.copy()

cdef double dmicdTcmb=0.0
cdef double phase_core=0.0
cdef double dmicdPic=0.0
cdef double dPicdTic=0.0
cdef double dTicdTcmb=0.0
cdef double[:] Tc_adiabat=np.zeros(core_outer_index+1)
Tc_adiabat_np=np.zeros(zone)
cdef double[:] melting_pressure_GPa=np.zeros(zone)
cdef double[:] initial_pressure_GPa=np.zeros(zone)
for i in range(zone):
    melting_pressure_GPa[i]=melt_pressure[i]/10.0**9.0
    initial_pressure_GPa[i]=initial_pressure[i]/1e9
pressure_GPa_record=initial_pressure_GPa.copy()
cdef int pre_adia_p_idx
cdef double dTdT0_cmb
cdef double pre_adia_rho
cdef double pre_adia_K_T
cdef double pre_adia_gamma
cdef double pre_adia_alpha
cdef double delta_T
cdef double pre_adia_debye_T_value
cdef double pre_adia_debye_int
cdef double pre_adia_debye_int_0

cdef double dT0dPcmb
cdef double outer_adiabat_Pcmb
cdef double delta_Pcmb
cdef double outer_adiabat_value
cdef double[:] outer_adiabat_Pcmb_array=np.zeros(core_outer_index+1)
cdef double[:] outer_adiabat_Pi_array=np.zeros(core_outer_index+1)
cdef double outer_adiabat_Pi

cdef double PdV_core_sum
cdef double dEG_core_sum
cdef double dV_core=0.0
cdef double PdV_cmb=0.0
dV_core=4.0/3.0*math.pi*(initial_radius[core_outer_index]**3.0-old_radius[core_outer_index]**3.0)

# Melting temperature of iron throughout the core.
#cdef double[:] T_simon_array=np.zeros(zone)
#for i in range(core_outer_index+1):
#    T_simon_array[i]=T_simon(melting_pressure_GPa[i])

cdef double[:] aa=np.zeros(zone-1)
cdef double[:] bb=np.zeros(zone-1)
cdef double[:] cc=np.zeros(zone-1)
cdef double[:] dd=np.zeros(zone-1)
cdef double[:] ee=np.zeros(zone-1)
cdef double[:] ff=np.zeros(zone-1)

cdef double[:] A_r=np.zeros(zone)
cdef double[:] A_p=np.zeros(zone)
cdef double[:] a=np.zeros(zone)
cdef double[:] b=np.zeros(zone)
cdef double[:] c=np.zeros(zone)
cdef double[:] d=np.zeros(zone)
cdef double[:] A=np.zeros(zone)
cdef double[:] B=np.zeros(zone)
cdef double[:] C=np.zeros(zone)
cdef double[:] D=np.zeros(zone) # lower case is for pressure. Upper case is for radius

cdef double[:] alp=np.zeros(zone)
cdef double[:] gam=np.zeros(zone)
cdef double[:] delta_y=np.zeros(zone)
cdef double[:] delta_x=np.zeros(zone)

cdef double[:] dr=np.zeros(zone)
for i in range(1,zone):
    dr[i]=initial_radius[i]-initial_radius[i-1]

cdef double rho_center, T_center
cdef double gamma, debye_T_value, debye_int, debye_int_0, initial_dqdy_center
T_center=initial_temperature[0]

cdef double[:] solution=np.zeros(zone-core_outer_index-1)
cdef double[:] solution_T=np.zeros(core_outer_index+1)
cdef double[:] new_Scell=np.zeros(zone)
cdef double[:] Fconv=np.zeros(zone)
cdef double[:] Fcond=np.zeros(zone)
cdef double[:] Ftot=np.zeros(zone)
cdef double[:] Fcond1=np.zeros(zone)
cdef double[:] Fcond2=np.zeros(zone)

cdef double[:] dynamo_magma_bot=np.ones(zone)*(core_outer_index+1)
cdef double[:] dynamo_magma_top=np.ones(zone)*(zone-1)
cdef int i_dy_magma_top,i_dy_magma_bot

cdef double[:] GMms=np.zeros(zone)
for i in range(zone):
    if i<core_outer_index+1:
        GMms[i]=mass[i]*h[i]*G
    else:
        GMms[i]=mass[i]*h[i]*G

cdef int iteration=0

cdef double buoy_T_value, buoy_x_value, rho_core, old_Ric, delta_r, delta_T_ra, old_delta_r,delta_r_flag,M_base_cond_flag
M_base_cond_flag=0.0
cdef double Ra_T,Ra_P,Ra_nu,Ra_g
old_delta_r=1.0
cdef double Ra_r_cs,Ra_T_s,delta_T_ra_s,Ra_P_s,Ra_nu_s,Ra_g_s,Ra_S_s,Ra_x_s,Ra_rho_s,Ra_CP_s,Ra_alpha_s
cdef double Ra_Sliq_s, Ra_Ssol_s, Ra_y_s, dm_s, r_cs, Area_b, S_s, y_s
cdef double delta_T_ra_s_raw
cdef double old_Ra_r_s=initial_radius[zone-1]-1.0
cdef double T_cs
cdef double old_Ra_T_s=temperature_cell[zone-1]
cdef double old_S_s=initial_S[zone-1]
cdef double T_s=initial_temperature[zone-1]
cdef double dt_Ra_flag=0.0
cdef double dt_Ra_flag1=0.0
cdef double r_c_m1=old_Ra_r_s-(old_Ra_r_s-initial_radius[zone-2])/2.0
cdef double g_c_m1,p_c_m1,S_liq_c_m1,S_sol_c_m1,rho_c_m1
cdef double surf_flag=1.0
cdef double old_Ra_P_s=1e5
cdef double Ttol=1.0
cdef double rtol=1e-6
cdef double old_T_s=T_s-1000.0
Ra_P_s=1e5
S_s=old_S_s
cdef double[:] old_S_liquidus=S_liquidus.copy()
cdef double[:] old_S_solidus=S_solidus.copy()
cdef double debug1, debug2, debug3, debug4, debug5, debug6, debug7

cdef double L_sigma, D_MO_dynamo, flag_top, flag_bot, MO_dynamo_bot, MO_dynamo_top
cdef int S_MO_index=core_outer_index+1
cdef double[:] sigma_MO=np.zeros(zone)
cdef double[:] v_MO=np.zeros(zone)
cdef double[:] Rem_MO=np.zeros(zone)

cdef double t_val

cdef double[:] t_save1=10.0**np.linspace(1.0,8.9,80)
cdef double[:] t_save2=np.linspace(1.0,20.0,77)*1e9
cdef double[:] t_save=np.concatenate((np.asarray(t_save1),np.asarray(t_save2)))

cdef Py_ssize_t ind

cdef double core_m

cdef double old_delta_r_s, old_vis_surf_value, old_Fsurf

cdef double[:] save_test=np.linspace(12500.0,50000.0,7501)

cdef double width
cdef double solid_core_flag=0.0
cdef double t_final=0.0
cdef Py_ssize_t solid_core_end_ind=core_outer_index-2

testing_delta_T_ra=[]
testing_delta_r=[]
testing_t=[]
testing_dt=[]
testing_x=[]

#while iteration<end_ite:
#while solid_index<core_outer_index:
while t<end_time:
    #if solid_index>core_outer_index-2:
    if initial_radius[solid_index]>initial_radius[solid_core_end_ind]:
        solid_core_flag=1.0
        t_final=t
        break
    dsdr=np.zeros(zone)
    for i in range(core_outer_index+1,zone-1):
        dsdr[i]=(S_cell[i]-S_cell[i+1])/(radius_cell[i]-radius_cell[i+1])
    
    viscosity=np.zeros(zone) # evaluated at the cell center
    for i in range(core_outer_index+1, zone):
        width=0.15
        viscosity[i]=f_viscosity(initial_temperature[i], initial_pressure[i], initial_density[i], initial_phase[i], melt_frac[i], 3700.0, 4000.0, width, ppv_eta_model)

    convection=np.zeros(zone)
    for i in range(zone):
        if i<core_outer_index+1:
            convection[i]=0.0
        else:
            if dsdr[i]<0.0:
                convection[i]=1.0
            else:
                convection[i]=0.0

    l_mlt=np.zeros(zone)
    for i in range(zone):
        if convection[i]==1.0:
            if (initial_radius[i]-initial_radius[core_outer_index])<=(initial_radius[zone-1]-initial_radius[core_outer_index])/2.0*l_beta:
                l_mlt[i]=l_alpha*(initial_radius[i]-initial_radius[core_outer_index])/l_beta
            else:
                l_mlt[i]=l_alpha*(initial_radius[zone-1]-initial_radius[i])/(2.0-l_beta)

    eddy_T_high_nu=np.zeros(zone)
    eddy_T_low_nu=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        eddy_T_high_nu[i]=f_eddy_T_high_nu(initial_temperature[i],CP[i],alpha[i],initial_gravity[i],l_mlt[i],viscosity[i],dsdr[i])
        eddy_T_low_nu[i]=f_eddy_T_low_nu(initial_temperature[i],CP[i],alpha[i],initial_gravity[i],l_mlt[i],viscosity[i],dsdr[i])

    Re=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        Re[i]=eddy_T_low_nu[i]/viscosity[i]

    eddy_k=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        if Re[i]>9.0/8.0:
            eddy_k[i]=eddy_T_low_nu[i]
        else:
            eddy_k[i]=eddy_T_high_nu[i]


    Ra_T=initial_temperature[core_outer_index+1]-(initial_temperature[core_outer_index+2]-initial_temperature[core_outer_index+1])/(initial_radius[core_outer_index+2]-initial_radius[core_outer_index+1])*((initial_radius[core_outer_index+1]-initial_radius[core_outer_index])-old_delta_r)
    delta_T_ra=initial_temperature[core_outer_index]-Ra_T
    Ra_P=initial_pressure[core_outer_index]+(initial_pressure[core_outer_index+1]-initial_pressure[core_outer_index])/(initial_radius[core_outer_index+1]-initial_radius[core_outer_index])*old_delta_r
    #Ra_x=(initial_tempearture[core_outer_index+1]-)/(-)
    Ra_nu=f_viscosity(Ra_T, Ra_P, initial_density[core_outer_index+1], initial_phase[core_outer_index+1], melt_frac[core_outer_index+1], 3700.0, 4000.0, 0.15,ppv_eta_model)
    Ra_g=initial_gravity[core_outer_index]+(initial_gravity[core_outer_index+1]-initial_gravity[core_outer_index])/(initial_radius[core_outer_index+1]-initial_radius[core_outer_index])*old_delta_r
    old_delta_r=delta_r
    if delta_T_ra<1e-3:
        delta_T_ra=1e-3
    delta_r=(10.0*Ra_nu*660.0/(initial_density[core_outer_index+1]*CP[core_outer_index+1]*alpha[core_outer_index+1]*Ra_g*delta_T_ra))**(1.0/3.0)
    if delta_r>initial_radius[core_outer_index+1]-initial_radius[core_outer_index]:
        delta_r_flag=1.0
        delta_r=initial_radius[core_outer_index+1]-initial_radius[core_outer_index]
        delta_T_ra=initial_temperature[core_outer_index]-initial_temperature[core_outer_index+1]
        Ra_T=initial_temperature[core_outer_index+1]

    Fcmb=-10.0*(delta_T_ra)/(-delta_r)
    Q_rad_m=Q_radm_Th*math.exp(math.log(2.0)*(4.5-t/86400.0/365.0/10.0**9.0)/tau_m_Th)+Q_radm_K*math.exp(math.log(2.0)*(4.5-t/86400.0/365.0/10.0**9.0)/tau_m_K)+Q_radm_U5*math.exp(math.log(2.0)*(4.5-t/86400.0/365.0/10.0**9.0)/tau_m_U5)+Q_radm_U8*math.exp(math.log(2.0)*(4.5-t/86400.0/365.0/10.0**9.0)/tau_m_U8)

    aa=np.zeros(zone)
    bb=np.zeros(zone)
    cc=np.zeros(zone)
    dd=np.zeros(zone)
    ee=np.zeros(zone)
    ff=np.zeros(zone)

    i=core_outer_index+1
    aa[i]=0.0
    bb[i]=0.0
    cc[i]=1.0/dt-Area[i]/(h[i]*temperature_cell[i])*initial_density[i]*initial_temperature[i]*(kappa[i]+eddy_k[i])/(radius_cell[i]-radius_cell[i+1])
    dd[i]=Area[i]/(h[i]*temperature_cell[i])*initial_density[i]*initial_temperature[i]*(kappa[i]+eddy_k[i])/(radius_cell[i]-radius_cell[i+1])
    ee[i]=0.0
    v1=Area[i-1]*Fcmb/h[i]/temperature_cell[i]
    v2=Area[i]/(h[i]*temperature_cell[i])*k_array[i]*dTdP[i]*dPdr[i]
    v3=0.0
    ff[i]=S_cell[i]/dt+v1+v2+v3+Q_rad_m/temperature_cell[i]

    old_vis_surf_value=vis_surf_value
    old_Fsurf=Fsurf
    if S_cell[-1]>sgrid[-1] and surf_flag==1.0:
    #if surf_flag==1.0:
        Fsurf=f_Fsurf_s(S_cell[-1]).tolist()
        dFds=f_dFds_s(S_cell[-1]).tolist()
        delta_r_s=f_deltaUBL_s(S_cell[-1]).tolist()
        T_s=f_Tsurf_s(S_cell[-1]).tolist()
        vis_surf_value=f_vissurf_s(S_cell[-1]).tolist()
    else:
        surf_flag=0.0
        delta_r_s=initial_radius[zone-1]-radius_cell[zone-1]
        delta_T_ra_s=temperature_cell[-1]-T_s
        Fsurf=k_array[-1]*delta_T_ra_s/delta_r_s


    i=zone-1
    #if surf_flag==1.0:
    if S_cell[-1]>sgrid[-1] and surf_flag==1.0:
        aa[i]=0.0
        bb[i]=Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])
        cc[i]=1.0/dt-Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])+dFds*Area[i]/(h[i]*temperature_cell[i])
        dd[i]=0.0
        ee[i]=0.0
        v1=0.0
        v2=-Area[i-1]/(h[i]*temperature_cell[i])*k_array[i]*dTdP[i-1]*dPdr[i-1]
        v3=-Fsurf*Area[i]/(h[i]*temperature_cell[i])+dFds*S_cell[i]*Area[i]/(h[i]*temperature_cell[i])
        ff[i]=S_cell[i]/dt+v1+v2+v3+Q_rad_m/temperature_cell[i]
    else:
        aa[i]=0.0
        bb[i]=Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])
        cc[i]=1.0/dt-Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])
        dd[i]=0.0
        ee[i]=0.0
        v1=0.0
        v2=-Area[i-1]/(h[i]*temperature_cell[i])*k_array[i]*dTdP[i-1]*dPdr[i-1]
        v3=-Fsurf*Area[i]/(h[i]*temperature_cell[i])
        ff[i]=S_cell[i]/dt+v1+v2+v3+Q_rad_m/temperature_cell[i]


    for i in range(core_outer_index+2,zone-1):
        aa[i]=0.0
        bb[i]=Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])
        cc[i]=1.0/dt-Area[i-1]/(h[i]*temperature_cell[i])*initial_density[i-1]*initial_temperature[i-1]*(kappa[i-1]+eddy_k[i-1])/(radius_cell[i-1]-radius_cell[i])-Area[i]/(h[i]*temperature_cell[i])*initial_density[i]*initial_temperature[i]*(kappa[i]+eddy_k[i])/(radius_cell[i]-radius_cell[i+1])
        dd[i]=Area[i]/(h[i]*temperature_cell[i])*initial_density[i]*initial_temperature[i]*(kappa[i]+eddy_k[i])/(radius_cell[i]-radius_cell[i+1])
        ee[i]=0.0
        v1=-Area[i-1]/(h[i]*temperature_cell[i])*k_array[i]*dTdP[i-1]*dPdr[i-1]
        v2=Area[i]/(h[i]*temperature_cell[i])*k_array[i]*dTdP[i]*dPdr[i]
        v3=0.0
        ff[i]=S_cell[i]/dt+v1+v2+v3+Q_rad_m/temperature_cell[i]
    solution=penta_solver(aa[core_outer_index+1:],bb[core_outer_index+1:],cc[core_outer_index+1:],dd[core_outer_index+1:],ee[core_outer_index+1:],ff[core_outer_index+1:])#,zone-core_outer_index-1)

    #if surf_flag==0.0:
    #if S_cell[-1]<=sgrid[-1]:
    T_s=(Fsurf/sigma+Teq**4.0)**0.25

    new_Scell=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        new_Scell[i]=solution[i-core_outer_index-1]
    new_S=np.zeros(zone)
    for i in range(core_outer_index+1,zone-1):
        new_S[i]=(new_Scell[i]+new_Scell[i+1])/2.0
    new_S[zone-1]=new_Scell[zone-1]

    Fcond1=np.zeros(zone)
    Fcond2=np.zeros(zone)
    Fcond=np.zeros(zone)
    Fconv=np.zeros(zone)
    Ftot=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        if i==zone-1:
            Ftot[i]=Fsurf
            Fconv[i]=0.0
            Fcond[i]=Fsurf
            Fcond1[i]=0.0
            Fcond2[i]=0.0
        else:
            Fcond1[i]=-10.0*dTdP[i]*dPdr[i]#*Area[i]
            Fcond2[i]=-initial_density[i]*initial_temperature[i]*kappa[i]*dsdr[i]#*Area[i]
            Fconv[i]=-initial_density[i]*initial_temperature[i]*eddy_k[i]*dsdr[i]#*Area[i]
            Fcond[i]=Fcond1[i]+Fcond2[i]
            Ftot[i]=Fconv[i]+Fcond[i]
    new_T_cell=np.zeros(zone)
    new_x_cell=np.zeros(zone)
    for i in range(core_outer_index+1,zone):
        if new_Scell[i]>=S_liquidus_cell[i]:
            y_array[i]=(new_Scell[i]-S_liquidus_cell[i])/(S_max-S_liquidus_cell[i])
            new_x_cell[i]=1.0
        elif new_Scell[i]<=S_solidus_cell[i]:
            y_array[i]=(new_Scell[i]-S_min)/(S_solidus_cell[i]-S_min)
            new_x_cell[i]=0.0
        else:
            y_array[i]=(new_Scell[i]-S_solidus_cell[i])/(S_liquidus_cell[i]-S_solidus_cell[i])
            new_x_cell[i]=y_array[i]
    new_T_cell_l[core_outer_index+1:]=T_interp_2d_liq((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_pv[core_outer_index+1:]=T_interp_2d_sol_pv((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_ppv[core_outer_index+1:]=T_interp_2d_sol_ppv((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_en[core_outer_index+1:]=T_interp_2d_sol_en((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_pv_mix[core_outer_index+1:]=T_interp_2d_mix_pv((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_ppv_mix[core_outer_index+1:]=T_interp_2d_mix_ppv((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_cell_en_mix[core_outer_index+1:]=T_interp_2d_mix_en((pressure_cell_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    for i in range(core_outer_index+1,zone):
        if new_Scell[i]>=S_liquidus_cell[i]:
            new_T_cell[i]=new_T_cell_l[i]
        elif new_Scell[i]<=S_solidus_cell[i]:
            if pressure_cell[i]<23.0*10.0**9.0:
                new_T_cell[i]=new_T_cell_en[i]
            elif pressure_cell[i]>=23.0e9 and pressure_cell[i]<125.0e9:
                new_T_cell[i]=new_T_cell_pv[i]
            else:
                new_T_cell[i]=new_T_cell_ppv[i]
        else:
            if pressure_cell[i]<23.0*10.0**9.0:
                new_T_cell[i]=new_T_cell_en_mix[i]
            elif pressure_cell[i]>=23.0e9 and pressure_cell[i]<125.0e9:
                new_T_cell[i]=new_T_cell_pv_mix[i]
            else:
                new_T_cell[i]=new_T_cell_ppv_mix[i]

    new_T=np.zeros(zone)
    new_x=np.zeros(zone)
    new_phase=np.zeros(zone)
    new_dqdy=np.zeros(zone)
    new_density=np.zeros(zone)
    new_alpha=np.zeros(zone)
    new_CP=np.zeros(zone)
    new_dTdP=np.zeros(zone)
    new_kappa=np.zeros(zone)
    new_dPdr=np.zeros(zone)
    new_rho_m=np.zeros(zone)
    new_rho_s=np.zeros(zone)

    for i in range(core_outer_index+1,zone):
        if new_S[i]>=S_liquidus[i]:
            y_array[i]=(new_S[i]-S_liquidus[i])/(S_max-S_liquidus[i])
            new_x[i]=1.0
            new_phase[i]=ph_pv_liq
            if initial_pressure[i]<23.0*10.0**9.0:
                k_array[i]=k_ol
            else:
                k_array[i]=k_pv

        elif new_S[i]<=S_solidus[i]:
            y_array[i]=(new_S[i]-S_min)/(S_solidus[i]-S_min)
            if initial_pressure[i]<23.0*10.0**9.0:
                k_array[i]=k_ol
            else:
                k_array[i]=k_pv
            new_x[i]=0.0
            new_phase[i]=ph_pv_sol
        else:
            y_array[i]=(new_S[i]-S_solidus[i])/(S_liquidus[i]-S_solidus[i])
            new_x[i]=y_array[i]#x_PS_mix(y,initial_pressure[i])[0]
            new_phase[i]=ph_pv_mix
            if initial_pressure[i]<23.0*10.0**9.0:
                k_array[i]=k_ol
            else:
                k_array[i]=k_pv

    new_T_l[core_outer_index+1:] = T_interp_2d_liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_pv[core_outer_index+1:] = T_interp_2d_sol_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_ppv[core_outer_index+1:] = T_interp_2d_sol_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_en[core_outer_index+1:] = T_interp_2d_sol_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_pv_mix[core_outer_index+1:] = T_interp_2d_mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_ppv_mix[core_outer_index+1:] = T_interp_2d_mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_T_en_mix[core_outer_index+1:] = T_interp_2d_mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    new_dqdy_l[core_outer_index+1:] = T_interp_2d_dy_liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_pv[core_outer_index+1:] = T_interp_2d_dy_sol_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_ppv[core_outer_index+1:] = T_interp_2d_dy_sol_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_en[core_outer_index+1:] = T_interp_2d_dy_sol_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_pv_mix[core_outer_index+1:] = T_interp_2d_dy_mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_ppv_mix[core_outer_index+1:] = T_interp_2d_dy_mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dqdy_en_mix[core_outer_index+1:] = T_interp_2d_dy_mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    new_density_l[core_outer_index+1:] = T_interp_2d_o_liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_pv[core_outer_index+1:] = T_interp_2d_o_sol_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_ppv[core_outer_index+1:] = T_interp_2d_o_sol_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_en[core_outer_index+1:] = T_interp_2d_o_sol_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_pv_mix[core_outer_index+1:] = T_interp_2d_o_mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_ppv_mix[core_outer_index+1:] = T_interp_2d_o_mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_density_en_mix[core_outer_index+1:] = T_interp_2d_o_mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    new_CP_l[core_outer_index+1:] = T_interp_2d__liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_CP_pv[core_outer_index+1:]=CP_s*np.ones(mantle_zone)
    new_CP_ppv[core_outer_index+1:]=CP_s*np.ones(mantle_zone)
    new_CP_en[core_outer_index+1:]=CP_s*np.ones(mantle_zone)
    new_CP_pv_mix[core_outer_index+1:] = T_interp_2d__mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_CP_ppv_mix[core_outer_index+1:] = T_interp_2d__mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_CP_en_mix[core_outer_index+1:] = T_interp_2d__mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    new_alpha_l[core_outer_index+1:] = T_interp_2d_pha_liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_pv[core_outer_index+1:] = T_interp_2d_pha_sol_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_ppv[core_outer_index+1:] = T_interp_2d_pha_sol_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_en[core_outer_index+1:] = T_interp_2d_pha_sol_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_pv_mix[core_outer_index+1:] = T_interp_2d_pha_mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_ppv_mix[core_outer_index+1:] = T_interp_2d_pha_mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_alpha_en_mix[core_outer_index+1:] = T_interp_2d_pha_mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    new_dTdP_l[core_outer_index+1:] = T_interp_2d_dP_liq((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_pv[core_outer_index+1:] = T_interp_2d_dP_sol_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_ppv[core_outer_index+1:] = T_interp_2d_dP_sol_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_en[core_outer_index+1:] = T_interp_2d_dP_sol_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_pv_mix[core_outer_index+1:] = T_interp_2d_dP_mix_pv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_ppv_mix[core_outer_index+1:] = T_interp_2d_dP_mix_ppv((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))
    new_dTdP_en_mix[core_outer_index+1:] = T_interp_2d_dP_mix_en((pressure_np[core_outer_index+1:],y_array[core_outer_index+1:]))

    for i in range(core_outer_index+1,zone):
        if new_phase[i]==ph_pv_liq:
            new_T[i]=new_T_l[i]
            new_dqdy[i]=new_dqdy_l[i]
            new_density[i]=new_density_l[i]
            new_alpha[i]=new_alpha_l[i]
            new_CP[i]=new_CP_l[i]
            new_dTdP[i]=new_dTdP_l[i]
            new_kappa[i]=k_array[i]/(new_density[i]*new_CP[i])
            new_dPdr[i]=-G*mass[i]*new_density[i]/initial_radius[i]**2.0
        elif new_phase[i]==ph_pv_sol:
            if initial_pressure[i]<23.0*10.0**9.0:
                new_T[i]=new_T_en[i]
                new_dqdy[i]=new_dqdy_en[i]
                new_density[i]=new_density_en[i]
                new_alpha[i]=new_alpha_en[i]
                new_CP[i]=new_CP_en[i]
                new_dTdP[i]=new_dTdP_en[i]
            elif initial_pressure[i]>=23.0e9 and initial_pressure[i]<125.0e9:
                new_T[i]=new_T_pv[i]
                new_dqdy[i]=new_dqdy_pv[i]
                new_density[i]=new_density_pv[i]
                new_alpha[i]=new_alpha_pv[i]
                new_CP[i]=new_CP_pv[i]
                new_dTdP[i]=new_dTdP_pv[i]
            else:
                new_T[i]=new_T_ppv[i]
                new_dqdy[i]=new_dqdy_ppv[i]
                new_density[i]=new_density_ppv[i]
                new_alpha[i]=new_alpha_ppv[i]
                new_CP[i]=new_CP_ppv[i]
                new_dTdP[i]=new_dTdP_ppv[i]
            new_kappa[i]=k_array[i]/(new_density[i]*new_CP[i])
            new_dPdr[i]=-G*mass[i]*new_density[i]/initial_radius[i]**2.0
        else:
            if initial_pressure[i]<23.0*10.0**9.0:
                new_T[i]=new_T_en_mix[i]
                new_dqdy[i]=new_dqdy_en_mix[i]
                new_density[i]=new_density_en_mix[i]
                new_alpha[i]=new_alpha_en_mix[i]
                new_CP[i]=new_CP_en_mix[i]
                new_dTdP[i]=new_dTdP_en_mix[i]
                new_rho_s[i]=4000.0
            elif initial_pressure[i]>=23.0e9 and initial_pressure[i]<125.0e9:
                new_T[i]=new_T_pv_mix[i]
                new_dqdy[i]=new_dqdy_pv_mix[i]
                new_density[i]=new_density_pv_mix[i]
                new_alpha[i]=new_alpha_pv_mix[i]
                new_CP[i]=new_CP_pv_mix[i]
                new_dTdP[i]=new_dTdP_pv_mix[i]
                new_rho_s[i]=4000.0
            else:
                new_T[i]=new_T_ppv_mix[i]
                new_dqdy[i]=new_dqdy_ppv_mix[i]
                new_density[i]=new_density_ppv_mix[i]
                new_alpha[i]=new_alpha_ppv_mix[i]
                new_CP[i]=new_CP_ppv_mix[i]
                new_dTdP[i]=new_dTdP_ppv_mix[i]
                new_rho_s[i]=4000.0
            new_rho_m[i]=3700.0
            new_kappa[i]=k_array[i]/(new_density[i]*new_CP[i])
            new_dPdr[i]=-G*mass[i]*new_density[i]/initial_radius[i]**2.0

    i=0
    aa[i]=0.0
    bb[i]=0.0
    cc[i]=1.0/dt+(k_array[i]*Area[i])/(h[i]*CP[i])/(radius_cell[i+1]-radius_cell[i])
    dd[i]=-(k_array[i]*Area[i])/(h[i]*CP[i])/(radius_cell[i+1]-radius_cell[i])
    ee[i]=0.0
    ff[i]=temperature_cell[i]/dt+alpha[i]*initial_temperature[i]/initial_density[i]/CP[i]*(initial_pressure[i]-old_pressure[i])/dt
    i=core_outer_index
    aa[i]=0.0
    bb[i]=-(k_array[i-1]*Area[i-1])/(h[i]*CP[i-1])/(radius_cell[i]-radius_cell[i-1])
    cc[i]=1.0/dt+(k_array[i-1]*Area[i-1])/(h[i]*CP[i-1])/(radius_cell[i]-radius_cell[i-1])
    dd[i]=0.0
    ee[i]=0.0
    ff[i]=temperature_cell[i]/dt+alpha[i]*initial_temperature[i]/initial_density[i]/CP[i]*(initial_pressure[i]-old_pressure[i])/dt
    for i in range(1,core_outer_index):
        aa[i]=0.0
        bb[i]=-(k_array[i-1]*Area[i-1])/(h[i]*CP[i-1])/(radius_cell[i]-radius_cell[i-1])
        cc[i]=1.0/dt+(k_array[i]*Area[i])/(h[i]*CP[i])/(radius_cell[i+1]-radius_cell[i])+(k_array[i-1]*Area[i-1])/(h[i]*CP[i-1])/(radius_cell[i]-radius_cell[i-1])
        dd[i]=-(k_array[i]*Area[i])/(h[i]*CP[i])/(radius_cell[i+1]-radius_cell[i])
        ee[i]=0.0
        ff[i]=temperature_cell[i]/dt+alpha[i]*initial_temperature[i]/initial_density[i]/CP[i]*(initial_pressure[i]-old_pressure[i])/dt
    solution_T=penta_solver(aa[:core_outer_index+1],bb[:core_outer_index+1],cc[:core_outer_index+1],dd[:core_outer_index+1],ee[:core_outer_index+1],ff[:core_outer_index+1])#,core_outer_index+1)
    #for i in range(core_outer_index):

    # add core stuff
    Q_th=-Fcmb*Area[core_outer_index]
    if Mic>0.0:
        Q_ICB=-k_array[solid_index]*(solution_T[solid_index]-solution_T[solid_index+1])/(radius_cell[solid_index]-radius_cell[solid_index+1])*Area[solid_index]
    else:
        Q_ICB=0.0

    pre_adia_p_idx=find_nearest(np.asarray(P_core_grid),initial_pressure[core_outer_index])
    pre_adia_p=np.zeros(pre_adia_p_idx+1)
    for i in range(len(pre_adia_p)-1):
        pre_adia_p[i]=P_core_grid[i]
    pre_adia_p[pre_adia_p_idx]=initial_pressure[core_outer_index]
    pre_adia_p=pre_adia_p[::-1]

    # Heating related to change in the pressure level in the core due to cooling
    x_alloy=x_core/mf_l

    dTdT0_cmb = dTdT0_cmb_interp((
        [x_alloy],
        [min_pre_adia_T],
        [initial_pressure[core_outer_index]],
    ))[0]
    #print(x_alloy,min_pre_adia_T,initial_temperature[core_outer_index])
    dT0dPcmb=f_dT0dP([x_alloy,min_pre_adia_T,initial_temperature[core_outer_index]])[0]
    delta_Pcmb=initial_pressure[core_outer_index]-old_pressure[core_outer_index]
    dTdT0_array=dTdT0_cmb_interp((
        np.ones(core_outer_index+1)*x_alloy,
        np.ones(core_outer_index+1)*min_pre_adia_T,
        pressure_np[:core_outer_index+1],
    ))
    for i in range(core_outer_index+1):
        outer_adiabat_value=h[i]*C_P_Fe*dTdT0_array[i]
        outer_adiabat_array[i]=outer_adiabat_value/dTdT0_cmb
        outer_adiabat_Pcmb_array[i]=outer_adiabat_value*dT0dPcmb
    outer_adiabat=np.sum(outer_adiabat_array[solid_index:])
    outer_adiabat_Pcmb=np.sum(outer_adiabat_Pcmb_array[solid_index:])*delta_Pcmb
    # Latent heat when there's ongoing inner core solidification
    if initial_phase[0]==ph_Fe_sol and initial_phase[core_outer_index]==ph_Fe_liq:
        dTdT0_ic=dTdT0_cmb_interp((
            [x_alloy],
            [min_pre_adia_T],
            [Pic],
        ))[0]
        dmicdPic=-4.0*math.pi/G*Ric**4.0/Mic
        dPicdTic=1.0/(alpha_ic/rho_ic*Tic/C_P_Fe)
        dTicdTcmb=dTdT0_ic/dTdT0_cmb
        dmicdTcmb=dmicdPic*dPicdTic*dTicdTcmb
    else:
        dmicdTcmb=0.0
    outer_adiabat=outer_adiabat-dmicdTcmb*L_Fe*(core_zone/(core_zone-0.0))

    PdV_core_sum=0.0
    dEG_core_sum=0.0
    PdV_cmb=0.0

    Q_rad_c=Q_rad_c_0*math.exp(-t/86400.0/365.0/1e9/1.2)
    Q=Q_th+Q_rad_c+Q_ICB
    #delta_Tcmb=(dt*Q+outer_adiabat_Pi-dEG_core_sum+PdV_core_sum-PdV_cmb-outer_adiabat_Pcmb)/outer_adiabat # Change in the temperature at the core mantle boundary.
    delta_Tcmb=(dt*Q-outer_adiabat_Pcmb)/outer_adiabat
    T_cmb=T_cmb+delta_Tcmb

    # Look for the reference temperature, min_pre_adia_T, for the core adiabat.
    pre_adia_T=[T_cmb]
    for i in range(0,len(pre_adia_p)-1):
        rho_liquid=f_rho_Fel(pre_adia_T[i],pre_adia_p[i])[0]
        rho_alloy=f_rho_Fea(pre_adia_T[i],pre_adia_p[i])[0]
        alpha_liquid=f_alpha_Fel(pre_adia_T[i],pre_adia_p[i])[0]
        alpha_alloy=f_alpha_Fea(pre_adia_T[i],pre_adia_p[i])[0]
        pre_adia_rho=rho_mix(x_alloy,rho_alloy,rho_liquid)
        pre_adia_alpha=alpha_mix(x_alloy,alpha_alloy,alpha_liquid,pre_adia_rho,rho_alloy,rho_liquid)
        delta_T=f_dTdP(pre_adia_alpha,pre_adia_rho,C_P_Fe,pre_adia_T[i])*(pre_adia_p[i+1]-pre_adia_p[i])
        pre_adia_T.append(pre_adia_T[i]+delta_T)
    min_pre_adia_T=pre_adia_T[len(pre_adia_T)-1]

    # update the temperature profile in the core.
    adiabat_array = T_interp((
        np.ones(core_outer_index+1)*x_alloy,
        np.ones(core_outer_index+1)*min_pre_adia_T,
        pressure_np[:core_outer_index+1],
    ))
    for i in range(core_outer_index+1):
        if initial_phase[i]==0.0:
            Tc_array[i]=f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist()#T_simon(initial_pressure[i]/10.0**9.0,x_core)
            Tc_adiabat[i]=adiabat_array[i]
        elif initial_phase[i]==1.0:
            Tc_array[i]=adiabat_array[i]
            Tc_adiabat[i]=Tc_array[i]

    if initial_phase[0]==ph_Fe_liq:
        phase_core=ph_Fe_liq
        T_center=Tc_array[0]
    else:
        phase_core=ph_Fe_sol
        T_center=f_interp_Tsimon(melt_pressure_Fe_GPa[0]).tolist()#T_simon(P_center/10.0**9.0)
    if Tc_array[0]>f_interp_Tsimon(melt_pressure_Fe_GPa[0]).tolist():#T_simon(melt_pressure[0]/10.0**9.0):
        initial_phase[0]=ph_Fe_liq
    else:
        initial_phase[0]=ph_Fe_sol
        Tc_array[0]=f_interp_Tsimon(melt_pressure_Fe_GPa[0]).tolist()#T_simon(melt_pressure[0]/10.0**9.0)

    old_Ric=Ric
    if initial_phase[0]==0.0:
        f_interp=interpolate.interp1d(melt_pressure_Fe_GPa[0:core_outer_index+1],Tc_adiabat[0:core_outer_index+1])
        if Tc_adiabat[0]>f_interp_Tsimon(melt_pressure_Fe_GPa[0]).tolist():#T_simon(melting_pressure_GPa[0]):
            Pic=melt_pressure_Fe[0]#melting_pressure_GPa[0]
        else:
            Pic=fsolve(func, x0=0.5*(melt_pressure_Fe_GPa[solid_index]+melt_pressure_Fe_GPa[solid_index+1]))[0]
        Pic=Pic*10.0**9.0
        for i in range(0, core_outer_index+1):
            if Pic>=melt_pressure_Fe[i+1] and Pic<melt_pressure_Fe[i]:
                solid_index=i
                Pic=f_ic(melt_pressure_Fe[i], melt_pressure_Fe[i+1], Pic, initial_pressure[i], initial_pressure[i+1])
                Tic=f_ic(initial_pressure[i], initial_pressure[i+1], Pic, Tc_adiabat[i], Tc_adiabat[i+1])
                rho_ic=f_ic(initial_pressure[i], initial_pressure[i+1], Pic, initial_density[i], initial_density[i+1])
                Ric=f_ic(initial_pressure[i], initial_pressure[i+1], Pic, initial_radius[i], initial_radius[i+1])
                Mic=f_ic(initial_pressure[i], initial_pressure[i+1], Pic, mass[i], mass[i+1])
                alpha_ic=f_ic(initial_pressure[i], initial_pressure[i+1], Pic, alpha[i], alpha[i+1])

    for i in range(core_outer_index+1):
        if i==0:
            if initial_phase[0]==ph_Fe_liq:
                initial_temperature[i]=Tc_array[i]
            else:
                initial_temperature[i]=f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist()#T_simon(pressure_GPa_record[i],x_record[i])
        if i>solid_index:
            initial_temperature[i]=Tc_array[i]

        if phase_change[i]==0.0:
            if i==solid_index:
                if i==0 and initial_temperature[i]<=f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist():#T_simon(pressure_GPa_record[i],x_record[i]):
                    initial_phase[i]=ph_Fe_sol
                    initial_temperature[i]=f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist()#T_simon(initial_pressure_GPa[i],x_core)
                if i>0:
                    initial_phase[i]=ph_Fe_sol
                    initial_temperature[i]=f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist()#T_simon(initial_pressure_GPa[i],x_core)
            else:
                if initial_temperature[i]>f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist():#T_simon(initial_pressure_GPa[i],x_core):
                    initial_phase[i]=ph_Fe_liq
                else:
                    initial_phase[i]=ph_Fe_sol
    for i in range(core_outer_index+1):
        if i<solid_index and initial_phase[0]==0.0:
            new_T_cell[i]=solution_T[i]
            initial_temperature[i]=(solution_T[i]+solution_T[i+1])/2.0
        else:
            if i>0:
                new_T_cell[i]=(initial_temperature[i-1]+initial_temperature[i])/2.0
            else:
                new_T_cell[i]=((initial_temperature[i]-(initial_temperature[i+1]-initial_temperature[i]))+initial_temperature[i])/2.0


    for i in range(core_outer_index+1):
        new_T[i]=initial_temperature[i]
        new_T_np[i]=initial_temperature[i]
        Tc_adiabat_np[i]=Tc_adiabat[i]

    if initial_phase[0]==1.0:
        x_core=x_init
    else:
        if Mic<mass[0]:
            Mic=mass[0]
            alpha_ic=alpha[0]
            rho_ic=initial_density[0]
        x_core=x_init*(M_pl*CMF-mass[0])/(M_pl*CMF-Mic)
        if x_core>max_impurities_concentration:
            x_core=max_impurities_concentration

    # calculating dT/dr. Borrow the first portion of dsdr_array for the mantle to save dTdr in the core.
    for i in range(core_outer_index):
        dsdr[i]=(temperature_cell[i]-temperature_cell[i+1])/(radius_cell[i]-radius_cell[i+1])

    for i in range(core_outer_index+1):
        if i>=solid_index:
            Fconv[i]=Fcmb#*4.0*np.pi*initial_radius[i]**2.0
            Fcond[i]=0.0 # Fconv here is the sum of conduction + convection
            Ftot[i]=Fconv[i]
        else:
            Fconv[i]=0.0
            Fcond[i]=-k_array[i]*dsdr[i]#*4.0*np.pi*initial_radius[i]**2.0
            Ftot[i]=Fcond[i]

    rho_liquid_array=interp_2d_rho_Fel((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    rho_solid_array=interp_2d_rho_Fes((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    rho_alloy_array=interp_2d_rho_Fea((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    alpha_liquid_array=interp_2d_alpha_Fel((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    alpha_solid_array=interp_2d_alpha_Fes((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    alpha_alloy_array=interp_2d_alpha_Fea((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    dqdy_liquid_array=interp_2d_dqdy_Fel((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    dqdy_solid_array=interp_2d_dqdy_Fes((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))
    dqdy_alloy_array=interp_2d_dqdy_Fea((pressure_np[:core_outer_index+1], new_T_np[:core_outer_index+1]))

    rho_liquid_array_a=interp_2d_rho_Fel_a((pressure_np[:core_outer_index+1], Tc_adiabat_np[:core_outer_index+1]))
    rho_alloy_array_a=interp_2d_rho_Fea_a((pressure_np[:core_outer_index+1], Tc_adiabat_np[:core_outer_index+1]))
    alpha_liquid_array_a=interp_2d_alpha_Fel_a((pressure_np[:core_outer_index+1], Tc_adiabat_np[:core_outer_index+1]))
    alpha_alloy_array_a=interp_2d_alpha_Fea_a((pressure_np[:core_outer_index+1], Tc_adiabat_np[:core_outer_index+1]))

    for i in range(0,core_outer_index+1):
        if i==0:
            if f_interp_Tsimon(melt_pressure_Fe_GPa[i]).tolist()>=T_center:
                rho_center=f_rho_Fes(T_center,P_center)[0]
                initial_dqdy_center=f_dqdy_Fes(T_center,P_center)[0]
            else:
                rho_liquid=f_rho_Fel(T_center,P_center)[0]
                rho_alloy=f_rho_Fea(T_center,P_center)[0]
                rho_center=rho_mix(x_core/mf_l,rho_alloy,rho_liquid)
                dqdy_liquid=f_dqdy_Fel(T_center,P_center)[0]
                dqdy_alloy=f_dqdy_Fea(T_center,P_center)[0]
                initial_dqdy_center=dqdy_mix(x_core/mf_l,rho_center,rho_alloy,rho_liquid,dqdy_alloy,dqdy_liquid,initial_pressure[i])
        if initial_phase[i]==ph_Fe_sol:
            new_density[i]=rho_solid_array[i]
            new_dqdy[i]=dqdy_solid_array[i]
            new_alpha[i]=alpha_solid_array[i]

            rho_liquid=rho_liquid_array_a[i]
            rho_alloy=rho_alloy_array_a[i]
            alpha_liquid=alpha_liquid_array_a[i]
            alpha_alloy=alpha_alloy_array_a[i]
            rho_Fe_l[i]=rho_mix(x_core/mf_l,rho_alloy,rho_liquid)
            alpha_Fe_l[i]=alpha_mix(x_core/mf_l,alpha_alloy,alpha_liquid,rho_Fe_l[i],rho_alloy,rho_liquid)

        else:
            rho_liquid=rho_liquid_array[i]
            rho_alloy=rho_alloy_array[i]
            alpha_liquid=alpha_liquid_array[i]
            alpha_alloy=alpha_alloy_array[i]
            dqdy_liquid=dqdy_liquid_array[i]
            dqdy_alloy=dqdy_alloy_array[i]
            new_density[i]=rho_mix(x_core/mf_l,rho_alloy,rho_liquid)
            new_dqdy[i]=dqdy_mix(x_core/mf_l,new_density[i],rho_alloy,rho_liquid,dqdy_alloy,dqdy_liquid,initial_pressure[i])
            new_alpha[i]=alpha_mix(x_core/mf_l,alpha_alloy,alpha_liquid,new_density[i],rho_alloy,rho_liquid)
            rho_Fe_l[i]=new_density[i]
            alpha_Fe_l[i]=new_alpha[i]

        if iteration<2:
            melt_frac[i]=0.0

    if initial_phase[0]==ph_Fe_sol:
        for i in range(core_outer_index+1):
            if Mic>mass[i] and Mic<=mass[i+1]:
                x=(mass[i+1]-Mic)/h[i]
                rho_liquid=f_rho_Fel(new_T[i],initial_pressure[i])[0]
                rho_alloy=f_rho_Fea(new_T[i],initial_pressure[i])[0]
                dqdy_liquid=f_dqdy_Fel(new_T[i],initial_pressure[i])[0]
                dqdy_alloy=f_dqdy_Fea(new_T[i],initial_pressure[i])[0]
                rho_m_v=rho_mix(x_core/mf_l,rho_alloy,rho_liquid)
                rho_s_v=f_rho_Fes(new_T[i],initial_pressure[i])[0]
                rho_tot=rho_mix(x, rho_m_v, rho_s_v)
                dqdy_m_v=dqdy_mix(x_core/mf_l,rho_m_v,rho_alloy,rho_liquid,dqdy_alloy,dqdy_liquid,initial_pressure[i])
                dqdy_s_v=f_dqdy_Fes(new_T[i],initial_pressure[i])[0]
                dqdy_tot=dqdy_mix(x, rho_tot, rho_m_v, rho_s_v, dqdy_m_v, dqdy_s_v, initial_pressure[i])
                new_density[i]=rho_tot
                new_dqdy[i]=dqdy_tot

    if iteration%25==0:
        t_array.append(t)
        dt_array.append(dt)
        average_Tm.append(np.sum(new_T[core_outer_index+1:])/mantle_zone)
        Fcmb_array.append(Fcmb)
        Fsurf_array.append(Fsurf)
        Q_ICB_array.append(Q_ICB)
        average_S.append(np.sum(new_S[core_outer_index+1:])/mantle_zone)
        average_x.append(np.sum(melt_frac[core_outer_index+1:])/mantle_zone)
        Qrad_array.append(Q_rad_m*mantle_mass)
        Rp.append(initial_radius[zone-1])
        Rc.append(initial_radius[core_outer_index])
        P_center_array.append(initial_pressure[0])
        P_cmb_array.append(initial_pressure[core_outer_index])
        Fcond_cmb.append(k_Fe*(initial_gravity[core_outer_index]*new_alpha[core_outer_index]*new_T[core_outer_index]/C_P_Fe))
        average_Tc.append(np.sum(new_T[:core_outer_index+1])/core_zone)
        Tcmb_array.append(new_T[core_outer_index])
        Tmbase_array.append(new_T[core_outer_index+1])
        T_s_array.append(new_T[zone-1])
        Ric_array.append(Ric)
        Mic_array.append(Mic)
        Tic_array.append(Tic)
        Pic_array.append(Pic)
        x_core_array.append(x_core)
        T_Fe_en.append(min_pre_adia_T)
        delta_r_array.append(delta_r)
        Tsurf_array.append(T_s)
        
        delta_r_s_array.append(delta_r_s)
        Tcmb_list.append(T_cmb)
        Qrad_c_array.append(Q_rad_c)
        T_center_array.append(new_T[0])
        vis_surf_array.append(vis_surf_value)

        Qsurf_array.append(Fsurf*4.0*math.pi*initial_radius[zone-1]**2.0)
        Qcmb_array.append(Fcmb*4.0*math.pi*initial_radius[core_outer_index]**2.0)
        Urey_array.append(Qrad_array[-1]/Qsurf_array[-1])
        L_Fe_array.append(dmicdTcmb*L_Fe*delta_Tcmb/dt)

        Fconv1_m1.append(Fconv[-2])
        Fcond2_m1.append(Fcond2[-2])
        Fcond1_m1.append(Fcond1[-2])
        T_m1_array.append(new_T[-1])
        T_m2_array.append(new_T[-2])
        T_c_m1_array.append(new_T_cell[-1])
        T_c_m2_array.append(new_T_cell[-2])
        S_m1_array.append(new_S[-1])
        S_c_m1_array.append(new_Scell[-1])
        S_m2_array.append(new_S[-2])
        S_c_m2_array.append(new_Scell[-2])
        A_m1_array.append(Area[-1])
        A_m2_array.append(Area[-2])
        v_m2_array.append(viscosity[-2])
        P_m2_array.append(initial_pressure[-2])

    old_S=initial_S.copy()
    old_Scell=S_cell.copy()
    old_density=initial_density.copy()
    old_T=initial_temperature.copy()
    old_T_cell=temperature_cell.copy()
    old_kappa=kappa.copy()
    old_dTdP=dTdP.copy()
    old_dPdr=dPdr.copy()

    initial_temperature[core_outer_index+1:]=new_T[core_outer_index+1:].copy()
    temperature_cell[core_outer_index+1:]=new_T_cell[core_outer_index+1:].copy()
    CP[core_outer_index+1:]=new_CP[core_outer_index+1:].copy()
    kappa[core_outer_index+1:]=new_kappa[core_outer_index+1:].copy()
    initial_density[core_outer_index+1:]=new_density[core_outer_index+1:].copy()
    dTdP[core_outer_index+1:]=new_dTdP[core_outer_index+1:].copy()
    dPdr[core_outer_index+1:]=new_dPdr[core_outer_index+1:].copy()
    initial_phase[core_outer_index+1:]=new_phase[core_outer_index+1:].copy()
    initial_S[core_outer_index+1:]=new_S[core_outer_index+1:].copy()
    S_cell[core_outer_index+1:]=new_Scell[core_outer_index+1:].copy()
    melt_frac[core_outer_index+1:]=new_x[core_outer_index+1:].copy()
    x_cell=new_x_cell.copy()
    initial_dqdy[core_outer_index+1:]=new_dqdy[core_outer_index+1:].copy()

    initial_temperature[:core_outer_index+1]=new_T[:core_outer_index+1].copy()
    temperature_cell[:core_outer_index+1]=new_T_cell[:core_outer_index+1].copy()
    alpha=new_alpha.copy()
    initial_density[:core_outer_index+1]=new_density[:core_outer_index+1].copy()
    initial_dqdy[:core_outer_index+1]=new_dqdy[:core_outer_index+1].copy()
    rho_m_array=new_rho_m.copy()
    rho_s_array=new_rho_s.copy()

    #### save old radius pressure radius_cell pressure_cell gravity
    old_pressure=initial_pressure.copy()
    old_pressure_cell=pressure_cell.copy()
    old_radius=initial_radius.copy()
    old_radius_cell=radius_cell.copy()
    old_gravity=initial_gravity.copy()
    old_Area=Area.copy()
    old_V=new_V.copy()
    old_EG=new_EG.copy()
    old_v_top=new_v_top.copy()

    ## Henyey code
    initial_pressure[zone-1]=P_surf
    A_r=np.zeros(zone); A_p=np.zeros(zone)
    for i in range(zone):
        if i==0:
            A_r[i]=(math.log(initial_radius[i])-1.0/3.0*(math.log(3.0*mass[i]/(4.0*np.pi))-math.log(rho_center)))
            A_p[i]=(math.log(initial_pressure[i])-math.log(P_center)+G/2.0*math.pow((4.0*math.pi*math.pow(mass[i],2.0/3.0)),1.0/3.0)*math.exp(4.0*math.log(rho_center)/3.0-math.log(P_center)))
        elif i>0 and i<zone:
            A_r[i]=(math.log(initial_radius[i])-math.log(initial_radius[i-1])-1.0/(4.0*np.pi)*(mass[i]-mass[i-1])*math.exp(-0.5*(math.log(initial_density[i])+math.log(initial_density[i-1]))-1.5*(math.log(initial_radius[i])+math.log(initial_radius[i-1]))))
            A_p[i]=(math.log(initial_pressure[i])-math.log(initial_pressure[i-1])+G/(8.0*np.pi)*(math.pow(mass[i],2.0)-math.pow(mass[i-1],2.0))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(initial_pressure[i-1]))-2.0*(math.log(initial_radius[i])+math.log(initial_radius[i-1]))))
    a=np.zeros(zone); b=np.zeros(zone); c=np.zeros(zone); d=np.zeros(zone); A=np.zeros(zone); B=np.zeros(zone); C=np.zeros(zone); D=np.zeros(zone) # lower case is for pressure. Upper case is for radius

    for i in range(0, zone):
        if i==0:
            a[i]=((G/(8.0*math.pi))*(math.pow(mass[i],2.0))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(P_center)))*math.exp(-2.0*math.log(initial_radius[i]))*(-2.0))
            b[i]=(-1.0+(G/(8.0*math.pi))*(math.pow(mass[i],2.0))*math.exp(-2.0*math.log(initial_radius[i]))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(P_center)))*(-0.5))
            c[i]=((G/(8.0*math.pi))*(math.pow(mass[i],2.0))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(P_center)))*math.exp(-2.0*math.log(initial_radius[i]))*(-2.0))
            d[i]=(1.0+(G/(8.0*math.pi))*(math.pow(mass[i],2.0))*math.exp(-2.0*math.log(initial_radius[i]))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(P_center)))*(-0.5))
            A[i]=(-1.0-(1.0/(4.0*math.pi))*(mass[i])*math.exp(-0.5*(math.log(initial_density[i])+math.log(rho_center)))*math.exp(-1.5*(math.log(initial_radius[i])))*(-1.5))
            B[i]=(-(1.0/(4.0*math.pi))*(mass[i])*math.exp(-0.5*(math.log(initial_density[i])+math.log(rho_center)))*math.exp(-1.5*(math.log(initial_radius[i])))*(-0.5)*initial_dqdy_center)
            C[i]=(1.0-(1.0/(4.0*math.pi))*(mass[i])*math.exp(-0.5*(math.log(initial_density[i])+math.log(rho_center)))*math.exp(-1.5*(math.log(initial_radius[i])))*(-1.5))
            D[i]=(-(1.0/(4.0*math.pi))*(mass[i])*math.exp(-0.5*(math.log(initial_density[i])+math.log(rho_center)))*math.exp(-1.5*(math.log(initial_radius[i])))*(-0.5)*initial_dqdy[i])
        else:
            a[i]=((G/(8.0*math.pi))*(math.pow(mass[i],2.0)-math.pow(mass[i-1],2.0))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(initial_pressure[i-1])))*math.exp(-2.0*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-2.0))
            b[i]=(-1.0+(G/(8.0*math.pi))*(math.pow(mass[i],2.0)-math.pow(mass[i-1],2.0))*math.exp(-2.0*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(initial_pressure[i-1])))*(-0.5))
            c[i]=((G/(8.0*math.pi))*(math.pow(mass[i],2.0)-math.pow(mass[i-1],2.0))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(initial_pressure[i-1])))*math.exp(-2.0*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-2.0))
            d[i]=(1.0+(G/(8.0*math.pi))*(math.pow(mass[i],2.0)-math.pow(mass[i-1],2.0))*math.exp(-2.0*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*math.exp(-0.5*(math.log(initial_pressure[i])+math.log(initial_pressure[i-1])))*(-0.5))
            A[i]=(-1.0-(1.0/(4.0*math.pi))*(mass[i]-mass[i-1])*math.exp(-0.5*(math.log(initial_density[i])+math.log(initial_density[i-1])))*math.exp(-1.5*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-1.5))
            B[i]=(-(1.0/(4.0*math.pi))*(mass[i]-mass[i-1])*math.exp(-0.5*(math.log(initial_density[i])+math.log(initial_density[i-1])))*math.exp(-1.5*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-0.5)*initial_dqdy[i-1])
            C[i]=(1.0-(1.0/(4.0*math.pi))*(mass[i]-mass[i-1])*math.exp(-0.5*(math.log(initial_density[i])+math.log(initial_density[i-1])))*math.exp(-1.5*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-1.5))
            D[i]=(-(1.0/(4.0*math.pi))*(mass[i]-mass[i-1])*math.exp(-0.5*(math.log(initial_density[i])+math.log(initial_density[i-1])))*math.exp(-1.5*(math.log(initial_radius[i])+math.log(initial_radius[i-1])))*(-0.5)*initial_dqdy[i])

    alp=np.zeros(zone); gam=np.zeros(zone)
    alp_0=0.0; gam_0=0.0
    for i in range(zone):
        if i==0:
            alp[i]=(d[i]*(B[i]-A[i]*alp_0)-D[i]*(b[i]-a[i]*alp_0))/(c[i]*(B[i]-A[i]*alp_0)-C[i]*(b[i]-a[i]*alp_0))
            gam[i]=((B[i]-A[i]*alp_0)*(A_p[i]-a[i]*gam_0)-(b[i]-a[i]*alp_0)*(A_r[i]-A[i]*gam_0))/(c[i]*(B[i]-A[i]*alp_0)-C[i]*(b[i]-a[i]*alp_0))
        else:
            alp[i]=(d[i]*(B[i]-A[i]*alp[i-1])-D[i]*(b[i]-a[i]*alp[i-1]))/(c[i]*(B[i]-A[i]*alp[i-1])-C[i]*(b[i]-a[i]*alp[i-1]))
            gam[i]=((B[i]-A[i]*alp[i-1])*(A_p[i]-a[i]*gam[i-1])-(b[i]-a[i]*alp[i-1])*(A_r[i]-A[i]*gam[i-1]))/(c[i]*(B[i]-A[i]*alp[i-1])-C[i]*(b[i]-a[i]*alp[i-1]))
    delta_y=np.zeros(zone)
    delta_x=np.zeros(zone)
    delta_y[zone-1]=0.0
    delta_x[zone-1]=-gam[zone-1]
    for i in range(zone-1, -1, -1):
        if i>0-1 and i<zone-1:
            delta_y[i]=-(A_r[i+1]-A[i+1]*gam[i]+C[i+1]*delta_x[i+1]+D[i+1]*delta_y[i+1])/(B[i+1]-A[i+1]*alp[i])
            delta_x[i]=-gam[i]-alp[i]*delta_y[i]
    delta_P_center=(A_r[0]+C[0]*delta_x[0]+D[0]*delta_y[0])/B[0]
    P_center=math.exp(math.log(P_center)+delta_P_center)
    for i in range(zone):
        initial_radius[i]=math.exp(math.log(initial_radius[i])+delta_x[i])
        initial_pressure[i]=math.exp(math.log(initial_pressure[i])+delta_y[i])
        initial_gravity[i]=(G*mass[i]/initial_radius[i]**2.0)
        initial_pressure_GPa[i]=initial_pressure[i]/1e9
        pressure_np[i]=initial_pressure[i]

    radius_cell=np.zeros(zone)
    pressure_cell=np.zeros(zone)
    Area=np.zeros(zone)
    for i in range(zone):
        if i==0:
            radius_cell[i]=initial_radius[i]/2.0
            pressure_cell[i]=(initial_pressure[i]+P_center)/2.0
            pressure_cell_np[i]=pressure_cell[i]
        else:
            radius_cell[i]=(initial_radius[i]+initial_radius[i-1])/2.0
            pressure_cell[i]=(initial_pressure[i]+initial_pressure[i-1])/2.0
            pressure_cell_np[i]=pressure_cell[i]
            dr[i]=initial_radius[i]-initial_radius[i-1]
        Area[i]=4.0*np.pi*initial_radius[i]**2.0

    for i in range(core_outer_index+1):
        if old_phase[i]==1.0 and initial_phase[i]==0.0:
            phase_change[i]=1.0
            x_record[i]=x_core
        if old_phase[i]==1.0 and initial_phase[i]==0.0:
            pressure_record[i]=initial_pressure[i]
            pressure_GPa_record[i]=pressure_record[i]/1e9
        elif old_phase[i]==1.0 and initial_phase[i]==1.0:
            pressure_record[i]=initial_pressure[i]
            pressure_GPa_record[i]=pressure_record[i]/1e9
    old_S_liquidus=S_liquidus.copy()
    old_S_solidus=S_solidus.copy()
    for i in range(core_outer_index+1,zone):
        S_liquidus[i]=S_liq_P(initial_pressure[i])
        S_liquidus_cell[i]=S_liq_P(pressure_cell[i])
        S_solidus[i]=S_sol_P(initial_pressure[i])
        S_solidus_cell[i]=S_sol_P(pressure_cell[i])

    sigma_MO=np.zeros(zone)
    v_MO=np.zeros(zone)
    Rem_MO=np.zeros(zone)
    L_sigma=0.0
    D_MO_dynamo=0.0
    flag_top=0.0
    flag_bot=0.0
    for i in range(core_outer_index+1,zone-10):
        if initial_phase[i]==ph_pv_sol and S_MO_index==core_outer_index+1:
            S_MO_index=i
    for i in range(core_outer_index+1,zone):
        if melt_frac[i]>0.0:
            sigma_MO[i]=sigma_silicate(initial_pressure[i],initial_temperature[i])
        else:
            sigma_MO[i]=0.0
        v_MO[i]=eddy_k[i]/(l_mlt[i]+1e-10)
        if melt_frac[i]>0.0 and i>S_MO_index-1:
            L_sigma=L_sigma+dr[i]

    # calculate convective velocity and Magnetic Reynolds number in the core.
    if Fcmb>=0.0 and solid_index<core_outer_index:
        for i in range(core_outer_index+1):
            if i>=solid_index:
                v_MO[i]=(((alpha[i]*initial_gravity[i])
                    *Fcmb*(initial_radius[core_outer_index]-initial_radius[solid_index]))/
                (initial_density[i]*C_P_Fe))**0.33
                Rem_MO[i]=mu_0*v_MO[i]*(initial_radius[core_outer_index]-initial_radius[solid_index])*k_Fe*2.0/(initial_temperature[i]*L0)
    for i in range(core_outer_index+1,zone):
        Rem_MO[i]=mu_0*v_MO[i]*L_sigma*sigma_MO[i]
    for i in range(core_outer_index+1,zone):
        if Rem_MO[i]>50.0 and i>S_MO_index-1:
            D_MO_dynamo=D_MO_dynamo+dr[i]
    for i in range(S_MO_index,zone-1):
        if Rem_MO[i]<=50.0 and Rem_MO[i+1]>50.0 and flag_bot==0.0:
            MO_dynamo_bot=initial_radius[i+1]
            flag_bot=1.0
    if Rem_MO[core_outer_index+1]>=50.0 and Rem_MO[core_outer_index+2]>=50.0 and MO_dynamo_bot==0.0:
        MO_dynamo_bot=initial_radius[core_outer_index+1]
    for i in range(zone,S_MO_index,-1):
        if Rem_MO[i]>50.0 and Rem_MO[i+1]<=50.0 and flag_top==0.0:
            MO_dynamo_top=initial_radius[i]
            flag_top=1.0

    if iteration%25==0:
        L_sigma_array.append(L_sigma)
        D_MO_dynamo_array.append(D_MO_dynamo)
        MO_dynamo_bot_array.append(MO_dynamo_bot)
        MO_dynamo_top_array.append(MO_dynamo_top)
    
    Buoy_T_value=alpha[core_outer_index]*initial_gravity[core_outer_index]/(initial_density[core_outer_index]*C_P_Fe)*(Fcmb-k_Fe*(initial_gravity[core_outer_index]*new_alpha[core_outer_index]*new_T[core_outer_index]/C_P_Fe))
    rho_core=M_pl*CMF/(4.0/3.0*math.pi*initial_radius[core_outer_index]**3.0)
    if solid_index==0:
        Buoy_x_value=initial_gravity[solid_index]*(f_rho_Fes(T_center,P_center)[0][0]-initial_density[solid_index+1])/rho_core*(Ric/(initial_radius[core_outer_index]))**2.0#*(Ric-old_Ric)/dt
    else:
        Buoy_x_value=initial_gravity[solid_index]*(initial_density[solid_index-1]-initial_density[solid_index+1])/rho_core*(Ric/(initial_radius[core_outer_index]))**2.0#*(Ric-old_Ric)/dt
    if Buoy_T_value+Buoy_x_value>0.0:
        core_m=4.0*math.pi*initial_radius[core_outer_index+1]**3.0*0.2*(initial_density[solid_index-1]/(2.0*4.0*math.pi*1e-7))**0.5*((initial_radius[core_outer_index]-initial_radius[solid_index]))**(1.0/3.0)
    else:
        core_m=0.0

    if iteration%25==0:
        solid_index_arr.append(solid_index)
        Buoy_T.append(Buoy_T_value)
        Buoy_x.append(Buoy_x_value)
        core_dipole_m.append(core_m)

    if old_vis_surf_value==0.0:
            old_vis_surf_value=1e-2
    if old_delta_r==0.0:
        old_delta_r=1e-2
    rdiff[0]=np.max(np.abs((np.asarray(old_Scell[core_outer_index+2:-1])-np.asarray(S_cell[core_outer_index+2:-1]))/np.asarray(old_Scell[core_outer_index+2:-1])))
    rdiff[1]=np.abs(old_delta_r-delta_r)/old_delta_r
    rdiff[2]=np.abs(old_vis_surf_value-vis_surf_value)/old_vis_surf_value
    rdiff[3]=np.abs(delta_Tcmb)/T_cmb
    rdiff[4]=np.abs(old_Fsurf-Fsurf)/Fsurf

    dt_thres=np.max(rdiff)
    ds_thres=ds_thres_xl

    if dt_thres<ds_thres:
        if dt_thres<0.975*ds_thres:
            if iteration<200:
                dt=dt*1.001
            elif t>1e8*86400.0*365.0 and dt_thres<0.8*ds_thres:
                dt=dt*1.2
            else:
                dt=dt*1.01
        else:
            if iteration<200.0:
                dt=dt
            else:
                dt=dt+30.0
    else:
        if dt_thres>1.05*ds_thres:
            dt=dt*0.8
        else:
            dt=dt*0.9

    if dt>5e6*86400.0*365.0:    
        dt=5e6*86400.0*365.0
    if dt<86400.0*365.0*0.1:
        dt=86400.0*365.0*0.1

    if iteration%50==0:
        if t/86400.0/365.0<1e3:
            t_val=t/86400.0/365.0
            print('time:%2.2fyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa T_surface:%2.2fK' %(t_val,Fcmb,Fsurf,Ric/1e3,new_T[core_outer_index-1],initial_pressure[0]/1e9,initial_pressure[core_outer_index-1]/1e9, T_s))
        elif t/86400.0/365.0>=1e3 and t/86400.0/365.0<1e6:
            t_val=t/86400.0/365.0/1e3
            print('time:%2.2fkyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa T_surface:%2.2fK' %(t_val,Fcmb,Fsurf,Ric/1e3,new_T[core_outer_index-1],initial_pressure[0]/1e9,initial_pressure[core_outer_index-1]/1e9, T_s))
        elif t/86400.0/365.0>=1e6 and t/86400.0/365.0<1e9:
            t_val=t/86400.0/365.0/1e6
            print('time:%2.2fMyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa T_surface:%2.2fK' %(t_val,Fcmb,Fsurf,Ric/1e3,new_T[core_outer_index-1],initial_pressure[0]/1e9,initial_pressure[core_outer_index-1]/1e9, T_s))
        else:
            t_val=t/86400.0/365.0/1e9
            print('time:%2.2fGyrs Fcmb:%2.2fW/m^2 Fsurf:%2.2fW/m^2 Ric:%2.2fkm Tcmb:%2.2fK Pc:%2.2fGPa Pcmb:%2.2fGPa T_surface:%2.2fK' %(t_val,Fcmb,Fsurf,Ric/1e3,new_T[core_outer_index-1],initial_pressure[0]/1e9,initial_pressure[core_outer_index-1]/1e9, T_s))
        
    for ind in range(len(t_save)):
        if t<t_save[ind]*86400.0*365.0+dt and t>t_save[ind]*86400.0*365.0-dt:
            np.savetxt(results_foldername+'/profile/StructureProfile_'+str(int(t_save[ind]))+'.txt',np.transpose([initial_radius,initial_pressure,initial_density,initial_gravity,
                initial_temperature,alpha,CP,Fconv,Fcond,Ftot,v_MO,Rem_MO,viscosity, mass, melt_frac]), header='radius, pressure, density, gravitational acceleration, temperature, thermal expansion coefficient, specific heat, convective heat flux, conductivt heat flux, total heat flux, convective velocity, mantle magnetic Reynolds number, mantle viscosity, mass, mantle melt fraction')
            np.savetxt(results_foldername+'/evolution.txt',np.transpose([t_array,dt_array,average_Tm,average_Tc,Tsurf_array,Tcmb_array,T_center_array,Fsurf_array,Fcmb_array,Fcond_cmb,Rp,Rc,P_center_array,P_cmb_array,Ric_array,Mic_array,D_MO_dynamo_array,Qrad_array,Qrad_c_array,Q_ICB_array,Buoy_T,Buoy_x,core_dipole_m,Qsurf_array,Qcmb_array,L_Fe_array,Urey_array]),
                header='time, time stepsize, mass averaged mantle temperature, mass averaged core temperature, surface temperature, core mantle boundary temperature, central temperature,surface heat flux, core mantle boundary heat flux, conductive heat flux along core adiabat, planet radius, core radius, central pressure, core mantle boundary pressure, inner core radius, inner core mass, thickness of dynamo source region in magma ocean, mantle radiogenic heating, core radiogenic heating, inner core conductive heat flow, core thermal buoyancy flux, core compositional buouyancy flux, core magnetic dipole moment, surface heat flow, CMB heat flow, core latent heat release, Urey ratio ')
    if iteration%10000==0:
        np.savetxt(results_foldername+'/evolution.txt',np.transpose([t_array,dt_array,average_Tm,average_Tc,Tsurf_array,Tcmb_array,T_center_array,Fsurf_array,Fcmb_array,Fcond_cmb,Rp,Rc,P_center_array,P_cmb_array,Ric_array,Mic_array,D_MO_dynamo_array,Qrad_array,Qrad_c_array,Q_ICB_array,Buoy_T,Buoy_x,core_dipole_m,Qsurf_array,Qcmb_array,L_Fe_array,Urey_array]),
            header='time, time stepsize, mass averaged mantle temperature, mass averaged core temperature, surface temperature, core mantle boundary temperature, central temperature,surface heat flux, core mantle boundary heat flux, conductive heat flux along core adiabat, planet radius, core radius, central pressure, core mantle boundary pressure, inner core radius, inner core mass, thickness of dynamo source region in magma ocean, mantle radiogenic heating, core radiogenic heating, inner core conductive heat flow, core thermal buoyancy flux, core compositional buouyancy flux, core magnetic dipole moment, surface heat flow, CMB heat flow, core latent heat release, Urey ratio ')      
        
    iteration=iteration+1
    t=t+dt
   
cdef int t_end_ind=find_nearest(t_save,t/86400.0/365.0)
if solid_core_flag==1.0:
    np.savetxt(results_foldername+'/profile/files_saved_at_these_time_list.txt',t_save[:t_end_ind])
elif solid_core_flag==0.0:
    np.savetxt(results_foldername+'/profile/files_saved_at_these_time_list.txt',t_save[:t_end_ind+1])

cdef Py_ssize_t start_ind
if sum(Buoy_x)>0.0:
    for i in range(1,len(t_array)):
        if Ric_array[i-1]==0.0 and Ric_array[i]>0.0:
            start_ind=i
    f_Ric_t=CubicSpline(t_array[start_ind:],Ric_array[start_ind:])
    f_dRicdt=f_Ric_t.derivative()
    dRicdt=Ric_array.copy()
    dRicdt[start_ind:]=f_dRicdt(t_array[start_ind:])
    log10_dRicdt=dRicdt.copy()
    log10_dRicdt_hat=dRicdt.copy()
    log10_dRicdt[start_ind:]=np.log10(dRicdt[start_ind:])
    log10_dRicdt_hat[start_ind:]=savgol_filter(log10_dRicdt[start_ind:], window_length=99,polyorder=9) 
    for i in range(len(t_array)):
        Buoy_x[i]=Buoy_x[i]*10.0**log10_dRicdt_hat[i]
        if Buoy_x[i]+Buoy_T[i]>0.0:
            core_dipole_m[i]=core_dipole_m[i]*(Buoy_T[i]+Buoy_x[i])**(1.0/3.0)
        else:
            core_dipole_m[i]=0.0
else:
    for i in range(len(t_array)):
        Buoy_x[i]=0.0
        if Buoy_T[i]>0.0:
            core_dipole_m[i]=core_dipole_m[i]*(Buoy_T[i]+Buoy_x[i])**(1.0/3.0)
        else:
            core_dipole_m[i]=0.0   
smooth_QICB=np.asarray(Q_ICB_array).copy()
smooth_L=np.asarray(L_Fe_array).copy()
smooth_QICB[start_ind:]=savgol_filter(Q_ICB_array[start_ind:], window_length=99,polyorder=1)
smooth_L[start_ind:]=savgol_filter(L_Fe_array[start_ind:], window_length=99,polyorder=1)

np.savetxt(results_foldername+'/evolution.txt',np.transpose([t_array,dt_array,average_Tm,average_Tc,Tsurf_array,Tcmb_array,T_center_array,Fsurf_array,Fcmb_array,Fcond_cmb,Rp,Rc,P_center_array,P_cmb_array,Ric_array,Mic_array,D_MO_dynamo_array,Qrad_array,Qrad_c_array,smooth_QICB,Buoy_T,Buoy_x,core_dipole_m,Qsurf_array,Qcmb_array,smooth_L,Urey_array]),
    header='time, time stepsize, mass averaged mantle temperature, mass averaged core temperature, surface temperature, core mantle boundary temperature, central temperature, surface heat flux, core mantle boundary heat flux, conductive heat flux along core adiabat, planet radius, core radius, central pressure, core mantle boundary pressure, inner core radius, inner core mass, thickness of dynamo source region in magma ocean, mantle radiogenic heating, core radiogenic heating, inner core conductive heat flow, core thermal buoyancy flux, core compositional buouyancy flux, core magnetic dipole moment, surface heat flow, CMB heat flow, core latent heat release, Urey ratio ')
