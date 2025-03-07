import numpy as np
from scipy import interpolate
import os

# Set verbosity based on an environment variable. Default to True.
EOS_TABLES_VERBOSE = os.getenv('EOS_TABLES_VERBOSE', '1') == '1'
os.makedirs('EoS/binary/', exist_ok=True)
DATA_FILE = 'EoS/binary/eos_raw_data.npz'

# --- Load raw EOS arrays ---
if os.path.exists(DATA_FILE):
    if EOS_TABLES_VERBOSE:
        print("Loading EOS raw data from binary files...")
    data = np.load(DATA_FILE)

    T_sol_ppv   = data['T_sol_ppv']
    rho_sol_ppv = data['rho_sol_ppv']
    alpha_sol_ppv= data['alpha_sol_ppv']
    dTdP_sol_ppv= data['dTdP_sol_ppv']
    dqdy_sol_ppv= data['dqdy_sol_ppv']

    T_mix_ppv   = data['T_mix_ppv']
    rho_mix_ppv = data['rho_mix_ppv']
    CP_mix_ppv  = data['CP_mix_ppv']
    alpha_mix_ppv= data['alpha_mix_ppv']
    dTdP_mix_ppv= data['dTdP_mix_ppv']
    dqdy_mix_ppv= data['dqdy_mix_ppv']

    T_mix_en    = data['T_mix_en']
    rho_mix_en  = data['rho_mix_en']
    CP_mix_en  = data['CP_mix_en']
    alpha_mix_en= data['alpha_mix_en']
    dTdP_mix_en = data['dTdP_mix_en']
    dqdy_mix_en = data['dqdy_mix_en']

    T_mix_pv    = data['T_mix_pv']
    rho_mix_pv  = data['rho_mix_pv']
    CP_mix_pv  = data['CP_mix_pv']
    alpha_mix_pv= data['alpha_mix_pv']
    dTdP_mix_pv = data['dTdP_mix_pv']
    dqdy_mix_pv = data['dqdy_mix_pv']

    T_sol_en    = data['T_sol_en']
    rho_sol_en  = data['rho_sol_en']
    alpha_sol_en= data['alpha_sol_en']
    dTdP_sol_en = data['dTdP_sol_en']
    dqdy_sol_en = data['dqdy_sol_en']

    T_sol_pv    = data['T_sol_pv']
    rho_sol_pv  = data['rho_sol_pv']
    alpha_sol_pv= data['alpha_sol_pv']
    dTdP_sol_pv = data['dTdP_sol_pv']
    dqdy_sol_pv = data['dqdy_sol_pv']

    T_liq       = data['T_liq']
    rho_liq     = data['rho_liq']
    CP_liq      = data['CP_liq']
    alpha_liq   = data['alpha_liq']
    dTdP_liq    = data['dTdP_liq']
    dqdy_liq    = data['dqdy_liq']

    y_grid      = data['y_grid']
    P_grid      = data['P_grid']  
    P_grid_en   = data['P_grid_en']
    P_grid_pv   = data['P_grid_pv']
    P_grid_ppv  = data['P_grid_ppv']
    S_sol_array = data['S_sol_array']
    S_liq_array = data['S_liq_array']

    rho_Fel     = data['rho_Fel']
    alpha_Fel   = data['alpha_Fel']
    dqdy_Fel    = data['dqdy_Fel']
    T_Fel       = data['T_Fel']
    P_Fel       = data['P_Fel']

    rho_Fea     = data['rho_Fea']
    alpha_Fea   = data['alpha_Fea']
    dqdy_Fea    = data['dqdy_Fea']
    T_Fea       = data['T_Fea']
    P_Fea       = data['P_Fea']

    rho_Fes     = data['rho_Fes']
    alpha_Fes   = data['alpha_Fes']
    dqdy_Fes    = data['dqdy_Fes']
    T_Fes       = data['T_Fes']
    P_Fes       = data['P_Fes']

    rho_Fel60   = data['rho_Fel60']
    alpha_Fel60 = data['alpha_Fel60']
    dqdy_Fel60  = data['dqdy_Fel60']
    T_Fel60     = data['T_Fel60']
    P_Fel60     = data['P_Fel60']

    rho_Fea60   = data['rho_Fea60']
    alpha_Fea60 = data['alpha_Fea60']
    dqdy_Fea60  = data['dqdy_Fea60']
    T_Fea60     = data['T_Fea60']
    P_Fea60     = data['P_Fea60']

    rho_Fes60   = data['rho_Fes60']
    alpha_Fes60 = data['alpha_Fes60']
    dqdy_Fes60  = data['dqdy_Fes60']
    T_Fes60     = data['T_Fes60']
    P_Fes60     = data['P_Fes60']

    x_core_grid = data['x_core_grid']
    Tref_core_grid = data['Tref_core_grid']
    Tgrid_core_grid = data['Tgrid_core_grid']
    P_core_grid = data['P_core_grid']
    load_original_T = data['load_original_T']
    load_original_dTdT0 = data['load_original_dTdT0']
    load_original_dT0dP = data['load_original_dT0dP']

    x_core_grid60 = data['x_core_grid60']
    Tref_core_grid60 = data['Tref_core_grid60']
    Tgrid_core_grid60 = data['Tgrid_core_grid60']
    P_core_grid60 = data['P_core_grid60']
    load_original_T60= data['load_original_T60']
    load_original_dTdT060 = data['load_original_dTdT060']
    load_original_dT0dP60 = data['load_original_dT0dP60']
else:
    if EOS_TABLES_VERBOSE:
        print("Reading EOS tables from text files...")

    # --- Read EOS tables ---
    print('Read EoS tables')
    print('MgSiO3 eos')
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
    CP_mix_pv=np.loadtxt('EoS/mantle/CP_mix_pv_Py.txt')
    alpha_mix_pv=np.loadtxt('EoS/mantle/alpha_mix_pv_Py.txt')
    dTdP_mix_pv=np.loadtxt('EoS/mantle/dTdP_mix_pv_Py.txt')
    dqdy_mix_pv=np.loadtxt('EoS/mantle/dqdy_mix_pv_Py.txt')

    T_mix_ppv=np.loadtxt('EoS/mantle/T_mix_ppv_Py.txt')
    rho_mix_ppv=np.loadtxt('EoS/mantle/rho_mix_ppv_Py.txt')
    CP_mix_ppv=np.loadtxt('EoS/mantle/CP_mix_ppv_Py.txt')
    alpha_mix_ppv=np.loadtxt('EoS/mantle/alpha_mix_ppv_Py.txt')
    dTdP_mix_ppv=np.loadtxt('EoS/mantle/dTdP_mix_ppv_Py.txt')
    dqdy_mix_ppv=np.loadtxt('EoS/mantle/dqdy_mix_ppv_Py.txt')

    T_mix_en=np.loadtxt('EoS/mantle/T_mix_en_Py.txt')
    rho_mix_en=np.loadtxt('EoS/mantle/rho_mix_en_Py.txt')
    CP_mix_en=np.loadtxt('EoS/mantle/CP_mix_en_Py.txt')
    alpha_mix_en=np.loadtxt('EoS/mantle/alpha_mix_en_Py.txt')
    dTdP_mix_en=np.loadtxt('EoS/mantle/dTdP_mix_en_Py.txt')
    dqdy_mix_en=np.loadtxt('EoS/mantle/dqdy_mix_en_Py.txt')

    T_liq=np.loadtxt('EoS/mantle/T_liq_Py_1500GPa.txt')
    rho_liq=np.loadtxt('EoS/mantle/rho_liq_Py_1500GPa.txt')
    CP_liq=np.loadtxt('EoS/mantle/CP_liq_Py_1500GPa.txt')
    alpha_liq=np.loadtxt('EoS/mantle/alpha_liq_Py_1500GPa.txt')
    dTdP_liq=np.loadtxt('EoS/mantle/dTdP_liq_Py_1500GPa.txt')
    dqdy_liq=np.loadtxt('EoS/mantle/dqdy_liq_Py_1500GPa.txt')

    y_grid=np.loadtxt('EoS/mantle/y.txt')
    P_grid_en=np.loadtxt('EoS/mantle/P_en.txt')
    P_grid_pv=np.loadtxt('EoS/mantle/P_pv.txt')
    P_grid_ppv=np.loadtxt('EoS/mantle/P_ppv.txt')

    P_solidus_liquidus=np.loadtxt('EoS/mantle/solid_P.txt')
    P_grid=P_solidus_liquidus[:,0][:1500].copy()
    S_sol_array=P_solidus_liquidus[:,1][:1500].copy()
    S_liq_array=P_solidus_liquidus[:,2][:1500].copy()
    #P_grid_np=P_solidus_liquidus[:,0][:1500]

    print('Fe and Fe-Si alloy eos')
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
    P_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_Pgrid.txt')
    loaded_dTdT0=np.loadtxt('EoS/Fe_core/Fe_dTdT0.txt')
    load_original_dTdT0=loaded_dTdT0.reshape(loaded_dTdT0.shape[0],loaded_dTdT0.shape[1]//989,989)
    loaded_dT0dP=np.loadtxt('EoS/Fe_core/Fe_dT0dP.txt')
    load_original_dT0dP=loaded_dT0dP.reshape(loaded_dT0dP.shape[0],loaded_dT0dP.shape[1]//826,826)
    Tgrid_core_grid=np.loadtxt('EoS/Fe_core/Fe_adiabat_P_Tgridgrid.txt')
    rho_Fes=np.loadtxt('EoS/Fe_core/rho_Fes_dew06.txt')
    alpha_Fes=np.loadtxt('EoS/Fe_core/alpha_Fes_dew06.txt')
    dqdy_Fes=np.loadtxt('EoS/Fe_core/dqdy_Fes_dew06.txt')
    T_Fes=np.loadtxt('EoS/Fe_core/T_Fes_dew06.txt')
    P_Fes=np.loadtxt('EoS/Fe_core/P_Fes_dew06.txt')

    rho_Fel60=np.loadtxt('EoS/Fe_core/rho_Fel_60GPa.txt')
    alpha_Fel60=np.loadtxt('EoS/Fe_core/alpha_Fel_60GPa.txt')
    dqdy_Fel60=np.loadtxt('EoS/Fe_core/dqdy_Fel_60GPa.txt')
    T_Fel60=np.loadtxt('EoS/Fe_core/T_Fel_60GPa.txt')
    P_Fel60=np.loadtxt('EoS/Fe_core/P_Fel_60GPa.txt')
    rho_Fea60=np.loadtxt('EoS/Fe_core/rho_Fe16Si_60GPa.txt')
    alpha_Fea60=np.loadtxt('EoS/Fe_core/alpha_Fe16Si_60GPa.txt')
    dqdy_Fea60=np.loadtxt('EoS/Fe_core/dqdy_Fe16Si_60GPa.txt')
    T_Fea60=np.loadtxt('EoS/Fe_core/T_Fe16Si_60GPa.txt')
    P_Fea60=np.loadtxt('EoS/Fe_core/P_Fe16Si_60GPa.txt')
    loaded_T60=np.loadtxt('EoS/Fe_core/Fe_adiabat_60GPa.txt')
    load_original_T60=loaded_T60.reshape(loaded_T60.shape[0],loaded_T60.shape[1]//995,995)#141
    x_core_grid60=np.loadtxt('EoS/Fe_core/Fe_adiabat_xgrid_60GPa.txt')
    Tref_core_grid60=np.loadtxt('EoS/Fe_core/Fe_adiabat_Tgrid_60GPa.txt')
    P_core_grid60=np.loadtxt('EoS/Fe_core/Fe_adiabat_Pgrid_60GPa.txt')
    loaded_dTdT060=np.loadtxt('EoS/Fe_core/Fe_dTdT0_60GPa.txt')
    load_original_dTdT060=loaded_dTdT060.reshape(loaded_dTdT060.shape[0],loaded_dTdT060.shape[1]//995,995)
    loaded_dT0dP60=np.loadtxt('EoS/Fe_core/Fe_dT0dP_60GPa.txt')
    load_original_dT0dP60=loaded_dT0dP60.reshape(loaded_dT0dP60.shape[0],loaded_dT0dP60.shape[1]//951,951)
    Tgrid_core_grid60=np.loadtxt('EoS/Fe_core/Fe_adiabat_P_Tgridgrid_60GPa.txt')
    rho_Fes60=np.loadtxt('EoS/Fe_core/rho_Fes_dew06_60GPa.txt')
    alpha_Fes60=np.loadtxt('EoS/Fe_core/alpha_Fes_dew06_60GPa.txt')
    dqdy_Fes60=np.loadtxt('EoS/Fe_core/dqdy_Fes_dew06_60GPa.txt')
    T_Fes60=np.loadtxt('EoS/Fe_core/T_Fes_dew06_60GPa.txt')
    P_Fes60=np.loadtxt('EoS/Fe_core/P_Fes_dew06_60GPa.txt')

    data = dict(T_mix_ppv    = T_mix_ppv,
    rho_mix_ppv  = rho_mix_ppv,
    rho_sol_ppv  = rho_sol_ppv,
    CP_mix_ppv   = CP_mix_ppv,
    T_sol_ppv    = T_sol_ppv,
    alpha_mix_ppv= alpha_mix_ppv,
    alpha_sol_ppv= alpha_sol_ppv,
    dTdP_mix_ppv = dTdP_mix_ppv,
    dTdP_sol_ppv = dTdP_sol_ppv,
    dqdy_mix_ppv = dqdy_mix_ppv,
    dqdy_sol_ppv = dqdy_sol_ppv,
    T_mix_en     = T_mix_en,
    rho_mix_en   = rho_mix_en,
    CP_mix_en    = CP_mix_en,
    alpha_mix_en = alpha_mix_en,
    dTdP_mix_en  = dTdP_mix_en,
    dqdy_mix_en  = dqdy_mix_en,
    T_mix_pv     = T_mix_pv,
    rho_mix_pv   = rho_mix_pv,
    CP_mix_pv    = CP_mix_pv,
    alpha_mix_pv = alpha_mix_pv,
    dTdP_mix_pv  = dTdP_mix_pv,
    dqdy_mix_pv  = dqdy_mix_pv,
    T_sol_en     = T_sol_en,
    rho_sol_en   = rho_sol_en,
    alpha_sol_en = alpha_sol_en,
    dTdP_sol_en  = dTdP_sol_en,
    dqdy_sol_en  = dqdy_sol_en,
    T_sol_pv     = T_sol_pv,
    rho_sol_pv   = rho_sol_pv,
    alpha_sol_pv = alpha_sol_pv,
    dTdP_sol_pv  = dTdP_sol_pv,
    dqdy_sol_pv  = dqdy_sol_pv,
    T_liq        = T_liq,
    rho_liq      = rho_liq,
    CP_liq       = CP_liq,
    alpha_liq    = alpha_liq,
    dTdP_liq     = dTdP_liq,
    dqdy_liq     = dqdy_liq,

    y_grid       = y_grid,
    P_grid       = P_grid,
    P_grid_en    = P_grid_en,
    P_grid_pv    = P_grid_pv,
    P_grid_ppv    = P_grid_ppv,
    S_sol_array  = S_sol_array,
    S_liq_array  = S_liq_array,
    
    rho_Fel      = rho_Fel,
    alpha_Fel    = alpha_Fel,
    dqdy_Fel     = dqdy_Fel,
    T_Fel        = T_Fel,
    P_Fel        = P_Fel,

    rho_Fea      = rho_Fea,
    alpha_Fea    = alpha_Fea,
    dqdy_Fea     = dqdy_Fea,
    T_Fea        = T_Fea,
    P_Fea        = P_Fea,

    rho_Fes      = rho_Fes,
    alpha_Fes    = alpha_Fes,
    dqdy_Fes     = dqdy_Fes,
    T_Fes        = T_Fes,
    P_Fes        = P_Fel,
    
    rho_Fel60    = rho_Fel60,
    alpha_Fel60  = alpha_Fel60,
    dqdy_Fel60   = dqdy_Fel60,
    T_Fel60      = T_Fel60,
    P_Fel60      = P_Fel60,

    rho_Fea60    = rho_Fea60,
    alpha_Fea60  = alpha_Fea60,
    dqdy_Fea60   = dqdy_Fea60,
    T_Fea60      = T_Fea60,
    P_Fea60      = P_Fea60,

    rho_Fes60    = rho_Fes60,
    alpha_Fes60  = alpha_Fes60,
    dqdy_Fes60   = dqdy_Fes60,
    T_Fes60      = T_Fes60,
    P_Fes60      = P_Fel60,
    
    load_original_T = load_original_T,
    load_original_T60 = load_original_T60,
    load_original_dTdT0 = load_original_dTdT0,
    load_original_dTdT060 = load_original_dTdT060,
    load_original_dT0dP = load_original_dT0dP,
    load_original_dT0dP60 = load_original_dT0dP60,
    x_core_grid  = x_core_grid,
    Tref_core_grid = Tref_core_grid,
    P_core_grid = P_core_grid,
    Tgrid_core_grid = Tgrid_core_grid,
    x_core_grid60 = x_core_grid60,
    Tref_core_grid60 = Tref_core_grid60,
    P_core_grid60 = P_core_grid60,
    Tgrid_core_grid60 = Tgrid_core_grid60,
    )

    np.savez(DATA_FILE , **data) 

if EOS_TABLES_VERBOSE:
    print('Creating eos interpolators')
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

Tlg,Plg=np.meshgrid(T_Fel,P_Fel,sparse=True)
Tag,Pag=np.meshgrid(T_Fea,P_Fea,sparse=True)
Tsg,Psg=np.meshgrid(T_Fes,P_Fes,sparse=True)
f_rho_Fel=interpolate.RectBivariateSpline(Tlg,Plg,rho_Fel.T)
f_rho_Fea=interpolate.RectBivariateSpline(Tag,Pag,rho_Fea.T)
f_rho_Fes=interpolate.RectBivariateSpline(Tsg,Psg,rho_Fes.T)
f_alpha_Fel=interpolate.RectBivariateSpline(Tlg,Plg,alpha_Fel.T)
f_alpha_Fea=interpolate.RectBivariateSpline(Tag,Pag,alpha_Fea.T)
f_alpha_Fes=interpolate.RectBivariateSpline(Tsg,Psg,alpha_Fes.T)
f_dqdy_Fel=interpolate.RectBivariateSpline(Tlg,Plg,dqdy_Fel.T)
f_dqdy_Fea=interpolate.RectBivariateSpline(Tag,Pag,dqdy_Fea.T)
f_dqdy_Fes=interpolate.RectBivariateSpline(Tsg,Psg,dqdy_Fes.T)

Tlg,Plg=np.meshgrid(T_Fel60,P_Fel60,sparse=True)
Tag,Pag=np.meshgrid(T_Fea60,P_Fea60,sparse=True)
Tsg,Psg=np.meshgrid(T_Fes60,P_Fes60,sparse=True)
f_rho_Fel60=interpolate.RectBivariateSpline(Tlg,Plg,rho_Fel60.T)
f_rho_Fea60=interpolate.RectBivariateSpline(Tag,Pag,rho_Fea60.T)
f_rho_Fes60=interpolate.RectBivariateSpline(Tsg,Psg,rho_Fes60.T)
f_alpha_Fel60=interpolate.RectBivariateSpline(Tlg,Plg,alpha_Fel60.T)
f_alpha_Fea60=interpolate.RectBivariateSpline(Tag,Pag,alpha_Fea60.T)
f_alpha_Fes60=interpolate.RectBivariateSpline(Tsg,Psg,alpha_Fes60.T)
f_dqdy_Fel60=interpolate.RectBivariateSpline(Tlg,Plg,dqdy_Fel60.T)
f_dqdy_Fea60=interpolate.RectBivariateSpline(Tag,Pag,dqdy_Fea60.T)
f_dqdy_Fes60=interpolate.RectBivariateSpline(Tsg,Psg,dqdy_Fes60.T)

Plg,Tlg=np.meshgrid(P_Fel,T_Fel,sparse=True)
Pag,Tag=np.meshgrid(P_Fea,T_Fea,sparse=True)
f_rho_Fel_structure=interpolate.RectBivariateSpline(Plg,Tlg,rho_Fel)
f_rho_Fea_structure=interpolate.RectBivariateSpline(Pag,Tag,rho_Fea)
f_alpha_Fel_structure=interpolate.RectBivariateSpline(Plg,Tlg,alpha_Fel)
f_alpha_Fea_structure=interpolate.RectBivariateSpline(Pag,Tag,alpha_Fea)
f_dqdy_Fel_structure=interpolate.RectBivariateSpline(Plg,Tlg,dqdy_Fel)
f_dqdy_Fea_structure=interpolate.RectBivariateSpline(Pag,Tag,dqdy_Fea)


Plg,Tlg=np.meshgrid(P_Fel60,T_Fel60,sparse=True)
Pag,Tag=np.meshgrid(P_Fea60,T_Fea60,sparse=True)
f_rho_Fel60_structure=interpolate.RectBivariateSpline(Plg,Tlg,rho_Fel60)
f_rho_Fea60_structure=interpolate.RectBivariateSpline(Pag,Tag,rho_Fea60)
f_alpha_Fel60_structure=interpolate.RectBivariateSpline(Plg,Tlg,alpha_Fel60)
f_alpha_Fea60_structure=interpolate.RectBivariateSpline(Pag,Tag,alpha_Fea60)
f_dqdy_Fel60_structure=interpolate.RectBivariateSpline(Plg,Tlg,dqdy_Fel60)
f_dqdy_Fea60_structure=interpolate.RectBivariateSpline(Pag,Tag,dqdy_Fea60)

interp_kwargs = dict(bounds_error=False, fill_value=None)

f_adiabat=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, P_core_grid), load_original_T)
f_dT0dP=interpolate.RegularGridInterpolator((x_core_grid, Tref_core_grid, Tgrid_core_grid), load_original_dT0dP)
f_adiabat60=interpolate.RegularGridInterpolator((x_core_grid60, Tref_core_grid60, P_core_grid60), load_original_T60)
f_dT0dP60=interpolate.RegularGridInterpolator((x_core_grid60, Tref_core_grid60, Tgrid_core_grid60), load_original_dT0dP60)

dTdT0_cmb_interp = interpolate.RegularGridInterpolator(
    (
        x_core_grid,
        Tref_core_grid,
        P_core_grid,
    ),
    load_original_dTdT0,
    **interp_kwargs,
)
T_interp = interpolate.RegularGridInterpolator(
    (
        x_core_grid,
        Tref_core_grid,
        P_core_grid,
    ),
    load_original_T,
    **interp_kwargs,
)

dTdT0_cmb_interp60 = interpolate.RegularGridInterpolator(
    (
        x_core_grid60,
        Tref_core_grid60,
        P_core_grid60,
    ),
    load_original_dTdT060,
    **interp_kwargs,
)
T_interp60 = interpolate.RegularGridInterpolator(
    (
        x_core_grid60,
        Tref_core_grid60,
        P_core_grid60,
    ),
    load_original_T60,
    **interp_kwargs,
)

T_interp_2d_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), T_liq, **interp_kwargs)
T_interp_2d_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), T_sol_pv, **interp_kwargs)
T_interp_2d_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), T_sol_ppv, **interp_kwargs)
T_interp_2d_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), T_sol_en, **interp_kwargs)
T_interp_2d_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), T_mix_pv, **interp_kwargs)
T_interp_2d_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), T_mix_ppv, **interp_kwargs)
T_interp_2d_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), T_mix_en, **interp_kwargs)
T_interp_2d_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), T_liq, **interp_kwargs)
T_interp_2d_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), T_sol_pv, **interp_kwargs)
T_interp_2d_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), T_sol_ppv, **interp_kwargs)
T_interp_2d_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), T_sol_en, **interp_kwargs)
T_interp_2d_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), T_mix_pv, **interp_kwargs)
T_interp_2d_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), T_mix_ppv, **interp_kwargs)
T_interp_2d_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), T_mix_en, **interp_kwargs)
T_interp_2d_dy_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), dqdy_liq, **interp_kwargs)
T_interp_2d_dy_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), dqdy_sol_pv, **interp_kwargs)
T_interp_2d_dy_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), dqdy_sol_ppv, **interp_kwargs)
T_interp_2d_dy_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), dqdy_sol_en, **interp_kwargs)
T_interp_2d_dy_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), dqdy_mix_pv, **interp_kwargs)
T_interp_2d_dy_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), dqdy_mix_ppv, **interp_kwargs)
T_interp_2d_dy_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), dqdy_mix_en, **interp_kwargs)
T_interp_2d_o_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), rho_liq, **interp_kwargs)
T_interp_2d_o_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), rho_sol_pv, **interp_kwargs)
T_interp_2d_o_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), rho_sol_ppv, **interp_kwargs)
T_interp_2d_o_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), rho_sol_en, **interp_kwargs)
T_interp_2d_o_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), rho_mix_pv, **interp_kwargs)
T_interp_2d_o_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), rho_mix_ppv, **interp_kwargs)
T_interp_2d_o_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), rho_mix_en, **interp_kwargs)
T_interp_2d__liq = interpolate.RegularGridInterpolator((P_grid, y_grid), CP_liq, **interp_kwargs)
T_interp_2d__mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), CP_mix_pv, **interp_kwargs)
T_interp_2d__mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), CP_mix_ppv, **interp_kwargs)
T_interp_2d__mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), CP_mix_en, **interp_kwargs)
T_interp_2d_pha_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), alpha_liq, **interp_kwargs)
T_interp_2d_pha_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), alpha_sol_pv, **interp_kwargs)
T_interp_2d_pha_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), alpha_sol_ppv, **interp_kwargs)
T_interp_2d_pha_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), alpha_sol_en, **interp_kwargs)
T_interp_2d_pha_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), alpha_mix_pv, **interp_kwargs)
T_interp_2d_pha_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), alpha_mix_ppv, **interp_kwargs)
T_interp_2d_pha_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), alpha_mix_en, **interp_kwargs)
T_interp_2d_dP_liq = interpolate.RegularGridInterpolator((P_grid, y_grid), dTdP_liq, **interp_kwargs)
T_interp_2d_dP_sol_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), dTdP_sol_pv, **interp_kwargs)
T_interp_2d_dP_sol_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), dTdP_sol_ppv, **interp_kwargs)
T_interp_2d_dP_sol_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), dTdP_sol_en, **interp_kwargs)
T_interp_2d_dP_mix_pv = interpolate.RegularGridInterpolator((P_grid_pv, y_grid), dTdP_mix_pv, **interp_kwargs)
T_interp_2d_dP_mix_ppv = interpolate.RegularGridInterpolator((P_grid_ppv, y_grid), dTdP_mix_ppv, **interp_kwargs)
T_interp_2d_dP_mix_en = interpolate.RegularGridInterpolator((P_grid_en, y_grid), dTdP_mix_en, **interp_kwargs)
interp_2d_rho_Fel = interpolate.RegularGridInterpolator((P_Fel, T_Fel), rho_Fel, **interp_kwargs)
interp_2d_rho_Fes = interpolate.RegularGridInterpolator((P_Fes, T_Fes), rho_Fes, **interp_kwargs)
interp_2d_rho_Fea = interpolate.RegularGridInterpolator((P_Fea, T_Fea), rho_Fea, **interp_kwargs)
interp_2d_alpha_Fel = interpolate.RegularGridInterpolator((P_Fel, T_Fel), alpha_Fel, **interp_kwargs)
interp_2d_alpha_Fes = interpolate.RegularGridInterpolator((P_Fes, T_Fes), alpha_Fes, **interp_kwargs)
interp_2d_alpha_Fea = interpolate.RegularGridInterpolator((P_Fea, T_Fea), alpha_Fea, **interp_kwargs)
interp_2d_dqdy_Fel = interpolate.RegularGridInterpolator((P_Fel, T_Fel), dqdy_Fel, **interp_kwargs)
interp_2d_dqdy_Fes = interpolate.RegularGridInterpolator((P_Fes, T_Fes), dqdy_Fes, **interp_kwargs)
interp_2d_dqdy_Fea = interpolate.RegularGridInterpolator((P_Fea, T_Fea), dqdy_Fea, **interp_kwargs)
interp_2d_rho_Fel_a = interpolate.RegularGridInterpolator((P_Fel, T_Fel), rho_Fel, **interp_kwargs)
interp_2d_rho_Fea_a = interpolate.RegularGridInterpolator((P_Fea, T_Fea), rho_Fea, **interp_kwargs)
interp_2d_alpha_Fel_a = interpolate.RegularGridInterpolator((P_Fel, T_Fel), alpha_Fel, **interp_kwargs)
interp_2d_alpha_Fea_a = interpolate.RegularGridInterpolator((P_Fea, T_Fea), alpha_Fea, **interp_kwargs)
interp_2d_rho_Fel60 = interpolate.RegularGridInterpolator((P_Fel60, T_Fel60), rho_Fel60, **interp_kwargs)
interp_2d_rho_Fes60 = interpolate.RegularGridInterpolator((P_Fes60, T_Fes60), rho_Fes60, **interp_kwargs)
interp_2d_rho_Fea60 = interpolate.RegularGridInterpolator((P_Fea60, T_Fea60), rho_Fea60, **interp_kwargs)
interp_2d_alpha_Fel60 = interpolate.RegularGridInterpolator((P_Fel60, T_Fel60), alpha_Fel60, **interp_kwargs)
interp_2d_alpha_Fes60 = interpolate.RegularGridInterpolator((P_Fes60, T_Fes60), alpha_Fes60, **interp_kwargs)
interp_2d_alpha_Fea60 = interpolate.RegularGridInterpolator((P_Fea60, T_Fea60), alpha_Fea60, **interp_kwargs)
interp_2d_dqdy_Fel60 = interpolate.RegularGridInterpolator((P_Fel60, T_Fel60), dqdy_Fel60, **interp_kwargs)
interp_2d_dqdy_Fes60 = interpolate.RegularGridInterpolator((P_Fes60, T_Fes60), dqdy_Fes60, **interp_kwargs)
interp_2d_dqdy_Fea60 = interpolate.RegularGridInterpolator((P_Fea60, T_Fea60), dqdy_Fea60, **interp_kwargs)
interp_2d_rho_Fel_a60 = interpolate.RegularGridInterpolator((P_Fel60, T_Fel60), rho_Fel60, **interp_kwargs)#
interp_2d_rho_Fea_a60 = interpolate.RegularGridInterpolator((P_Fea60, T_Fea60), rho_Fea60, **interp_kwargs)#
interp_2d_alpha_Fel_a60 = interpolate.RegularGridInterpolator((P_Fel60, T_Fel60), alpha_Fel60, **interp_kwargs)#
interp_2d_alpha_Fea_a60 = interpolate.RegularGridInterpolator((P_Fea60, T_Fea60), alpha_Fea60, **interp_kwargs)# check these
