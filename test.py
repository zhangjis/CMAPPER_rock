import subprocess
from tqdm import tqdm
import os
from PIL import Image
import glob
import contextlib

import cv2
import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np

load_file=np.loadtxt('input.txt')
results_foldername='results_Mass'+str(load_file[0])+'EarthMass_CoreMassFraction'+str(load_file[1])+'_TotalTime'+str(load_file[2])+'Gyr_RadiogenicHeatingKThU238U235RatioToEarth'+str(load_file[3])+'_'+str(load_file[4])+'_'+str(load_file[5])+'_'+str(load_file[6])+'_EquilibriumTemperature'+str(load_file[8])+'K'
os.makedirs(results_foldername+'/profile/t0', exist_ok=True)

# filepaths
filefolders = [
    results_foldername+'/image/TemperatureVsPressure/step',
    results_foldername+'/image/TemperatureVsMass/step',
    results_foldername+'/image/MantleConvectiveHeatFluxVsMass/step',
    results_foldername+'/image/DensityVsMass/step',
    results_foldername+'/image/MagneticReynoldsNumberInMagmaOceanVsMass/step',
    results_foldername+'/image/MantleViscosityVsMass/step',
    results_foldername+'/image/MantleConvectiveVelocityVsMass/step',
    results_foldername+'/image/RadiusVsMass/step',
    results_foldername+'/image/PressureVsMass/step',
    results_foldername+'/image/GravityVsMass/step'
]
for filefolder in filefolders:
    os.makedirs(filefolder, exist_ok=True)
program_list=[
'rocky_class.py','heat_transport.py']
for program in program_list:
    print("Start:" + program)
    subprocess.check_call(['python3', program])
    print("Simulation finished")

previous=np.loadtxt(results_foldername+'/profile/t0/previous0.txt')
core_zone=previous[-1]

N_PLOTS = 312
print('Plotting %d timesteps...' % N_PLOTS)

evo=np.loadtxt(results_foldername+'/evolution.txt')
plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/86400.0/365.0,evo[:,7],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Surface heat flux ($Wm^{-2}$)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/SurfaceHeatFlux.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,4],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Surface temperature (K)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/SurfaceTemperature.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,8],color='black',linewidth=2.0,label='Heat flux at core mantle boundary')
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,9],color='black',linestyle='dashed',linewidth=2.0, label='Conductive heat flux along core adiabat')
plt.yscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'Flux ($Wm^{-2}$)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.legend(frameon=True, fontsize=12)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/HeatFluxAtCoreMantleBoundary.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,10]/6371000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'Planet radius ($R_{\oplus}$)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/PlanetRadius.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,11]/6371000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'Core radius ($R_{\oplus}$)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/CoreRadius.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,12]/1e9,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Central pressure (GPa)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/CentralPressure.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,6],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Central temperature (K)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/CentralTemperature.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,13]/1e9,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Core mantle boundary pressure (GPa)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/PressureAtCoreMantleBoundary.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,5],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Core mantle boundary temperature (K)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.savefig(results_foldername+'/image/TemperatureAtCoreMantleBoundary.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,14]/1e3,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Inner core size (km)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/RadiusOfSolidInnerCore.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,15]/5.97e24,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel(r'Inner core mass ($M_{\oplus})$',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/MassOfSolidInnerCore.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,16]/1000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xscale('log')
plt.ylabel('Thickness of dynamo source region in magma ocean (km)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/ThicknessOfDynamoSourceRegionInMagmaOcean.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,17]/1e12,color='tomato',linewidth=2.0,label='Mantle')
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,18]/1e12,color='royalblue',linewidth=2.0,label='Core')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Radiogenic heating (TW)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.legend(frameon=True,fontsize=12)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/RadiogenicHeatingInCoreAndMantle.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,2],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Mass-averaged mantle temperature (K)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/MassAveragedMantleTemperature.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,3],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Mass-averaged core temperature (K)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/MassAveragedCoreTemperature.png',dpi=200)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,22]/1e21,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.yscale('log')
plt.ylabel(r'Core dipolar magnetic moment ($\times 10^{21} Am^2$)',fontsize=16.5)
plt.xlabel('Time (Gyr)',fontsize=16.5)
plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
plt.savefig(results_foldername+'/image/CoreDipolarMagneticMoment.png',dpi=200)
plt.close()

save_t=[1.0]
for i in range(1,312):
    if save_t[i-1]<5000.0:
        save_t.append(save_t[i-1]+60.0)
    elif save_t[i-1]<1e8:
        save_t.append(save_t[i-1]+int(save_t[i-1]/8.0))
    else:
        save_t.append(save_t[i-1]+int(save_t[i-1]/30.0))
save_t_title=save_t.copy()
t_title=[]
for i in range(len(save_t)):
    if save_t[i]<100.0 and save_t[i]>1.0:
        save_t_title[i]=save_t_title[i]-1.0
    elif save_t[i]>=100.0 and save_t[i]<10000.0:
        round_value=round(save_t[i], -2)
        save_t_title[i]=round_value
    else:
        round_value=round(save_t[i], -len(str(int(save_t[i])))+2)
        save_t_title[i]=round_value

    if save_t_title[i]<1000.0:
        t_title.append('t = '+str(int(save_t_title[i]))+' years')
    elif save_t_title[i]>=1000.0 and save_t_title[i]<1000000.0:
        t_title.append('t = '+str((save_t_title[i])/1000.0)+' kyr')
    elif save_t_title[i]>=1000000.0 and save_t_title[i]<1000000000.0:
        t_title.append('t = '+str((save_t_title[i])/1000000.0)+' Myr')
    elif save_t_title[i]>=1000000000.0 and save_t_title[i]<1000000000000.0:
        t_title.append('t = '+str((save_t_title[i])/1000000000.0)+' Gyr')

    st=t_title[i]
    if len(t_title[i])==11:
        t_title[i]='  '+st
    elif len(t_title[i])==12:
        t_title[i]=' '+st

mass_profile=np.loadtxt(results_foldername+'/profile/t0/structure0.txt')
mass=mass_profile[:,8]

s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[0]))+'.txt')
T_max=max(s[:,4])+max(s[:,4])*0.01
R_max=max(s[:,0])+max(s[:,0])*0.01
Rem_max=max(s[:,9])+max(s[:,9])*0.01
vconv_max=max(s[:,8][int(core_zone+5):-2])+max(s[:,8][int(core_zone+5):-2])*0.01
Fconv_max=(max(s[:,7][int(core_zone+5):-2])+max(s[:,7][int(core_zone+5):-2])*0.01)/(4.0*np.pi*s[:,0][-1]**2.0)
s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[N_PLOTS-1]))+'.txt')
T_min=min(s[:,4])-50.0
P_max=max(s[:,1])/1e9+max(s[:,1])/1e11
P_min=-1.0
rho_max=max(s[:,2])+max(s[:,2])*0.01
g_max=max(s[:,3])+max(s[:,3])*0.01
eta_max=max(s[:,10])+max(s[:,10])*0.01

for i in tqdm(range(N_PLOTS)):
    s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[i]))+'.txt')
    plt.figure(figsize=(8,6))
    plt.ylim(0.0,T_max)
    plt.plot(mass/5.972e24,s[:,4],color='black',linewidth=2.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.savefig(results_foldername+'/image/TemperatureVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(s[:,1]/1e9,s[:,4],color='black',linewidth=2.0)
    plt.xlim(0.0,P_max)
    plt.ylim(0.0,T_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel('Pressure (GPa)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/TemperatureVsPressure/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,3],color='black',linewidth=2.0)
    plt.ylim(0.01,g_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Gravitational acceleration ($ms^{-2}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/GravityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,7]/(4.0*np.pi*s[:,0]**2.0),color='black',linewidth=2.0)
    plt.yscale('log')
    plt.ylim(1e-5,Fconv_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Convective heat flux ($Wm^{-2}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MantleConvectiveHeatFluxVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,2],color='black',linewidth=2.0)
    plt.ylim(0.01,rho_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Density (kg m$^{-3}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/DensityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,0]/6371000.0,color='black',linewidth=2.0)
    plt.ylim(0.0,R_max/6371000.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Radius ($R_{\oplus}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/RadiusVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,1]/1e9,color='black',linewidth=2.0)
    plt.ylim(P_min,P_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Pressure (GPa)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/PressureVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,9],color='black',linewidth=2.0)
    plt.ylim(0.01,Rem_max)
    #plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Magnetic Reynolds number',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MagneticReynoldsNumberInMagmaOceanVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,10],color='black',linewidth=2.0)
    plt.ylim(0.01,eta_max)
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Viscosity (Pa $\cdot$ s)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MantleViscosityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(mass/5.972e24,s[:,8],color='black',linewidth=2.0)
    plt.ylim(1e-14,vconv_max)
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Convective velocity ($ms^{-1}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MantleConvectiveVelocityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

print('Making movies')
out_names=['TemperatureVsPressure','TemperatureVsMass','MantleConvectiveHeatFluxVsMass','DensityVsMass','MagneticReynoldsNumberInMagmaOceanVsMass','MantleViscosityVsMass', 'MantleConvectiveVelocityVsMass','RadiusVsMass','PressureVsMass','GravityVsMass']
for i in range(len(out_names)):
    outnames=os.path.join(filefolders[i][:len(results_foldername+'/image')],out_names[i]+'.mp4')
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default=outnames, help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = filefolders[i]
    ext = args['extension']
    output = args['output']

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    images=sorted(images)

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))
    shutil.rmtree(filefolders[i][:-5])



