import subprocess
from tqdm import tqdm
import os
from PIL import Image
import glob
import contextlib

import matplotlib.pyplot as plt
import numpy as np

os.makedirs('initial', exist_ok=True)
os.makedirs('results/profile', exist_ok=True)

# filepaths
filefolders = [
    "results/image/PT/step",
    "results/image/MT/step",
    "results/image/Fconv/step",
    "results/image/rho/step",
    "results/image/Rem/step",
    "results/image/viscosity/step",
    "results/image/vconv/step",
    "results/image/MR/step",
    "results/image/MP/step",
    "results/image/gravity/step"
]
for filefolder in filefolders:
    os.makedirs(filefolder, exist_ok=True)

program_list=[
'rocky_class.py','heat_transport.py']
for program in program_list:
    print("Start:" + program)
    subprocess.call(['python3', program])
    print("Simulation finished")


N_PLOTS = 643
print('Plotting %d timesteps...' % N_PLOTS)


evo=np.loadtxt('results/evolution.txt')
plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/86400.0/365.0,evo[:,7],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Surface heat flux (W/m^2)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Fsurf.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,5],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Surface temperature (K)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Tsurf.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,8],color='black',linewidth=2.0,label='Heat flux at core mantle boundary')
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,9],color='black',linestyle='dashed',linewidth=2.0, label='Conductive heat flux along core adiabat')
plt.yscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Flux (W/m^2)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.legend(frameon=True, fontsize=12)
plt.savefig('results/image/Fcmb.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,10]/6371000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Planet radius (R_Earth)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Planet_radius.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,11]/6371000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Core radius (R_Earth)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Core_radius.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,12]/1e9,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Central pressure (GPa)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Pc.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,20],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Central temperature (K)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Tc.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,13]/1e9,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Core mantle boundary pressure (GPa)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Pcmb.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,6],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Core mantle boundary temperature (K)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Tcmb.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,14]/1e3,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Inner core size (km)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Ric.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,15]/5.97e24,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Inner core mass (M_Earth)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/Mic.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,16]/1000.0,color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xscale('log')
plt.ylabel('MO dynamo thickness (km)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/D_MO.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,17]/1e12,color='tomato',linewidth=2.0,label='Mantle')
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,18]/1e12,color='royalblue',linewidth=2.0,label='Core')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Radiogenic_heating (TW)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.legend(frameon=True,fontsize=12)
plt.savefig('results/image/Radiogenic_heating.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,3],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Average mantle temperature (K)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/average_Tm.png',dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,4],color='black',linewidth=2.0)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Average core temperature (K)',fontsize=16.5)
plt.xlabel('Gyr',fontsize=16.5)
plt.savefig('results/image/average_Tc.png',dpi=200)
plt.close()


save_t=[1.0]
for i in range(1,643):
    if save_t[i-1]<5000.0:
        save_t.append(save_t[i-1]+25.0)
    elif save_t[i-1]<1e8:
        save_t.append(save_t[i-1]+int(save_t[i-1]/20.0))
    else:
        save_t.append(save_t[i-1]+int(save_t[i-1]/50.0))
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

mass_profile=np.loadtxt('initial/structure0.txt')
mass=mass_profile[:,8]

s=np.loadtxt('results/profile/structure_'+str(int(save_t[0]))+'.txt')
T_max=max(s[:,2])+max(s[:,2])*0.01
R_max=max(s[:,0])+max(s[:,0])*0.01
s=np.loadtxt('results/profile/structure_'+str(int(save_t[N_PLOTS-1]))+'.txt')
p=np.loadtxt('results/profile/property_'+str(int(save_t[0]))+'.txt')
T_min=min(s[:,2])-50.0
P_max=max(s[:,1])/1e9+max(s[:,1])/1e11
P_min=-1.0
Rem_max=max(p[:,5])+max(p[:,5])*0.01
rho_max=max(s[:,3])+max(s[:,3])*0.01
g_max=max(s[:,4])+max(s[:,4])*0.01
vconv_max=max(p[:,4][352:-2])+max(p[:,4][352:-2])*0.01
Fconv_max=(max(p[:,3][352:-2])+max(p[:,3][352:-2])*0.01)/(4.0*np.pi*s[:,0][-1]**2.0)
p=np.loadtxt('results/profile/property_'+str(int(save_t[N_PLOTS-1]))+'.txt')
eta_max=max(p[:,6])+max(p[:,6])*0.01

for i in tqdm(range(N_PLOTS)):
    s=np.loadtxt('results/profile/structure_'+str(int(save_t[i]))+'.txt')
    p=np.loadtxt('results/profile/property_'+str(int(save_t[i]))+'.txt')
    plt.figure(figsize=(8,5))
    plt.ylim(0.0,T_max)
    plt.plot(mass/5.972e24,s[:,2],color='black',linewidth=2.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/MT/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(s[:,1]/1e9,s[:,2],color='black',linewidth=2.0)
    plt.xlim(0.0,P_max)
    plt.ylim(0.0,T_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel('Pressure (GPa)',fontsize=16.5)
    plt.savefig('results/image/PT/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,s[:,4],color='black',linewidth=2.0)
    plt.ylim(0.01,g_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Gravitational acceleration (m/s^2)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/gravity/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,p[:,3]/(4.0*np.pi*s[:,0]**2.0),color='black',linewidth=2.0)
    plt.yscale('log')
    plt.ylim(1e-5,Fconv_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Convective heat flux (W/m^2)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/Fconv/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,s[:,3],color='black',linewidth=2.0)
    plt.ylim(0.01,rho_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Density (kg/m^3)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/rho/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,s[:,0]/6371000.0,color='black',linewidth=2.0)
    plt.ylim(0.0,R_max/6371000.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Radius (R_Earth)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/MR/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,s[:,1]/1e9,color='black',linewidth=2.0)
    plt.ylim(P_min,P_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Pressure (GPa)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/MP/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,p[:,5],color='black',linewidth=2.0)
    plt.yscale('log')
    plt.ylim(0.01,Rem_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Magnetic Reynolds number',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/Rem/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,p[:,6],color='black',linewidth=2.0)
    plt.yscale('log')
    plt.ylim(0.01,eta_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Viscosity (Pa s)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/viscosity/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(mass/5.972e24,p[:,4],color='black',linewidth=2.0)
    plt.yscale('log')
    plt.ylim(1e-14,vconv_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(t_title[i], fontsize=16)
    plt.ylabel('Convective velocity (m/s)',fontsize=16.5)
    plt.xlabel('Mass (M_Earth)',fontsize=16.5)
    plt.savefig('results/image/vconv/step/%04d.png' % i,dpi=200)
    plt.close()

out_names=['PT','MT','Fconv','rho','Rem','viscosity', 'vconv','MR','MP','gravity']
for i in range(len(filefolders)):
    fp_in = os.path.join(filefolders[i], '*.png')
    fp_out = os.path.join(filefolders[i][:13],out_names[i]+'.gif')
    print('Creating video:', fp_out)

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=20, loop=0)
