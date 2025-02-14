import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

def plot_for_index(args):
    (
        i, save_t, t_title, mass, core_zone,
        P_min, P_max, R_min, R_max, rho_min, rho_max,
        g_min, g_max, T_min, T_max, P_min, P_max,
        Fconv_min, Fconv_max, vconv_min, vconv_max, Rem_min, Rem_max,
        eta_min, eta_max, results_foldername, load_file,
    ) = args
    s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[i]))+'.txt')

    plt.figure(figsize=(9,6))
    plt.plot(s[:,0]/6371000.0,s[:,1]/1e9,color='black',linewidth=2.0)
    plt.ylim(P_min,P_max)
    plt.xlim(R_min/6371000.0,R_max/6371000.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Pressure (GPa)',fontsize=16.5)
    plt.xlabel(r'Radius ($R_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/PressureVsRadius/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,2],color='black',linewidth=2.0)
    plt.ylim(rho_min,rho_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Density (kg m$^{-3}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/DensityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,3],color='black',linewidth=2.0)
    plt.ylim(g_min,g_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Gravitational acceleration ($ms^{-2}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/GravityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.ylim(T_min,T_max)
    plt.plot(mass/5.972e24,s[:,4],color='black',linewidth=2.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.savefig(results_foldername+'/image/TemperatureVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(s[:,1]/1e9,s[:,4],color='black',linewidth=2.0)
    plt.xlim(P_min,P_max)
    plt.ylim(T_min,T_max)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Temperature (K)',fontsize=16.5)
    plt.xlabel('Pressure (GPa)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/TemperatureVsPressure/step/%04d.png' % i,dpi=200)
    plt.close()

    solid_core_index=0
    for j in range(int(core_zone-1)):
        if s[:,8][j]>0 and s[:,8][j+1]==0.0:
            solid_core_index=j
    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,9],color='black',linewidth=4.0,label='Total flux')
    plt.plot(mass[int(core_zone+1):-2]/5.972e24,s[:,7][int(core_zone+1):-2],color='tomato',linewidth=1.5,label='Convective heat flux in mantle')
    plt.plot(mass[int(core_zone):]/5.972e24,s[:,8][int(core_zone):],color='royalblue',linewidth=1.5,label='Conductive heat flux in mantle')
    plt.plot(mass[int(solid_core_index+1):int(core_zone)]/5.972e24,s[:,7][int(solid_core_index+1):int(core_zone)],linestyle='dashed',color='tomato',linewidth=1.5,label='Total heat flux in liquid core')
    plt.plot(mass[:int(solid_core_index)]/5.972e24,s[:,8][:int(solid_core_index)],linestyle='dashed',color='royalblue',linewidth=1.5,label='Conductive heat flux in solid core')
    plt.yscale('log')
    plt.ylim(Fconv_min,Fconv_max*10000.0)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Convective heat flux ($Wm^{-2}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.legend(frameon=True,loc='upper left',fontsize=12)
    plt.savefig(results_foldername+'/image/HeatFluxVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,10],color='black',linewidth=2.0)
    plt.ylim(vconv_min,vconv_max)
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Convective velocity ($ms^{-1}$)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/ConvectiveVelocityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,11],color='white',linewidth=2.0)
    plt.plot(mass/5.972e24,s[:,11],color='black',linewidth=2.0,label='Profile of magnetic Reynolds Number')
    plt.ylim(Rem_min,Rem_max*5.0)
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    threshold_Bfield=np.ones(len(mass))*50.0
    plt.plot(mass/5.972e24,threshold_Bfield,color='black',linestyle=':',label='Threshold for potential dynamo')
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Magnetic Reynolds number',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.legend(frameon=True,loc='upper right',fontsize=12)
    plt.savefig(results_foldername+'/image/MagneticReynoldsNumberVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,12],color='white',linewidth=2.0)
    plt.plot(mass[int(core_zone):]/5.972e24,s[:,12][int(core_zone):],color='black',linewidth=2.0)
    plt.ylim(eta_min,eta_max)
    plt.yscale('log')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel(r'Viscosity (Pa $\cdot$ s)',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MantleViscosityVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(mass/5.972e24,s[:,14],color='black',linewidth=2.0)
    plt.ylim(-0.05,1.05)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1])+'   '+t_title[i] ,fontsize=16)
    plt.ylabel('Melt fraction by mass',fontsize=16.5)
    plt.xlabel(r'Mass ($M_{\oplus}$)',fontsize=16.5)
    plt.savefig(results_foldername+'/image/MantleMeltFractionVsMass/step/%04d.png' % i,dpi=200)
    plt.close()

def update_input_parameters():
    print("Please enter new simulation parameters:")
    
    params_def = [
        {
            "prompt": "Planet Mass (Unit: M_Earth. Valid range: 0.5 - 8.0): ",
            "comments": ["Planet mass (M_pl) in earth mass (Valid range for the current version: 0.5 - 8.0)"],
            "num_values": 1
        },
        {
            "prompt": "Core Mass Fraction (Valid range: 0.1 - 0.7): ",
            "comments": ["Core mass fraction (Valid range for the current version: 0.1 and 0.7)"],
            "num_values": 1
        },
        {
            "prompt": "Evolutionary Time (Unit: Gyr. Default choice: 14): ",
            "comments": ["Evolutionary time in billion years (Gyr)"],
            "num_values": 1
        },
        {
            # This parameter block holds four related values for the mantle’s radiogenic heating.
            "prompt": "Mantle radiogenic heating relative to that of Earth's mantle (enter values for each as prompted below):\n",
            "comments": ["Radiogenic heating relative to that of Earth's mantle. We currently don't consider Al24 for its short half life time."],
            "num_values": 0,  # Will be handled via subparams.
            "subparams": [
                {
                    "prompt": "  Potassium - 40 ",
                    "comments": ["K"],
                    "num_values": 1
                },
                {
                    "prompt": "  Thorium - 232 ",
                    "comments": ["Th"],
                    "num_values": 1
                },
                {
                    "prompt": "  Uranium - 238 ",
                    "comments": ["U238"],
                    "num_values": 1
                },
                {
                    "prompt": "  Uranium - 235 ",
                    "comments": ["U235"],
                    "num_values": 1
                }
            ]
        },
        {
            "prompt": "Viscosity model for post-perovskite (enter 1 for diffusion creep (default choice) or 2 for dislocation creep): ",
            "comments": [
                "viscosity models. For the current version, two rheology models for the viscosity of ppv is included. 1 is for diffusion creep (default) and 2 is for dislocation creep. See Tackley et al. 2013."
            ],
            "num_values": 1
        },
        {
            "prompt": "Equilibrium temperature (Unit: K. Default choice: 255): ",
            "comments": ["Equilibrium temperature (Valid range for the current version: 255.0 (default) - 2700.0)"],
            "num_values": 1
        },
        {
            "prompt": "Core radiogenic heating (Value of 1 corresponds to a concentration of potassium with 1 TW heating in Earth's core at 4.5 Gyr. Default choice: 0): ",
            "comments": [
                "radiogenic heating from potassium alone in core. 1.0 being the concentration of potassium with 1 TW heating at 4.5 Gyr."
            ],
            "num_values": 1
        }
    ]
    
    # This will hold the numerical values for each parameter.
    # For parameters with a single value, we store a list of one number.
    # For subparameter blocks, we store a list of numbers.
    params = []
    
    # Helper function: repeatedly prompt the user until a valid float is entered.
    def get_float_input(prompt_text):
        while True:
            s = input(prompt_text)
            try:
                return float(s)
            except ValueError:
                print("Invalid input. Please enter a valid number (integer or float).")
    
    # Iterate over each parameter definition and prompt the user.
    for param_def in params_def:
        if "subparams" in param_def:
            subvalues = []
            # Optionally, you might print the main prompt once (if you wish).
            print(param_def["prompt"])
            for sub in param_def["subparams"]:
                value = get_float_input(sub["prompt"])
                subvalues.append(value)
            params.append(subvalues)
        else:
            value = get_float_input(param_def["prompt"])
            params.append([value])
    
    # Write the header and all parameters to 'input.txt'
    with open('input.txt', 'w') as f:
        # Write header line
        f.write("##### Do not change the order of input parameters\n")
        
        # Loop over each parameter definition along with the values we just obtained.
        for idx, param_def in enumerate(params_def):
            # Write the comment lines for this parameter.
            for comment in param_def["comments"]:
                f.write("# " + comment + "\n")
            # If there is a subparameter block, write each subparameter on its own line.
            if "subparams" in param_def:
                for sub_idx, sub in enumerate(param_def["subparams"]):
                    # If a subparameter has its own comment, write it.
                    if sub.get("comments"):
                        for c in sub["comments"]:
                            f.write("# " + c + "\n")
                    f.write(f"{params[idx][sub_idx]}\n")
            else:
                # Otherwise, just write the single value.
                f.write(f"{params[idx][0]}\n")
        f.write(f"1.0\n")
    print("Input parameters have been updated and written to 'input.txt'.")

def prompt_new_simulation():
    if not hasattr(prompt_new_simulation, "first_time"):
        prompt_new_simulation.first_time = True

    if prompt_new_simulation.first_time:
        prompt_new_simulation.first_time = False
        answer = input(
            "Do you want to use the existing input.txt file or create a new one? "
            "(Enter 'e' for existing, 'n' for new): "
        ).strip().lower()
        if answer.startswith('n'):
            update_input_parameters()
        # Always proceed with the first simulation.
        return True
    else:    
        answer = input("Do you want to run another simulation? (y/n): ").strip().lower()
        if answer.startswith('y'):
            update_input_parameters()
            return True
        else:
            return False

def run():
    while prompt_new_simulation():
        
        load_file=np.loadtxt('input.txt')
        results_foldername='results_Mpl'+str(load_file[0])+'_CMF'+str(load_file[1])+'_time'+str(load_file[2])+'_Qrad'+str(load_file[3])+'_'+str(load_file[4])+'_'+str(load_file[5])+'_'+str(load_file[6])+'_Teq'+str(load_file[8])+'_Qradc'+str(load_file[9])+'_eta'+str(load_file[7])+'_mzmulti'+str(load_file[10])
        os.makedirs(results_foldername+'/profile/t0', exist_ok=True)
        # filepaths
        filefolders = [
            results_foldername+'/image/TemperatureVsPressure/step',
            results_foldername+'/image/TemperatureVsMass/step',
            results_foldername+'/image/HeatFluxVsMass/step',
            results_foldername+'/image/DensityVsMass/step',
            results_foldername+'/image/MagneticReynoldsNumberVsMass/step',
            results_foldername+'/image/MantleViscosityVsMass/step',
            results_foldername+'/image/ConvectiveVelocityVsMass/step',
            results_foldername+'/image/PressureVsRadius/step',
            results_foldername+'/image/GravityVsMass/step',
            results_foldername+'/image/MantleMeltFractionVsMass/step'
        ]
        for filefolder in filefolders:
            os.makedirs(filefolder, exist_ok=True)
        program_list=['rocky_class.py','heat_transport.py']
        # Create a copy of the current environment variables
        env = os.environ.copy()
        # Set EOS_TABLES_VERBOSE to '0' for the subprocess
        env['EOS_TABLES_VERBOSE'] = '0'
        for program in program_list:
            print("Start:" + program, time.time())
            subprocess.check_call(['python3', program],env=env)
            if program==program_list[0]:
                print("Initial profiles obtained")
            else:
                print("Simulation finished")
        print(time.time())
        def f_axis_max_min(v_min,v_max,axis_scale):
            if axis_scale=='log':
                v_max=np.log10(v_max)
                if v_min<=0.0:
                    v_min=1e-3
                v_min=np.log10(v_min)
                dv=np.abs(v_max-v_min)/10.0
                v_max=10.0**(v_max+dv)
                v_min=10.0**(v_min-dv)
            else:
                dv=np.abs(v_max-v_min)/10.0
                v_max=v_max+dv
                v_min=v_min-dv
            return v_min,v_max

        previous=np.loadtxt(results_foldername+'/profile/t0/previous0.txt')
        core_zone=previous[-2]
        
        # re-ad in list of time at which results are saved
        save_t=np.loadtxt(results_foldername+'/profile/files_saved_at_these_time_list.txt')

        N_PLOTS = len(save_t)
        print('Plotting %d timesteps...' % N_PLOTS)
        
        evo=np.loadtxt(results_foldername+'/evolution.txt')
        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/86400.0/365.0,evo[:,7],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'Surface heat flux ($Wm^{-2}$)',fontsize=16.5)
        plt.xlabel('Time (years)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/SurfaceHeatFlux.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,4],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Surface temperature (K)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/SurfaceTemperature.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0][15:]/1e9/86400.0/365.0,evo[:,8][15:],color='black',linewidth=2.0,label='Heat flux at core mantle boundary')
        plt.plot(evo[:,0][15:]/1e9/86400.0/365.0,evo[:,9][15:],color='black',linestyle='dashed',linewidth=2.0, label='Conductive heat flux along core adiabat')
        plt.yscale('log')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel(r'Flux ($Wm^{-2}$)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.legend(frameon=True, fontsize=12)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/HeatFluxAtCoreMantleBoundary.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,10]/6371000.0,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel(r'Planet radius ($R_{\oplus}$)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/PlanetRadius.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,11]/6371000.0,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel(r'Core radius ($R_{\oplus}$)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/CoreRadius.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,12]/1e9,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Central pressure (GPa)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/CentralPressure.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,6],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Central temperature (K)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/CentralTemperature.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,13]/1e9,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Core mantle boundary pressure (GPa)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/PressureAtCoreMantleBoundary.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,5],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Core mantle boundary temperature (K)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.savefig(results_foldername+'/image/TemperatureAtCoreMantleBoundary.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,14]/1e3,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Inner core size (km)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/RadiusOfSolidInnerCore.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,15]/5.97e24,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel(r'Inner core mass ($M_{\oplus})$',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/MassOfSolidInnerCore.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,16]/1000.0,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.xscale('log')
        plt.ylabel('Thickness of dynamo source region in magma ocean (km)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/ThicknessOfDynamoSourceRegionInMagmaOcean.png',dpi=200)
        plt.close()

        #plt.figure(figsize=(9,6))
        #plt.yticks(fontsize=14)
        #plt.xticks(fontsize=14)
        #plt.ylabel('Radiogenic heating (TW)',fontsize=16.5)
        #plt.xlabel('Time (Gyr)',fontsize=16.5)
        #plt.legend(frameon=True,fontsize=12)
        #plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        #plt.savefig(results_foldername+'/image/RadiogenicHeatingInCoreAndMantle.png',dpi=200)
        #plt.close()
        
        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,2],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Mass-averaged mantle temperature (K)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/MassAveragedMantleTemperature.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,3],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel('Mass-averaged core temperature (K)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/MassAveragedCoreTemperature.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,22]/1e21,color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yscale('log')
        plt.ylabel(r'Core dipolar magnetic moment ($\times 10^{21} Am^2$)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/CoreDipolarMagneticMoment.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,23]/1e12,color='#a6cee3',linewidth=2.0,label='Surface heat flow')
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,24]/1e12,color='#1f78b4',linewidth=2.0,label='Core mantle boundary heat flow')
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,25]/1e12,color='#f08080',linewidth=2.0,label='Latent heat release due to inner core growth')
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,19]/1e12,color='#c71585',linewidth=2.0,label='Inner core boundary heat flow')
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,17]/1e12,color='#c49c94',linewidth=2.0,label='Radiogenic heating in the mantle')
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,18]/1e12,color='#8c564b',linewidth=2.0,label='Radiogenic heating in the core')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yscale('log')
        plt.ylabel('Heat flow (TW)',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.ylim(0.05, (evo[:,23][-100])/1e12*500.0)
        plt.legend(frameon=True)
        plt.savefig(results_foldername+'/image/HeatFlow.png',dpi=200)
        plt.close()

        plt.figure(figsize=(9,6))
        plt.plot(evo[:,0]/1e9/86400.0/365.0,evo[:,26],color='black',linewidth=2.0)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylim(1e-3, 1.5)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'Urey ratio',fontsize=16.5)
        plt.xlabel('Time (Gyr)',fontsize=16.5)
        plt.title(r'$M_{\mathrm{pl}}= $'+str(load_file[0])+r'$ M_{\oplus}$, Core mass fraction = '+str(load_file[1]) ,fontsize=16)
        plt.savefig(results_foldername+'/image/UreyRatio.png',dpi=200)
        plt.close()
        
        
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
                if save_t[i]==1.0:
                    t_title.append('t = '+str(int(save_t_title[i]))+' year')
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
        T_max=max(s[:,4])#+max(s[:,4])*0.01
        R_max=max(s[:,0])#+max(s[:,0])*0.01
        Fconv_max=max(s[:,7][5:-2])#+max(s[:,7][5:-2])*0.01
        eta_min=min(s[:,12][int(core_zone+5):])
        g_min=min(s[:,3])
        rho_min=min(s[:,2])

        s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[-1]))+'.txt')
        T_min=min(s[:,4])#-50.0
        P_max=max(s[:,1])/1e9#+max(s[:,1])/1e11
        P_min=1e5/1e9
        rho_max=max(s[:,2])#+max(s[:,2])*0.01
        g_max=max(s[:,3])#+max(s[:,3])*0.01
        vconv_min=min(s[:,10][int(core_zone+10):-10])
        eta_max=max(s[:,12])#+max(s[:,12])*0.01
        Rem_min=min(s[:,11]+1e-3)
        R_min=min(s[:,0])
        Fconv_min=min(s[:,7][5:-2]+1e-7)

        s=np.loadtxt(results_foldername+'/profile/StructureProfile_'+str(int(save_t[1]))+'.txt')
        Rem_max=max(s[:,11])#+max(s[:,11])*0.01
        vconv_max=max(s[:,10][5:-2])#+max(s[:,10][5:-2])*0.01

        axis_scale='linear'
        T_min,T_max=f_axis_max_min(T_min,T_max,axis_scale)
        R_min,R_max=f_axis_max_min(R_min,R_max,axis_scale)
        g_min,g_max=f_axis_max_min(g_min,g_max,axis_scale)
        rho_min,rho_max=f_axis_max_min(rho_min,rho_max,axis_scale)
        P_min,P_max=f_axis_max_min(P_min,P_max,axis_scale)
        axis_scale='log'
        eta_min,eta_max=f_axis_max_min(eta_min,eta_max,axis_scale)
        Fconv_min,Fconv_max=f_axis_max_min(Fconv_min,Fconv_max,axis_scale)
        vconv_min,vconv_max=f_axis_max_min(vconv_min,vconv_max,axis_scale)
        Rem_min,Rem_max=f_axis_max_min(Rem_min,Rem_max,axis_scale)

        ncpus = os.cpu_count()
        nthreads = int(max(os.getenv('NTHREADS', ncpus - 1), 1))
        print(f'Plotting with {nthreads} simultaneous threads (out of {ncpus} detected cpus)')
        args = [
            (
                i, save_t, t_title, mass, core_zone,
                P_min, P_max, R_min, R_max, rho_min, rho_max,
                g_min, g_max, T_min, T_max, P_min, P_max,
                Fconv_min, Fconv_max, vconv_min, vconv_max, Rem_min, Rem_max,
                eta_min, eta_max, results_foldername, load_file,
            )
            for i in range(1, N_PLOTS)
        ]
        #with Pool(nthreads) as p:
        #    for _ in tqdm(p.imap_unordered(plot_for_index, args), total=len(args)):
        #        pass  # We don't need the result, just waiting for tasks to finish.
        #    #process_map(plot_for_index, args, max_workers=nthreads)
        process_map(plot_for_index, args, max_workers=nthreads)
        
        print('Making movies')
        os.makedirs(results_foldername+'/movie', exist_ok=True)
        out_names=['TemperatureVsPressure','TemperatureVsMass','HeatFluxVsMass','DensityVsMass','MagneticReynoldsNumberVsMass','MantleViscosityVsMass', 'ConvectiveVelocityVsMass','PressureVsRadius','GravityVsMass','MantleMeltFractionVsMass']
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
            # remove individual frames of movies
            shutil.rmtree(filefolders[i][:-5])
            # move movies into the movie folder
            source_path = os.path.join(filefolders[0][:len(results_foldername+'/image')],out_names[i]+'.mp4')
            destination_path = os.path.join(filefolders[0][:len(results_foldername)],'movie/'+out_names[i]+'.mp4')
            shutil.move(source_path, destination_path)
        
if __name__ == '__main__':
    from multiprocessing import Pool
    import subprocess
    from tqdm.contrib.concurrent import process_map
    from tqdm import tqdm
    import os
    os.environ['EOS_TABLES_VERBOSE'] = '1'
    import cv2
    import argparse
    import shutil
    import eos_tables 
    run()
