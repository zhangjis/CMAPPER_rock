#
!!!! The individual files for EoS tables in EoS and mixture_table are compressed into gz files, so the files are small enough to be uploaded onto github. Please unzip the files in these two folders before compiling and running the code. (this issue will be fixed shortly by the end of this week). ----- Note by Jisheng, this may not be needed. np.loadtxt decompresses gz files automatically. 

# CMAPPER

This is a repository for an 1D model of thermal evolution of rocky exoplanets
with masses between 0.5 and 6 Earth-mass and core mass fractions between 0.1 and
0.8. Additioanlly, the range for planet equilibrium temperature is between 255 and 1800K/ We are currently extending the equations of state tables so the code can be applied to planets with a wider range of planet mass with higher equilbrium temperature. The model outputs detailed planet thermal history and tracks possible phase transitions including magma ocean and inner core solidification (see results).

## Quickstart

Note that this is just an abbreviated version of the below instructions. Please
read on if you have any additional difficulties.

1) Make sure you have `python3` installed (>3.10 has been tested)
    - If you don't, installing via
      [Anaconda](https://www.anaconda.com/download/) is easiest, by running
      `conda install python`
2) Download the code (`git clone https://github.com/zhangjis/CMAPPER_rock.git`
    is recommended
    - Or you just click `Download ZIP` under the green "Code" button on the
      Github website)
3) Change into the code directory and type `make`
    - This will install dependencies, build the code, and run a test script
4) View evolutionary plots and movies in the `results_case/image` and `results_case/movie` folders. 
    - `case` indicates the specific planet the results saved in the folder are for.  For example, the directory in which evolution results of a planet with a mass of 3 Earth mass, core mass fraction of 0.326, evolving time of 12 Gyr, same level of radiogenic elements concentration to that of Earth, and equilibrium temperature of 255 K, the directory will be named `/results_Mpl3.0_CMF0.326_time12.0_Qrad1.0_1.0_1.0_1.0_Teq255.0`.

5) (Optional) Use routines provided in the jupyter notebook, `plot.ipynb`, to make some plots yourself. The notebook includes all the routines used in the file `test.py` where we generated plots and movies. Feel free to use your own graphing tools or routines in the jupyter notebook to make plots and/or movies for future simulations. 

## Full Installation Instructions
### 1) Python
We recommend installing Python using Anaconda. This distribution of Python comes
with the Cython bundle and other popular libraries required by CMAPper, such as
NumPy and Scipy.

To download Anaconda installer, visit https://www.anaconda.com/download/, and
download the version appropriate for your operating system.

To verify installation, type the following command in the terminal,
   ```sh
   python3 --version
   conda --version
   ```
You should see the installed Python and conda version numbers if the
installation was successful.

### 2) Clone the repository
Go to a directory to which you want to download the repository by typing `cd`
followed by the path to the directory in the terminal. Copy and paste `git clone
https://github.com/zhangjis/CMAPPER_rock.git` in the terminal. You can then enter the
directory by typing `cd CMAPPER_rock` in the terminal.

Alternatively, you can click `Download ZIP` on the Github UI, find and extract
the ZIP file, then `cd` into the correct folder.

### 3a) Python Packages

To set up all dependencies automatically, type:
```sh
make dependencies
```

Alternatively, to set up by hand:
- *Recommended:* set up a [Python virtual
  environment](https://docs.python.org/3/library/venv.html): `python3 -m venv
  venv && source venv/bin/activate`
- To install all required packages, type in the terminal:
```sh
pip install -r requirements.pip
```

### 3b) Compiling and Running
To compile the code, run the evolution, and generate plots, run the command in
the terminal
```sh
make run
```
Alternatively, to compile the code by hand, run
```sh
python3 setup_pre_adiabat.py build_ext --inplace
```
and to run the code, run
```sh
python3 test.py
```

### 4) Results
The output of CMAPper includes planet thermal and structural profiles of the
planet, as well as thermophysical quantities at pre-selected timesteps. The
thermal and structural profiles are saved in the sub-directory
`/results_case/profile/` with file names `structure_timestep.txt`, timestep being
years into the simulation. Each column in the txt files represent 

column 1-5: radius, pressure, density, gravitational acceleration,temperature, 

column 6-10: thermal expansion coefficient, specific heat, convective heat flux, conductive heat flux, total heat flux  

column 11-13: mantle convective velocity, mantle magnetic Reynolds number,mantle viscosity, and mass. 


In addition, planet thermal history is saved in `evolution.txt` in the
sub-directory `/results_case`. Each column from left to right represents 


column 1-5: time, time stepsize, mass averaged mantle temperature, mass averaged core temperature, surface temperature, 

column 6-10: core mantle boundary temperature, central temperature,surface heat flux, core mantle boundary heat flux, conductive heat flux along core adiabat, 

column 11-15: planet radius, core radius, central pressure, core mantle boundary pressure, inner core radius, 

column 16-20: inner core mass, thickness of dynamo source region in magma ocean, mantle radiogenic heating, core radiogenic heating, inner core conductive heat flow, 

column 21-23: core thermal buoyancy flux, core compositional buouyancy flux, core magnetic dipole moment

All saved quantities are in SI units.

The code saves mp4 movies and png figures showing the thermal history of the
planets in the sub-directories `/results_case/movie/` and `/results_case/image/` respectively. Additionally, we provide a
tutorial in Jupyter notebook for visualization of saved text files.

A Jupyter notebook "plot.ipynb" contains code to visualize saved txt files in
the sub-directory `/results_case/profile/` and `/results_case/evolution.txt`

## Configuring CMAPper
### Input parameters
Main input parameters for the simulations include planet mass, initial central
temperature, radiogenic heating relative to that of Earth's mantle and the
lifetime of the planet to be simulated. Additionally there is also the option to change the value of planet equilibrium temperature, which is the average temperature a planet would have if it absorbed energy from the star and emitted it back out into space evenly. The default value is 255K. These values can be set in `input.txt`.

Mantle viscosity has a strong influence over the dynamics and the cooling
history of the mantle. The exact viscosity of mantle silicate, especially for
high pressure phases such as post-perovskite (ppv), remains the largest
uncertainty in CMAPper. Currently, we include two models for the viscosity of
ppv in CMAPper (see [Tackley et al.
2013](https://www.sciencedirect.com/science/article/abs/pii/S0019103513001231)).
The two options are for two deep mantle rheologies with diffusion creep and
dislocation creep as the dominant deformation mechanisms. Users can choose the
viscosity model by setting `nu_ppv_model` to 1 for diffusion creep and 2 for
dislocation creep in `input.txt`.

We provide a default choice of equations of state (EoS) for individual phases of
mantle silicate, including the [liquid
phase](https://www.sciencedirect.com/science/article/abs/pii/S0031920117301449),
[enstatite](https://www.sciencedirect.com/science/article/pii/S0019103507001601?via%3Dihub),
[olivine](https://www.sciencedirect.com/science/article/abs/pii/S0031920108002227?via%3Dihub),
[perovskite](https://www.nature.com/articles/35082048) and
[post-perovskite](https://www.nature.com/articles/nature02701), as well as
[liquid Fe](https://www.nature.com/articles/srep41863),
[ε-Fe](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.215504),
[FeS
alloy](https://www.sciencedirect.com/science/article/pii/S0019103507001601?via%3Dihub)
and [FeSi
alloy](https://www.sciencedirect.com/science/article/pii/S0012821X12005183?via%3Dihub).
To choose specific choices of EoS in the mantle and the core, modify the
varibles `eos_ma`, `eos_um`, `eos_lm`, `eos_dm`, `eos_lc` and `eos_sc` in
`input.txt`. The following table summarizes the code for default choices of
EoS.

| Layers         | Varible in input file | Phases                 | Code      |
| -------------- | --------------------- | ---------------------- | --------- |
| Magma ocean    | eos_ma                | liquid MgSiO3          | l_mgsio3  |
| Upper mantle   | eos_um                | enstatite              | en        |
|                |                       | olivine                | ol        |
| Lower mantle   | eos_lm                | perovskite             | pv        |
| Deep mantle    | eos_dm                | post-perovskite        | ppv       |
| Outer core     | eos_lc                | Liquid Fe + FeS alloy  | l_S_Fe    |
|                |                       | Liquid Fe + FeSi alloy | l_Si_Fe   |
| Inner core     | eos_sc                | ε-Fe + FeS alloy       | epi_S_Fe  |
|                |                       | ε-Fe + FeSi alloy      | epi_Si_Fe |


See examples and comments in `input.txt` for additional information.

In future versions, users will be able to provide additional models for mantle viscosity and choices of EoS in addition to the default options provided.