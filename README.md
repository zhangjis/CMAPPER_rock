# CMAPPER

## Installation
### Python
We recommend installing Python using Anaconda. This distribution of Python comes with the Cython bundle and other popular libraries required by CMAPper, such as NumPy and Scipy.

To download Anaconda installer, visit https://www.anaconda.com/download/, and download the version appropriate for your operating system.

To verify installation, type the following command in the terminal,
   ```sh
   python3 --version
   conda --version
   ```
You should see the installed Python and conda version numbers if the installation was successful.

### Python Packages

- Optionally, first set up a [Python virtual environment](https://docs.python.org/3/library/venv.html): `virtualenv venv && source venv/bin/activate`

To install all required packages, type in the terminal:
    ```sh
    pip install -r requirements.pip
    ```

## Compiling and running
### Input parameters
Main input parameters for the simulations include planet mass, initial central temperature, radiogenic heating relative to that of Earth's mantle and the lifetime of the planet to be simulated. These values can be set in `input_file.pyx`.
### Compiling
To compile the code, run the command in the terminal
   ```sh
   python3 setup_pre_adiabat.py build_ext --inplace
   ```
Note that the code must be recompiled after changing `input_file.pyx`.
### Running
To run the evolution and generate plots, run the command in the terminal
   ```sh
   python3 test.py
   ```
### Results
The output of CMAPper includes planet thermal profiles and profiles of planet structure (radius, pressure, density and gravitational acceleration), thermophysical quantities (thermal expansion coefficient, specific heat at constant pressure, adiabatic temperature gradient, viscosity, eddy diffusivity, convective heat flux and magnetic Reynolds number) at pre-selected timesteps.

In addition, the following quantities are saved throughout the simulation, temperatures at the planet center, core mantle boundary and planet surface, heat fluxes at the core mantle boundary and planet surface, pressure levels at the planet center and the core mantle boundary, radii of the planet, the core and the solid inner core, the depth and the thickness of the dynamo source region, as well as the strength of B-field at the planet surface.

The output and figures/movies for visualization are saved in the `results` folder.

TBD
(prepare a jupyter notebook to visualize saved text files).
