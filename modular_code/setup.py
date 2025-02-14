# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        '*.pyx',
        compiler_directives={'language_level' : "3"},
    ),
    include_dirs=[numpy.get_include()],
)

"""
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("main", ["main.pyx"], include_dirs=[np.get_include()]),
    Extension("structure", ["structure.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="thermal_evolution",
    ext_modules=cythonize(extensions),
)
"""