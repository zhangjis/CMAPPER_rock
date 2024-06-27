from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('*.pyx'),include_dirs=[numpy.get_include()])
#setup(ext_modules = cythonize('speed_5ME_Earth_001HHe.pyx'),include_dirs=[numpy.get_include()])