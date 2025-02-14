
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
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        '*.pyx',
        compiler_directives={
            'language_level': "3",
            'warn.undeclared': True,  # Warn about undeclared variables
            'warn.unused': True,  # Warn about unused variables
            'warn.unused_arg': True,  # Warn about unused function arguments
            'warn.maybe_uninitialized': True  # Warn about possibly uninitialized variables
        },
    ),
    include_dirs=[numpy.get_include()],
)
"""