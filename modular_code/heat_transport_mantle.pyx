# integration.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc cimport math
cimport cython

from planet cimport Planet

cpdef void update_mantle_entropy(Planet planet, ):
    planet.dsdr
