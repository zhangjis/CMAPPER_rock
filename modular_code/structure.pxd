# structure.pxd

from planet cimport Planet

# Expose only the public function "shooting" from structure.pyx.
cpdef void RK4(Planet planet,
                double m_array_start,
                double c_array_start,
                double d_Pc,
                double rtol)

cpdef void henyey_solver(Planet planet, double dsdr_c, double initial, double rtol)