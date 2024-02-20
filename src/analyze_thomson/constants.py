#
# Created on Mon Feb 19 2024
#
# Copyright (c) 2024 Parker Roberts
#
# Constants module for analyze_thomson

import numpy as np

# Fundamental constants
e = 1.602e-19  # electron charge, C
c = 299792458  # speed of light, m/s
kB = 1.381e-23  # Boltzmann constant, J/s
me = 9.109e-31  # electron mass, kg

# Experiment parameters
mi = 1.66e-27*131.293 # ion mass (xenon), kg

# Laser parameters
lambda_0 = 532e-9  # laser wavelength, nm
P_i = 0.7  # Shot energy, J

# Detector settings
theta = 30*np.pi/180  # scattering angle, radians
