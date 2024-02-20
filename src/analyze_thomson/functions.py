### Parker J. Roberts - 2024-02-19
### Functions for analyze_thomson package

import numpy as np
import constants as c

def add_one(number):
    return number + 1


def parse_lightfield_raw(filepath):
    """Load data from lightfield raw images stored as .csv.
        !!! write actual code for this
    
    Args:
        filepath (str): location of data file

    Returns:
        wl (arr): (c,) array of wavelengths
        frame (arr): (s,r,c) array of intensity values (shot, row, column)
    """
    print(filepath)
    wl = np.zeros((50,1))
    frame = np.zeros((50,50))
    return wl, frame


def generate_dataset(ne,Te,ue,noise):
    """Simulate Thomson spectrum based on 1D Maxwellian distribution.
        For testing analysis scripts and robustness to noise. For
        nonzero noise (> 0), corrupts data with pseudorandom gaussian 
        noise of standard deviation (noise).

    Args:
        ne (float): electron density (m^-3)
        Te (float): electron temperature (eV)
        ue (float): electron velocity (m/s)
        noise (float): noise magnitude (as fraction of ne)
        
    Returns:
        wl (float arr): wavelengths (!!! hard-coded in for now)
        spectrum (float arr): 1D spectrum
    """
    # Compute wavelength/velocity axis 
    wl = np.linspace(522e-9,542e-9,102)  # !!! hard-coded for now
    u = wl2velocity(wl)

    # Compute Maxwellian distribution
    vT = np.sqrt(2*c.e*Te/c.me) # Thermal electron velocity
    spectrum = ne*np.exp(-((u - ue)/vT)**2)

    # Corrupt data with noise
    spectrum = spectrum + noise*ne*0  # !!! add pseudorandom sampling

    return wl,spectrum


def velocity2wl(velocities):
    """convert velocity to wavelength (vectorized)

    Args:
        velocities (n,): array of velocities (m/s)

    Returns:
        wls (n,): array of wavelengths (m)
    """
    wls = velocities
    print('Error: velocity2wl not complete')
    return wls


def wl2velocity(wls):
    """convert wavelength to velocity (vectorized)

    Args:
         wls (n,): array of wavelengths (m)

    Returns:
        velocities(n,): array of velocities (m/s)
    """
    velocities = c.c*(c.lambda_0/wls - 1)/2/np.sin(c.theta/2)
    return velocities