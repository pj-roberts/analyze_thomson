### Parker J. Roberts - 2024-02-19
### Functions for analyze_thomson package

import csv
import numpy as np

import constants as c

def add_one(number):
    return number + 1


def parse_lightfield_raw(filepath):
    """Load data from lightfield raw images stored as .csv.
        These are the detector/camera image files.
    
    Args:
        filepath (str): location of data file

    Returns:
        wl (arr): (c,) array of wavelengths
        frame (arr): (s,r,c) array of intensity values (shot, row, column)
    """
    
    # Initialize arrays
    startrows = []
    endrows = []
    wls = 0

    # Loop once: detect array dimensions & preallocate
    with open(filepath) as csvfile:
        filereader = csv.reader(csvfile)
        for ii,row in enumerate(filereader):
            if row[0] == '0': # detect start of ccd image
                startrows.append(ii)
            if row[0] == 'End Region 1': # detect end of image
                # !!! If csvread doesn't see the strings, this is risky!
                endrows.append(ii-1)
            if (row[0] == 'Wavelength') and (wls == 0):
                wls = row[1:-1]

    # Get array sizes
    nframes = len(startrows)
    nwl = len(wls)
    nrows = endrows[0] - startrows[0] + 1

    # Convert wavelength strings to floats
    wls = [float(ii) for ii in wls]

    # preallocate frames array
    frames = np.zeros((nframes,nrows,nwl))
    framecount = 0
    rowcount = 0

    # Loop again: detect array dimensions & preallocate
    with open(filepath) as csvfile:
        filereader = csv.reader(csvfile)
        for ii,row in enumerate(filereader):
            if framecount < nframes: # End loop when loaded all frames
                if ii == startrows[framecount]:
                    # Start counting distance from startrow == matrix row
                    frames[framecount,rowcount,:] = row[1:-1]
                    rowcount += 1

                elif startrows[framecount] < ii < endrows[framecount]:
                    frames[framecount,rowcount,:] = row[1:-1]
                    rowcount += 1
                    
                elif ii == endrows[framecount]:
                    frames[framecount,rowcount,:] = row[1:-1]
                    framecount += 1
                    rowcount = 0

    return wls, frames


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