#%% Parker J. Roberts - U of M 2024-03

# Script for function development: import/process raw LVTF Thomson files

# We have just switched to acquiring at 20 Hz, with background shots
# in-between laser shots. This script parses these files.

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.ndimage import median_filter
import scipy.optimize as op
from scipy.signal import savgol_filter

import functions as fn
import constants as cn

#%%

#%% 
# Constants/Initialize

datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\University of'\
    ' Michigan\\2024\\Experiments\\Spring 2024 Laser'\
        ' Testing\\Sorted Data\\Xenon\\600 V 15 A B 1\\'

# datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\University of Michigan\\2024\\Experiments\\Spring 2024 Laser Testing\\Sorted Data\\Xenon\\300 V 15 A Radial Sweeps\\3 mm\\'

# get list of files in data folder
# files = [f for f in listdir(datafolder) if isfile(join(datafolder, f))]

# num_files = len(files)

#% Loop through files

# !!! Need to get correct binned wavelengths

# Pull positions from filenames or use a data key
# Read data key file to map shot numbers to positions
keypath = datafolder + 'posn_key.csv'
keyframe = pd.read_csv(keypath)
posns = keyframe.loc[:,'Axial Position (mm)'].values
filenames = keyframe.loc[:,'File Name'].values
num_files = len(posns)

# Preallocate spectrum array
nwl = 256
ne_arr = np.zeros((num_files,))
Te_arr = np.zeros((num_files,))
Te_err_arr = np.zeros((num_files,))
ue_arr = np.zeros((num_files,))
ue_err_arr = np.zeros((num_files,))
Me_arr = np.zeros((num_files,))
spectra_arr = np.zeros((num_files,nwl))
# spectra_fit_arr = np.zeros((n_shots,nwl))

# Loop through files and load/process spectra
for ii, filename in enumerate(filenames):
    filepath = datafolder + filename

    wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version

    # Separate thomson from background (20Hz concurrent background data)
    net_spec = fn.subtract_bg_concurrent(frames)
    spec_smth = savgol_filter(net_spec,51,3)

    spectra_arr[ii,:] = net_spec

    # Plot net spectrum
    wls = np.linspace(517.88,546.1,256)
    plt.plot(wls,net_spec)
    plt.plot(wls,spec_smth)

    # Fit spectrum:

    # Generate velocities corresponding to wavelength bins
    v = fn.wl2velocity(wls*1.0e-9)

    filt_rad = 1  # filter "radius" (half-width)
    wl_mask, v_mask, its_mask = fn.mask_filter(wls,v,net_spec,filt_rad)
    # Fit Gaussian to masked spectrum
    model = lambda v, a, b, c: a*np.exp(-((v - b)/c)**2)
    Bounds = ([0.2, -1e6, 0], [2, 1e6, 1e7])
    popt,pcov = op.curve_fit(model, v_mask[1:175], its_mask[1:175]/np.max(its_mask[1:175]),\
                             p0=[1,0,1e6],bounds=Bounds)
    
    # Generate Figure/Axis
    fig = plt.figure(figsize=(3.37,3.37))
    ax = fig.add_axes([0.22,0.22,0.7,0.7])
    # ax.set_box_aspect(1.0)
    # Plot Data
    ax.plot(v,net_spec)
    ax.plot(v,spec_smth)
    ax.plot(v,np.max(its_mask)*model(v,popt[0],popt[1],popt[2]),'k')
    ax.vlines(fn.wl2velocity(cn.lambda_0+filt_rad*1.0e-9),np.min(its_mask),np.max(its_mask),'k')
    ax.vlines(fn.wl2velocity(cn.lambda_0-filt_rad*1.0e-9),np.min(its_mask),np.max(its_mask),'k')
    # Set Axes
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Photon Counts')
    # ax.set_xlim((528,536))
    # ax.set_ylim((-0.005,0.02))
    # ax.set_xticks(np.arange(520,544+4,4))
    ax = plt.gca()
    plt.show()


    # Output physical parameters from fit parameters
    v_mean = popt[1]
    v_therm = popt[2]
    Te = 0.5*cn.me*v_therm**2/cn.e
    Te_err = cn.me*v_therm*np.sqrt(pcov[2,2])/cn.e
    ue_err = np.sqrt(pcov[1,1])
    Me = np.abs(v_mean/v_therm)
    print(filename)
    print(f'Electron Temperature: {Te:2.2f} +/- {Te_err:2.2f} eV')
    print(f'Mach Number: {Me:2.2f}')

    # Save data as arrays
    Te_arr[ii] = Te
    Te_err_arr[ii] = Te_err
    ne_arr[ii] = popt[0]*np.sqrt(Te)
    ue_arr[ii] = v_mean
    ue_err_arr[ii] = ue_err
    Me_arr[ii] = Me


#%%
# Generate Reduced Data Plots

fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)
ax.contourf(v,posns,spectra_arr,5)
plt.show()

fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(wls, posns)
ax.plot_surface(X,Y,spectra_arr)
plt.show()

fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)
ax.errorbar(posns,Te_arr,Te_err_arr,0.5,capsize=3,marker='o',linestyle='')
plt.show()


fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)
ax.errorbar(posns,ue_arr,ue_err_arr,0.5,capsize=3,marker='o',linestyle='')
plt.show()

fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)
ax.plot(posns,ne_arr,marker='o')
plt.show()
