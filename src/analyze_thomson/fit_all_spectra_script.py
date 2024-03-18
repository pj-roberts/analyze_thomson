#%%
# Parker J. Roberts - U of M - 2024-02-20
# Python script to prototype loading ITS spectra and test model fitting 

#%% 
# Load Packages

# import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import scipy.optimize as op
# import matplotlib as mpl

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import functions as fn
import constants as cn

# Initialize matplotlib settings
fn.initialize_plotting()

#%% 
# Settings/Initialize

# data/saving directories
spectrafolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
    'University of Michigan\\2023\\ITS CTF\\' +\
        '2024 Python Analysis\\Sorted Data\\20 A 15 sccm\\Spectra\\'


savefolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
    'University of Michigan\\2023\\ITS CTF\\' +\
        '2024 Python Analysis\\Sorted Data\\20 A 20 sccm\\'

plotfolder = savefolder + 'Plots\\'


#%% Read Data Key

# Read data key file to map shot numbers to positions
keypath = spectrafolder + 'posn_key.csv'
keyframe = pd.read_csv(keypath)
posns = keyframe.loc[:,'Axial Position (mm)'].values
shots = keyframe.loc[:,'Shot Number'].values

#%% 
# Load Spectrum Data

# Preallocate save arrays
nwl = 760
n_shots = len(shots)
ne_arr = np.zeros((n_shots,1))
Te_arr = np.zeros((n_shots,1))
ue_arr = np.zeros((n_shots,1))
Me_arr = np.zeros((n_shots,1))
spectra_arr = np.zeros((n_shots,nwl))
spectra_fit_arr = np.zeros((n_shots,nwl))

# Loop through all shots in folder and generate plots/save reduced data
for ii,shot_num in enumerate(shots):
    
    # Load Spectrum
    wl,its_pc,pbg_pc = fn.load_spectrum(shot_num,spectrafolder)

    # Fit with reduced data

    # Generate velocities corresponding to wavelength bins
    v = fn.wl2velocity(wl*1e-9)
    # Median filter to remove spikes - maybe need more smoothing
    its_smth = median_filter(its_pc,(3,))
    # Mask filter bandwidth 
    filt_rad = 0.25  # filter "radius" (half-width)
    wl_mask, v_mask, its_mask = fn.mask_filter(wl,v,its_smth,filt_rad)
    # Fit Gaussian to masked spectrum
    model = lambda v, a, b, c: a*np.exp(-((v - b)/c)**2)
    popt,pcov = op.curve_fit(model, v_mask, its_mask/np.max(its_mask),\
                             p0=[1,0,5e5])

    # Generate Figure/Axis
    fig = plt.figure(figsize=(3.37,3.37))
    ax = fig.add_axes([0.22,0.22,0.7,0.7])
    # ax.set_box_aspect(1.0)
    # Plot Data
    ax.plot(v,its_smth)
    ax.plot(v,np.max(its_smth)*model(v,popt[0],popt[1],popt[2]),'k')
    # Set Axes
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Photon Counts')
    # ax.set_xlim((528,536))
    # ax.set_ylim((-0.005,0.02))
    # ax.set_xticks(np.arange(520,544+4,4))
    ax = plt.gca()
    plt.legend()
    plt.show()

    # Output physical parameters from fit parameters
    v_mean = popt[1]
    v_therm = popt[2]
    Te = 0.5*cn.me*v_therm**2/cn.e
    Me = np.abs(v_mean/v_therm)
    print(f'Electron Temperature: {Te:2.2f} eV')
    print(f'Mach Number: {Me:2.2f}')

    # Save data as arrays
    Te_arr[ii] = Te
    ne_arr[ii] = popt[0]
    ue_arr[ii] = v_mean
    Me_arr[ii] = Me

    spectra_arr[ii,:] = its_smth
    spectra_fit_arr[ii,:] = np.max(its_smth)*model(v,popt[0],popt[1],popt[2])

    # Save to spectrum array
# %%
