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
            '2024 Python Analysis\\Sorted Data\\20 A 15 sccm\\'

plotfolder = savefolder + 'Plots\\'



#%% 
# Load Spectrum Data

# For now, test by processing only a single file.
filenum = 45

# Generate Filename from number
filename_its = f'{filenum:03.0f}_its_spectrum.csv'
filename_pbg = f'{filenum:03.0f}_pbg_spectrum.csv'

# Load data into dataframe 
df_its = pd.read_csv(spectrafolder + filename_its)
df_pbg = pd.read_csv(spectrafolder + filename_pbg)

# Extract columns as np arrays
wl = df_its.loc[:,'Wavelength (nm)'].values
its_pc = df_its.loc[:,'Photon Counts'].values
pbg_pc = df_pbg.loc[:,'Photon Counts'].values

v = fn.wl2velocity(wl*1e-9)

#%% 
# Process + Fit Spectrum Data

# Median filter to remove spikes
its_smth = median_filter(its_pc,(3,))

# Mask filter bandwidth 
filt_rad = 0.25  # filter "radius" (half-width)
wl_mask, v_mask, its_mask = fn.mask_filter(wl,v,its_smth,filt_rad)

# Fit Gaussian to masked spectrum
model = lambda v, a, b, c: a*np.exp(-((v - b)/c)**2)
popt,pcov = op.curve_fit(model, v_mask, its_mask/np.max(its_mask),p0=[1,0,1e6])

#%% 
# Generate Plots of Spectrum Data & Save

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

#%% 
# Output physical parameters from fit parameters

v_mean = popt[1]
v_therm = popt[2]

Te = 0.5*cn.me*v_therm**2/cn.e
Me = np.abs(v_mean/v_therm)

print(f'Electron Temperature: {Te:2.2f} eV')
print(f'Mach Number: {Me:2.2f}')