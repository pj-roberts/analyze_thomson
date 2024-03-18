#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script for prototyping better photon counting algorithm


#%% Load Packages

# import csv
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import functions as fn

#%% 
# Initialize matplotlib

fn.initialize_plotting()

#%% Settings/Initialize

# data/saving directories
datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\Raw Data\\Processed Data\\'

filename = '043_its.csv'
filepath = datafolder + filename

savefolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\' +\
            '2024 Python Analysis\\Spectra\\'

plotfolder = savefolder + 'Plots\\'

#%% Load Data & count photons

# Import Frames using parsing function
wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version

#%% 
# Count photons from 2D frames

# Photon count settings 
pct = 180
crt = 0
rad = 5
crrad = 10

spectrum_pc = fn.count_photons_mask(frames,pct,crt,rad,crrad)

# frames_pc = fn.count_photons_rad(frames,pct,crt)

#%% Bin over pixel columns to compute mean spectrum      

# Bin over rows/shots
spectrum = np.sum(np.mean(frames,0),0) # Mean shot, sum each column

# Save wavelengths
savename_wl = savefolder + filename[0:-4] + '_wl.npy'
np.save(savename_wl,wls)

# Save mean spectrum
savename_spectrum = savefolder + filename[0:-4] + '_spectrum.npy'
np.save(savename_spectrum,spectrum)

# Save photon count spectrum
savename_spectrum_pc = savefolder + filename[0:-4] + '_spectrum_pc.npy'
np.save(savename_spectrum_pc,spectrum_pc)


#%% 
# Plot Results

# Generate Figure/Axis for line plot
fig = plt.figure(figsize=(3.37,2))
ax = fig.add_axes([0.3,0.3,0.6,0.6])
# ax.set_box_aspect(1.0)

# Line plot of spectrum
plt.plot(wls,spectrum_pc,linewidth=1,color='k')
plt.ylim((-0.001,0.01))

plt.xlabel('Wavelength (nm)')
plt.ylabel('Counts')

# plt.tight_layout()
plt.savefig(plotfolder +  filename[0:-4] + '_spectrum_pc.pdf', format='pdf')
plt.show()
# plt.close()

#%%


# plt.imshow(frames[ii,:,:])
# ii += 1
# plt.colorbar()

plt.imshow(np.mean(frames,1))
plt.colorbar()