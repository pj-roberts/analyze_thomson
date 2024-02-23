#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script for prototyping importing and resaving a single CCD file

#%% Load Packages

# import csv
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

filename = '010_its.csv'
filepath = datafolder + filename

savefolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\' +\
            '2024 Python Analysis\\Spectra\\'

plotfolder = savefolder + 'Plots\\'

# Photon count settings 
pct = 150
crt = 0
rad = 1


#%% Load Data & count photons

# Import Frames using parsing function
wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version

# Count photons from 2D frames
frames_pc = fn.count_photons_simple(frames,pct,crt)

#%% Bin over pixel columns to compute mean spectrum

# Bin over rows/shots
spectrum = np.sum(np.mean(frames,0),0) # Mean shot, sum each column
spectrum_pc = np.sum(np.mean(frames_pc,0),0) # Mean shot, sum each column
# this gives units of photons/shot for the pc spectrum

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
# plt.plot(wls,np.mean(np.mean(frames,0),0),linewidth=1)
plt.plot(wls,np.sum(np.mean(frames_pc,0),0),linewidth=1,color='k')
# plt.plot(wls,np.mean(spectra_pc,0),linewidth=1,color='b')
# plt.xlim((530,535))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Counts')
# plt.xticks(np.arange(526,542,4))
# plt.ylim((0,0.06))
# plt.yticks(np.arange(0,0.08,0.02))

# plt.tight_layout()
plt.savefig(plotfolder +  filename[0:-4] + '_spectrum_pc.pdf', format='pdf')
# plt.show()
plt.close()

# Generate Figure/Axis for raw contour
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.1,0.1,0.6,0.6])
ax.set_box_aspect(1.0)

# Contour plot of raw mean frame
plt.contourf(np.mean(frames,0),levels=np.linspace(-3,3,20))
plt.colorbar(fraction=0.045)
plt.savefig(plotfolder +  filename[0:-4] + '_mean_frame.png', format='png',dpi=600)
# plt.show()
plt.close()

# Generate Figure/Axis for counted frame
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.1,0.1,0.6,0.6])
ax.set_box_aspect(1.0)

# Contour plot of counted mean frame
plt.contourf(np.mean(frames_pc*1000,0),levels=np.linspace(0,1,20))
plt.colorbar(fraction=0.045,ticks=np.linspace(0,1,5))
plt.savefig(plotfolder +  filename[0:-4] + '_mean_frame_pc.png', format='png', dpi=600)
# plt.show()
plt.close()