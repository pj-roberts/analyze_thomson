#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script for prototyping photon counting and saving spectra

#%% Load Packages

# import csv
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import functions as fn

#%% Initialize matplotlib

# Font Settings
mpl.rcParams.update({'font.size': 12})  # Adjust font size as needed
mpl.rcParams.update({'font.family': 'Arial'})  # Choose appropriate font

# Axis settings
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['xtick.major.width'] = 2.0
mpl.rcParams['ytick.major.width'] = 2.0

# Line/Color Settings
mpl.rcParams['lines.linewidth'] = 2.0


#%% Settings/Initialize

# Where to look for data
datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\Raw Data\\Processed Data\\'

filepath = datafolder + '043_its.csv'

#%% Load Pre-Parsed Spectra

# Import Frames using parsing function
wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version


#%% Visualize raw frames

# plt.plot(wls,np.transpose(frames[1,:,:]))
# plt.show()

# plt.contourf(frames[-2,:,:])
# plt.show()

#%% Count Photons

pct = 150
crt = 1000
rad = 1
frames_pc = fn.count_photons_simple(frames,pct,crt)

#%% Plot contour of mean frame

# Generate Figure/Axis for raw contour
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)

# Contour plot of raw mean frame
plt.contourf(np.mean(frames,0),levels=np.linspace(-5,5,20))
plt.colorbar(fraction=0.045)
plt.show()

# Generate Figure/Axis for counted frame
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)

# Contour plot of counted mean frame
plt.contourf(np.mean(frames_pc*1000,0),levels=np.linspace(0,1,20))
plt.colorbar(fraction=0.045,ticks=np.linspace(0,1,5))
plt.show()


#%% Plot mean spectrum

# Generate Figure/Axis
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)

# Line plot of spectrum
# plt.plot(wls,np.mean(np.mean(frames,0),0),linewidth=1)
plt.plot(wls,np.mean(np.mean(frames_pc,0),0),linewidth=1)
plt.ylim((0,0.001))
plt.xlim((530,535))
plt.show()

# spectrum = fn.compute_mean_spectrum(frames)

# fig, ax = fn.plot_spectrum(wls,spectrum)
