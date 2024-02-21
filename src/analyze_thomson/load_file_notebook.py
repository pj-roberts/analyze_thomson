#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script for prototyping parsing of raw lightfield data

#%% Load Packages

import csv
import numpy as np
# import pandas as pd
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
data_folder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\Raw Data\\Processed Data\\'

filepath = data_folder + '078_raman.csv'

#%% Load Data

# Import Frames using parsing function
wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version

# Should save these as intermediate data types

#%% Visualize raw frames

plt.plot(wls,np.transpose(frames[1,:,:]))
plt.show()

plt.contourf(frames[-2,:,:])
plt.show()

#%% Compute mean frame and plot resulting spectrum

mean_frame = np.mean(frames,0)

plt.contourf(mean_frame,levels=np.linspace(0,5,20))
plt.colorbar()
plt.show()

#%%
spectrum = np.mean(mean_frame,0)


# Generate Figure/Axis
fig = plt.figure(figsize=(3.37,1.69))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
# ax.set_box_aspect(1.0)

plt.plot(wls,spectrum,linewidth=1)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Counts')
plt.ylim((-0.1,4))
plt.xlim((532,537))
plt.show()


