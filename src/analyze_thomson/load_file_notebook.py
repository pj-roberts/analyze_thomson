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

plt.contourf(mean_frame,levels=np.linspace(-10,20,20))
plt.colorbar()
plt.show()

#%%
spectrum = np.mean(mean_frame,0)

plt.plot(wls,spectrum)
plt.show()


