#%%
## Main entry point for analyze_thomson scripts

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

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

#%% Set Filepath

filepath = "C:\\Users\\pjrob\\Desktop\\"

#%% Generate datset

ne = 1e16
Te = 80
u = 0
noise = 0
wl,spectrum = fn.generate_dataset(ne,Te,u,noise)

#%%  Plot resulting spectrum

# Generate Figure/Axis
fig = plt.figure(figsize=(3.7,3.7))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)

# Plot Data
ax.plot(wl/1e-9,spectrum/max(spectrum),'-')

# Set Axes
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity (norm.)')
ax.set_xlim((520,544))
ax.set_ylim((-0.2,1.2))
ax.set_xticks(np.arange(520,544+4,4))
ax = plt.gca()

plt.show()

