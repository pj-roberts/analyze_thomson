#%%
# Parker J. Roberts - U of M - 2024-03-14
# Python script to fit Langmuir Probe

#%% 
# Load Packages

# import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import scipy.optimize as op
# import matplotlib as mpl

# from os import listdir
# from os.path import isfile, join
# from tqdm import tqdm

import functions as fn
# import constants as cn

# Initialize matplotlib settings
fn.initialize_plotting()

#%% Load and plot data 

datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\'\
    'University of Michigan\\2024\\Experiments\\'\
        'Spring 2024 Laser Testing\\Langmuir Probe Test Analysis\\'
filename = 'langmuirprobe_2.csv'

filepath = datafolder + filename

df = pd.read_csv(filepath, header=None)

voltage = df.iloc[:,0].values
current = df.iloc[:,1].values


#%% Analyze Data to get electron temperature

# Find zero-crossing (floating potential)

# Differentiate 
dI_dV = np.diff(current)/np.diff(voltage)
d2I_dV = np.diff(dI_dV)/np.diff(voltage[:-1])

# Max point is just end of array in this case, but should find it 
# programatically in general.
start_idx = 200
end_idx = 201

# linear fit to log of current
log_current = np.log(current)
diff_logI = log_current[end_idx] - log_current[start_idx]
diff_V = voltage[end_idx] - voltage[start_idx]
slope = diff_logI/diff_V
Te = 1/slope
print(Te)

#%%

# Generate Figure/Axis
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.22,0.22,0.7,0.7])
ax.set_box_aspect(1.0)
# Plot Data
ax.semilogy(voltage,current)
ax.semilogy(voltage,np.exp((voltage - voltage[end_idx])*slope + log_current[end_idx]))
# ax.plot(current)
# ax.plot((voltage - voltage[end_idx])*slope + current[end_idx])

# ax.set_ylim((-10,100))

# Set Axes
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A)')
ax.grid()

plt.show()