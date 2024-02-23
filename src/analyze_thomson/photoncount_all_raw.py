#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script to load, photon count, and save all raw data in dir.

#%% 
# Load Packages

# import csv
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import functions as fn

#%% 
# Initialize matplotlib

fn.initialize_plotting()

#%% 
# Settings/Initialize

# data/saving directories
datafolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\Raw Data\\Processed Data\\'

savefolder = 'C:\\Users\\pjrob\\Documents\\Research\\' +\
      'University of Michigan\\2023\\ITS CTF\\' +\
            '2024 Python Analysis\\Spectra\\'

plotfolder = savefolder + 'Plots\\'

# Photon count settings 
pct = 40
crt = 0
rad = 5
crrad = 0

#%% Loop through large files, photon count, and save spectra

# get list of files in data folder
files = [f for f in listdir(datafolder) if isfile(join(datafolder, f))]

# Loop through list of files, count photons, and plot/save
for ii in tqdm(range(len(files))):

      filename = files[ii]
      filepath = datafolder + filename
      
      # Import Frames using parsing function
      wls, frames = fn.parse_lightfield_raw(filepath)

      # Compute and save mean spectrum
      spectrum = np.mean(np.mean(frames,0),0)
      savename_spectrum = savefolder + filename[0:-4] + '_spectrum.npy'
      np.save(savename_spectrum,spectrum)

      # Count photons from 2D frames and save
      spectrum_pc = fn.count_photons_mask(frames,pct,crt,rad,crrad)
      savename_spectrum_pc = savefolder + filename[0:-4] + '_spectrum_pc.npy'
      np.save(savename_spectrum_pc,spectrum_pc)

      # Save wavelengths
      savename_wl = savefolder + filename[0:-4] + '_wl.npy'
      np.save(savename_wl,wls)

      # Generate Figure/Axis for line plot
      fig = plt.figure(figsize=(3.37,2))
      ax = fig.add_axes([0.3,0.3,0.6,0.6])
      # ax.set_box_aspect(1.0)

      # Line plot of spectrum
      plt.plot(wls,spectrum_pc,linewidth=1,color='k')

      plt.xlabel('Wavelength (nm)')
      plt.ylabel('Counts')

      # plt.tight_layout()
      plt.savefig(plotfolder +  filename[0:-4] + '_spectrum_pc.pdf', format='pdf')
      # plt.show()
      plt.close()
