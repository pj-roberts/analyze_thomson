#%%

# Parker J. Roberts - U of M - 2024-02-20
# Python script for prototyping better photon counting algorithm

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
rad = 5
crrad = 10

#%% Load Data & count photons

# Import Frames using parsing function
wls, frames = fn.parse_lightfield_raw(filepath) # Use imported version

#%% Count photons from 2D frames

def count_photons_mask(frames,pct,crt,rad,crrad):
      """Counts photon events on the array of CCD images frames via 
        a thresholding algorithm (sets value to 1 if > pct, 0 if < pct).
        Also has option to remove cosmic rays (detections > crt set to
        zero) of crt > 0.

        This version of the function processes the points on each frame 
        in descending order of intensity, masking points within "rad" of
        a detection. This ensures the most intense point is counted as 
        a photon, but gets rid of blooming so that the statistics 
        represent the actual number of photons detected. However, the 
        resulting data is more descrete and needs smoothing.

        A similar masking is done for the CR detections.

      Args:
          frames (arr): CCD image matrix, (shot, row, wavelength)
          pct (float): photon counting threshold
          crt (float): cosmic ray threshold
          rad (float): radius (radius for masking nearby detections)
          crrad (float): radius (radius for masking nearby detections)

      Returns:
          frames_pc (arr): (wavelength,) histogram vector, value 
            corresponds to # of counts at that wavelength
      """
      # First pass: ignore radius and just logically index
      frames_pc = frames.copy()

      # Detect cosmic rays
      if crt > 0:
            crt_idx = frames_pc > crt
            frames_pc[crt_idx] = 0
      
      # Detect photons:
      # pc_idx = frames_pc > pct
      # frames_pc[pc_idx] = 1
      # frames_pc[np.logical_not(pc_idx)] = 0   

      # First, set all pixels below the threshold to zero
      frames_pc[frames_pc < pct] = 0

      # Then, count nearby counts as single detections, in descending order.
      # This probably could be made faster, but I think the loading
      # time is still the limiting factor.
      for ii in range(len(frames[:,0,0])):  # for each shot:
            frame = frames_pc[ii,:,:]  # get this frame
            frame_vec = np.ravel(frame)  # represent as vector
            sort_vec = np.flip(np.sort(frame_vec)) # values in reverse order
            sort_idx = np.flip(np.argsort(frame_vec)) # corresponding indices
            for jj in range(len(sort_vec)): # For each pixel:
                  if sort_vec[jj] > 0:  # If not already filtered out:
                        # Get 2d indices for this pixel
                        frame_idx = np.unravel_index(sort_idx[jj],\
                                                     np.shape(frame))
                        # Scan every pixel with weaker signal 
                        for kk in range(jj,len(sort_vec)):
                              # Get indices of scan pixel
                              frame_idx_2 = np.unravel_index(sort_idx[kk],\
                                                             np.shape(frame))
                              # filter if within square of side length rad
                              x_dist = frame_idx[0] - frame_idx_2[0]
                              y_dist = frame_idx[1] - frame_idx_2[1]
                              if (x_dist < rad) or (y_dist < rad):
                                    # set to zero to filter out point
                                    sort_vec[kk] = 0
                                    frame[frame_idx_2] = 0
                  frames_pc[ii,:,:] = frame
      # Very slow...


      ## I thought of a better, or at least more readable, way to do this.
       # Rather than define the whole loop, just take the max and set all 
       # nearby points to zero `while` the max is greater than 1. 
       # (Assumes points below pct have already been set to zero)
      
      return frames_pc

frames_pc = count_photons_mask(frames,pct,crt,rad,crrad)

# frames_pc = fn.count_photons_rad(frames,pct,crt)

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

plt.tight_layout()
plt.savefig(plotfolder +  filename[0:-4] + '_spectrum_pc.pdf', format='pdf')
plt.show()
# plt.close()

# Generate Figure/Axis for raw contour
fig = plt.figure(figsize=(3.37,3.37))
ax = fig.add_axes([0.1,0.1,0.6,0.6])
ax.set_box_aspect(1.0)

# Contour plot of raw mean frame
plt.contourf(np.mean(frames,0),levels=np.linspace(-3,3,20))
plt.colorbar(fraction=0.045)
plt.savefig(plotfolder +  filename[0:-4] + '_mean_frame.png', format='png',dpi=600)
plt.show()
# plt.close()

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