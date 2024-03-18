### Parker J. Roberts - 2024-02-19
### Functions for analyze_thomson package
#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import constants as c


def parse_lightfield_raw(filepath):
    """Load data from lightfield raw images stored as .csv.
        These are the detector/camera image files.
    
    Args:
        filepath (str): location of data file

    Returns:
        wl (arr): (c,) array of wavelengths
        frame (arr): (s,r,c) array of intensity values (shot, row, column)
    """
    
    # Initialize arrays
    startrows = []
    endrows = []
    wls = 0

    # Loop once: detect array dimensions & preallocate
    with open(filepath) as csvfile:
        filereader = csv.reader(csvfile)
        for ii,row in enumerate(filereader):
            if row[0] == '0': # detect start of ccd image
                startrows.append(ii)
                rowlen = len(row) - 1
            if row[0] == 'End Region 1': # detect end of image
                # !!! If csvread doesn't see the strings, this is risky!
                endrows.append(ii-1)
            if (row[0] == 'Wavelength') and (wls == 0):
                wls = row[1:]

    # Handle case where wavelengths are binned and there is no
    # wavelength information
    if wls == 0:
        nwl = rowlen
        wls = np.linspace(0,nwl-1,nwl)
    else:
        nwl = len(wls)

    # Get array sizes
    nframes = len(startrows)
    nrows = endrows[0] - startrows[0] + 1

    # Convert wavelength strings to floats
    wls = np.array([float(ii) for ii in wls])

    # preallocate frames array
    frames = np.zeros((nframes,nrows,nwl))
    framecount = 0
    rowcount = 0

    # Loop again: transfer frames to array structure
    # Handle case for single-row frames:

    if nrows == 1:
        # Get 1D Binned Spectra
        with open(filepath) as csvfile:
            filereader = csv.reader(csvfile)
            for ii,row in enumerate(filereader):
                if framecount < nframes: # End loop when loaded all frames
                    if ii == startrows[framecount]:
                        # Start counting distance from startrow == matrix row
                        frames[framecount,rowcount,:] = row[1:]
                        framecount += 1

    else:
        # Get 2D frame
        with open(filepath) as csvfile:
            filereader = csv.reader(csvfile)
            for ii,row in enumerate(filereader):
                if framecount < nframes: # End loop when loaded all frames
                    if ii == startrows[framecount]:
                        # Start counting distance from startrow == matrix row
                        frames[framecount,rowcount,:] = row[1:]
                        rowcount += 1

                    elif startrows[framecount] < ii < endrows[framecount]:
                        frames[framecount,rowcount,:] = row[1:]
                        rowcount += 1
                        
                    elif ii == endrows[framecount]:
                        frames[framecount,rowcount,:] = row[1:]
                        framecount += 1
                        rowcount = 0

    # For spectra, reduce to 2D 
    frames = np.squeeze(frames)

    return wls, frames


def compute_mean_spectrum(frames):
    """Compute the average spectrum corresponding to a matrix of CCD
      images (output of parse_lightfield_raw). Assumes the spectrum is 
      uniform w.r.t. row height, i.e. no spatial resolution on detector.

    Args:
        frames (arr): (shot,row,wavelength) array of CCD images

    Returns:
        spectrum (arr): 1D array (wavelength,) of average counts per pixel.
    """
    mean_frame = np.mean(frames,0)
    spectrum = np.mean(mean_frame,0)

    return spectrum


def count_photons_simple(frames,pct,crt):
      """Counts photon events on the array of CCD images frames via 
        a thresholding algorithm (sets value to 1 if > pct, 0 if < pct).
        Also has option to remove cosmic rays (detections > crt set to
        zero) of crt > 0.

      Args:
          frames (arr): CCD image matrix, (shot, row, wavelength)
          pct (float): photon counting threshold
          crt (float): cosmic ray threshold
          rad (float): radius (max 1 detection w/in rad pixels)

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
      
      # Detect photons
      pc_idx = frames_pc > pct
      frames_pc[pc_idx] = 1
      frames_pc[np.logical_not(pc_idx)] = 0            

      return frames_pc


def count_photons_mask(frames,pct,crt,rad,crrad):
    """Counts photon events on the array of CCD images frames via 
    a thresholding algorithm (sets value to 1 if >= pct, 0 if < pct).
    Also has option to remove cosmic rays (detections > crt set to
    zero) of crt > 0.

    This version of the function processes the points on each frame 
    in descending order of intensity, masking points within "rad" of
    a detection. This ensures the most intense point is counted as 
    a single photon, but gets rid of blooming so that the statistics 
    represent the actual number of photons detected. However, the 
    resulting data is more discrete and needs smoothing.

    !!! A similar masking could be added for the CR detections.

    Args:
        frames (arr): CCD image matrix, (shot, row, wavelength)
        pct (float): photon counting threshold
        crt (float): cosmic ray threshold
        rad (float): radius (radius for masking nearby detections)
        crrad (float): radius (radius for masking nearby detections)

    Returns:
        spectrum_pc (arr): (wavelength,) histogram vector, value\
            corresponds to # of counts at that wavelength
    """

    # Store frame shape in memory
    frame_shape = np.shape(frames) 

    # Preallocate histogram for storing counts
    spectrum_pc = np.zeros((frame_shape[2],))

    frames_pc = frames.copy()
    frames_pc = frames_pc - np.mean(frames_pc,0)

    # Detect cosmic rays
    if crt > 0:
        crt_idx = frames_pc > crt
        frames_pc[crt_idx] = 0
    
    # First, set all pixels below the threshold to zero
    frames_pc[frames_pc < pct] = 0

    for ii in range(frame_shape[0]):  # for each shot:
        frame = frames_pc[ii,:,:]  # get this frame
        frame_vec = np.ravel(frame)  # represent as vector
        last_max = 2  # init variable to track recent max value
        while last_max > 1:

                # !!! Consider making the following a function which 
                # can be reused for masking the CRT points with a 
                # different radius.

                # Get maximum unchecked pixel
                max_idx = np.argmax(frame_vec)
                last_max = np.max(frame_vec)

                if last_max > 0:  # Don't count frame if done

                    # Get matrix indices of maximum in frame array
                    frame_idx = np.unravel_index(max_idx,frame_shape[1:])

                    # Mask out nearby pixels in a pixel square w side rad
                    mask_row_min = np.max((0,frame_idx[0] - rad))
                    mask_row_max = np.min((frame_shape[1],frame_idx[0] + rad))
                    mask_col_min = np.max((0,frame_idx[1] - rad))
                    mask_col_max = np.min((frame_shape[2],frame_idx[1] + rad))
                    for row_idx in range(mask_row_min,mask_row_max):
                        for col_idx in range(mask_col_min,mask_col_max):
                                mask_idx = np.ravel_multi_index(\
                                    (row_idx,col_idx),frame_shape[1:])
                                frame_vec[mask_idx] = 0

                    # Count detection at maximum
                    spectrum_pc[frame_idx[1]] += 1

    spectrum_pc /= frame_shape[0]    

    return spectrum_pc


def subtract_bg_concurrent(frames):
    """Perform background subtraction when background frames are taken
      between laser frames, e.g. 20 Hz acquisition of 10 Hz laser pulses.
      Does not assume that the first frame is bg or signal - instead, 
      inverts output if area under curve is negative.

      This code takes the simple mean, but could be updated to allow 
      photon counting.

    Args:
        frames (float array): frame matrix with alternating signal, bg
        in first index

    Returns:
        net_spec: mean spectrum (its - pbg) 
    """
    
    frames_its = frames[::2,:]
    frames_pbg = frames[1::2,:]

    mean_its = np.mean(frames_its,0)
    mean_pbg = np.mean(frames_pbg,0)

    net_spec = mean_its - mean_pbg

    # Correct for switch in signal vs background frame order
    norm = sum(net_spec)
    net_spec = norm/abs(norm)*net_spec

    return net_spec


def load_spectrum(filenum,spectrafolder):
      """Use pandas dataframe to load photon count spectrum and bg from .csv

      Args:
          filenum (int): shot number below 1000
          spectrafolder (str): folder from which to read spectra

      Returns:
          wl (ndarray): Wavelength vector
          its_pc (ndarray): Thomson photon count vector
          pbg_pc (ndarray): Thomson photon count vector
      """
      # Generate Filename from number
      filename_its = f'{filenum:03.0f}_its_spectrum.csv'
      filename_pbg = f'{filenum:03.0f}_pbg_spectrum.csv'

      # Load data into dataframe 
      df_its = pd.read_csv(spectrafolder + filename_its) # incoherent Thomson
      df_pbg = pd.read_csv(spectrafolder + filename_pbg) # plasma background

      # Extract columns as np arrays
      wl = df_its.loc[:,'Wavelength (nm)'].values
      its_pc = df_its.loc[:,'Photon Counts'].values
      pbg_pc = df_pbg.loc[:,'Photon Counts'].values

      return wl,its_pc,pbg_pc


def mask_filter(wl,v,its,filt_rad):
      """Returns masked data not including points within filt_rad of 
      532 nm.

      Args:
          wl (ndarray): wavelength vector
          v (ndarray): velocity vector
          its (ndarray): signal vector
          filt_rad (float): radius around 532 nm to exclude points

      Returns:
          wl_mask (ndarray): masked wavelength vector
          v_mask (ndarray): masked velocity vector
          its_mask (ndarray): masked signal vector
      """
      filt_min = 532 - filt_rad
      filt_max = 532 + filt_rad
      mask_idx = (wl < filt_min) | (wl > filt_max)
      wl_mask = wl[mask_idx]
      v_mask = v[mask_idx]
      its_mask = its[mask_idx]
      return wl_mask, v_mask, its_mask


def generate_dataset(ne,Te,ue,noise):
    """Simulate Thomson spectrum based on 1D Maxwellian distribution.
        For testing analysis scripts and robustness to noise. For
        nonzero noise (> 0), corrupts data with pseudorandom gaussian 
        noise of standard deviation (noise).

    Args:
        ne (float): electron density (m^-3)
        Te (float): electron temperature (eV)
        ue (float): electron velocity (m/s)
        noise (float): noise magnitude (as fraction of ne)
        
    Returns:
        wl (float arr): wavelengths (!!! hard-coded in for now)
        spectrum (float arr): 1D spectrum
    """
    # Compute wavelength/velocity axis 
    wl = np.linspace(522e-9,542e-9,102)  # !!! hard-coded for now
    u = wl2velocity(wl)

    # Compute Maxwellian distribution
    vT = np.sqrt(2*c.e*Te/c.me) # Thermal electron velocity
    spectrum = ne*np.exp(-((u - ue)/vT)**2)

    # Corrupt data with noise
    spectrum = spectrum + noise*ne*0  # !!! add pseudorandom sampling

    return wl,spectrum


def velocity2wl(velocities):
    """convert velocity to wavelength (vectorized)

    Args:
        velocities (n,): array of velocities (m/s)

    Returns:
        wls (n,): array of wavelengths (m)
    """
    wls = velocities
    print('Error: velocity2wl not complete')
    return wls


def wl2velocity(wls):
    """convert wavelength to velocity (vectorized)

    Args:
         wls (n,): array of wavelengths (m)

    Returns:
        velocities(n,): array of velocities (m/s)
    """
    velocities = c.c*(c.lambda_0/wls - 1)/2/np.sin(c.theta/2)
    return velocities

# Plotting functions

def initialize_plotting():
    """
    Set nice-looking publication-ready linewidths and fonts
    """

    # Font Settings
    mpl.rcParams.update({'font.size': 12})  # Adjust font size as needed
    mpl.rcParams.update({'font.family': 'Arial'})  # Choose appropriate font

    # Axis settings
    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 2.0
    mpl.rcParams['ytick.major.width'] = 2.0
    # plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    # plt.rcParams['axes.xmargin'] = 0
    # plt.rcParams['axes.ymargin'] = 0


    # Line/Color Settings
    mpl.rcParams['lines.linewidth'] = 2.0


def plot_spectrum(wls,spectrum):
      """Generate line plot of spectrum

      Args:
          wl (_type_): _description_
          spectrum (_type_): _description_

      Returns:
          _type_: _description_
      """

      # Generate Figure/Axis
      fig = plt.figure(figsize=(3.37,1.69))
      ax = fig.add_axes([0.22,0.22,0.7,0.7])
      # ax.set_box_aspect(1.0)

      # Generate Plot
      plt.plot(wls,spectrum,linewidth=1,color='k')
      plt.xlabel('Wavelength (nm)')
      plt.ylabel('Counts')
      plt.ylim((-0.1,4))
      plt.xlim((526,538))
      plt.xticks(np.arange(526,540,2))
      plt.show()
      
      return fig,ax