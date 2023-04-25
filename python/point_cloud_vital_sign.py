# Load Library
import numpy as np
from isens_vtrigU import isens_vtrigU 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from scipy.constants import c
from scipy.signal import find_peaks
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from time import time

start = time()
# Load Data
current_case = 'test04102023'
current_scenario = 'human_2'
my_vtrig = isens_vtrigU(case=current_case)
my_vtrig.nframes = 500
calArr, recArr = my_vtrig.load_data(case=current_case, scenario=current_scenario)

# Parameter Setup
nframe = recArr.shape[0]
chosen_frame = 50
ntarget = 6             # number of 
bound = 2.5
Nfft = 512
y_offset_shift = 220 
x_offset_shift = -90
x_ratio = 20/30
y_ratio = 20/25
my_vtrig.dist_vec = my_vtrig.compute_dist_vec(Nfft=Nfft)

# Background Substraction
proArr = my_vtrig.calibration(calArr,recArr,method=0)

# Compute the Range Profile and Find the Target Range Bin (For one target)
range_profile = my_vtrig.range_pipeline(current_case,current_scenario, plot=False, Nfft=Nfft)[50,:]
range_profile[np.where(my_vtrig.dist_vec>bound)] = np.mean(range_profile)
range_peaks, _ = find_peaks(range_profile)
# Sort the peaks by their amplitudes in descending order and select the first 6 peaks
sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
top_range_peaks = range_peaks[sorted_peak_indices]
# Print the indices and angles of the top 6 peaks
print(f"Top {ntarget} range bin indices:", top_range_peaks)
print(f"Top {ntarget} range bins:", my_vtrig.dist_vec[top_range_peaks], '[m]')
# plt.figure(figsize=(20,10))
# plt.plot(my_vtrig.dist_vec,range_profile)
# plt.scatter(my_vtrig.dist_vec[top_range_peaks],range_profile[top_range_peaks])
# plt.show()
# target_range_bin = np.argmax(range_profile)
# range_bins = [target_range_bin-1, target_range_bin, target_range_bin+1]
# print(f'{my_vtrig.dist_vec[target_range_bin]} [m]')

# Generate the 3D map
tof = np.fft.ifft(proArr,n=Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
tof[:,:,np.where(my_vtrig.dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range

# tof = tof.reshape(nframe,20,20,-1)[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
tof = tof.reshape(nframe,20,20,-1)
tof = np.transpose(tof,axes=(1,2,3,0))
# tof = np.linalg.norm(tof,axis=0)
tof = np.fft.fft(tof,n=Nfft,axis=1)
tof = np.fft.fft(tof,n=Nfft,axis=0)
tof = my_vtrig.normalization(tof)
doppler = tof.copy()
# tof[:,:,top_range_peaks] *= 10
tof = np.roll(tof,shift=y_offset_shift,axis=1)
tof = np.roll(tof,shift=x_offset_shift,axis=0)
# new shape: (AoD, AoA, Range) = (x, y, z)

# Example 3D matrix (replace this with your matrix)
matrix = np.abs(tof[:,:,:,chosen_frame])

# Arrays to change the axis values (replace these with your arrays)
x_array = (my_vtrig.angle_vec-90)*x_ratio
y_array = (my_vtrig.angle_vec-90)*y_ratio
z_array = my_vtrig.dist_vec

# Channels to search for peaks (replace these with your desired channels)
channels_to_search = top_range_peaks
print(top_range_peaks)

# Create a masked copy of the matrix
masked_matrix = matrix.copy()
masked_matrix[:, :, [ch for ch in range(matrix.shape[2]) if ch not in channels_to_search]] = -np.inf

# Find the indices of the top 6 largest values
flat_indices = np.argsort(masked_matrix.flatten())[::-1][:ntarget]
indices_3d = np.unravel_index(flat_indices, matrix.shape)

# Extract x, y, and z coordinates of the top 6 peaks
x_peaks, y_peaks, z_peaks = indices_3d
doppler[x_peaks, y_peaks, z_peaks, :] *= 100
doppler = my_vtrig.normalization(doppler)
doppler = np.fft.fft(np.real(doppler), axis=3)
d=(2*60+2.8)/500
doppler_freq = np.fft.fftfreq(500,d)
freq_low = 10
freq_high = 230
range_low = 100
range_high = 350
doppler = doppler[freq_low:freq_high,range_low:range_high]
print(doppler.shape)
"""
plt.figure(figsize=(8,6))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Range [m]')
extent=[doppler_freq[freq_low],doppler_freq[freq_high],my_vtrig.dist_vec[range_low],my_vtrig.dist_vec[range_high]]
plt.imshow(np.abs(doppler).T,origin='lower',aspect='auto',extent=extent)
plt.colorbar()
plt.show()


print(matrix.shape)
print(x_peaks)
print(y_peaks)
print(z_peaks)
print(x_array[x_peaks])
print(y_array[y_peaks])
print(z_array[z_peaks])
# Create a 3D scatter plot for the original matrix
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Get x, y, z indices for the entire matrix
x, y, z = np.indices(matrix.shape)

# Add the entire matrix as blue points with changed axis values
# ax.scatter(x_array[x], y_array[y], z_array[z], c='b', alpha=0.3, label='Original matrix')

# Add the top 6 peaks to the plot with changed axis values
ax.scatter(x_array[x_peaks], y_array[y_peaks], z_array[z_peaks], c='r', marker='x', s=100, label=f'Top {ntarget} peaks')

# Set axis labels
ax.set_xlabel('X (AoD [deg])')
ax.set_ylabel('Y (AoA [deg])')
ax.set_zlabel('Z (Range [m])')

# Set plot title
plt.title(f"3D Matrix with Top {ntarget} Peaks (Changed Axis Values)")

# Set the axis limits to match the ranges of x_array, y_array, and z_array
ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())
ax.set_zlim(z_array.min(), z_array.max())

# Add legend
ax.legend()

end = time()
print(end-start, '[s]')

# Show the plot
plt.show()

"""