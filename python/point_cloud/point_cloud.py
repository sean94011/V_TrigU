# Load Library
from math import ceil, log
from time import time

import numpy as np
from isens_vtrigU import isens_vtrigU
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.constants import c
from scipy.io import savemat
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

def cfar(matrix, num_train_cells, threshold_scale_factor):
    """
    Apply the Cell Averaging CFAR algorithm to a 2D or 3D matrix.

    Parameters:
    - matrix: The 2D or 3D matrix to apply CFAR to.
    - num_train_cells: The number of training cells (in each direction from the cell under test).
    - threshold_scale_factor: The factor to scale the average noise level by to get the threshold.

    Returns:
    A binary matrix of the same shape as the input where cells with a value of 1 exceed the CFAR threshold.
    """
    # Calculate the average noise level in the neighbourhood of each cell
    avg_noise_level = uniform_filter(matrix, size=2*num_train_cells+1, mode='constant', cval=0)

    # Multiply the average noise level by the threshold scale factor to get the threshold
    threshold = avg_noise_level * threshold_scale_factor

    # Create a binary matrix where cells with a value of 1 exceed the CFAR threshold
    cfar_matrix = (matrix > threshold).astype(int)

    # Find the indices of cells that exceed the threshold
    cfar_indices = np.argwhere(cfar_matrix == 1)

    return cfar_indices

def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def range_doppler_peaks(proArr, range_Nfft, doppler_Nfft, dist_vec, plot=True):
    tof = np.fft.ifft(proArr,n=range_Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
    tof[:,:,np.where(dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range
    # tof = tof.reshape(nframe,400,-1)#[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
    print(tof.shape)
    range_doppler = tof.copy()
    range_doppler = np.linalg.norm(range_doppler, axis=1)
    print(range_doppler.shape)
    
    # Doppler FFT
    range_doppler = np.fft.fft(np.real(range_doppler),n=doppler_Nfft,axis=0)
    range_doppler = np.abs(range_doppler).T # (range, doppler)

    d=(2*60+2.8)/500
    # d = 1/fs
    doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    doppler_freq = doppler_freq[doppler_freq>=0]
    freq_low = np.where(doppler_freq>=0.3)[0][0]
    freq_high = np.where(doppler_freq<=2.0)[0][-1]
    range_low = np.where(dist_vec>=0.3)[0][0]
    range_high = np.where(dist_vec<=2.0)[0][-1]

    # Eliminate the high DC offset
    # range_low, range_high, freq_low, freq_high = 50, 400, 100, 300
    range_range = np.r_[0:range_low, range_high:range_doppler.shape[0]]
    doppler_range = np.r_[0:freq_low, freq_high:range_doppler.shape[1]]
    range_doppler[range_range,:] = np.min(range_doppler)
    range_doppler[:, doppler_range] = np.min(range_doppler)

    range_doppler = normalization(range_doppler)
    rd_peaks_indices = filter_and_cluster(matrix=range_doppler[range_low:range_high,freq_low:freq_high], threshold=0.5, n=6)
    range_bins = set((np.argwhere(dist_vec==dist_vec[range_low:range_high][rd_peaks_indices[:,0]]).flatten()))
    if plot:
        extent=[doppler_freq[freq_low],doppler_freq[freq_high],my_vtrig.dist_vec[range_low],my_vtrig.dist_vec[range_high]]
        # rd_peaks_indices = find_n_largest(range_doppler, 6)

        plt.figure(figsize=(10,10))
        plt.imshow(range_doppler[range_low:range_high,freq_low:freq_high],origin="lower",aspect="auto",extent=extent)
        plt.scatter(doppler_freq[freq_low:freq_high][rd_peaks_indices[:, 1]], my_vtrig.dist_vec[range_low:range_high][rd_peaks_indices[:, 0]], color='r', marker='x')
        plt.colorbar()
        plt.xlabel('Doppler Frequency [Hz]')
        plt.ylabel('Range [m]')
        plt.title(f'Setup: {current_scenario}')
        plt.show()

    
    return list(range_bins), range_doppler[range_low:range_high,freq_low:freq_high], range_low, range_high

def find_closest_index(arr, value):
    diff = np.abs(arr - value)
    return np.argmin(diff)

def filter_and_cluster(matrix, threshold=0.0, n=np.inf, eps=1, min_samples=3):
    indices = np.argwhere(matrix > threshold)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(indices)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    cluster_sizes = [(label, sum(labels == label)) for label in unique_labels if label != -1]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []
    for i in range(min(n, len(cluster_sizes))):
        label = cluster_sizes[i][0]
        cluster_indices = indices[labels == label]
        centroid = cluster_indices.mean(axis=0)
        selected_indices.append(cluster_indices[distance.cdist([centroid], cluster_indices).argmin()])

    return np.array(selected_indices)

start = time()
# Load Data
# current_case = 'test01242023'
# current_scenario = 'cf_x_angle_+20'
# current_case = 'test01242023'
# current_scenario = 'cf_x_angle_+20'
current_case = 'test04242023'
current_scenario = '2cf_yy_angle_+-20_3'
rd_peaks_channels = False
cfar_channels = False
Nfft = 0
angle_Nfft = [64, 64] # [AoD, AoA]
range_Nfft = 512
my_vtrig = isens_vtrigU(case=current_case, range_Nfft=range_Nfft, angle_Nfft=angle_Nfft)
calArr, recArr = my_vtrig.load_data(case=current_case, scenario=current_scenario)
# threshold = 0.93 # corner reflectors
threshold = 0.98


# Parameter Setup
nframe = recArr.shape[0]
chosen_frame = 0
ntarget = 6             # number of 
bound = 2.5
if angle_Nfft[0] == 64:
    x_offset_shift = -11
    x_ratio = 20/(34.2857)
elif angle_Nfft[0] == 512: #(Nfft=512)
    x_offset_shift = -90
    x_ratio = 20/30
else: #(Nfft=512)
    x_offset_shift = 0
    x_ratio = 1

if angle_Nfft[1] == 64:
    y_offset_shift = 27
    y_ratio = 20/29
elif angle_Nfft[1] == 512: #(Nfft=512)
    y_offset_shift = 220 
    y_ratio = 20/25
else: #(Nfft=512)
    y_offset_shift = 0 
    y_ratio = 1


my_vtrig.dist_vec = my_vtrig.compute_dist_vec(Nfft=range_Nfft)

x_array = (my_vtrig.AoD_vec-90)*x_ratio
y_array = (my_vtrig.AoA_vec-90)*y_ratio
z_array = my_vtrig.dist_vec


# x_offset_shift = find_closest_index(x_array,x_offset_shift)
# print(x_offset_shift)

# Background Substraction
proArr = my_vtrig.calibration(calArr,recArr,method=0)
doppler_Nfft =  2**(ceil(log(proArr.shape[0],2))+1)

if rd_peaks_channels:
    rd_peaks_idx, range_doppler, range_low, range_high = range_doppler_peaks(proArr=proArr, range_Nfft=range_Nfft, doppler_Nfft=doppler_Nfft, dist_vec=my_vtrig.dist_vec)

if cfar_channels:
    num_train_cells = 5
    threshold_scale_factor = 1.2
    cfar_idx = cfar(range_doppler, num_train_cells, threshold_scale_factor)
    cfar_idx = list(set([int(np.where(my_vtrig.dist_vec == val)[0].flatten()) for val in my_vtrig.dist_vec[range_low:range_high][cfar_idx[:,0]]]))






# Compute the Range Profile and Find the Target Range Bin (For one target)
range_profile = my_vtrig.range_pipeline(current_case,current_scenario, plot=False, Nfft=range_Nfft)[chosen_frame,:]
range_profile[np.where(my_vtrig.dist_vec>bound)] = np.mean(range_profile)
range_peaks, _ = find_peaks(range_profile)
# Sort the peaks by their amplitudes in descending order and select the first 6 peaks
sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
top_range_peaks = range_peaks[sorted_peak_indices]
# Print the indices and angles of the top 6 peaks
print(f"Top {ntarget} range bin indices:", top_range_peaks)
print(f"Top {ntarget} range bins:", my_vtrig.dist_vec[top_range_peaks], '[m]')
# plt.figure(figsize=(20,10 ))
# plt.plot(my_vtrig.dist_vec,range_profile)
# plt.scatter(my_vtrig.dist_vec[top_range_peaks],range_profile[top_range_peaks])
# plt.show()
# Generate the 3D map
# Arrays to change the axis values (replace these with your arrays)


Nfft = 150
tof = np.fft.ifft(proArr,n=range_Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
tof[:,:,np.where(my_vtrig.dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range
tof = tof.reshape(nframe,20,20,-1)[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
Nfft = 20
tof = np.fft.fft(tof,n=angle_Nfft[1],axis=1)
tof = np.fft.fft(tof,n=angle_Nfft[0],axis=0)
tof = my_vtrig.normalization(tof)
tof = np.roll(tof,shift=y_offset_shift,axis=1)
tof = np.roll(tof,shift=x_offset_shift,axis=0)
tof = np.abs(tof)
# new shape: (AoD, AoA, Range) = (x, y, z)
# Example 3D matrix (replace this with your matrix)
matrix = tof

# Channels to search for peaks (replace these with your desired channels)
if rd_peaks_channels:
    if cfar_channels:
        channels_to_search = cfar_idx
    else:
        channels_to_search = rd_peaks_idx
else:
    channels_to_search = top_range_peaks
print(top_range_peaks)

# Create a masked copy of the matrix
masked_matrix = matrix.copy()
masked_matrix[:, :, [ch for ch in range(matrix.shape[2]) if ch not in channels_to_search]] = np.min(matrix)

# Find the indices of the top 6 largest values
flat_indices = np.argsort(masked_matrix.flatten())[::-1][:ntarget]
indices_3d = np.unravel_index(flat_indices, matrix.shape)
# Extract x, y, and z coordinates of the top 6 peaks
x_peaks, y_peaks, z_peaks = indices_3d

# Create a 3D scatter plot for the original matrix
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Get x, y, z indices for the entire matrix
normalized_matrix = my_vtrig.normalization(masked_matrix)

# Apply a threshold
mask = normalized_matrix > threshold

# Get x, y, z indices for the entire matrix
x, y, z = np.indices(matrix.shape)

# Add the entire matrix as blue points with changed axis values
ax.scatter(x_array[x[mask]], y_array[y[mask]], z_array[z[mask]], alpha=0.3, label='All points',)
print('x_peaks indices', x[mask])
print('y_peaks indices', y[mask])
print('z_peaks indices', z[mask])
print('AoD', x_array[x[mask]], '[deg]')
print('AoA', y_array[y[mask]],'[deg]')
print('Range', z_array[z[mask]], '[m]')
peak_indices = filter_and_cluster(normalized_matrix, threshold, n=6 ,eps=10, min_samples=1)
print(peak_indices)
if len(peak_indices) != 0:
    for i in range(len(peak_indices)):
        print('AoD, AoA, Range: ',x_array[peak_indices[i,0]],y_array[peak_indices[i,1]],z_array[peak_indices[i,2]])
        ax.scatter(x_array[peak_indices[i,0]],y_array[peak_indices[i,1]],z_array[peak_indices[i,2]],marker='x', color='r')


# Set axis labels
ax.set_xlabel('X (AoD [deg])')
ax.set_ylabel('Y (AoA [deg])')
ax.set_zlabel('Z (Range [m])')
# Set plot title
plt.title(f"3D Point Cloud: {current_scenario}")
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

for channel in channels_to_search:
    plt.figure(figsize=(8,6))
    plt.title(f'Scenario: {current_scenario} | Range NFFT Bins: {range_Nfft} | Range Bin: {np.round(my_vtrig.dist_vec[channel],3)} [m]')
    extent = [x_array[0],x_array[-1],y_array[0],y_array[-1]]
    plt.imshow(masked_matrix[:,:,channel].T,origin='lower',aspect='auto', extent=extent)
    # clutter_centers = filter_and_cluster(normalization(normalized_matrix[:,:,channels_to_search]),threshold=0.5, min_samples=1)
    # print(clutter_centers)
    # plt.scatter(x_array[clutter_centers[]])
    plt.xlabel('X (AoD [deg])')
    plt.ylabel('Y (AoA [deg])')
    plt.colorbar()
    plt.grid()
    plt.show()