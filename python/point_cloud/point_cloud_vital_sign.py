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
import threading
from queue import Queue
import json
import os

# Helper Functions
def save_dict_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

# Load dictionary from a JSON file
def load_dict_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

start = time()
# Load Data
current_case = 'test04102023'
current_scenario = 'human_longer'
my_vtrig = isens_vtrigU(case=current_case)
my_vtrig.nframes = 500
calArr, recArr, dataPath = my_vtrig.load_data(case=current_case, scenario=current_scenario, return_path=True)

# Parameter Setup
recArr = recArr[100:200,:,:]
nframe = recArr.shape[0]
chosen_frame = 50
ntarget = 6             # number of 
bound = 2.5
Nfft = 64
y_offset_shift = 220 
x_offset_shift = -90
x_ratio = 20/30
y_ratio = 20/25
threshold=0.98
num_threads = 32
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
map_3d = []
for frame in range(nframe):
    tof = proArr[frame,:,:]
    tof = np.fft.ifft(tof,n=Nfft,axis=1)                      # Do IFFT across frequency steps to get the range bins
    tof[:,np.where(my_vtrig.dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range

    # tof = tof.reshape(nframe,20,20,-1)[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
    tof = tof.reshape(20,20,-1)
    # tof = np.transpose(tof,axes=(1,2,3,0))
    # tof = np.linalg.norm(tof,axis=0)
    tof = np.fft.fft(tof,n=Nfft,axis=1)
    tof = np.fft.fft(tof,n=Nfft,axis=0)
    map_3d.append(tof)
tof = np.stack(map_3d,axis=-1)
print(tof.shape)
print('test')
# tof = my_vtrig.normalization(tof)
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

# Get x, y, z indices for the entire matrix
normalized_matrix = my_vtrig.normalization(masked_matrix)

# Apply a threshold
mask = normalized_matrix > threshold

# Get x, y, z indices for the entire matrix
x, y, z = np.indices(matrix.shape)
# x, y, z = np.indices(matrix.shape)

# Create a colormap to map the normalized values to colors
# colormap = plt.cm.viridis

# Add the entire matrix as blue points with changed axis values
# ax.scatter(x_array[x[mask]], y_array[y[mask]], z_array[z[mask]], alpha=0.6, label='All points')
# x_peaks, y_peaks, z_peaks = x[mask], y[mask], z[mask] 
# Doppler Helper Functions
def doppler_fft(key, array, output):
    result = np.fft.fft(array)
    output[key] = result

def doppler_worker(queue, output):
    while not queue.empty():
        item = queue.get()
        doppler_fft(*item, output)
        queue.task_done()

possible_doppler = {}
work_queue = Queue()
for x_peak_idx in x[mask]:
    for y_peak_idx in y[mask]:
        for z_peak_idx in z[mask]:
            possible_doppler[(x_peak_idx, y_peak_idx, z_peak_idx)] = doppler[x_peak_idx, y_peak_idx, z_peak_idx, :]
# doppler = my_vtrig.normalization(doppler)
for key, value in possible_doppler.items():
    work_queue.put((key,value))
threads = []
possible_doppler_output={}
for _ in range(num_threads):
    t = threading.Thread(target=doppler_worker, args=(work_queue, possible_doppler_output))
    t.start()
    threads.append(t)

work_queue.join()
for t in threads:
    t.join()
save_dict_json(possible_doppler_output, os.path.join(dataPath,'doppler_dict.json'))
d=(2*60+2.8)/500
doppler_freq = np.fft.fftfreq(500,d)
freq_low = 10
freq_high = 230
# range_low = 100
# range_high = 350
for x_peak_idx in x[mask]:
    for y_peak_idx in y[mask]:
        for z_peak_idx in z[mask]:
            plt.figure(figsize=(20,10))
            cropped_doppler = possible_doppler_output[(x_peak_idx, y_peak_idx, z_peak_idx)]
            cropped_doppler = cropped_doppler[freq_low:freq_high]
            cropped_doppler = my_vtrig.normalization(np.abs(cropped_doppler))
            plt.plot(doppler_freq[freq_low:freq_high],cropped_doppler)
            plt.title(f'(AoD, AoA, Range) = ({(x_array[x_peak_idx])} [deg], {(y_array[y_peak_idx])} [deg], {(z_array[z_peak_idx])} [m])')
            plt.xlabel('doppler frequency [Hz]')
            plt.ylabel('magnitude')
            plt.savefig(os.path.join(dataPath,'tmp_plots',f'({(x_array[x_peak_idx])}, {(y_array[y_peak_idx])}, {(z_array[z_peak_idx])}).png'))
# print(doppler.shape)
# np.save(dataPath, doppler.npy) 
end = time()
print(end-start, '[s]')
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