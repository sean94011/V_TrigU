# Load Library
import os
from time import time

import numpy as np
from isens_vtrigU import isens_vtrigU
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.constants import c
from scipy.io import savemat
from scipy.signal import find_peaks

# Load Data
# current_case = 'test01242023'
# current_scenario = 'cf_x_angle_+20'
current_case = 'test04242023'
current_scenario = 'human_stretch_stand'
my_vtrig = isens_vtrigU(case=current_case)
calArr, recArr, dataPath = my_vtrig.load_data(case=current_case, scenario=current_scenario, return_path=True)
print(dataPath)

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
threshold = 0.971
my_vtrig.dist_vec = my_vtrig.compute_dist_vec(Nfft=Nfft)

# Background Substraction
proArr = my_vtrig.calibration(calArr,recArr,method=0)

# Compute the Range Profile and Find the Target Range Bin (For one target)
range_profile = my_vtrig.range_pipeline(current_case,current_scenario, plot=True, Nfft=Nfft)[chosen_frame,:]
range_profile[np.where(my_vtrig.dist_vec>bound)] = np.mean(range_profile)
range_peaks, _ = find_peaks(range_profile)
# Sort the peaks by their amplitudes in descending order and select the first 6 peaks
sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
top_range_peaks = range_peaks[sorted_peak_indices]
# Print the indices and angles of the top 6 peaks
print(f"Top {ntarget} range bin indices:", top_range_peaks)
print(f"Top {ntarget} range bins:", my_vtrig.dist_vec[top_range_peaks], '[m]')
plt.figure(figsize=(20,10))
plt.plot(my_vtrig.dist_vec,range_profile)
plt.scatter(my_vtrig.dist_vec[top_range_peaks],range_profile[top_range_peaks])
plt.show()
# target_range_bin = np.argmax(range_profile)
# range_bins = [target_range_bin-1, target_range_bin, target_range_bin+1]
# print(f'{my_vtrig.dist_vec[target_range_bin]} [m]')
if 'all_point_cloud.npy' not in os.listdir(dataPath):
    all_point_cloud = []

    for chosen_frame in range(nframe):
        # Generate the 3D map
        tof = np.fft.ifft(proArr,n=Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
        tof[:,:,np.where(my_vtrig.dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range
        tof = tof.reshape(nframe,20,20,-1)[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
        # tof = np.linalg.norm(tof,axis=0)
        tof = np.fft.fft(tof,n=Nfft,axis=1)
        tof = np.fft.fft(tof,n=Nfft,axis=0)
        tof = my_vtrig.normalization(tof)
        # tof[:,:,top_range_peaks] *= 10
        tof = np.roll(tof,shift=y_offset_shift,axis=1)
        tof = np.roll(tof,shift=x_offset_shift,axis=0)
        tof = np.abs(tof)
        # new shape: (AoD, AoA, Range) = (x, y, z)
        matrix = tof

        # Channels to search for peaks (replace these with your desired channels)
        channels_to_search = top_range_peaks
        # print(top_range_peaks)

        # Create a masked copy of the matrix
        masked_matrix = matrix.copy()
        masked_matrix[:, :, [ch for ch in range(matrix.shape[2]) if ch not in channels_to_search]] = np.min(matrix)
        normalized_matrix = my_vtrig.normalization(masked_matrix)

        all_point_cloud.append(normalized_matrix)

    all_point_cloud = np.stack(all_point_cloud, axis=0)
    np.save(f'{dataPath}/all_point_cloud.npy',all_point_cloud)
else:
    all_point_cloud = np.load(f'{dataPath}/all_point_cloud.npy')




def interactive_point_cloud(arr):
    init_fnum = 0
    normalized_matrix = arr[init_fnum,:,:,:]
    # Arrays to change the axis values (replace these with your arrays)
    x_array = (my_vtrig.angle_vec-90)*x_ratio
    y_array = (my_vtrig.angle_vec-90)*y_ratio
    z_array = my_vtrig.dist_vec

    # Create a 3D scatter plot for the original matrix
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Apply a threshold
    mask = normalized_matrix > threshold

    # Get x, y, z indices for the entire matrix
    x, y, z = np.indices(normalized_matrix.shape)

    # Add the entire matrix as blue points with changed axis values
    img = ax.scatter(x_array[x[mask]], y_array[y[mask]], z_array[z[mask]], alpha=0.6, label='All points')

    # Add the top 6 peaks to the plot with changed axis values
    # ax.scatter(x_array[x_peaks], y_array[y_peaks], z_array[z_peaks], c='r', marker='x', s=100, label=f'Top {ntarget} peaks')

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
    ax.grid()

    fig.subplots_adjust(left=0.25, bottom=0.25)
    axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label='frame',
        valmin=0,
        valmax=arr.shape[0]-1,
        valinit=init_fnum,
        valstep=1,
    )
    def update(val):
        img.set_data(my_vtrig.normalization(np.abs(arr[int(frame_slider.val)].T)))
        fig.canvas.draw_idle()
    frame_slider.on_changed(update)
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        frame_slider.reset()

    button.on_clicked(reset)
    plt.show(block=True)

interactive_point_cloud(all_point_cloud)