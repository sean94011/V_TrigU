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
import os

def main(plot=True):
    start = time()
    # Load Data
    # current_case = 'test01242023'
    # current_scenario = 'cf_x_angle_+20'
    current_case = 'test04102023'
    current_scenario = 'human_longer'
    chosen_frame = 50

    # Background Substraction
    point_cloud, axis_value, target_idx = gen_3D_data(
                                                chosen_frame,
                                                current_case,
                                                current_scenario,
                                                threshold=0.99
                                            )
    if plot:
        ground_truth = np.array([[20,0,1]])
        plot_data_path = os.path.join('./data/', current_case, current_scenario,"")
        # point_cloud_plot(axis_value, target_idx, current_scenario, ground_truth) # Threshold Points
        point_cloud_plot(axis_value, target_idx, current_scenario, point_cloud, all_points=True, ground_truth=ground_truth, plot_data_path=plot_data_path) # All Points

    end = time()
    print(end-start, '[s]')

def gen_3D_data(
        chosen_frame,
        current_case,
        current_scenario,
        ntarget = 6, 
        bound = 2.5, 
        Nfft = 512,
        y_offset_shift = 220 ,
        x_offset_shift = -90,
        x_ratio = 20/30,
        y_ratio = 20/25,
        threshold = 0.971,
    ):
    print('Processing...')
    my_vtrig = isens_vtrigU(case=current_case)
    calArr, recArr = my_vtrig.load_data(case=current_case, scenario=current_scenario)
    proArr = my_vtrig.calibration(calArr,recArr,method=0)


    # Parameter Setup
    nframe = recArr.shape[0]
 
    my_vtrig.dist_vec = my_vtrig.compute_dist_vec(Nfft=Nfft)
    # Compute the Range Profile and Find the Target Range Bin (For one target)
    range_profile = my_vtrig.range_pipeline(current_case,current_scenario, plot=False, Nfft=Nfft)[chosen_frame,:]
    range_profile[np.where(my_vtrig.dist_vec>bound)] = np.mean(range_profile)
    range_peaks, _ = find_peaks(range_profile)
    # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
    sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
    top_range_peaks = range_peaks[sorted_peak_indices]
    # Print the indices and angles of the top 6 peaks
    # print(f"Top {ntarget} range bin indices:", top_range_peaks)
    # print(f"Top {ntarget} range bins:", my_vtrig.dist_vec[top_range_peaks], '[m]')
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

    # Example 3D matrix (replace this with your matrix)
    matrix = tof

    # Arrays to change the axis values (replace these with your arrays)
    x_array = (my_vtrig.angle_vec-90)*x_ratio
    y_array = (my_vtrig.angle_vec-90)*y_ratio
    z_array = my_vtrig.dist_vec

    # Channels to search for peaks (replace these with your desired channels)
    channels_to_search = top_range_peaks
    # print(top_range_peaks)

    # Create a masked copy of the matrix
    masked_matrix = matrix.copy()
    masked_matrix[:, :, [ch for ch in range(matrix.shape[2]) if ch not in channels_to_search]] = np.min(matrix)

    # Find the indices of the top 6 largest values
    # flat_indices = np.argsort(masked_matrix.flatten())[::-1][:ntarget]
    # indices_3d = np.unravel_index(flat_indices, matrix.shape)

    # Extract x, y, and z coordinates of the top 6 peaks
    # x_peaks, y_peaks, z_peaks = indices_3d
    # print(x_peaks)
    # print(y_peaks)
    # print(z_peaks)
    # print(x_array[x_peaks])
    # print(y_array[y_peaks])
    # print(z_array[z_peaks])
    # Get x, y, z indices for the entire matrix
    normalized_matrix = my_vtrig.normalization(tof)

    # Apply a threshold
    mask = normalized_matrix > threshold

    # Get x, y, z indices for the entire matrix
    x, y, z = np.indices(matrix.shape)
    # x, y, z = np.indices(matrix.shape)

    return normalized_matrix, [x_array, y_array, z_array], [x[mask].tolist(),y[mask].tolist(),z[mask].tolist()] 

def point_cloud_plot(axis_value, target_idx, scenario, point_cloud=None, all_points=False, ground_truth=None, plot_data_path=None):
    print('Plotting')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_array, y_array, z_array = axis_value
    # Create a colormap to map the normalized values to colors
    # colormap = plt.cm.viridis

    # Add the entire matrix as blue points with changed axis values
    if all_points and point_cloud is not None:
        # Create a colormap to map the normalized values to colors
        colormap = plt.cm.viridis
        x, y, z = np.indices(point_cloud.shape)
        ax.scatter(x_array[x], y_array[y], z_array[z], c=colormap(point_cloud.flatten()), alpha=0.01, label='All points')
    else:
        ax.scatter(x_array[target_idx[0]], y_array[target_idx[1]], z_array[target_idx[2]], alpha=0.6, label='Measured Points')

    # Add Ground Truth
    if ground_truth is not None:
            ax.scatter(x_array[x_array==find_nearest(x_array,ground_truth[:,0])], y_array[target_idx[1][0]], z_array[target_idx[2][0]], c='r', marker='x', label='Ground Truth')

        # n_points = len(ground_truth)
        # for point in range(n_points):
        #     ax.scatter(x_array[x_array==find_nearest(x_array,ground_truth[point,0])], y_array[y_array==find_nearest(y_array,ground_truth[point,0])], z_array[z_array==find_nearest(z_array,ground_truth[point,0])], c='r')

    # Add the top 6 peaks to the plot with changed axis values
    # ax.scatter(x_array[x_peaks], y_array[y_peaks], z_array[z_peaks], c='r', marker='x', s=100, label=f'Top {ntarget} peaks')

    # Set axis labels
    ax.set_xlabel('X (AoD [deg])')
    ax.set_ylabel('Y (AoA [deg])')
    ax.set_zlabel('Z (Range [m])')

    # Set plot title
    if all_points:
        point_display = 'All Points'
    else:
        point_display = 'Threshold Points'
    plt.title(f"3D Point Cloud ({point_display}): {scenario}")

    # Set the axis limits to match the ranges of x_array, y_array, and z_array
    ax.set_xlim(x_array.min(), x_array.max())
    ax.set_ylim(y_array.min(), y_array.max())
    ax.set_zlim(z_array.min(), z_array.max())

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    if plot_data_path is not None:
        print(os.listdir(plot_data_path))
        if 'tmp_plot' not in os.listdir(plot_data_path):
            os.mkdir(os.path.join(plot_data_path,'tmp_plot'))
        plot_data_path = os.path.join(plot_data_path,'tmp_plot',f'Point_Cloud_{scenario}_{point_display}_{time()}.png')
        plt.savefig(plot_data_path)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == '__main__':
     main()