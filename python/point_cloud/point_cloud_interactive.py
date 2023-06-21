# Load Library
import os
from time import time

import numpy as np
from isens_vtrigU import isens_vtrigU
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.constants import c
# from point_cloud import gen_3D_data
import pickle
import multiprocessing as mp
from scipy.signal import find_peaks


global x_ratio, y_ratio, Nfft
x_ratio = 20/30
y_ratio = 20/25
Nfft = 64
# current_data_folder = '/Volumes/SCKY/V_TrigU/python/data'
current_data_folder = './data'
current_case = 'test04242023'
current_scenario = '2cf_yy_angle_+-20_3'
my_vtrig = isens_vtrigU(case=current_case, data_folder=current_data_folder)
nframe = (my_vtrig.load_data(case=current_case,scenario=current_scenario)[1]).shape[0]
chosen_frame = 50
threshold = 0.98

def main(interactive_point_cloud=True):
    # Load Data
    start = time()
    processed = False
    print('Data Path: ', os.path.join(current_data_folder,current_case,current_scenario))
    if not processed:
        all_point_cloud, all_axis_value, all_target_idx, dataPath = point_cloud_process(
                                                                        current_case, 
                                                                        current_scenario, 
                                                                        current_data_folder=current_data_folder,
                                                                        save_3d_data=False,
                                                                        force_process=True
                                                                    )
    else:
        all_point_cloud = []
        fpath = os.path.join(current_data_folder,current_case,current_scenario,f'frames_point_cloud_NFFT={Nfft}')
    only_thres = True
    # if only_thres:
    #     all_point_cloud = [np.load(os.path.join(fpath,f'Point_Cloud:frame={chosen_frame}.npy')),]
    # else:
    #     for frame in range(nframe):
    #         fname = f'Point_Cloud:frame={frame}.npy'
    #         all_point_cloud.append(np.load(os.path.join(fpath,fname)))
    if interactive_point_cloud:
        gen_interactive_point_cloud(current_scenario,axis_value=all_axis_value,all_target_idx=all_target_idx,dataPath=dataPath, save_plot=False)
        # gen_interactive_point_cloud(current_scenario, all_point_cloud=all_point_cloud, dataPath=os.path.join(current_data_folder,current_case,current_scenario), save_plot=True)

    end = time()
    print(end-start, '[s]')

# Save the list of lists to a Pickle file
def save_list_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Load the list of lists from a Pickle file
def load_list_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def point_cloud_process(current_case, current_scenario, current_data_folder, save_3d_data=True, force_process=False):
    my_vtrig = isens_vtrigU(case=current_case, data_folder=current_data_folder)
    _, recArr, dataPath = my_vtrig.load_data(case=current_case, scenario=current_scenario, return_path=True)
    nframe = recArr.shape[0]
    print(nframe)
    tmp_data_path = os.path.join(dataPath,f'frames_point_cloud_NFFT={Nfft}')
    if f'frames_point_cloud_NFFT={Nfft}' not in os.listdir(dataPath):
        os.mkdir(tmp_data_path)

    
    if force_process or ('axis_value.npy' not in os.listdir(tmp_data_path) or 'all_target_idx.pkl' not in os.listdir(tmp_data_path)):
        print('Processing...')
        all_point_cloud = []
        all_target_idx = []
        pool = mp.Pool()
        for chosen_frame in range(nframe):
            print(f'Frame: {chosen_frame}')
            if save_3d_data:
                if f'point_cloud_frame_{chosen_frame}.npy' in os.listdir(tmp_data_path):
                    print('Processed, proceed to next frame...')
                    continue
            output = pool.apply_async(gen_3D_data,(chosen_frame, current_case, current_scenario, current_data_folder, 0.98))
            point_cloud, axis_value, target_idx = output.get()
            if save_3d_data:
                np.save(f'{tmp_data_path}/point_cloud_frame_{chosen_frame}.npy',point_cloud)
            all_target_idx.append(target_idx)
        pool.close()
        if save_3d_data:
            for chosen_frame in range(nframe):
                all_point_cloud.append(np.load(f'{tmp_data_path}/point_cloud_frame_{chosen_frame}.npy'))
            all_point_cloud = np.stack(all_point_cloud, axis=0)
            np.save(f'{tmp_data_path}/all_point_cloud.npy',all_point_cloud)

        # all_target_idx = np.stack(all_target_idx, axis=0)
        np.save(f'{tmp_data_path}/axis_value.npy',axis_value)
        save_list_pickle(f'{tmp_data_path}/all_target_idx.pkl',all_target_idx)
    else:
        print('All Files Exist, Loading...')
        # all_point_cloud = np.load(f'{tmp_data_path}/all_point_cloud.npy')
        axis_value = np.load(f'{tmp_data_path}/axis_value.npy')
        all_target_idx = load_list_pickle(f'{tmp_data_path}/all_target_idx.pkl')
    print(np.array(all_point_cloud).shape)
    return all_point_cloud, axis_value, all_target_idx, dataPath


def gen_interactive_point_cloud(scenario, all_point_cloud=None, axis_value=None, all_target_idx=None, dataPath=None, save_plot=False):
    init_frame = 0
    init_thres = 0.97
    current_frame = init_frame
    current_thres = init_thres
    if all_target_idx is not None:
        target_idx = all_target_idx[init_frame]
    else:
        if all_point_cloud is not None:
            point_cloud = all_point_cloud[init_frame]
        # Apply a threshold
        mask = point_cloud > init_thres

        # Get x, y, z indices for the entire matrix
        x, y, z = np.indices(point_cloud.shape)

        target_idx = [x[mask].tolist(),y[mask].tolist(),z[mask].tolist()] 

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(scenario)

    if axis_value is not None:
        x_array, y_array, z_array = axis_value
    else:
        x_array = (np.linspace(0,180,Nfft)-90)*x_ratio
        y_array = (np.linspace(0,180,Nfft)-90)*y_ratio
        z_array = my_vtrig.compute_dist_vec(Nfft=Nfft)
    # Create a colormap to map the normalized values to colors
    # colormap = plt.cm.viridis

    # Add the entire matrix as blue points with changed axis values
    def plot_points(target_idx):
        # Clear all points
        ax.clear()

        # Add new points with changed axis values
        ax.scatter(x_array[target_idx[0]], y_array[target_idx[1]], z_array[target_idx[2]], alpha=0.6, label='All points')

        # Set axis labels and limits
        ax.set_xlabel('X (AoD [deg])')
        ax.set_ylabel('Y (AoA [deg])')
        ax.set_zlabel('Z (Range [m])')
        ax.set_xlim(x_array.min(), x_array.max())
        ax.set_ylim(y_array.min(), y_array.max())
        ax.set_zlim(z_array.min(), z_array.max())

        # Add legend and grid
        ax.legend()
        ax.grid()
    # # Add the entire matrix as blue points with changed axis values
    # img = ax.scatter(x_array[target_idx[0]], y_array[target_idx[1]], z_array[target_idx[2]], alpha=0.6, label='All points')

    # # Set axis labels
    # ax.set_xlabel('X (AoD [deg])')
    # ax.set_ylabel('Y (AoA [deg])')
    # ax.set_zlabel('Z (Range [m])')

    # Set plot title
    plt.title(f"Interactive 3D Point Cloud: {scenario}")

    # # Set the axis limits to match the ranges of x_array, y_array, and z_array
    # ax.set_xlim(x_array.min(), x_array.max())
    # ax.set_ylim(y_array.min(), y_array.max())
    # ax.set_zlim(z_array.min(), z_array.max())

    # # Add legend
    # ax.legend()
    # ax.grid()

    # Plot initial points
    plot_points(target_idx)



    tmp_plot_path = os.path.join(dataPath,'tmp_plot')
    if 'tmp_plot' not in os.listdir(dataPath):
        os.mkdir(tmp_plot_path)
    tmp_plot_fname = f'tmp_plot_{time()}.png'
    plt.savefig(os.path.join(tmp_plot_path,tmp_plot_fname))
    fig.subplots_adjust(left=0.25, bottom=0.3)
    axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axthres = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    if (all_point_cloud is not None and len(all_point_cloud) != 1) or all_target_idx is not None:
        if all_point_cloud is not None:
            val_max = len(all_point_cloud) - 1
        else:
            val_max = len(all_target_idx) - 1
        frame_slider = Slider(
            ax=axframe,
            label='frame',
            valmin=0,
            valmax=val_max,
            valinit=init_frame,
            valstep=1,
        )
    thres_slider = Slider(
        ax=axthres,
        label='threshold',
        valmin=0.970,
        valmax=0.999,
        valinit=init_thres,
        valstep=0.001,
    )
    def update_frame(val):
        current_frame = int(val)
        if all_target_idx is not None:
            target_idx = all_target_idx[current_frame]
        else:
            if all_point_cloud is not None:
                point_cloud = all_point_cloud[current_frame]
            # Apply a threshold
            mask = point_cloud > current_thres

            # Get x, y, z indices for the entire matrix
            x, y, z = np.indices(point_cloud.shape)

            target_idx = [x[mask].tolist(),y[mask].tolist(),z[mask].tolist()] 
        plot_points(target_idx)
        fig.canvas.draw_idle()

    def update_thres(val):
        current_thres = val
        if all_target_idx is not None:
            target_idx = all_target_idx[current_frame]
        else:
            if all_point_cloud is not None:
                point_cloud = all_point_cloud[current_frame]
            # Apply a threshold
            mask = point_cloud > current_thres

            # Get x, y, z indices for the entire matrix
            x, y, z = np.indices(point_cloud.shape)

            target_idx = [x[mask].tolist(),y[mask].tolist(),z[mask].tolist()] 
        plot_points(target_idx)
        fig.canvas.draw_idle()
    if (all_point_cloud is not None and len(all_point_cloud) != 1) or all_target_idx is not None:    
        frame_slider.on_changed(update_frame)
    thres_slider.on_changed(update_thres)
    resetax = fig.add_axes([0.8, 0.005, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        frame_slider.reset()
        thres_slider.reset()

    button.on_clicked(reset)
    plt.show(block=True)

def gen_3D_data(
        chosen_frame,
        current_case,
        current_scenario,
        current_data_folder='./data',
        threshold = 0.971,
        ntarget = 6, 
        bound = 2.5, 
        Nfft = 64,
        y_offset_shift = 220 ,
        x_offset_shift = -90,
        x_ratio = 20/30,
        y_ratio = 20/25,
    ):
    print(f'Processing Frame {chosen_frame}...')
    my_vtrig = isens_vtrigU(data_folder=current_data_folder, case=current_case)
    calArr, recArr = my_vtrig.load_data(data_folder=current_data_folder, case=current_case, scenario=current_scenario)
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

    return [normalized_matrix], [x_array, y_array, z_array], [x[mask].tolist(),y[mask].tolist(),z[mask].tolist()] 

if __name__ == '__main__':
     main()