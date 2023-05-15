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

global x_ratio, y_ratio, Nfft
x_ratio = 20/30
y_ratio = 20/25
Nfft = 512
current_case = 'test04102023'
current_scenario = 'cf_y_angle_0'
my_vtrig = isens_vtrigU(case=current_case)

def main(interactive_point_cloud=True):
    # Load Data
    start = time()
    
    # all_point_cloud, all_axis_value, all_target_idx, dataPath = point_cloud_process(
    #                                                                 current_case, 
    #                                                                 current_scenario, 
    #                                                                 save_3d_data=True, 
    #                                                                 force_process=True
    #                                                             )
    all_point_cloud = []
    fpath = os.path.join('./data',current_case,current_scenario,'frames_point_cloud')
    for frame in os.listdir(fpath):
        all_point_cloud.append(np.load(os.path.join(fpath,frame)))
    if interactive_point_cloud:
        # gen_interactive_point_cloud(current_scenario, all_axis_value, all_target_idx, dataPath, save_plot=True)
        gen_interactive_point_cloud(current_scenario, all_point_cloud=all_point_cloud, dataPath=os.path.join('./data',current_case,current_scenario), save_plot=True)

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

def point_cloud_process(current_case, current_scenario, save_3d_data=True, force_process=False):
    my_vtrig = isens_vtrigU(case=current_case)
    _, recArr, dataPath = my_vtrig.load_data(case=current_case, scenario=current_scenario, return_path=True)
    nframe = recArr.shape[0]
    print(nframe)
    tmp_data_path = os.path.join(dataPath,'tmp_data')
    if 'tmp_data' not in os.listdir(dataPath):
        os.mkdir(tmp_data_path)

    
    if force_process or ('axis_value.npy' not in os.listdir(tmp_data_path) or 'all_target_idx.pkl' not in os.listdir(tmp_data_path)):
        print('Processing...')
        all_point_cloud = []
        all_target_idx = []
        pool = mp.Pool()
        for chosen_frame in range(10):
            print(f'Frame: {chosen_frame}')
            output = pool.apply_async(gen_3D_data,(chosen_frame, current_case, current_scenario, 0.98))
            point_cloud, axis_value, target_idx = output.get()
            all_point_cloud.append(point_cloud)
            all_target_idx.append(target_idx)
        pool.close()
        if save_3d_data:
            all_point_cloud = np.stack(all_point_cloud, axis=0)
            np.save(f'{tmp_data_path}/all_point_cloud.npy',all_point_cloud)

        # all_target_idx = np.stack(all_target_idx, axis=0)
        np.save(f'{tmp_data_path}/axis_value.npy',axis_value)
        save_list_pickle(f'{tmp_data_path}/all_target_idx.pkl',all_target_idx)
    else:
        print('All Files Exist, Loading...')
        all_point_cloud = np.load(f'{tmp_data_path}/all_point_cloud.npy')
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
    frame_slider = Slider(
        ax=axframe,
        label='frame',
        valmin=0,
        valmax=len(all_point_cloud)-1,
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
        
    frame_slider.on_changed(update_frame)
    thres_slider.on_changed(update_thres)
    resetax = fig.add_axes([0.8, 0.005, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        frame_slider.reset()
        thres_slider.reset()

    button.on_clicked(reset)
    plt.show(block=True)

if __name__ == '__main__':
     main()