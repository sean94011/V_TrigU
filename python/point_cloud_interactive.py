# Load Library
import os
from time import time

import numpy as np
from isens_vtrigU import isens_vtrigU
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.constants import c
from point_cloud import gen_3D_data
import pickle

def main(interactive_point_cloud=True):
    # Load Data
    start = time()
    current_case = 'test04102023'
    current_scenario = 'human_longer'
    all_point_cloud, all_axis_value, all_target_idx, dataPath = point_cloud_process(
                                                                    current_case, 
                                                                    current_scenario, 
                                                                    save_3d_data=True, 
                                                                    force_process=False
                                                                )
    if interactive_point_cloud:
        gen_interactive_point_cloud(current_scenario, all_axis_value, all_target_idx, dataPath, save_plot=True)

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
    recArr, _, dataPath = my_vtrig.load_data(case=current_case, scenario=current_scenario, return_path=True)
    nframe = recArr.shape[0]
    tmp_data_path = os.path.join(dataPath,'tmp_data')
    if 'tmp_data' not in os.listdir(dataPath):
        os.mkdir(tmp_data_path)

    
    if force_process or ('axis_value.npy' not in os.listdir(tmp_data_path) or 'all_target_idx.pkl' not in os.listdir(tmp_data_path)):
        all_point_cloud = []
        all_target_idx = []
        for chosen_frame in range(nframe):
            point_cloud, axis_value, target_idx = gen_3D_data(
                                                    chosen_frame,
                                                    current_case,
                                                    current_scenario,
                                                )
            all_point_cloud.append(point_cloud)
            all_target_idx.append(target_idx)

        if save_3d_data:
            all_point_cloud = np.stack(all_point_cloud, axis=0)
            np.save(f'{tmp_data_path}/all_point_cloud.npy',all_point_cloud)

        # all_target_idx = np.stack(all_target_idx, axis=0)
        np.save(f'{tmp_data_path}/axis_value.npy',axis_value)
        save_list_pickle(f'{tmp_data_path}/all_target_idx.pkl',all_target_idx)
    else:
        all_point_cloud = np.load(f'{tmp_data_path}/all_point_cloud.npy')
        axis_value = np.load(f'{tmp_data_path}/axis_value.npy')
        all_target_idx = load_list_pickle(f'{tmp_data_path}/all_target_idx.pkl')

    return all_point_cloud, axis_value, all_target_idx, dataPath


def gen_interactive_point_cloud(scenario, axis_value, all_target_idx, dataPath=None, save_plot=False):
    init_frame = 0
    target_idx = all_target_idx[init_frame]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_array, y_array, z_array = axis_value
    # Create a colormap to map the normalized values to colors
    # colormap = plt.cm.viridis

    # Add the entire matrix as blue points with changed axis values
    img = ax.scatter(x_array[target_idx[0]], y_array[target_idx[1]], z_array[target_idx[2]], alpha=0.6, label='All points')

    # Set axis labels
    ax.set_xlabel('X (AoD [deg])')
    ax.set_ylabel('Y (AoA [deg])')
    ax.set_zlabel('Z (Range [m])')

    # Set plot title
    plt.title(f"Interactive 3D Point Cloud: {scenario}")

    # Set the axis limits to match the ranges of x_array, y_array, and z_array
    ax.set_xlim(x_array.min(), x_array.max())
    ax.set_ylim(y_array.min(), y_array.max())
    ax.set_zlim(z_array.min(), z_array.max())

    # Add legend
    ax.legend()
    ax.grid()
    tmp_plot_path = os.path.join(dataPath,'tmp_plot')
    if 'tmp_plot' not in os.listdir(dataPath):
        os.mkdir(tmp_plot_path)
    tmp_plot_fname = f'tmp_plot_{time()}.png'
    plt.savefig(os.path.join(tmp_plot_path,tmp_plot_fname))
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label='frame',
        valmin=0,
        valmax=len(target_idx[0])-1,
        valinit=init_frame,
        valstep=1,
    )
    def update(val):
        target_idx = all_target_idx[val]
        img.set_data((x_array[target_idx[0]], y_array[target_idx[1]], z_array[target_idx[2]]))
        fig.canvas.draw_idle()
    frame_slider.on_changed(update)
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        frame_slider.reset()

    button.on_clicked(reset)
    plt.show(block=True)

if __name__ == '__main__':
     main()