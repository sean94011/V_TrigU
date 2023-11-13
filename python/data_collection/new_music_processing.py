# Import Libraries
import os
import time
from math import ceil, log

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from parameter_setup import normalization
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from pyargus.directionEstimation import *


# Main Function
def main():
    collected_data_folder = '../../../../../Library/CloudStorage/Box-Box/Vayyar Radar/vitalsign1019' 
    for data in sorted(os.listdir(collected_data_folder)):

        data_folder = os.path.join(collected_data_folder,data)

        if data[-10:] == 'background':
            # Compute calibration frame
            cal_frame = np.load(os.path.join(data_folder,'recording.npy'))
            cal_frame = np.mean(cal_frame,axis=0)
            continue

        if '+' in data:
            continue

        # load data & parameters
        data_arr = np.load(os.path.join(data_folder,'recording.npy'))
        params = np.load(os.path.join(data_folder, "config.npy"), allow_pickle=True).item()

        # Setup data path
        cur_folder_name = 'Infant_Only'
        if cur_folder_name not in os.listdir('./processed_data_plots'):
            os.mkdir(f'./processed_data_plots/{cur_folder_name}')
        cur_fig_name = f'./processed_data_plots/{cur_folder_name}/{data}.jpg'

        # Record the starting time
        start = time.time()

        # Extract target frame
        cur_frame = 100
        rec_arr = data_arr[cur_frame]#np.mean(np.array(data_arr), axis=0)






















        # Setup Clustering Parameters
        vmin = 0.0#6
        ntarget = 1
        thres=0.9
        eps=10
        n=10

        # Setup bound for eliminating DC offset
        upper_bound = 2.8
        lower_bound = 0.2

        # Compute FFT Parameters
        n_rx = 20
        n_tx = rec_arr.shape[0] // n_rx
        params['range_Nfft'] = 2**(ceil(log(data_arr.shape[-1],2))+1)
        params['angle_Nfft'] = [2**(ceil(log(n_tx,2))+1),2**(ceil(log(n_rx,2))+1)]
        
        
        # Perform Calbration
        pro_arr = rec_arr - cal_frame


        # Compute Range Profiles
        range_profile = np.abs(np.fft.ifft(pro_arr, n=params['range_Nfft'], axis=1))[110] # w/ calibration
        range_profile[np.where(params['dist_vec']>upper_bound)]=np.min(range_profile)
        range_profile[np.where(params['dist_vec']<lower_bound)]=np.min(range_profile)

        raw_range_profile = np.abs(np.fft.ifft(rec_arr, n=params['range_Nfft'], axis=1))[110] # w/o calibration
        raw_range_profile[np.where(params['dist_vec']>upper_bound)]=np.mean(raw_range_profile)
        
        cal_range_profile = np.abs(np.fft.ifft(cal_frame, n=params['range_Nfft'], axis=1))[110]
        cal_range_profile[np.where(params['dist_vec']>upper_bound)]=np.mean(cal_range_profile)

        # Compute 3D FFT Profile
        pro_arr_3D = pro_arr.reshape(-1,n_rx,150)
        pro_arr_3D = np.fft.ifft(pro_arr_3D, n=params['range_Nfft'], axis=2)
        pro_arr_3D[:,:,np.where(params['dist_vec']>upper_bound)]=np.mean(pro_arr_3D)
        pro_arr_3D[:,:,np.where(params['dist_vec']<lower_bound)]=np.mean(pro_arr_3D)
        pro_arr_3D = np.fft.fft2(pro_arr_3D, s=params['angle_Nfft'], axes=[0,1])

        target_range_bin = np.argsort(range_profile)[::-1][:ntarget]#params['dist_vec'][np.argmax(range_profile)]
        range_bins_within_bounds = np.where(np.logical_and(params['dist_vec']<=upper_bound, params['dist_vec']>=lower_bound))
        target_range_bin = np.intersect1d(target_range_bin,range_bins_within_bounds)
        print(params['dist_vec'][target_range_bin])

        # Define the coordinate values as per your actual data
        params['AoD_vec'] = np.linspace(-90,90,params['angle_Nfft'][0])
        params['AoA_vec'] = np.linspace(-90,90,params['angle_Nfft'][1])
        theta_values = params['AoD_vec']
        phi_values = params['AoA_vec']
        r_values = params['dist_vec']  # replace with actual range ifhbgn different

        theta, phi, r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        x, y, z = spherical_to_rectangular(r, theta, phi)

        # Interpolate data to rectangular coordinates.
        print(theta_values.shape)
        print(phi_values.shape)
        print(r_values.shape)
        print(pro_arr_3D.shape)
        interpolator = RegularGridInterpolator((theta_values, phi_values, r_values), pro_arr_3D)

        # Create a grid in spherical coordinates
        grid_theta, grid_phi, grid_r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        # Convert the grid to Cartesian coordinates
        grid_x, grid_y, grid_z = spherical_to_rectangular(grid_r, grid_theta, grid_phi)

        rect_data = interpolator((grid_theta, grid_phi, grid_r))

        # Project along the xy, xz, and yz planes
        xy_projection = np.linalg.norm(rect_data[:,:,target_range_bin], axis=2)#np.linalg.norm()#np.abs(rect_data[:,:,target_range_bin].mean(axis=2))
        xz_projection = np.linalg.norm(rect_data, axis=1)#np.abs(rect_data.mean(axis=1))
        yz_projection = np.linalg.norm(rect_data, axis=0)#np.abs(rect_data.mean(axis=0))
        # xy_projection = np.sum(np.abs(rect_data[:,:,target_range_bin]), axis=2)#np.abs(rect_data[:,:,target_range_bin].mean(axis=2))
        # xz_projection = np.abs(np.sum(rect_data, axis=1))#np.abs(rect_data.mean(axis=1))
        # yz_projection = np.abs(np.sum(rect_data, axis=0))#np.abs(rect_data.mean(axis=0))
        # params['x_offset_shift'] = 0
        # params['y_offset_shift'] = 0
        params['y_offset_shift'] = -11
        params['x_offset_shift'] = 27
        xy_projection = np.roll(xy_projection,shift=params['y_offset_shift'],axis=1)
        xy_projection = np.roll(xy_projection,shift=params['x_offset_shift'],axis=0)
        xz_projection = np.roll(xz_projection,shift=params['x_offset_shift'],axis=0)[:,np.where(np.logical_and(params['dist_vec']<=upper_bound, params['dist_vec']>=lower_bound))].squeeze()
        yz_projection = np.roll(yz_projection,shift=params['y_offset_shift'],axis=0)[:,np.where(np.logical_and(params['dist_vec']<=upper_bound, params['dist_vec']>=lower_bound))].squeeze()

        # peak_indices = filter_and_cluster(xy_projection, threshold=0.02, n=4)
        x_axis = np.linspace(grid_x.min(), grid_x.max(), params['angle_Nfft'][0])
        y_axis = np.linspace(grid_y.min(), grid_y.max(), params['angle_Nfft'][1])
        z_axis = np.linspace(grid_z.min(), grid_z.max(), params['range_Nfft'])

        n_occupants, occupancy_map, peak_indices = count_occupants(xy_projection, grid_x, grid_y, thres=thres, eps=eps, n=n)
        seat_for_title = []
        for i in range(len(occupancy_map)):
            if occupancy_map[i]:
                seat_for_title.append(f'#{i+1}')
        seat_for_title = ', '.join(seat_for_title)
        fig = plt.figure(figsize=(12,10))
        fig.suptitle(f'{data[:-4]}\nAntennas: {n_tx}x{n_rx}\nDetection Result: {n_occupants} people in the vehicle | Location: Seat {seat_for_title}')

        plt.subplot(2,2,2)
        plt.title('XY Perspective')
        extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        plt.imshow((xy_projection).T,origin='lower',aspect='auto', extent=extent, vmin=vmin, interpolation='nearest')
        plt.scatter(x_axis[peak_indices[:,0]], y_axis[peak_indices[:,1]], color='r')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.grid()
        plt.subplot(2,2,1)
        plt.title('Range Profile')

        # range_profile[top_range_peaks] = range_profile[top_range_peaks]*10
        plt.plot(params['dist_vec'],range_profile,label='w/ background subtraction')
        # plt.plot(params['dist_vec'],cal_range_profile,label='background')
        # plt.plot(params['dist_vec'],raw_range_profile,label='w/o background subtraction')
        plt.xlabel('Range [m]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend()
        plt.subplot(2,2,4)
        plt.title('YZ Perspective')
        extent = [grid_y.min(), grid_y.max(), lower_bound, upper_bound]
        plt.imshow(yz_projection.T,origin='lower',aspect='auto', extent=extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Y [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        plt.subplot(2,2,3)
        plt.title('XZ Perspective')
        extent = [grid_x.min(), grid_x.max(), lower_bound, upper_bound]
        plt.imshow((xz_projection).T,origin='lower',aspect='auto', extent=extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        plt.savefig(cur_fig_name)
        # plt.show()
        plt.close
        print('Occupancy Detection Frame Duration: ',time.time()-start, '[s]')
        # break

def compute_tof_scanning_vector(freq, time_vec):
    # Compute the scanning vector
    df = freq[1] - freq[0]
    freq_shift = np.arange(0,len(freq))
    freq_shift =  df*freq_shift
    scanning_vectors = np.zeros((len(freq),len(time_vec)),dtype='complex')
    for i in range(len(time_vec)):
        scanning_vectors[:,i] = np.exp(-1j*2*np.pi*freq_shift*time_vec[i])
    # scanning_vectors = np.real(scanning_vectors)
    # scanning_vectors = np.exp(1j*2*np.pi*time_vec).reshape(-1,1)
    return scanning_vectors


def compute_tof_music(test_signal, scanning_vectors, dist_vec, signal_dimension=3, plot=False):
    # test_signal = test_signal.reshape(-1,1)

    # Compute the correlation matrix
    # R = np.dot(test_signal, test_signal.conj().T)
    P = 150
    R = spatial_smoothing(test_signal, P=P, direction="forward-backward")

    # Implement MUSIC for ToF
    tof = np.zeros(np.size(scanning_vectors, 1))
    M = np.size(R, 0)

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    # Sorting    
    eig_array = []
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0], reverse=False)

    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension    
    E = np.zeros((M,noise_dimension),dtype=complex)
    for i in range(noise_dimension):     
        E[:,i] = eig_array[i][1]     
        
    E = np.matrix(E)    
    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):             
        S_theta_ = scanning_vectors[:, i]
        S_theta_  = np.matrix(S_theta_).getT() 
        tof[theta_index]=  1/np.abs(S_theta_.getH()*(E*E.getH())*S_theta_)
        theta_index += 1

    log_scale_min = -50

    tof = np.divide(np.abs(tof),np.max(np.abs(tof))) # normalization 
    tof = 10*np.log10(tof)                
    time_index = 0        
    for t in time_vec:                    
        if tof[time_index] < log_scale_min:
            tof[time_index] = log_scale_min
        time_index += 1   
    # tof = normalization(np.abs(tof))
    tof = normalization(tof-np.min(tof))
    if plot:
        plt.figure()
        axes = plt.axes()
        axes.plot(dist_vec,tof)    
        axes.set_title('ToF MUSIC estimation ',fontsize = 16)
        axes.set_xlabel('Range [m]')
        axes.set_ylabel('Amplitude [dB]')   
        axes.legend(["MUSIC"])
    # print(tof)
    return tof

if __name__ == '__main__':
    main()