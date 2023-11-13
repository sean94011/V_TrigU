# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from parameter_setup import normalization
import time
from PIL import Image
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FuncFormatter
from math import ceil, log

def main():
    # Choose the data & frame
    # num_passenger = 1
    # seat_combination = 1
    # used_rbw = 10
    # data_folder = f'./collected_data/{num_passenger}p_seat_{seat_combination}_rbw_{used_rbw}_06-26-2023--19-10-43_1687824643607988500/'
    collected_data_folder = '../../../../../Library/CloudStorage/Box-Box/Vayyar Radar/vitalsign1019' 
    # collected_data_folder = './collected_data'
    for data in sorted(os.listdir(collected_data_folder)):
        # n_target = 2
        # if data[0] != f'{n_target}':
        #     continue
        # if data[15+n_target-1:17+n_target-1] == 'cf':
        #     continue
        # if data[15+n_target-1:17+n_target-1] == 'mp':
        #     continue
        # cur_rbw = '10'
        # if cur_rbw == '10':
        #     cur_rbw = '10_'
        # if data[14+n_target-1:17+n_target-1] != cur_rbw:
        #     continue
        # if data[18:20] == '1b':
        #     continue
        if '+' in data:
            continue
        file_name = data
        # plt.title(f'Experiment: {data}')
        cur_frame = 100

        data_folder = os.path.join(collected_data_folder,data)

        # Setup the data path
        # data_queue_folder = os.path.join(data_folder, 'data_queue')
        # data_queue = sorted(os.listdir(data_queue_folder)) 
        data_arr = np.load(os.path.join(data_folder,'recording.npy'))
        # load parameters
        params = np.load(os.path.join(data_folder, "config.npy"), allow_pickle=True).item()#load_params(data_folder)
        print(params.keys())
        params['range_Nfft'] = 2**(ceil(log(len(data_arr),2))+1)
        vmin = 0.0#6
        ntarget = 3
        thres=0.6
        eps=10
        n=2
        enhance_rate = 1
        bound = 3
        n_rx = 20
        # time_window_len = 5
        # if time_window_len < cal_frame_len:
        #     time_window_len = cal_frame_len
        choose_center = False
        if choose_center:
            ants_loc = 'center'
        else:
            ants_loc = 'front'
        cur_folder_name = 'Infant_Only'#f'{n_target}p_with_cf_{n_rx}x{n_rx}_antenna_{ants_loc}'
        if cur_folder_name not in os.listdir('./processed_data_plots'):
            os.mkdir(f'./processed_data_plots/{cur_folder_name}')
        cur_fig_name = f'./processed_data_plots/{cur_folder_name}/{file_name}.jpg'
        # Record the starting time
        start = time.time()
        # cal_arr = params['cal_arr']
        # cal_frame = np.fft.ifft(cal_arr, axis=2, n=params['range_Nfft'])
        # cal_frame = np.mean(cal_frame, axis=0)
        # cal_arr = cal_frame[len(cal_frame)//2]#np.mean(cal_frame, axis=0)
        # data_arr = []
        # for cur_frame_data in data_queue:
        #     # cur_frame_data = data # data_queue[cur_frame]
        #     data_arr.append(np.load(os.path.join(data_queue_folder,cur_frame_data)))
        rec_arr = data_arr[cur_frame]#np.mean(np.array(data_arr), axis=0)
        n_tx = rec_arr.shape[0] // n_rx
        params['angle_Nfft'] = [2**(ceil(log(n_tx,2))+1),2**(ceil(log(n_rx,2))+1)]
        # data_arr = np.stack(data_arr,axis=0)
        # cal_frame = data_arr[cur_frame-time_window_len:cur_frame-time_window_len+cal_frame_len]
        # cal_frame = np.fft.ifft(cal_frame, n=params['range_Nfft'], axis=2)
        # cal_frame = np.mean(cal_frame, axis=0)
        cal_frame_len = 5
        cal_frame = np.mean(data_arr[:cal_frame_len,:,:], axis=0)
        doppler_cal_frame = data_arr[0:cal_frame_len]
        doppler_cal_frame = np.fft.ifft(doppler_cal_frame, n=params['range_Nfft'], axis=2)
        doppler_cal_frame = np.mean(doppler_cal_frame,axis=0)
        
        
        # rec_arr = np.load(os.path.join(data_queue_folder,data_queue[cur_frame]))

        # cur_frame_data = data_queue[cur_frame]
        # rec_arr = np.load(os.path.join(data_queue_folder,cur_frame_data))
        print(rec_arr.shape, cal_frame.shape)
        pro_arr = rec_arr - cal_frame
        print(pro_arr.shape)

        # Extract Certain Number of antennas
        
        if choose_center:
            center_antenna = 10
            start_antenna = center_antenna - n_rx//2
            end_antenna = center_antenna + n_rx//2
        else:
            start_antenna = 0
            end_antenna = n_rx
        # pro_arr = pro_arr.reshape(20,20,-1)
        # pro_arr = pro_arr[start_antenna:end_antenna,start_antenna:end_antenna,:]
        # pro_arr = pro_arr.reshape(-1,150)


        range_profile = np.linalg.norm(pro_arr,axis=0)
        # # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
        # range_peaks, _ = find_peaks(range_profile)
        # sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
        # top_range_peaks = range_peaks[sorted_peak_indices]
        # # load 
        range_profile = np.linalg.norm(np.fft.ifft(pro_arr, n=512, axis=1),axis=0)
        # range_profile[np.where(params['dist_vec']>bound)]=np.min(np.abs(range_profile))
        raw_range_profile = np.linalg.norm(np.fft.ifft(rec_arr, n=512, axis=1),axis=0)
        # raw_range_profile[np.where(params['dist_vec']>bound)]=np.min(np.abs(raw_range_profile))
        cal_range_profile = np.linalg.norm(np.fft.ifft(cal_frame, n=512, axis=1),axis=0)
        # cal_range_profile[np.where(params['dist_vec']>bound)]=np.min(np.abs(cal_range_profile))

        pro_arr_3D = pro_arr.reshape(-1,n_rx,150)#[chosen_frame,:,:,:]
        pro_arr_3D = np.fft.ifft(pro_arr_3D, n=params['range_Nfft'], axis=2)
        # pro_arr_3D[:,:,np.where(params['dist_vec']>bound)]=np.min(np.abs(pro_arr_3D))

        
        # pro_arr_3D = np.roll(pro_arr_3D,shift=params['y_offset_shift'],axis=1)
        # pro_arr_3D = np.roll(pro_arr_3D,shift=params['x_offset_shift'],axis=0)

        # # pro_arr_3D[:,:,top_range_peaks] = pro_arr_3D[:,:,top_range_peaks]*enhance_rate


        pro_arr_3D = np.fft.fft2(pro_arr_3D, s=params['angle_Nfft'], axes=[0,1])

        target_range_bin = np.argsort(range_profile)[::-1][:ntarget]#params['dist_vec'][np.argmax(range_profile)]
        print(params['dist_vec'][target_range_bin])

        # Define the coordinate values as per your actual data
        params['AoD_vec'] = np.linspace(0,180,params['angle_Nfft'][0])
        params['AoA_vec'] = np.linspace(0,180,params['angle_Nfft'][1])
        theta_values = params['AoD_vec']
        phi_values = params['AoA_vec']
        r_values = params['dist_vec']  # replace with actual range if different

        theta, phi, r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        x, y, z = spherical_to_rectangular(r, theta, phi)

        # Interpolate data to rectangular coordinates.
        interpolator = RegularGridInterpolator((theta_values, phi_values, r_values), pro_arr_3D)

        # Create a grid in spherical coordinates
        grid_theta, grid_phi, grid_r = np.meshgrid(theta_values, phi_values, r_values, indexing='ij')

        # Convert the grid to Cartesian coordinates
        grid_x, grid_y, grid_z = spherical_to_rectangular(grid_r, grid_theta, grid_phi)
        print(np.min(grid_y))

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
        xz_projection = np.roll(xz_projection,shift=params['x_offset_shift'],axis=0)[:,np.where(params['dist_vec']<=bound)].squeeze()
        yz_projection = np.roll(yz_projection,shift=params['y_offset_shift'],axis=0)[:,np.where(params['dist_vec']<=bound)].squeeze()

        # peak_indices = filter_and_cluster(xy_projection, threshold=0.02, n=4)
        x_axis = np.linspace(grid_x.min(), grid_x.max(), 64)
        y_axis = np.linspace(grid_y.min(), grid_y.max(), 64)
        z_axis = np.linspace(grid_z.min(), grid_z.max(), 512)

        n_occupants, occupancy_map, peak_indices = count_occupants(xy_projection, grid_x, grid_y, thres=thres, eps=eps, n=n)
        seat_for_title = []
        for i in range(len(occupancy_map)):
            if occupancy_map[i]:
                seat_for_title.append(f'#{i+1}')
        seat_for_title = ', '.join(seat_for_title)
        fig = plt.figure(figsize=(12,10))
        fig.suptitle(f'{file_name[:-4]}\nAntennas: {n_tx}x{n_rx}\nDetection Result: {n_occupants} people in the vehicle | Location: Seat {seat_for_title}')

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
        # plt.plot(params['dist_vec'],np.linalg.norm(pro_arr_3D.reshape(-1,params['range_Nfft']),axis=0),label='w/ background subtraction')
        plt.xlabel('Range [m]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend()
        plt.subplot(2,2,4)
        plt.title('YZ Perspective')
        extent = [grid_y.min(), grid_y.max(), grid_z.min(), bound]
        plt.imshow(yz_projection.T,origin='lower',aspect='auto', extent=extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Y [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        plt.subplot(2,2,3)
        plt.title('XZ Perspective')
        extent = [grid_x.min(), grid_x.max(), grid_z.min(), bound]
        plt.imshow((xz_projection).T,origin='lower',aspect='auto', extent=extent, vmin=vmin, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.grid()
        plt.show()
        plt.savefig(cur_fig_name)
        plt.close()
        print('Occupancy Detection Frame Duration: ',time.time()-start, '[s]')
        break
        # Doppler Extraction
        # doppler_arr = np.linalg.norm(data_arr,axis=1)
        doppler_arr = np.fft.ifft(data_arr,n=params['range_Nfft'],axis=2)
        doppler_arr = doppler_arr - doppler_cal_frame[np.newaxis,:,:]
        params['doppler_Nfft'] = 2**(ceil(log(doppler_arr.shape[0],2))+1)
        start = time.time()
        # doppler_arr[:,:,np.where(params['dist_vec']>bound)]=np.min(np.abs(doppler_arr))
        # print(time.time()-start)
        doppler_arr = doppler_arr.reshape(-1,20,20,params['range_Nfft'])
        print(time.time()-start)
        # doppler_arr = np.fft.fft(doppler_arr,n=params['doppler_Nfft'],axis=0)
        doppler_arr = np.fft.fft2(doppler_arr, s=params['angle_Nfft'], axes=[1,2])
        print(time.time()-start)
        doppler_arr = np.abs(doppler_arr)
        print(time.time()-start)
        doppler_arr = np.fft.fft(doppler_arr,n=params['doppler_Nfft'],axis=0)
        print(time.time()-start)
        # print(doppler_arr.shape)
        # doppler_arr = np.mean(doppler_arr,axis=3)
        # print(time.time()-start)
        # print(doppler_arr.shape)
        doppler_arr_aoa = np.mean(doppler_arr,axis=1)
        doppler_arr_aoa = np.mean(doppler_arr_aoa,axis=2)
        print(doppler_arr.shape)
        doppler_arr = np.abs(doppler_arr)
        r_bound = 10
        d_bound = 50
        extent = [np.min(params['doppler_freq']), np.max(params['doppler_freq']), params['AoA_vec'][r_bound], params['AoA_vec'][-r_bound]]
        # plt.subplot(1,2,2)
        plt.figure(figsize=(12,10))
        plt.suptitle(data)
        plt.subplot(2,2,1)
        plt.imshow(np.abs(doppler_arr_aoa[d_bound:-d_bound,r_bound:-r_bound].T), aspect='auto', origin='lower', extent=extent)
        plt.xlabel('Doppler Frequency [Hz]')
        plt.ylabel('AoA [deg]')
        plt.title('AoA v.s. Doppler Frequency')
        plt.colorbar()

        doppler_arr_aod = np.mean(doppler_arr,axis=2)
        doppler_arr_aod = np.mean(doppler_arr_aod,axis=2)
        plt.subplot(2,2,2)
        extent = [np.min(params['doppler_freq']), np.max(params['doppler_freq']), params['AoD_vec'][r_bound], params['AoD_vec'][-r_bound]]
        plt.imshow(np.abs(doppler_arr_aod[d_bound:-d_bound,r_bound:-r_bound].T), aspect='auto', origin='lower', extent=extent)
        plt.xlabel('Doppler Frequency [Hz]')
        plt.ylabel('AoD [deg]')
        plt.title('AoD v.s. Doppler Frequency')
        plt.colorbar()
        plt.grid()

        doppler_arr_range = np.mean(doppler_arr,axis=1)
        doppler_arr_range = np.mean(doppler_arr_range,axis=1)
        plt.subplot(2,2,3)
        extent = [np.min(params['doppler_freq']), np.max(params['doppler_freq']), params['dist_vec'][r_bound], params['dist_vec'][-r_bound]]
        plt.imshow(np.abs(doppler_arr_range[d_bound:-d_bound,r_bound:-r_bound].T), aspect='auto', origin='lower', extent=extent)
        plt.xlabel('Doppler Frequency [Hz]')
        plt.ylabel('Range [m]')
        plt.title('Range v.s. Doppler Frequency')
        plt.colorbar()
        plt.grid()

        plt.show()
        break
        

def load_params(data_folder):
    parameters = {}
    parameters['start_freq'] = np.load(f'./{data_folder}/parameters/start_freq.npy')
    parameters['stop_freq'] = np.load(f'./{data_folder}/parameters/stop_freq.npy')
    parameters['num_freq_step'] = np.load(f'./{data_folder}/parameters/num_freq_step.npy')
    parameters['rbw'] = np.load(f'./{data_folder}/parameters/rbw.npy')
    parameters['TxRxPairs'] = np.load(f'./{data_folder}/parameters/TxRxPairs.npy')
    parameters['freq'] = np.load(f'./{data_folder}/parameters/freq.npy')
    parameters['dist_vec'] = np.load(f'./{data_folder}/parameters/dist_vec.npy')
    parameters['AoD_vec'] = np.load(f'./{data_folder}/parameters/AoD_vec.npy')
    parameters['AoA_vec'] = np.load(f'./{data_folder}/parameters/AoA_vec.npy')
    parameters['doppler_freq'] = np.load(f'./{data_folder}/parameters/doppler_freq.npy')
    parameters['range_Nfft'] = np.load(f'./{data_folder}/parameters/range_Nfft.npy')
    parameters['angle_Nfft'] = np.load(f'./{data_folder}/parameters/angle_Nfft.npy')
    parameters['doppler_Nfft'] = np.load(f'./{data_folder}/parameters/doppler_Nfft.npy')
    parameters['x_offset_shift'] = np.load(f'./{data_folder}/parameters/x_offset_shift.npy')
    parameters['y_offset_shift'] = np.load(f'./{data_folder}/parameters/y_offset_shift.npy')
    parameters['x_ratio'] = np.load(f'./{data_folder}/parameters/x_ratio.npy')
    parameters['y_ratio'] = np.load(f'./{data_folder}/parameters/y_ratio.npy')
    parameters['ant_loc'] = np.load(f'./{data_folder}/parameters/ant_loc.npy')
    parameters['doppler_window_size'] = np.load(f'./{data_folder}/parameters/doppler_window_size.npy')
    parameters['cal_arr'] = np.load(f'./{data_folder}/parameters/cal_arr.npy')

    return parameters

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

# Conversion function
def spherical_to_rectangular(r, theta, phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def count_occupants(xy_projection, grid_x, grid_y, thres=0.03, eps=1, n=np.inf):
    x_axis = np.linspace(grid_x.min(), grid_x.max(), 64)
    y_axis = np.linspace(grid_y.min(), grid_y.max(), 64)

    occupancy_map = np.array([False, False, False, False])

    peak_indices = filter_and_cluster(normalization(xy_projection), threshold=thres, eps=eps, n=n)
    peak_locs = []
    for idx in peak_indices:
        peak_locs.append((x_axis[idx[0]], y_axis[idx[1]]))
    peak_locs = np.array(peak_locs)

    if np.any(np.logical_and(peak_locs[:,0]<0, peak_locs[:,1]<0)):
        occupancy_map[0] = True
        print('Seat #1 Occupied')
    if np.any(np.logical_and(peak_locs[:,0]>=0, peak_locs[:,1]<0)):
        occupancy_map[1] = True
        print('Seat #2 Occupied')
    if np.any(np.logical_and(peak_locs[:,0]<0, peak_locs[:,1]>=0)):
        occupancy_map[2] = True
        print('Seat #3 Occupied')
    if np.any(np.logical_and(peak_locs[:,0]>=0, peak_locs[:,1]>=0)):
        occupancy_map[3] = True
        print('Seat #4 Occupied')

    n_occupants = len(peak_indices)
    print(f'There are {n_occupants} occupants in the vehicle.')
    return n_occupants, occupancy_map, peak_indices

if __name__ == '__main__':
    main()