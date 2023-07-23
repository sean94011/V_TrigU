# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from parameter_setup import load_params, normalization
import time
from PIL import Image
from scipy.signal import find_peaks



def main():
    # load parameters
    params = load_params()
    print('waiting for pre-doppler data collections...')
    while(len(os.listdir('./data_queue'))<params['doppler_window_size']*2):
        continue
    print('Done')
    print('Start Processing & Visualizing...')
    vmin = 0.0#6
    # cal_frame = sorted(os.listdir('./data_queue'))[0]
    # cal_frame = np.load(os.path.join('./data_queue',cal_frame))
    cal_frame = np.load('./parameters/cal_arr.npy')
    ntarget = 3
    enhance_rate = 1
    while(True):
        start = time.time()
        data_queue = sorted(os.listdir('./data_queue')) 
        # doppler_arr = []
        # for doppler_frame in data_queue[:-10]:
        #     doppler_arr.append(np.load(os.path.join('.\data_queue',doppler_frame)))
        # frame_window = data_queue[-10:-1 ]
        # cur_frame_data = []
        # for data in data_queue:
        #     cur_frame_data.append(np.load(os.path.join('./data_queue',data)))
        # cur_frame_data = np.stack(cur_frame_data, axis=0)
        # rec_arr = np.mean(cur_frame_data, axis=0)
        cur_frame_data = data_queue[-1]
        # pre_frame_data = data_queue[len(data_queue)-2]
        # pre_frame_data = data_queue[0]
        rec_arr = np.load(os.path.join('./data_queue',cur_frame_data))
        pro_arr = rec_arr - cal_frame #np.load(os.path.join('./data_queue',pre_frame_data))
        range_profile = np.fft.ifft(pro_arr, axis=1)
        range_profile = np.linalg.norm(pro_arr,axis=0)
        # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
        range_peaks, _ = find_peaks(range_profile)
        sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
        top_range_peaks = range_peaks[sorted_peak_indices]
        # load 
        pro_arr_3D = pro_arr.reshape(20,20,150)#[chosen_frame,:,:,:]
        pro_arr_3D = np.fft.ifft(pro_arr_3D, n=params['range_Nfft'], axis=2)
        pro_arr_3D[:,:,np.where(params['dist_vec']>3.)]=np.mean((pro_arr_3D))
        pro_arr_3D[:,:,top_range_peaks] = pro_arr_3D[:,:,top_range_peaks]*enhance_rate


        pro_arr_3D = np.fft.fft2(pro_arr_3D, s=params['angle_Nfft'], axes=[0,1])
        pro_arr_yz = np.linalg.norm(pro_arr_3D, axis=0)
        pro_arr_xy = np.linalg.norm(pro_arr_3D,axis=2)
        pro_arr_xz = np.linalg.norm(pro_arr_3D,axis=1)
        # pro_arr_yz = np.abs(np.mean(pro_arr_3D,axis=0))
        # pro_arr_xy = np.abs(np.mean(pro_arr_3D,axis=2))
        # pro_arr_xz = np.abs(np.mean(pro_arr_3D,axis=1))
        pro_arr_xy = np.roll(pro_arr_xy,shift=params['y_offset_shift'],axis=1)
        pro_arr_xy = np.roll(pro_arr_xy,shift=params['x_offset_shift'],axis=0)
        pro_arr_yz = np.roll(pro_arr_yz,shift=params['y_offset_shift'],axis=0)
        pro_arr_xz = np.roll(pro_arr_xz,shift=params['x_offset_shift'],axis=0)


        # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
        AoD_profile = np.linalg.norm(pro_arr_xz, axis=1)
        AoD_peaks, _ = find_peaks(AoD_profile)
        sorted_peak_indices = np.argsort(AoD_profile[AoD_peaks])[::-1][:ntarget]
        top_AoD_peaks = AoD_peaks[sorted_peak_indices]
        # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
        AoA_profile = np.linalg.norm(pro_arr_yz, axis=1)
        AoA_peaks, _ = find_peaks(AoA_profile)
        sorted_peak_indices = np.argsort(AoA_profile[AoA_peaks])[::-1][:ntarget]
        top_AoA_peaks = AoA_peaks[sorted_peak_indices]

        pro_arr_xy[top_AoD_peaks,:] = pro_arr_xy[top_AoD_peaks,:]*enhance_rate
        pro_arr_xy[:,top_AoA_peaks] = pro_arr_xy[:,top_AoA_peaks]*enhance_rate
        pro_arr_yz[top_AoA_peaks,:] = pro_arr_yz[top_AoA_peaks,:]*enhance_rate
        pro_arr_xz[top_AoD_peaks,:] = pro_arr_xz[top_AoD_peaks,:]*enhance_rate

        # pro_arr_3D = pro_arr_3D[:,:,people]


        
        # range_doppler = np.fft.ifft(doppler_arr, n=params['range_Nfft'], axis=2)
        # range_doppler = np.fft.fft(range_doppler,n=params['doppler_Nfft'],axis=0)[:,210,:]
        # # range_doppler = np.linalg.norm(range_doppler, axis=1)   

        # range_doppler = np.abs(range_doppler).T

        # # print(doppler_freq)
        # freq_low = np.where(params['doppler_freq']>=0.1)[0][0]
        # freq_high = np.where(params['doppler_freq']<=2.0)[0][-1]
        # range_low = np.where(params['dist_vec']>=0.3)[0][0]
        # range_high = np.where(params['dist_vec']<=2.5)[0][-1]
        # # Eliminate the high DC offset
        # # range_low, range_high, freq_low, freq_high = 50, 400, 100, 300
        # range_range = np.r_[0:range_low, range_high:range_doppler.shape[0]]
        # doppler_range = np.r_[0:freq_low, freq_high:range_doppler.shape[1]]
        # range_doppler[range_range,:] = np.min(range_doppler)
        # range_doppler[:, doppler_range] = np.min(range_doppler)

        # range_doppler = normalization(range_doppler)
        extent = [np.min(params['AoD_vec']), np.max(params['AoD_vec']), np.min(params['AoA_vec']), np.max(params['AoA_vec'])]
        plt.figure(figsize=(12,10))
        plt.subplot(2,2,2)
        plt.title('XY Perspective')
        plt.imshow((pro_arr_xy).T,origin='lower',aspect='auto', extent=extent, vmin=vmin)
        plt.colorbar()
        plt.xlabel('AoD [deg]')
        plt.ylabel('AoA [deg]')
        plt.grid()
        plt.subplot(2,2,1)
        plt.title('Range Profile')
        range_profile = np.linalg.norm(np.fft.ifft(pro_arr, n=512, axis=1),axis=0)
        range_profile[np.where(params['dist_vec']>3)]=np.mean((range_profile))
        raw_range_profile = np.linalg.norm(np.fft.ifft(rec_arr, n=512, axis=1),axis=0)
        raw_range_profile[np.where(params['dist_vec']>3)]=np.mean((raw_range_profile))
        cal_range_profile = np.linalg.norm(np.fft.ifft(cal_frame, n=512, axis=1),axis=0)
        cal_range_profile[np.where(params['dist_vec']>3)]=np.mean((cal_range_profile))
        # range_profile[top_range_peaks] = range_profile[top_range_peaks]*10

        plt.plot(params['dist_vec'],cal_range_profile,label='background')
        plt.plot(params['dist_vec'],raw_range_profile,label='w/o background subtraction')
        plt.plot(params['dist_vec'],range_profile,label='w/ background subtraction')
        plt.xlabel('Range [m]')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend()
        plt.subplot(2,2,4)
        plt.title('YZ Perspective')
        extent = [np.min(params['AoA_vec']), np.max(params['AoA_vec']), np.min(params['dist_vec']), np.max(params['dist_vec'])]
        plt.imshow((pro_arr_yz).T,origin='lower',aspect='auto', extent=extent, vmin=vmin)
        plt.colorbar()
        plt.xlabel('AoA [deg]')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.subplot(2,2,3)
        plt.title('XZ Perspective')
        extent = [np.min(params['AoD_vec']), np.max(params['AoD_vec']), np.min(params['dist_vec']), np.max(params['dist_vec'])]
        plt.imshow((pro_arr_xz).T,origin='lower',aspect='auto', extent=extent, vmin=vmin)
        plt.colorbar()
        plt.xlabel('AoD [deg]')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.savefig('real_time_visualization.jpg')
        plt.close()
        print('Occupancy Detection Frame Duration: ',time.time()-start, '[s]')
        # print(time.time()-start, '[s]')
if __name__ == '__main__':
    main()