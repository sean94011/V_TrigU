# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from parameter_setup import load_params, normalization
import time



def main():
    # load parameters
    params = load_params()
    print('waiting for pre-doppler data collections...')
    while(len(os.listdir('./data_queue'))<params['doppler_window_size']*2):
        continue
    print('Done')
    print('Start Processing Doppler...')
    
    while(True):
        start = time.time()
        data_queue = sorted(os.listdir('./data_queue'))
        doppler_arr = []
        for doppler_frame in data_queue[:-10]:
            doppler_arr.append(np.load(os.path.join('.\data_queue',doppler_frame)))
        
        range_doppler = np.fft.ifft(doppler_arr, n=params['range_Nfft'], axis=2)
        range_doppler = np.fft.fft(range_doppler,n=params['doppler_Nfft'],axis=0)[:,210,:]
        range_doppler = np.abs(range_doppler).T

        # print(doppler_freq)
        freq_low = np.where(params['doppler_freq']>=0.4)[0][0]
        freq_high = np.where(params['doppler_freq']<=2.0)[0][-1]
        range_low = np.where(params['dist_vec']>=0.3)[0][0]
        range_high = np.where(params['dist_vec']<=2.5)[0][-1]
        # Eliminate the high DC offset
        # range_low, range_high, freq_low, freq_high = 50, 400, 100, 300
        range_range = np.r_[0:range_low, range_high:range_doppler.shape[0]]
        doppler_range = np.r_[0:freq_low, freq_high:range_doppler.shape[1]]
        range_doppler[range_range,:] = np.min(range_doppler)
        # range_doppler[:, doppler_range] = np.min(range_doppler)

        # range_doppler = normalization(range_doppler)
        plt.figure(figsize=(8,6))


        extent=[params['doppler_freq'][freq_low],params['doppler_freq'][freq_high],params['dist_vec'][range_low],params['dist_vec'][range_high]]
        plt.imshow(range_doppler[range_low:range_high,freq_low:freq_high],origin="lower",aspect="auto",extent=extent)
        plt.colorbar()
        plt.xlabel('Doppler Frequency [Hz]')
        plt.ylabel('Range [m]')
        plt.grid()
        plt.savefig('real_time_doppler.jpg')
        plt.close()
        print('Doppler Frame Duration: ',time.time()-start, '[s]')
        # print(time.time()-start, '[s]')
if __name__ == '__main__':
    main()