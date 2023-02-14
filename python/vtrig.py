"""     
    iSens Lab: Step-CW Radar Data Processing with FFT
    Author: Sean Yao 
"""

""" Import Librarys """
import numpy as np
import sys
import multiprocessing as mp
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from scipy.constants import c
from os import listdir
from math import ceil, log
from matplotlib.widgets import Slider, Button
from collections import OrderedDict
from pyargus.directionEstimation import *


class isens_vtrigU:

    def __init__(self, peak_height=0.3, enhance=True, enhance_rate=100, interpolate=True) -> None:
        """ Load Parameters """
        # load setup parameters
        self.freq = np.load('./data/test01162023/constants/freq.npy')
        self.nframes = np.load('./data/test01162023/constants/nframes.npy')
        self.TxRxPairs = np.load('./data/test01162023/constants/TxRxPairs.npy')
        self.ants_locations = np.load('./data/test01162023/constants/ants_locations.npy')
        self.Nfft = 2**(ceil(log(self.freq.shape[0],2))+1)
        self.center_ant = 10
        self.peak_height = peak_height
        self.enhance_rate = enhance_rate
        self.enhance = enhance
        self.interpolate = interpolate
        self.dist_vec = self.compute_dist_vec()
        self.d = 0.5
        self.aoa_offset = 90-83.83561643835615-6.34050881
        self.aod_offset = 90-50.0196#131.5068493
        self.BW = 7e9
        self.Rres = c/(2*self.BW)
        

        self.angle_vec = np.linspace(0,180,self.Nfft)
        self.N_freq = self.freq.shape[0]

        print(f'Freq Points: {self.freq.shape[0]} ')
        print(f'TxRxPairs Shape: {self.TxRxPairs.shape}')
        print(f'Nfft = {self.Nfft}')
        print(f'Number of Recorded Frames: {self.nframes}')
        print('')


    """ Helper Functions """
    # Compute Distance Vector
    def compute_dist_vec(self):
        Ts = 1/self.Nfft/(self.freq[1]-self.freq[0]+1e-16) # Avoid nan checks
        time_vec = np.linspace(0,Ts*(self.Nfft-1),num=self.Nfft)
        return time_vec*(c/2) # distance in meters

    # Load collected Data
    def load_data(self, scenario='move_z', case = 'test/'):
        # specify data path components
        data_path = './data/' + case

        if scenario in listdir(data_path):
            raw_data = scenario + '/recording.npy'
            cal_data = scenario + '/calibration.npy'

            # combine data paths
            raw_path = data_path + raw_data
            cal_path = data_path + cal_data
            # processed_path = data_path + processed_data

            # load data
            print('Current scenario: ' + scenario)
            print('')
            recArr = np.load(raw_path)
            calArr = np.load(cal_path)
            print(f'calArr Shape: {calArr.shape}')
            print(f'recArr Shape: {recArr.shape}')
            print('')
            print('recArr Channels: (frame, Tx*Rx, freqs)')
            print('')

        else:
            print(f'Scenario: {scenario} does not exist!')
            print('')
            sys.exit()
        
        return calArr, recArr, data_path+scenario+'/'

    # Plot the Antenna Array
    def plot_antennas(self):
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.plot(self.ants_locations[ 0:20,0],self.ants_locations[ 0:20,1],'^')
        plt.plot(self.ants_locations[20:40,0],self.ants_locations[20:40,1],'^')
        # plt.plot(np.mean(antsLocations[ 0:20,0]),np.mean(antsLocations[20:40,1]),'*')
        plt.title('Uniform Linear Array')
        plt.legend(['Tx Antenna','Rx Antenna'])
        plt.grid()
        plt.axis('scaled')
        plt.xlim([-0.04,0.04])
        plt.ylim([-0.04,0.04])
        plt.show(block=True)

    # Compute ToF, AoA, AoD
    def calibration(self, calArr, recArr, method = 0):
        """
        calibration method:
            0 - subtract the pre-collected background noise
            1 - subtract each frame by the previous frame
        """
        if len(calArr.shape) > 2:
            calFrame = np.mean(calArr,axis=0)
        else:
            calFrame = calArr

        if method == 0:
            proArr = recArr - calFrame
        elif method == 1:
            proArr = recArr[1:recArr.shape[0],:,:] - recArr[0:recArr.shape[0]-1,:,:]
            
        return proArr


    def normalization(self, x):
        return (x - np.min(x))/(np.max(x)-np.min(x))

    def interpolation(self, signal):
        interpolated_signal = np.zeros([signal.shape[0],signal.shape[1],signal.shape[2]*2],dtype='complex')
        interpolated_signal[:,:,::2] = signal
        for i in np.arange(interpolated_signal.shape[2])[1::2]:
            left = 0
            right = 0
            if i-1 >= 0 and i-1 < interpolated_signal.shape[2]:
                left = interpolated_signal[:,:,i-1]
            if i+1 < interpolated_signal.shape[2] and i+1 >= 0:
                right = interpolated_signal[:,:,i+1]
            interpolated_signal[:,:,i] = (left+right)/2
        interpolated_signal = interpolated_signal[:,:,:-1]
        # Adjust other parameters correspondingly
        self.Nfft = 2**(ceil(log(interpolated_signal.shape[2],2))+1)
        xp = np.arange(len(self.dist_vec))
        fp = self.dist_vec
        x = np.arange(self.Nfft)
        self.dist_vec = np.interp(x,xp,fp)
        return interpolated_signal

    def enhance_target(self, signal):
        if len(signal.shape) == 4:
            for i in range(signal.shape[0]):
                peaks = find_peaks(self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:])),height=self.peak_height)[0]
                if i == 50:
                    plt.figure()
                    plt.plot(self.dist_vec,self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:])))
                    plt.plot(self.dist_vec[peaks],self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:]))[peaks],'o')
                    plt.xlabel('Distance [m]')
                    plt.ylabel('Normalized Magnitude')
                    plt.title('Peak extraction')
                    plt.show(block=True)
                    print(f'Enhance Peak at: {self.dist_vec[peaks]}')
                signal[i,:,:,peaks] *= self.enhance_rate
        elif len(signal.shape) == 2:
                peaks = find_peaks(self.normalization(np.abs(signal[self.center_ant,:])),height=self.peak_height)[0]
                plt.figure()
                plt.plot(self.dist_vec,self.normalization(np.abs(signal[self.center_ant,:])))
                plt.plot(self.dist_vec[peaks],self.normalization(np.abs(signal[self.center_ant,:]))[peaks],'o')
                plt.xlabel('Distance [m]')
                plt.ylabel('Normalized Magnitude')
                plt.title('Peak extraction')
                plt.show(block=True)
                print(f'Enhance Peak at: {self.dist_vec[peaks]}')
                signal[:,peaks] *= self.enhance_rate
        return signal, peaks

    def compute_tof(self, X, Nfft=512):
        x = np.fft.ifft(X,Nfft,2)
        return np.linalg.norm(x,axis=1)

    def compute_aoa_fft(self, X):
        print('Computing AoA...')
        print('')
        if self.interpolate:
            X = self.interpolation(X)
        x = np.fft.ifft(X,self.Nfft,2).reshape(X.shape[0],20,20,-1)
        if self.enhance:
            x = self.enhance_target(x)
        aoa = np.fft.fft(x[:,self.center_ant,:,:],self.Nfft,axis=1) # extract the center tx data and do the processing
        aoa = np.fft.fftshift(aoa,axes=1)
        return aoa

    def compute_aod_fft(self, X):
        print('Computing AoD...')
        print('')
        if self.interpolate:
            X = self.interpolation(X)
        x = np.fft.ifft(X,self.Nfft,2).reshape(X.shape[0],20,20,-1)
        plt.figure()
        plt.plot(self.dist_vec,self.normalization(np.abs(x[50,10,10,:])))
        plt.xlabel('Distance [m]')
        plt.ylabel('Normalized Magnitude')
        plt.title('Peak extraction')
        plt.show(block=True)
        if self.enhance:
            x = self.enhance_target(x)
        aod = np.fft.fft(x[:,:,self.center_ant,:],self.Nfft,axis=1) # extract the center rx data and do the processing
        aod = np.fft.fftshift(aod,axes=1)
        return aod
    
    """ MUSIC """

    def compute_aoa_music_single_frame(self, rec_signal, signal_dimension=3, smoothing=True, plot=False, n_rx=20): 
        if smoothing:
            # Calculate the forward-backward spatially smotthed correlation matrix
            R = spatial_smoothing(rec_signal.T, P=n_rx, direction="forward-backward")
        else:
            # Estimating the spatial correlation matrix without spatial smoothing
            R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

        # Regenerate the scanning vector for the sub-array
        array_alignment = np.arange(0, n_rx, 1)* self.d
        incident_angles= self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

        # Estimate DOA 
        aoa = DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension)
        print(f'Max Peak at: {incident_angles[np.argmax(aoa)]-90+self.aoa_offset} deg')
        print()
        if plot:
            # Get matplotlib axes object
            plt.figure()
            axes = plt.axes()
            
            # Plot results on the same fiugre
            DOA_plot(aoa, incident_angles+self.aoa_offset-90, log_scale_min = -50, axes=axes, alias_highlight=False)

            axes.legend(["MUSIC"])

            # Mark nominal incident angles
            # axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
            plt.show()
        return aoa

    def compute_aoa_music(self, X, n_rx=20, signal_dimension=3, smoothing=True, plot=False): 
        # Regenerate the scanning vector for the sub-array
        aoa = []
        array_alignment = np.arange(0, n_rx, 1)* self.d
        incident_angles= self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)
        for frame in range(X.shape[0]):
            rec_signal = X.reshape((X.shape[0],20,20,-1))[frame,self.center_ant,:,:]
            rec_signal = np.fft.ifft(rec_signal,axis=1,n=self.Nfft)
            if smoothing:
                # Calculate the forward-backward spatially smotthed correlation matrix
                R = spatial_smoothing(rec_signal.T, P=n_rx, direction="forward-backward")
            else:
                # Estimating the spatial correlation matrix without spatial smoothing
                R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")
            # Estimate DOA 
            aoa.append(DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension))

        return np.array(aoa)
    
    def compute_aod_music_single_frame(self,rec_signal, signal_dimension=3, smoothing=True, plot=False, n_tx=20): 
        if smoothing:
            # Calculate the forward-backward spatially smotthed correlation matrix
            R = spatial_smoothing(rec_signal.T, P=n_tx, direction="forward-backward")
        else:
            # Estimating the spatial correlation matrix without spatial smoothing
            R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

        # Regenerate the scanning vector for the sub-array
        array_alignment = np.arange(0, n_tx, 1)* self.d
        incident_angles= self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

        # Estimate DOA 
        aod = DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension)
        aod_angles = incident_angles-90+self.aod_offset
        print(f'Max Peak at: {aod_angles[np.where(np.abs(aod_angles) <= 41)][np.argmax(aod[np.where(np.abs(aod_angles) <= 41)])]} deg')
        print()
        if plot:
            # Get matplotlib axes object
            plt.figure()
            axes = plt.axes()
            
            # Plot results on the same fiugre
            DOA_plot(aod, incident_angles-90+self.aod_offset, log_scale_min = -50, axes=axes, alias_highlight=False)

            axes.legend(["MUSIC"])

            # Mark nominal incident angles
            # axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
            plt.show()
        return aod

    # Plot 2D heatmaps
    def heatmap_2D(self, arr, fnum):
        fig, ax = plt.subplots(figsize=(8,8))
        extent = [-90, 90, 0, np.max(self.dist_vec)]
        ax.imshow(np.abs(arr[fnum].T),origin='lower', interpolation='nearest', aspect='auto', extent=extent)
        ax.set_title("2-D Heat Map")
        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("Range [m]")

    def interactive_heatmap_2d(self, arr, title='2-D Heat Map', method='music'):
        init_fnum = 0
        cur_frame = np.abs(arr[init_fnum].T)
        fig, ax = plt.subplots(figsize=(10,8))
        plt.ion()
        if method=='music':
            extent = [0+self.aoa_offset, 180+self.aoa_offset, 0, np.max(self.dist_vec)]
        else:
            extent = [-90, 90, 0, np.max(self.dist_vec)] 
        img = ax.imshow(self.normalization(np.abs(arr[init_fnum,:,:].T)),origin='lower', interpolation=None, aspect='auto', extent=extent)

        fig.colorbar(img)
        ax.set_title(title)
        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("Range [m]")
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
            img.set_data(self.normalization(np.abs(arr[int(frame_slider.val)].T)))
            fig.canvas.draw_idle()
        frame_slider.on_changed(update)
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')
        def reset(event):
            frame_slider.reset()

        button.on_clicked(reset)
        plt.show(block=True)

    # Integrate everything into the processing pipeline
    def fft_pipeline(self, case='test/', scenario='move_z', cal_method=0, all_case=False, plot=None, test_mode=False, show_ants=False):
        if show_ants:
            self.plot_antennas()
        if all_case == False:
            # Load Data
            calArr, recArr, path = self.load_data(scenario=scenario, case=case)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method)
            # Compute AoA & ToF
            if f"fft_aoa_1frame_cal{cal_method}.npy" not in listdir(path):
                aoaArr = self.compute_aoa_fft(proArr)
                np.save(path+f"fft_aoa_1frame_cal{cal_method}.npy",aoaArr)
            else:
                if not test_mode:
                    aoaArr = np.load(path+f"fft_aoa_1frame_cal{cal_method}.npy")
                else:
                    aoaArr = self.compute_aoa_fft(proArr)

            
            # Compute AoD & ToF
            if f"fft_aod_1frame_cal{cal_method}.npy" not in listdir(path):
                aodArr = self.compute_aod_fft(proArr)
                np.save(path+f"fft_aod_1frame_cal{cal_method}.npy",aodArr)
            else:
                if not test_mode:
                    aodArr = np.load(path+f"fft_aod_1frame_cal{cal_method}.npy")
                else:
                    aodArr = self.compute_aod_fft(proArr)

            if plot:
                print('Interactive 2D heatmap displaying, please hit interrupt to stop displaying...')
            if plot == 'aoa':
                if self.interpolate:
                    self.interactive_heatmap_2d(aoaArr[:,:,0::2], 'AoA 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
            if plot == 'aod':
                if self.interpolate:
                    self.interactive_heatmap_2d(aodArr[:,:,0::2], 'AoD 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
            if plot == 'both':
                if self.interpolate:
                    self.interactive_heatmap_2d(aoaArr[:,:,0::2], 'AoA 2D Heatmap')
                    self.interactive_heatmap_2d(aodArr[:,:,0::2], 'AoD 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                    self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
        else:
            for scenario in listdir('./data/' + case):
                if scenario == 'constants':
                    continue
                # Load Data
                calArr, recArr, path = self.load_data(scenario=scenario, case=case)
                print(f'Path: {path}')
                print('')
                # Calibrate Data
                for i in [0]:
                    print(f'Current Calibration Method: {i}')
                    print('')
                    proArr = self.calibration(calArr,recArr,i)
                    # Compute AoA & ToF
                    if f"fft_aoa_1frame_cal{i}.npy" not in listdir(path):
                        aoaArr = self.compute_aoa_fft(proArr)
                        np.save(path+f"fft_aoa_1frame_cal{i}.npy",aoaArr)
                    else:
                        aoaArr = np.load(path+f"fft_aoa_1frame_cal{i}.npy")
                    
                    # Compute AoD & ToF
                    if f"fft_aod_1frame_cal{i}.npy" not in listdir(path):
                        aodArr = self.compute_aod_fft(proArr)
                        np.save(path+f"fft_aod_1frame_cal{i}.npy",aodArr)
                    else:
                        aodArr = np.load(path+f"fft_aod_1frame_cal{i}.npy")
                    
                    if plot:
                        print('Interactive 2D heatmap displaying, please hit interrupt to stop displaying...')
                    if plot == 'aoa':
                        self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                    if plot == 'aod':
                        self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
                    if plot == 'both':
                        self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                        self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')

    def range_pipeline(self, case='test/', scenario='move_z', cal_method=0):
        # Load Data
        calArr, recArr, _ = self.load_data(scenario=scenario, case=case)
        # Calibrate Data
        for cal_method in [0,1]:
            proArr = self.calibration(calArr,recArr,cal_method)
            # Compute Range Profile
            tof = self.compute_tof(proArr).T
            # plot the range profile vs frame
            extent = [0, 99, 0, np.max(self.dist_vec)]
            plt.figure(figsize=(10,5))
            plt.imshow(self.normalization(tof),origin='lower', interpolation='nearest', aspect='auto', extent=extent)
            plt.colorbar()
            plt.title('Frame vs Distance')
            plt.xlabel('Frame')
            plt.ylabel('Distance [m]')
            plt.show(block=True)

    def music_sim_signal(self, thetas = [50, 80, 110]):
        # Number of antenna elements
        M = 20
        # number of samples
        N = self.Nfft
        # Interelement spacing is half-wavelength
        d= 0.5
        a_list = []
        for theta in thetas:
            a_list.append(np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta))))
        # Generate multichannel test signal 
        soi = np.random.normal(0,1,N)  # Signal of Interest
        soi_matrix = np.zeros(np.size(np.outer( soi, a_list[0])))
        for a in a_list:
            soi_matrix  += np.outer( soi, a)
        soi_matrix = soi_matrix.T
        # Generate multichannel uncorrelated noise
        noise = np.random.normal(0,np.sqrt(10**-1),(M,N))
        # Create received signal array
        rec_signal = soi_matrix + noise
        return rec_signal 

    def music_aoa_single_frame_pipeline(self, case='test/', scenario='move_z', simulation=False, cal_method=0, signal_dimension=3, smoothing=True, plot_aoa_spectrum=False):
        if simulation:
            rec_signal = self.music_sim_signal()
            aoa = self.compute_aoa_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aoa_spectrum)

        else:
            # Load Data
            calArr, recArr, _ = self.load_data(scenario=scenario, case=case)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)
        
            range_profile = np.real(np.fft.ifft(proArr[50,self.center_ant,:,:],n=self.Nfft, axis=1))
            # range_profile = np.linalg.norm(range_profile,axis=0)
            range_profile[:,np.where(self.dist_vec>2.5)] = 0
            range_profile[:,np.where(self.dist_vec<0.3)] = 0
            if self.enhance:
                range_profile, peaks = self.enhance_target(range_profile)
            else:
                _, peaks = self.enhance_target(range_profile)
            aoa = []
            pool = mp.Pool()
            outputs = []
            for range_bin in range(range_profile.shape[1]):
                rec_signal = range_profile[:,range_bin].reshape((-1,1))
                # print(rec_signal.shape)
                # aoa.append(self.compute_aoa_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aoa_spectrum))
                outputs.append(pool.apply_async(self.compute_aoa_music_single_frame,(rec_signal,signal_dimension,smoothing,plot_aoa_spectrum)))
            pool.close()
            aoa = [res.get() for res in outputs]
            heat_map = np.array(aoa)
            heat_map[peaks,:] *= 1000
            heat_map = self.normalization(np.abs(heat_map))
            angle_peaks = []
            for i in range(len(peaks)):
                angle_peaks.append(find_peaks(np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))),height=self.peak_height)[0])

            # angle_peaks = np.squeeze(angle_peaks)
            # aoa_angles = self.angle_vec-90+self.aoa_offset

            # print(f'Target at: {self.dist_vec[peaks]} m & {aoa_angles[angle_peaks]} deg in y direction')
            # print(np.where(np.abs(heat_map)==np.max(np.abs(heat_map))))
            plt.figure()
            extent = [-90+self.aoa_offset, 90+self.aoa_offset, 0, np.max(self.dist_vec)]
            plt.imshow(heat_map,origin='lower', interpolation=None, aspect='auto', extent=extent)

            plt.colorbar()
            plt.title('MUSIC AoA')
            plt.xlabel("Angle [deg]")
            plt.ylabel("Range [m]")
            plt.grid()
            plt.show()


    def music_aod_single_frame_pipeline(self, case='test/', scenario='move_z', simulation=False, cal_method=0, signal_dimension=3, smoothing=True, plot_aod_spectrum=False):
        if simulation:
            rec_signal = self.music_sim_signal()
            aod = self.compute_aod_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aod_spectrum)
        else:
            # Load Data
            calArr, recArr, _ = self.load_data(scenario=scenario, case=case)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)
            rec_signal = np.fft.ifft(proArr[50,:,self.center_ant,:],axis=1,n=self.Nfft)
        
            range_profile = np.real(np.fft.ifft(proArr[50,:,self.center_ant,:],n=self.Nfft, axis=1))
            range_profile[:,np.where(self.dist_vec>2.5)] = 0
            range_profile[:,np.where(self.dist_vec<0.3)] = 0
            if self.enhance:
                range_profile, peaks = self.enhance_target(range_profile)
            else:
                _, peaks = self.enhance_target(range_profile)
            aod = []
            pool = mp.Pool()
            outputs = []
            for range_bin in range(range_profile.shape[1]):
                rec_signal = range_profile[:,range_bin].reshape((-1,1))
                # print(rec_signal.shape)
                # aod.append(self.compute_aod_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aod_spectrum))
                outputs.append(pool.apply_async(self.compute_aod_music_single_frame,(rec_signal,signal_dimension,smoothing,plot_aod_spectrum)))
            pool.close()
            aod = [res.get() for res in outputs]

            heat_map = np.array(aod)
            aod_angles = self.angle_vec-90+self.aod_offset
            # print(np.where(np.abs(aod_angles) <= 41) )
            heat_map = np.squeeze(heat_map[:,np.where(np.abs(aod_angles) <= 41)])

            heat_map[peaks,:] *= 1000
            plt.figure()
            # extent = [-90+self.aod_offset, 90+self.aod_offset, 0, np.max(self.dist_vec)]
            extent = [-41, 41, 0, np.max(self.dist_vec)]
            heat_map = self.normalization(np.abs(heat_map))
            angle_peaks = []
            for i in range(len(peaks)):
                angle_peaks.append(find_peaks(np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))),height=self.peak_height)[0])
            plt.figure()
            plt.plot(aod_angles[np.where(np.abs(aod_angles) <= 41)],np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))))
            plt.plot(aod_angles[np.where(np.abs(aod_angles) <= 41)][angle_peaks],np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:])))[angle_peaks],'o')
            plt.show(block=True)
            angle_peaks = np.squeeze(angle_peaks)
            print(f'Target at: {self.dist_vec[peaks]} m & {aod_angles[angle_peaks]+9.15854324853229} deg in x direction')
            plt.imshow(heat_map,origin='lower', interpolation=None, aspect='auto', extent=extent)

            plt.colorbar()
            plt.title('MUSIC AoD')
            plt.xlabel("Angle [deg]")
            plt.ylabel("Range [m]")
            plt.grid()
            plt.show()

    def music_pipeline(self, case='test/', scenario='move_z', cal_method=0, signal_dimension=3, smoothing=True):
        # Load Data
        calArr, recArr, _ = self.load_data(scenario=scenario, case=case)
        # Calibrate Data
        proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)
        aoa = self.compute_aoa_music(X=proArr, signal_dimension=signal_dimension, smoothing=smoothing)
        range_profile = np.fft.ifft(proArr[:,self.center_ant,self.center_ant,:],axis=1,n=self.Nfft)
        heat_map = np.zeros((self.nframes, range_profile.shape[1], aoa.shape[1]))
        for frame in range(self.nframes):
            heat_map[frame,:,:] =  aoa[frame,:].reshape((-1,1))*range_profile[frame,:].reshape((1,-1))
        self.interactive_heatmap_2d(heat_map)


def main():
    plt.close()
    # current_case = 'test01312023/' # 2cf
    current_case = 'test01242023/' # 1cf
    # current_case = 'test01162023/'
    my_vtrig = isens_vtrigU(peak_height=0.7, enhance_rate=100, enhance=True, interpolate=False)
    # my_vtrig.fft_pipeline(case=current_case, all_case=True, plot=None, test_mode=False, show_ants=False)
    # my_vtrig.fft_pipeline(case=current_case, scenario='human_y_angle_-20',cal_method=0 ,all_case=False, plot='both', test_mode=True, show_ants=False)
    # my_vtrig.range_pipeline(case=current_case, scenario='human_x_angle_0')
    # my_vtrig.music_pipeline(case=current_case, scenario='cf_x_angle_-20',signal_dimension=1)
    my_vtrig.music_aoa_single_frame_pipeline(case=current_case, scenario='cf_y_angle_+20',signal_dimension=1, plot_aoa_spectrum=False)
    # my_vtrig.music_aod_single_frame_pipeline(case=current_case, scenario='cf_x_angle_0',signal_dimension=3, plot_aod_spectrum=False)
    # current_case = 'test01222023/' # x calibration
    # my_vtrig.music_aod_single_frame_pipeline(case=current_case, scenario='2cf_yy_angle_+-20',signal_dimension=5, plot_aod_spectrum=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:       
        print('')
    finally:
        print('')
        print("...Program terminated")