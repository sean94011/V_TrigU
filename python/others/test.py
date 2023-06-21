# import library
import numpy as np
from isens_vtrigU import isens_vtrigU
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from math import log, ceil
from sklearn.cluster import DBSCAN
from scipy.spatial import distance



bound = 2.5
Nfft = 512
ntarget = 6
y_offset_shift = 220
x_offset_shift = -90
x_ratio = 20/30
y_ratio = 20/25
threshold = 0.971
doppler_thres = 0.8

def filter_and_cluster(matrix, threshold=0.0, n=np.inf):
    indices = np.argwhere(matrix > threshold)

    clustering = DBSCAN(eps=1, min_samples=3).fit(indices)
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




def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def find_n_largest(matrix, n):
    flat = matrix.flatten()
    indices = np.argpartition(flat, -n)[-n:]  # get the indices of the top n values
    indices = indices[np.argsort(-flat[indices])]  # sort the indices
    indices_2d = np.unravel_index(indices, matrix.shape)  # convert the indices to 2D
    return list(zip(indices_2d[0], indices_2d[1]))

def gen_range_doppler(proArr, range_Nfft, doppler_Nfft, dist_vec):
    tof = np.fft.ifft(proArr,n=Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
    tof[:,:,np.where(dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range
    # tof = tof.reshape(nframe,400,-1)#[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
    print(tof.shape)
    range_doppler = tof.copy()
    range_doppler = np.linalg.norm(range_doppler, axis=1)
    print(range_doppler.shape)
    
    # Doppler FFT
    range_doppler = np.fft.fft(np.real(range_doppler),n=doppler_Nfft,axis=0)
    range_doppler = np.abs(range_doppler).T # (range, doppler)

    d=(2*60+2.8)/500
    # d = 1/fs
    doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    doppler_freq = doppler_freq[doppler_freq>=0]
    freq_low = np.where(doppler_freq>=0.3)[0][0]
    freq_high = np.where(doppler_freq<=2.0)[0][-1]
    range_low = np.where(dist_vec>=0.3)[0][0]
    range_high = np.where(dist_vec<=2.0)[0][-1]

    # Eliminate the high DC offset
    # range_low, range_high, freq_low, freq_high = 50, 400, 100, 300
    range_range = np.r_[0:range_low, range_high:range_doppler.shape[0]]
    doppler_range = np.r_[0:freq_low, freq_high:range_doppler.shape[1]]
    range_doppler[range_range,:] = np.min(range_doppler)
    range_doppler[:, doppler_range] = np.min(range_doppler)
    range_doppler_norm = np.linalg.norm(range_doppler,axis=1)

    range_doppler = normalization(range_doppler)
    rd_peaks_indices = filter_and_cluster(matrix=range_doppler[range_low:range_high,freq_low:freq_high], threshold=0.5, n=6)
    range_bins = rd_peaks_indices[0]

    return range_bins

def main():
    current_case = "test05012023"
    current_scenario = "human_walk_radar"
    current_scenario_path = os.path.join("./data", current_case, current_scenario)

    my_vtrig = isens_vtrigU(case=current_case)
    

    # processed_data_dir = "frames_point_cloud_NFFT=512"
    # point_cloud_path = os.path.join("./data",current_case,current_scenario,processed_data_dir)
    calArr = np.load(os.path.join(current_scenario_path,"calibration.npy"))
    recArr = np.load(os.path.join(current_scenario_path,"recording.npy"))
    proArr = recArr - np.mean(calArr,axis=0)
    Nfft = 2**(ceil(log(proArr.shape[2],2))+1)
    doppler_Nfft =  2**(ceil(log(proArr.shape[0],2))+1)

    nframe = recArr.shape[0]
    chosen_frame = 50

    my_vtrig.dist_vec = my_vtrig.compute_dist_vec(Nfft=Nfft)

    # for chosen_frame in range(nframe):
    # Compute the Range Profile and Find the Target Range Bin (For one target)
    range_profile = my_vtrig.range_pipeline(current_case,current_scenario, plot=False, Nfft=Nfft)[chosen_frame,:]
    range_profile[np.where(my_vtrig.dist_vec>bound)] = np.mean(range_profile)
    range_peaks, _ = find_peaks(range_profile)
    # Sort the peaks by their amplitudes in descending order and select the first 6 peaks
    sorted_peak_indices = np.argsort(range_profile[range_peaks])[::-1][:ntarget]
    top_range_peaks = range_peaks[sorted_peak_indices]

    tof = np.fft.ifft(proArr,n=Nfft,axis=2)                      # Do IFFT across frequency steps to get the range bins
    tof[:,:,np.where(my_vtrig.dist_vec>bound)] = np.min(tof)       # Eliminate noise that is beyond the range
    # tof = tof.reshape(nframe,400,-1)#[chosen_frame,:,:,:]      # Reshape the range profile to Tx x Rx and extract a frame
    print(tof.shape)
    range_doppler = tof.copy()
    range_doppler = np.linalg.norm(range_doppler, axis=1)
    print(range_doppler.shape)
    
    for i in range(range_doppler.shape[0]-9):
        # Doppler FFT
        range_doppler = np.fft.fft(np.real(range_doppler[i:i+10,:]),n=doppler_Nfft,axis=0)
        range_doppler = np.abs(range_doppler).T # (range, doppler)
        np.save(f'./doppler_data/range_doppler_{current_scenario}_{i}.npy',range_doppler)

    d=(2*60+2.8)/500
    # d = 1/fs
    doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    doppler_freq = doppler_freq[doppler_freq>=0]
    freq_low = np.where(doppler_freq>=0.3)[0][0]
    freq_high = np.where(doppler_freq<=2.0)[0][-1]
    range_low = np.where(my_vtrig.dist_vec>=0.3)[0][0]
    range_high = np.where(my_vtrig.dist_vec<=2.0)[0][-1]

    # Eliminate the high DC offset
    # range_low, range_high, freq_low, freq_high = 50, 400, 100, 300
    range_range = np.r_[0:range_low, range_high:range_doppler.shape[0]]
    doppler_range = np.r_[0:freq_low, freq_high:range_doppler.shape[1]]
    range_doppler[range_range,:] = np.min(range_doppler)
    range_doppler[:, doppler_range] = np.min(range_doppler)
    range_doppler_norm = np.linalg.norm(range_doppler,axis=1)

    range_doppler = normalization(range_doppler)
    rd_peaks_indices = filter_and_cluster(matrix=range_doppler[range_low:range_high,freq_low:freq_high], threshold=0.5, n=6)
    print(rd_peaks_indices)
    
    extent=[doppler_freq[freq_low],doppler_freq[freq_high],my_vtrig.dist_vec[range_low],my_vtrig.dist_vec[range_high]]

    print(range_doppler.shape)
    range_doppler_norm = normalization(range_doppler_norm)
    # rd_peaks_indices = find_n_largest(range_doppler, 6)

    plt.figure(figsize=(10,10))
    plt.imshow(range_doppler[range_low:range_high,freq_low:freq_high],origin="lower",aspect="auto",extent=extent)
    plt.scatter(doppler_freq[freq_low:freq_high][rd_peaks_indices[:, 1]], my_vtrig.dist_vec[range_low:range_high][rd_peaks_indices[:, 0]], color='r', marker='x')
    plt.colorbar()
    plt.xlabel('Doppler Frequency [Hz]')
    plt.ylabel('Range [m]')
    plt.title(f'Setup: {current_scenario}')
    plt.show()

   
    plt.figure(figsize=(8,6))
    plt.plot(my_vtrig.dist_vec, range_doppler_norm)
    plt.xlabel('Range [m]')
    plt.ylabel('Doppler Magnitude')
    plt.grid()
    plt.show()

    # """ Test Clusttering Algorithm"""
    # # Set a random seed for reproducibility
    # np.random.seed(0)

    # # Generate some clusters of points
    # matrix = np.zeros((100, 100))
    # matrix[30:35, 40:45] = np.random.rand(5, 5)
    # matrix[60:65, 70:75] = np.random.rand(5, 5)
    # matrix[80:85, 20:25] = np.random.rand(5, 5)

    # # Set the threshold
    # threshold = 0.5

    # # Use your function to find the cluster centers
    # cluster_center_indices = filter_and_cluster(matrix, threshold, 3)

    # # Create the heatmap
    # plt.figure(figsize=(10, 10))
    # plt.imshow(matrix, origin='lower', aspect='auto')
    # plt.colorbar(label='Matrix values')

    # # Add the cluster centers to the plot
    # plt.scatter(cluster_center_indices[:, 1], cluster_center_indices[:, 0], color='r')

    # # Label axes and add a title
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Heatmap with Cluster Centers')

    # # Display the plot
    # plt.show()

    
    # doppler_mask = range_doppler_norm > doppler_thres
    # masked_range_bins = range_doppler_norm[doppler_mask]
    # print(masked_range_bins)



    
    # tof_target = tof[:,:,top_range_peaks]
        
    # for range_bin in range(ntarget):
    #     cur_bin = tof_target[:,:,range_bin]
    #     doppler = np.fft.fft(np.real(cur_bin), n=doppler_Nfft,  axis=0)
    #     doppler = doppler[0:len(doppler)//2,:]
    #     doppler = np.abs(doppler.T)

    #     d=(2*60+2.8)/500
    #     # d=(1*60+43)/200

    #     # d = 1/fs
    #     doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    #     doppler_freq = doppler_freq[doppler_freq>=0]
        

    #     # doppler_freq = np.fft.fftfreq()
    #     plt.figure(figsize=(8,6))
    #     freq_low = np.where(doppler_freq>=0.5)[0][0]
    #     freq_high = np.where(doppler_freq<=2.0)[0][-1]
    #     # range_low = np.where(my_vtrig.dist_vec>=0.5)[0][0]
    #     # range_high = np.where(my_vtrig.dist_vec<=2.0)[0][-1]

    #     extent=[doppler_freq[freq_low],doppler_freq[freq_high],0,399]
    #     plt.imshow(normalization(doppler[:, freq_low:freq_high]), origin='lower', extent=extent, aspect='auto')
    #     # plt.scatter(doppler_freq[np.where(doppler_freq>=0.1)[0][0]], my_vtrig.dist_vec[np.argmax(range_profile_norm[50,np.where(my_vtrig.dist_vec<1.3)])],c='r', alpha=0.8, marker='x',label='Ground Truth')
    #     # plt.legend()
    #     plt.colorbar()
    #     plt.xlabel("Doppler Frequency [Hz]")
    #     plt.ylabel("Antenna Pairs")
    #     plt.title(f"Doppler Vital Sign Heatmap: {current_scenario} [Hz], Range Bin: {my_vtrig.dist_vec[top_range_peaks[range_bin]]}")
    #     plt.grid()
    #     plt.show()

    
if __name__ == '__main__':
    main()