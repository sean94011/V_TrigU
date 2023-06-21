import os
from math import ceil, log

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c

def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

data_root = "./data"
cur_case = "test04242023"
cur_scenario = "human_stretch_0.1"
cur_constants = "constants"

data_path = os.path.join(data_root, cur_case, cur_scenario)
const_path = os.path.join(data_root, cur_case, cur_constants)

recArr = np.load(os.path.join(data_path, "recording.npy"))
calArr = np.load(os.path.join(data_path, "calibration.npy"))
freq = np.load(os.path.join(const_path, "freq.npy"))

proArr = recArr - np.mean(calArr, axis=0)
print(len(proArr))

range_Nfft = 2**(ceil(log(proArr.shape[2],2))+1)
doppler_Nfft =  2**(ceil(log(proArr.shape[0],2))+1)

Ts = 1/range_Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts*(range_Nfft-1),num=range_Nfft)
dist_vec = time_vec*(c/2)

range_profile = np.fft.ifft(proArr, n=range_Nfft, axis=2)
range_profile_norm = np.linalg.norm(range_profile, axis=1)
range_profile_norm[:,np.where(dist_vec>2.0)] = np.min(range_profile_norm)
# range_profile_norm = np.linalg.norm(proArr,axis=1)
# range_profile_norm = np.fft.ifft(range_profile_norm, n=range_Nfft, axis=1)
plt.figure()
plt.plot(dist_vec,range_profile_norm[50,:])
plt.show()
doppler = np.fft.fft(np.real(range_profile_norm), n=doppler_Nfft,  axis=0)
doppler = doppler[0:len(doppler)//2,:]
doppler = np.abs(doppler.T)

d=(2*60+2.8)/500
# d=(1*60+43)/200

# d = 1/fs
doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
doppler_freq = doppler_freq[doppler_freq>=0]

# doppler_freq = np.fft.fftfreq()
plt.figure(figsize=(8,6))
freq_low = np.where(doppler_freq>=0.05)[0][0]
freq_high = np.where(doppler_freq<=2.0)[0][-1]
range_low = np.where(dist_vec>=0.5)[0][0]
range_high = np.where(dist_vec<=2.0)[0][-1]

extent=[doppler_freq[freq_low],doppler_freq[freq_high],dist_vec[range_low],dist_vec[range_high]]
plt.imshow((doppler[range_low:range_high, freq_low:freq_high]), origin='lower', extent=extent, aspect='auto')
plt.scatter(doppler_freq[np.where(doppler_freq>=0.1)[0][0]], dist_vec[np.argmax(range_profile_norm[50,np.where(dist_vec<1.3)])],c='r', alpha=0.8, marker='x',label='Ground Truth')
plt.legend()
plt.colorbar()
plt.xlabel("Doppler Frequency [Hz]")
plt.ylabel("Range [m]")
plt.title(f"Range-Doppler Vital Sign Heatmap: {cur_scenario} [Hz]")
plt.grid()
plt.show()

