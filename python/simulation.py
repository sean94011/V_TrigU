## SFCW Simulation

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c, pi
from math import ceil, log

# Configuration Parameters
BW = 7e9

freq = np.load('./constants/freq.npy')
nframes = np.load('./constants/nframes.npy')
TxRxPairs = np.load('./constants/TxRxPairs.npy')

Nfft = 2**(ceil(log(freq.shape[0],2))+1)
Ts = 1 / Nfft / (freq[1] - freq[0] + 1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts * (Nfft-1), num=Nfft)
dist_vec = time_vec * (c/2) # distance in meters
N_freq = freq.shape[0]

r = 2 # Virtual item placed!

# Simulating received frequencies
omega = freq * 2 * pi
delta_t = 2 * r / c
phi = np.multiply(omega, delta_t)
iq_sampled = np.exp(-1j * phi)

# Pad with enough zeros to Nfft points
iq_sampled = np.pad(iq_sampled, pad_width=(0, Nfft - N_freq), mode='constant')

# IQ Samples --IFFT-> Range Profile
rp = np.fft.ifft(iq_sampled)

# Plotting the synthetic pulse
def get_vis_labels(arr, label_cnt=3, precision=0):
    N = arr.shape[0]
    vis_idx = np.arange(0, N, N // label_cnt)
    vis = list(map(lambda x: ('%.' + str(precision) + 'f') % x, arr[vis_idx]))
    return vis_idx, vis

plt.plot(np.abs(rp)),
rp_vis_idx, rp_vis = get_vis_labels(np.arange(Nfft) * c / (2 * Nfft * (freq[1] - freq[0])), label_cnt=5, precision=1)
plt.gca().set_xticks(rp_vis_idx, labels=rp_vis)
plt.gca().set_xlabel("Range (m)")

plt.show()