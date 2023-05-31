## SFCW Simulation

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, pi
from math import ceil, log
import scipy.signal
import os

# Configuration Parameters
BW = 7e9

current_path = os.path.dirname(os.path.abspath(__file__))
freq = np.load(os.path.join(current_path, '../constants/freq.npy'))
nframes = np.load(os.path.join(current_path, '../constants/nframes.npy'))
TxRxPairs = np.load(os.path.join(current_path, '../constants/TxRxPairs.npy'))

Nfft = 2**(ceil(log(freq.shape[0],2))+1)
Ts = 1 / Nfft / (freq[1] - freq[0] + 1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts * (Nfft-1), num=Nfft)
dist_vec = time_vec * (c/2) # distance in meters
N_freq = freq.shape[0]

r = 2 # Virtual item placed!
vel = 10 # m/s

# Simulating received frequencies
omega = freq * 2 * pi
delta_t = 2 * r / c
phi = np.multiply(omega, delta_t)
iq_sample = np.exp(-1j * phi)
iq_sample_vel = np.exp(-1j * phi * (1 - vel / c))
delta_phase = np.angle(iq_sample_vel) - np.angle(iq_sample)

rp = np.fft.ifft(iq_sample, n=Nfft)
rp_vel = np.fft.ifft(iq_sample_vel, n=Nfft)

fig, axes = plt.subplots(3, 1)

axes[0].plot(dist_vec, np.abs(rp))
axes[0].plot(dist_vec, np.abs(rp_vel))
axes[0].set_xlabel('Distance (m)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Range Profile')
axes[0].legend(['Static', 'Moving at {} m/s'.format(vel)])

axes[1].plot(dist_vec, np.angle(rp))
axes[1].plot(dist_vec, np.angle(rp_vel))
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Phase')
axes[1].set_title('Range Profile')
axes[1].legend(['Static', 'Moving at {} m/s'.format(vel)])

axes[2].plot(delta_phase)
axes[2].set_xlabel('Distance (m)')
axes[2].set_ylabel('Phase')
axes[2].set_title('Phase Difference from Doppler')

plt.show()