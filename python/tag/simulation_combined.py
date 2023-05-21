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
txRxPairs = np.load(os.path.join(current_path, '../constants/TxRxPairs.npy'))

nfft = 2**(ceil(log(freq.shape[0],2))+1)
ts = 1 / nfft / (freq[1] - freq[0] + 1e-16) # Avoid nan checks
time_vec = np.linspace(0,ts * (nfft-1), num=nfft)
dist_vec = time_vec * (c/2) # distance in meters
n_freq = freq.shape[0]

r = 2 # Virtual item placed!

# Simulating received frequencies
omega = freq * 2 * pi
delta_t = 2 * r / c
phi = np.multiply(omega, delta_t)
iq_sample = np.exp(-1j * phi)

# Repeat for multiple frames
n_frames = 10
iq_sample = np.tile(iq_sample, n_frames)
# iq_sample shape: (N_freq * n_frames,)

# Apply modulation
n_steps_total = n_freq * n_frames

fig, axes = plt.subplots(1, 3)
switch_slider_ax = plt.axes([0.25, 0.07, 0.65, 0.03])
switch_slider = Slider(switch_slider_ax, 'Steps per Switch', 0, 20, valinit=10, valstep=1)
switch_offset_slider_ax = plt.axes([0.25, 0.04, 0.65, 0.03])
switch_offset_slider = Slider(switch_offset_slider_ax, 'Large switching offset', 0, n_freq - 1, valinit=0, valstep=1)
switch_rest_offset_slider_ax = plt.axes([0.25, 0.01, 0.65, 0.03])
switch_rest_offset_slider = Slider(switch_rest_offset_slider_ax, 'Switching In-step Offset', n_freq // 2, n_freq * 2, valinit = n_freq, valstep=1)

def update(val):
    n_switch_steps = switch_slider.val
    n_switch_step_offset = switch_offset_slider.val
    n_rest_steps = switch_rest_offset_slider.val

    modulation = scipy.signal.square(2 * pi * np.arange(n_steps_total) / (n_switch_steps * 2)) > 0
    modulation *= scipy.signal.square(2 * pi * (np.arange(n_steps_total) / (n_rest_steps * 2) + n_switch_step_offset)) > 0
    modulated = np.multiply(iq_sample, modulation)

    modulated = np.reshape(modulated, (n_frames, n_freq)) # frames * steps
    rp = np.fft.ifft(modulated, axis=1) # time * range
    rd = np.fft.fft(rp, axis=0) # doppler * range
    dp = np.fft.fft(modulated, axis=0) # freq * steps

    axes[0].imshow(np.abs(rp), aspect='auto')
    axes[0].set_title('Range Profile')
    axes[0].set_xlabel('Range')
    axes[0].set_ylabel('Time')

    axes[1].imshow(np.abs(rd), aspect='auto')
    axes[1].set_title('Range Doppler')
    axes[1].set_xlabel('Range')
    axes[1].set_ylabel('Doppler')

    axes[2].imshow(np.abs(dp), aspect='auto')
    axes[2].set_title('?')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Frequency')

    fig.canvas.draw_idle()

switch_slider.on_changed(update)
switch_offset_slider.on_changed(update)
switch_rest_offset_slider.on_changed(update)

plt.show()