## SFCW Simulation

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, pi
from math import ceil, log
import scipy.signal

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
iq_sample = np.exp(-1j * phi)

# Pad with enough zeros to Nfft points
iq_sample = np.pad(iq_sample, pad_width=(0, Nfft - N_freq), mode='constant')

# IQ Samples --IFFT-> Range Profile
rp = np.fft.ifft(iq_sample)

# Plotting the synthetic pulse
def get_vis_labels(arr, label_cnt=3, precision=0):
    N = arr.shape[0]
    vis_idx = np.arange(0, N, N // label_cnt)
    vis = list(map(lambda x: ('%.' + str(precision) + 'f') % x, arr[vis_idx]))
    return vis_idx, vis

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.set_xlabel("Range (m)")

rp_vis_idx, rp_vis = get_vis_labels(np.arange(Nfft) * c / (2 * Nfft * (freq[1] - freq[0])), label_cnt=5, precision=1)
plt.gca().set_xticks(rp_vis_idx, labels=rp_vis)
rp_plot, = ax.plot(np.abs(rp))

freq_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_comp_slider = Slider(freq_slider_ax, 'Frequency Comp. Cnt', 0, N_freq-1, valinit=0, valstep=1)

switch_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
switch_slider = Slider(switch_slider_ax, 'Steps per Switch', 0, 10, valinit=0, valstep=1)

def calc_phase(val):
    trimmed_iq_sample = np.zeros_like(iq_sample)
    trimmed_iq_sample[:freq_comp_slider.val] = iq_sample[:freq_comp_slider.val]
    steps_per_switch = switch_slider.val
    if steps_per_switch > 0:
        switch_mask = scipy.signal.square(pi * np.arange(Nfft) / steps_per_switch) > 0
    else:
        switch_mask = np.zeros(Nfft, dtype="bool")
    switch_iq_sample = trimmed_iq_sample.copy()
    switch_iq_sample[switch_mask] = 0
    switch_rp_abs = np.abs(np.fft.ifft(switch_iq_sample))
    rp_plot.set_ydata(switch_rp_abs)
    fig.canvas.draw_idle()
    ax.set_ylim(np.min(switch_rp_abs), np.max(switch_rp_abs))

freq_comp_slider.on_changed(calc_phase)
switch_slider.on_changed(calc_phase)

plt.show()