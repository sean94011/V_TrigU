# import libraries
import datetime
import os
import time
import serial
from math import ceil, log

import matplotlib.pyplot as plt
import numpy as np
import vtrigU as vtrig
from numpy.linalg import norm
from scipy import constants

# define some constants
c = constants.c

BW = 7e9
Rmax = 75*c/BW
Rres = c/(2*BW)
print(f"Rres = {Rres} m")
print(f"Rmax = {Rmax} m")

RBW = 80.0 # in KHz
# initialize the device
vtrig.Init()
# set setting structure
vtrigSettings = vtrig.RecordingSettings(
        vtrig.FrequencyRange(62.0*1000, # Start Frequency (in MHz)
                             69.0*1000, # Stop  Frequency (in MHz) (66.5 for 5m) (68.0 for 3m)
                             150),      # Number of Frequency Points (Maximum: 150)
        RBW,                            # RBW (in KHz)
        vtrig.VTRIG_U_TXMODE__LOW_RATE  # Tx Mode (LOW: 20 Tx, MED: 10 Tx, HIGH: 4 Tx)
        )

# validate settings
vtrig.ValidateSettings(vtrigSettings)
# apply settings
vtrig.ApplySettings(vtrigSettings)

# get antenna pairs and convert to numpy matrix
TxRxPairs = np.array(vtrig.GetAntennaPairs(vtrigSettings.mode))
# get used frequencies in Hz
freq = np.array(vtrig.GetFreqVector_MHz()) * 1e6
# define constants
N_txrx = TxRxPairs.shape[0]
N_freq = freq.shape[0]

Nfft = 2**(ceil(log(freq.shape[0],2))+1)
Ts = 1/Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts*(Nfft-1),num=Nfft)
dist_vec = time_vec*(c/2) # distance in meters

# Record the calibration frames
def rec2arr(rec):
    recArr = []
    for key in rec.keys():
        recArr.append(rec[key])
    return np.array(recArr)

print("Calibrating...")
nrecs = 10
calFrame = []
for i in range(nrecs):
    vtrig.Record()
    rec = vtrig.GetRecordingResult()
    recArr = rec2arr(rec)
    calFrame.append(recArr)
calFrame = np.mean(calFrame,axis=0)
print(calFrame.shape)
print("Calibration matrix collected!")
input("Press any key to continue...")


# Live visualize
def get_vis_labels(arr, label_cnt=3, precision=0):
    N = arr.shape[0]
    vis_idx = np.arange(0, N, N // label_cnt)
    vis = list(map(lambda x: ('%.' + str(precision) + 'f') % x, arr[vis_idx]))
    return vis_idx, vis

print("Started visualizing...")
fig, ax = plt.subplots(1, 1)
vRange = np.arange(Nfft) * c / (2 * Nfft * (freq[1] - freq[0]))
rp_plot, = ax.plot(vRange, np.abs(np.fft.ifft(calFrame[0, :], Nfft)))

# rp_vis_idx, rp_vis = get_vis_labels(vRange, label_cnt=5, precision=1)
# ax.set_xticks(rp_vis_idx, labels=rp_vis)

plt.ion()
plt.show()

nframes = 100000
for i in range(nframes):
    # write_read(str(motion_stage[i]))
    vtrig.Record()
    rec = vtrig.GetRecordingResult()
    recArr = rec2arr(rec)
    # recCal = recArr[0] - calFrame[0]
    recCal = recArr[0]

    rp = np.abs(np.fft.ifft(recCal, Nfft))
    rp_plot.set_ydata(rp)
    ax.set_ylim(np.min(rp), np.max(rp))
    fig.canvas.draw_idle()

    plt.pause(0.001)



