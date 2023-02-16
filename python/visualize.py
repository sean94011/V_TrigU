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

from vtrigU_helper_functions import *

# define some constants
c = constants.c
antsLocations = ants_locations()

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
print("Calibrating...")
nrecs = 10
calFrame = []
for i in range(nrecs):
    vtrig.Record()
    rec = vtrig.GetRecordingResult()
    recArr = rec2arr(rec)
    calFrame.append(recArr)
calFrame = np.mean(calFrame,axis=0)
print("Calibration matrix collected!")

print("Started visualizing...")
nframes = 10000
for i in range(nframes):
    # write_read(str(motion_stage[i]))
    vtrig.Record()
    rec = vtrig.GetRecordingResult()



