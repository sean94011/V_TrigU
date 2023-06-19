from ...isens_vtrigU import isens_vtrigU
import numpy as np
from matplotlib import pyplot as plt
import time
import os

radar = isens_vtrigU()

bws = [1, 2, 3, 4, 5, 6, 7] # GHz
nframes = 100
elapsed_time = np.zeros(len(bws))

for i in range(len(bws)):
    radar.scan_setup(rbw = 100, start_freq = 62e3, stop_freq = (62 + bws[i]) * 1e3)
    print("Collecting data for Bandwidth: {} GHz".format(bws[i]))

    curr_time = time.time()
    recData = radar.scan_data(nframes=nframes)
    elapsed_time[i] = (time.time() - curr_time) / nframes

plt.plot(bws, elapsed_time)
plt.xlabel('Bandwidth (GHz)')
plt.ylabel('Time per frame (s)')

plt.show()