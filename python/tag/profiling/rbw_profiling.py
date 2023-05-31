from ...isens_vtrigU import isens_vtrigU
import numpy as np
from matplotlib import pyplot as plt
import time

radar = isens_vtrigU()

rbws = np.arange(10, 800, 20)
nframes = 50
elapsed_time = np.zeros(len(rbws))

for i in range(len(rbws)):
    radar.scan_setup(rbw=rbws[i])
    print("Collecting data for RBW: {} kHz".format(radar.rbw))

    curr_time = time.time()
    radar.scan_data(nframes=nframes)
    elapsed_time[i] = (time.time() - curr_time) / nframes

plt.plot(rbws, elapsed_time)
plt.xlabel('RBW (kHz)')
plt.ylabel('Time per frame (s)')

plt.show()