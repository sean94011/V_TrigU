from ...isens_vtrigU import isens_vtrigU
import numpy as np
from matplotlib import pyplot as plt
import time
import os

radar = isens_vtrigU()

rbws = [10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800]
nframes = 20
elapsed_time = np.zeros(len(rbws))

case = 'anechoic-cr'
scenario = 'test'
calibrate = True

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, '../../../data/{}/{}'.format(case, scenario))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for i in range(len(rbws)):
    radar.scan_setup(rbw=rbws[i])
    print("Collecting data for RBW: {} kHz".format(radar.rbw))

    curr_time = time.time()
    recData = radar.scan_data(nframes=nframes)
    elapsed_time[i] = (time.time() - curr_time) / nframes

    np.save(os.path.join(data_dir, 'recording_{}kHz.npy'.format(radar.rbw)), recData)

plt.plot(rbws, elapsed_time)
plt.xlabel('RBW (kHz)')
plt.ylabel('Time per frame (s)')

plt.show()