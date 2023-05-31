from .isens_vtrigU import isens_vtrigU
import numpy as np
import os
import time

case = 'radar-tests'
scenario = 'cr-moving'
calibrate = True

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, '../data/{}/{}'.format(case, scenario))

radar = isens_vtrigU()

radar.scan_setup()
cfg = {
    'start_freq': radar.start_freq,
    'stop_freq': radar.stop_freq,
    'n_freq': radar.n_freq,
    'rbw': radar.rbw,
    'scan_profile': radar.scan_profile,
    'txRxPairs': radar.txRxPairs,
    'freq': radar.freq,
    'nfft': radar.nfft,
    'dist_vec': radar.dist_vec,
    'collect_time': time.time()
}

print("Radar initialized with configuration:")
print("Start frequency: {} MHz".format(cfg['start_freq']))
print("Stop frequency: {} MHz".format(cfg['stop_freq']))
print("RBW: {} kHz".format(cfg['rbw']))
print()

if calibrate:
    input("Press Enter to start calibration...")
    calFrame = radar.scan_calibration()

input("Press Enter to start recording...")
record_time = time.time()
recData = radar.scan_data()
record_time = time.time() - record_time

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if calibrate:
    np.save(os.path.join(data_dir, 'calibration.npy'), calFrame)
np.save(os.path.join(data_dir, 'recording.npy'), recData)
np.save(os.path.join(data_dir, 'config.npy'), cfg)

print("Data collection finished in {} seconds".format(record_time))
print("Data saved to {}".format(data_dir))

