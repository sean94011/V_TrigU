# ! Run in base project directory: `python -m python.tag.multiple_measures`
from ..isens_vtrigU import isens_vtrigU
import numpy as np
import matplotlib.pyplot as plt
import time

radar = isens_vtrigU()

c = 2.99792458e8
n_freq = 150
n_txrx = radar.TxRxPairs.shape[0]

Rmax = 100 # max range in meters
Rres = 0.02 # range resolution in meters
df = c / (2 * Rmax) # step frequency in Hz
total_steps = int(Rmax / Rres) # number of steps
frames = int(np.ceil(total_steps / n_freq))

print("Frequency per step: {} MHz".format(df / 1e6))
print("Total steps: {}".format(total_steps))
print("Total bandwidth: {} MHz".format(total_steps * df / 1e6))
print("Total frames: {}".format(frames))

start_freqs = 62e3 + np.arange(frames) * df * n_freq / 1e6
end_freqs = start_freqs + df * n_freq / 1e6
print("Start frequencies: {} MHz".format(start_freqs))
print("End frequencies: {} MHz".format(end_freqs))

# Original configuration, for debugging purposes
# start_freqs = [62e3]
# end_freqs = [69e3]
# frames = 1

Nfft = int(2**(np.ceil(np.log(n_freq * frames)/np.log(2))+1))
Ts = 1/Nfft/(df+1e-16)
time_vec = np.linspace(0,Ts*(Nfft-1),num=Nfft)
dist_vec = time_vec*(c/2) # frequency bins -> range

if np.max(end_freqs) > 69e3:
    raise Exception("End frequency exceeds maximum")

start_time = time.time()
data = np.zeros((n_txrx, frames * n_freq), dtype=np.complex64)
for i in range(frames):
    radar.scan_setup(start_freqs[i], end_freqs[i])
    recArrs = radar.scan_data(nframes=1, print_progress=False) # shape (1, 400, 150)
    data[:, i*n_freq:(i+1)*n_freq] = recArrs[0]
    print("Finished collecting frame #{} ({} ~ {} MHz)".format(i, start_freqs[i], end_freqs[i]))
end_time = time.time()
print("Time elapsed: {} seconds".format(end_time - start_time))

rp = np.linalg.norm(np.fft.ifft(data, Nfft, axis=1), axis=0)
plt.plot(dist_vec, rp)
plt.xlabel('Range (m)')
plt.ylabel('Amplitude')
plt.show()