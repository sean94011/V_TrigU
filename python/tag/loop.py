import numpy as np
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '../data/20230516-cr/cr-2m')

data = np.load(os.path.join(data_path, 'recording.npy'))

# data in dimensions (nframes, n_txrx, n_freq)
freqs = np.load(os.path.join(current_path, '../constants/freq.npy'))
n_freq = freqs.shape[0]
nframes = data.shape[0]

sig = data[0, 0, :]
shifted = np.zeros((n_freq, n_freq), dtype=np.complex64)
for i in range(n_freq):
    shifted[i, :] = np.roll(sig, i)

rp_ifft = np.fft.ifft(shifted, axis=1)

plt.imshow(np.abs(rp_ifft), aspect='auto')
plt.show()