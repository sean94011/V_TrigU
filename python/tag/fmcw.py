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

RBW = 80 # kHz
df = freqs[1] - freqs[0]
slope = df * RBW
t = np.arange(n_freq) / RBW
sent_sig = np.exp(-1j * 2 * np.pi * (freqs[0] + slope * t[::-1]))


# mixed_sig = data[0, 0, :] * sent_
mixed_sig = data[0, 0, :]
# mixed_sig = sent_sig * data[0, 0, :] * np.exp(1j * 2 * np.pi * freqs * t)

rp_fft = np.fft.fft(mixed_sig)
# rp_fft = np.fft.fft(np.conj(data[0, 0, :]))
rp_ifft = np.fft.ifft(data[0, 0, :])

fig, axes = plt.subplots(2, 1)

# plt.plot(np.angle(sent_sig))

axes[0].plot(np.abs(rp_fft) / np.max(np.abs(rp_fft)))
axes[0].plot(np.abs(rp_ifft) / np.max(np.abs(rp_ifft)))
axes[0].legend(['IFFT', 'FFT'])

axes[1].plot(np.angle(rp_fft))
axes[1].plot(np.angle(rp_ifft))
axes[1].legend(['IFFT', 'FFT'])

plt.show()