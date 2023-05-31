import numpy as np
from matplotlib import pyplot as plt
import os

data_dir = 'data/tv-color/'
colors = ['black', 'red', 'green', 'blue']
# colors = ['black-far', 'red-far', 'green-far', 'blue-far']
n_colors = len(colors)

# np.load(os.path.join(data_dir, 'black/recording.npy'))

cfg = np.load(os.path.join(data_dir, 'black/config.npy'), allow_pickle=True).item()
print(cfg)
nfft = cfg['nfft']
dist_vec = cfg['dist_vec']
nfft_doppler = 100
doppler_vec = 10 / nfft_doppler * np.arange(0, nfft_doppler)

def process(scenario_dir):
    recData = np.load(os.path.join(scenario_dir, 'recording.npy'))
    calFrame = np.load(os.path.join(scenario_dir, 'calibration.npy'))
    calData = recData[:, 0, :]

    rp = np.fft.ifft(calData, axis=1, n=nfft) # frames, range bins
    rd = np.fft.fft(rp, axis=0, n=nfft_doppler) # doppler, range bins

    return rp, rd

reference = np.zeros((nfft_doppler, nfft), dtype=np.complex64)
fig, axes = plt.subplots(2, n_colors)
for i in range(n_colors):
    rp, rd = process(os.path.join(data_dir, '{}'.format(colors[i])))
    tv_bin = np.argmax(np.sum(np.abs(rp), axis=0)[:nfft//2], axis=0)

    # axes[0, i].imshow(np.abs(rp.T), aspect='auto')
    axes[0, i].plot(dist_vec, np.abs(rp[0, :]))
    axes[0, i].axvline(dist_vec[tv_bin], color='r')
    axes[0, i].set_title("Range Profile, {}".format(colors[i]))

    # img = axes[1, i].imshow(np.abs(rd[1:, :].T), aspect='auto')
    # axes[1, i].set_xlabel("Doppler Bins")
    # axes[1, i].set_ylabel("Range Bins")
    # axes[1, i].set_title("Range doppler, without DC, {}".format(colors[i]))
    # # fig.colorbar(img, ax=axes[0, i])

    start = 2
    end = nfft_doppler
    minusRef = np.abs(rd - reference)[start:end, tv_bin]
    # axes[1, i].vlines(doppler_vec[10:nfft_doppler//2:], 0, minusRef)
    axes[1, i].plot(doppler_vec[start:end], minusRef)
    axes[1, i].set_xlabel("Doppler Bins")
    axes[1, i].set_ylabel("Magnitude")
    axes[1, i].set_title("Range doppler {}".format(colors[i]))

    if n_colors == 0:
        reference = rd


plt.show()