import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, pi
from math import ceil, log
import scipy.signal
from isens_vtrigU import isens_vtrigU

RBW = 80 # in kHz

freq = np.load('./constants/freq.npy')
nframes = np.load('./constants/nframes.npy')
TxRxPairs = np.load('./constants/TxRxPairs.npy')

# define constants
N_txrx = TxRxPairs.shape[0]
N_freq = freq.shape[0]
Nfft = 2**(ceil(log(freq.shape[0],2))+1)
Ts = 1/Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts*(Nfft-1),num=Nfft)
dist_vec = time_vec*(c/2) # distance in meters

rad = isens_vtrigU()
# cal_data, rec_data = rad.load_data(case='20230306-tag-basic/', scenario='tag-0.7m-250us')
cal_data, rec_data = rad.load_data(case='20230410-horn', scenario='1s')
data = rad.calibration(cal_data, rec_data)
# data = rec_data
print(data.shape)

rp = rad.compute_tof_ifft(data)

def get_vis_labels(arr, label_cnt=3, precision=0):
    N = arr.shape[0]
    vis_idx = np.arange(0, N, N // label_cnt)
    vis = list(map(lambda x: ('%.' + str(precision) + 'f') % x, arr[vis_idx]))
    return vis_idx, vis

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1.set_xlabel("Range (m)")
ax2.set_xlabel("Frames")

vRange = np.arange(Nfft) * c / (2 * Nfft * (freq[1] - freq[0]))
rp_vis_idx, rp_vis = get_vis_labels(vRange, label_cnt=5, precision=1)
ax1.set_xticks(rp_vis_idx, labels=rp_vis)
rp_plot_per_frame, = ax1.plot(np.abs(rp[0]))
rp_plot_per_range, = ax2.plot(np.abs(rp[:, 0]))

frame_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(frame_slider_ax, 'Frame', 0, 99, valinit=0, valstep=1)
range_bin_slider_ax = plt.axes([0.25, 0.06, 0.65, 0.03])
range_bin_slider = Slider(range_bin_slider_ax, 'Range Bin', 0, Nfft-1, valinit=0, valstep=1)

rp_img = ax3.imshow(rp[:, ::-1].T, aspect='auto')
ax3.set_yticks(rp_vis_idx[::-1], labels=rp_vis)
ax3.set_ylabel("Range (m)")
ax3.set_xlabel("Frames")
fig.colorbar(rp_img, ax=ax3)

def rp_per_frame(val):
    frame_idx = frame_slider.val

    rp_single_frame = rp[frame_idx]
    rp_plot_per_frame.set_ydata(rp_single_frame)
    fig.canvas.draw_idle()
    ax1.set_ylim(np.min(rp_single_frame), np.max(rp_single_frame))

def rp_per_range(val):
    range_idx = range_bin_slider.val

    rp_single_range = rp[:, range_idx]
    rp_plot_per_range.set_ydata(rp_single_range)
    fig.canvas.draw_idle()
    ax2.set_ylim(np.min(rp_single_range), np.max(rp_single_range))
    ax2.set_title(f'Range: {vRange[range_idx]}m')

frame_slider.on_changed(rp_per_frame)
range_bin_slider.on_changed(rp_per_range)

plt.show()