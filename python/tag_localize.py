import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, pi
from math import ceil, log
import scipy.signal
from vtrig import isens_vtrigU

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
cal_data, rec_data, _ = rad.load_data(case='tag/', scenario='test')
data = rad.calibration(cal_data, rec_data)

rp = rad.compute_tof(data)

def get_vis_labels(arr, label_cnt=3, precision=0):
    N = arr.shape[0]
    vis_idx = np.arange(0, N, N // label_cnt)
    vis = list(map(lambda x: ('%.' + str(precision) + 'f') % x, arr[vis_idx]))
    return vis_idx, vis

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.set_xlabel("Range (m)")

rp_vis_idx, rp_vis = get_vis_labels(np.arange(Nfft) * c / (2 * Nfft * (freq[1] - freq[0])), label_cnt=5, precision=1)
ax.set_xticks(rp_vis_idx, labels=rp_vis)
rp_plot, = ax.plot(np.abs(rp[0]))

frame_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(frame_slider_ax, 'Frame', 0, 99, valinit=0, valstep=1)

def localize(val):
    frame_idx = frame_slider.val

    rp_single_frame = rp[frame_idx]
    rp_plot.set_ydata(rp_single_frame)
    fig.canvas.draw_idle()
    ax.set_ylim(np.min(rp_single_frame), np.max(rp_single_frame))

frame_slider.on_changed(localize)

plt.show()