from ..isens_vtrigU import isens_vtrigU
import numpy as np
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))

case = "20230306-tag-basic/tag-0.7m-2s"
data_path = os.path.join(current_path, '../data', case)
recording = np.load(os.path.join(data_path, 'recording.npy'))

fig, axes = plt.subplots(2, 1)

# Recording in shape (frames, n_txrx, n_freq)
phases = np.angle(recording[0])
axes[0].imshow(phases, aspect='auto')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Tx-Rx Pair')

axes[1].plot(phases[0])
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Phase')

plt.show()