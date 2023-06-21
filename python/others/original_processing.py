import clr
import sys
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

# Import necessary classes from DLL
import vtrigU

# Initialize the device
vtrigU.Init()

# Set the settings structure
freq_range = FreqRange()
freq_range.freqStartMHz = 62 * 1000
freq_range.freqStopMHz = 67 * 1000
freq_range.numFreqPoints = 150
rbw_khz = 40
tx_mode = TxMode.LOW_RATE
settings = Settings()
settings.freqRange = freq_range
settings.rbw_khz = rbw_khz
settings.txMode = tx_mode
vtrigU.ApplySettings(settings.freqRange.freqStartMHz,
                     settings.freqRange.freqStopMHz,
                     settings.freqRange.numFreqPoints,
                     settings.rbw_khz,
                     settings.txMode)

# Validate settings
my_settings = vtrigU.GetSettings()
vtrigU.ValidateSettings(my_settings)

# Get antenna pairs and convert to NumPy array
ants_pairs = vtrigU.GetAntennaPairs(my_settings.txMode)
tx_rx_pairs = np.zeros((len(ants_pairs), 2))
for ii in range(len(ants_pairs)):
    tx_rx_pairs[ii, 0] = ants_pairs[ii].tx
    tx_rx_pairs[ii, 1] = ants_pairs[ii].rx

# Get used frequencies in Hz
freq = np.array(vtrigU.GetFreqVector_MHz()) * 1e6

# Get antenna locations from script
sys.path.append(r'path/to/vtrigU_ants_location.py')
from vtrigU_ants_location import VtrigU_ants_location

# Define constants
N_txrx = tx_rx_pairs.shape[0]
N_freq = len(freq)
Nfft = 2**(np.ceil(np.log2(freq.shape[0]))+1)
Ts = 1/Nfft/(freq[1]-freq[0]+1e-16)
time_vec = np.arange(0, Ts*(Nfft-1)+Ts, Ts)
dist_vec = time_vec*1.5e8

xgrid = [0.18, 0.18] #-0.3:0.1:0.3
ygrid = np.arange(-0.3, 0.3+0.03, 0.03)
zgrid = np.arange(0.1, 1+0.02, 0.02)
Xgrid, Ygrid, Zgrid = np.meshgrid(xgrid, ygrid, zgrid)

src = np.concatenate((Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1), Zgrid.reshape(-1, 1)), axis=1)
src2 = np.transpose(src.reshape((src.shape[0],) + Xgrid.shape[::-1]), (2, 1, 3, 0))

# Use radar equation to find total loss
# Pe/Ps = (Gtx*Grx*RCS*lambda^2)/((4pi)^3*|Rtx|^2*|Rrx|^2)
Rvec = src2 - VtrigU_ants_location
Rmag = np.sqrt(np.sum(np.square(Rvec), axis=1))
Rtheta = np.arctan2(np.sqrt(np.sum(np.square(Rvec[:, 0:2, :, :]), axis=1)), Rvec[:, 2, :, :])
Rphi = np.arctan2(Rvec[:, 1, :, :], Rvec[:, 0, :, :])
Sphase = 2*np.pi*Rmag[:, :, :, np.newaxis] * freq[np.newaxis, np.newaxis, :, np.newaxis] / c  # Electrical Length in Radians
RCS = 1  # m^2
lambda_ = c / freq
csf = np.sqrt(RCS) * lambda_ / ((4*np.pi)**(3/2))
Smag = 10**(5.8/20) * RadiationPattern(Rtheta, Rphi) / Rmag[:, :, :, np.newaxis]
H2 = np.zeros((src.shape[0], tx_rx_pairs.shape[0], freq.shape[0], 1), dtype=np.complex)
for ii in range(tx_rx_pairs.shape[0]):
    tx = int(tx_rx_pairs[ii, 0])
    rx = int(tx_rx_pairs[ii, 1])
    H2[:, ii, :, :] = 1 / (csf * Smag[tx, :, :, :] * Smag[rx, :, :, :] *
                           np.exp(-1j * (Sphase[:, :, :, tx] + Sphase[:, :, :, rx])))
H2 = H2.reshape(src2.shape[0], -1)  # xyz x txrx x freq

# Start recording
nfig = 4
fig = [None] * nfig
for ii in range(nfig):
    fig[ii] = plt.figure()

while True:
    first_iter = True
    input('continue?')
    nrecs = 50
    recs = np.zeros((tx_rx_pairs.shape[0], freq.shape[0], nrecs))
    for kk in range(nrecs):
        vtrigU.Record()
        rec = np.array(vtrigU.GetRecordingResult(SignalCalibration.DEFAULT_CALIBRATION))
        X = rec.astype(np.complex128)

        # Convert and reshape result to complex matrix
        smat_size = (tx_rx_pairs.shape[0], freq.shape[0], 2)
        X = X.reshape(smat_size[0], smat_size[1], smat_size[2]).transpose((0, 2, 1))
        X = X[:, :, 0] + 1j * X[:, :, 1]
        recs[:, :, kk] = X

        if kk == 0:
            plt.figure(fig[0].number)
            plt.clf()
            for ll in range(smat_size[0]):
                plt.plot(freq, 20*np.log10(np.abs(X[ll, :])))
            plt.plot(freq, 20*np.log10(np.sqrt(np.sum(np.abs(X)**2, axis=0))) -
                     10*np.log10(smat_size[0]), 'k.', linewidth=4)
            plt.title('Channel response (per channel and rms average)')

            # Identify resonant frequencies
            thresh = 3
            lnconv = min(max(np.floor(freq.shape[0] / 8) * 2 + 1, np.floor(50 / (freq[1] - freq[0])) * 2 + 1), np.floor(3 * freq.shape[0] / 8) * 2 + 1)
            c2 = -np.ones((int(lnconv),)) / (lnconv - 1)
            c2[(int(lnconv) + 1) // 2] = 1
            padsig = 20 * np.log10(rssq(X, axis=0))
            padsig = np.concatenate([padsig[(int(lnconv) - 1) // 2::-1], padsig, padsig[-1:-int(lnconv) // 2 - 1:-1]])
            padsig = np.convolve(padsig, c2, mode='valid')
            f_res = padsig > thresh
        
        #Remove resonant frequencies
        X = X * (1-f_res)
        
        #convert to complex time domain signal
        x = np.fft.ifft(X,Nfft,1)
        y_cart = np.reshape(H2.dot(X.flatten()),Xgrid.shape)

        #Create and show power delay profile - non coherent summation
        PDP = rssq(x,1)
        plt.figure(fig[2])
        plt.plot(dist_vec,20*np.log10(np.abs(PDP/np.max(np.abs(PDP)))))
        
        if first_iter:
            plt.title('Power Delay Profile - non coherent summation')
            plt.ylim([0,2])
            plt.xlabel('Distance[m]')
            plt.ylabel('Normalized amplitude[dB]')
        
        #Y = permute(mean(H.*X,1),[3,4,2,1]);
        #y = ifft(Y,Nfft,3);

        #y_cart = interp3(Theta3,Phi3,R3,y,Thetagrid,Phigrid,Rgrid,'linear');
        #y_cart = y_cart./RadPattern.^2;
        
        ## Scatter Plot 
        if np.min([len(xgrid),len(ygrid),len(zgrid)])>2:
            th = np.abs(y_cart)>1
            fig = plt.figure(fig[1])
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Xgrid[th],Ygrid[th],Zgrid[th], c=20*np.log10(np.abs(y_cart[th])))
            if first_iter:
                ax.set_title('Point Cloud')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_xlim(xgrid[0],xgrid[-1])
                ax.set_ylim(ygrid[0],ygrid[-1])
                ax.set
        




