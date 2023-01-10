# import libraries
import os
import time
from math import ceil, log

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import constants

from vtrigU_helper_functions import *

file_start = time.time()
print("loading constants...")
# define some constants
c = constants.c
antsLocations = ants_locations()

# load configurations
freq = np.load("./constants/freq.npy")
TxRxPairs = np.load("./constants/TxRxPairs.npy")

# define constants
N_txrx = TxRxPairs.shape[0]
N_freq = freq.shape[0]

Nfft = 2**(ceil(log(freq.shape[0],2))+1)
Ts = 1/Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
time_vec = np.linspace(0,Ts*(Nfft-1),num=Nfft)
dist_vec = time_vec*(c/2) # distance in meters

print("done...")
print("")
print("start constructing the grid...")
# xgrid = np.array([0,0]
xgrid = np.arange(-0.3, 0.3+0.025, 0.0215)
ygrid = np.arange(-0.3, 0.3+0.025, 0.0215)
zgrid = np.arange( 0.05, 2+0.025,   0.0215)

[Xgrid,Ygrid,Zgrid] = np.meshgrid(xgrid,ygrid,zgrid)

src = np.stack((Xgrid, Ygrid, Zgrid),3).reshape((-1,3,1,1))
src2 = np.transpose(src,[2,1,3,0])

# Use Radar equation to find total loss
# Pe/Ps = (Gtx*Grx*RCS*lambda^2)/((4pi)^3*|Rtx|^2*|Rrx|^2)
Rvec = src2 - np.expand_dims(antsLocations,axis=[2,3])

Rmag = np.expand_dims(norm(Rvec,axis=1),axis=1)
Rtheta = np.expand_dims(np.arctan2(norm(Rvec[:,0:1,:,:],axis=1),Rvec[:,2,:,:]),axis=1)
Rphi = np.expand_dims(np.arctan2(Rvec[:,1,:,:],Rvec[:,0,:,:]),axis=1)
Sphase = 2*np.pi*Rmag*np.expand_dims(freq,axis=[0,2,3])/c #Electrical Length in Radians
curRCS = 1 #m^2
curLambda = c/freq 
csf = np.sqrt(curRCS)*curLambda/((4*np.pi)**(3/2)) 
Smag = 10**(5.8/20)*RadiationPattern(Rtheta,Rphi)/Rmag

H2 = np.zeros((len(TxRxPairs),len(freq),1,src2.shape[3]),dtype = 'complex_')
for i in range(len(TxRxPairs)):
    tx = TxRxPairs[i,0]-1
    rx = TxRxPairs[i,1]-1
    H2[i,:,:,:] = 1/(csf.reshape(1,-1,1,1)*Smag[tx,:,:,:]*Smag[rx,:,:,:]*np.exp(-1j*(Sphase[tx,:,:,:]+Sphase[rx,:,:,:])))

H2 = np.transpose(H2,[3,0,1,2]).reshape((src2.shape[3],-1))
print("construction done...")
print("")

# define how many frames for the recording
nframes = np.load("./constants/nframes.npy")

print("start processing data...")
# loop through the collected files
directory = "./data"
for scenario in os.listdir(directory):
    cur_folder = f"{directory}/{scenario}"
    print(f"current scenario: {scenario}")
    print("")
    calFrame = np.load(f"{cur_folder}/raw_data/calibration.npy")
    recArrs = np.load(f"{directory}/{scenario}/raw_data/recording.npy")
    # Record frame and then process it to cartesian coordinates
    for frame in range(nframes):
        print(f"frame: {frame}")
        print("computing...")
        if f"y_cart_{frame}.npy" not in os.listdir(f"{cur_folder}/processed_data"):
            start = time.time()
            recArr = recArrs[frame]
            X = (recArr-calFrame).astype(complex)
            if frame == 0:           
                # Identify resonant frequencies 
                thresh = 3
                lnconv = int(min(max(np.floor(N_freq/8)*2+1,np.floor(50/(freq[1]-freq[0]))*2+1),np.floor(3*N_freq/8)*2+1)); #conv length between 1/4 and 3/4 N_freq
                c2 = -np.ones(lnconv)/(lnconv-1)
                c2[(lnconv+1)//2] = 1
                padsig = 20*np.log10(norm(X,axis=0))
                padsig = np.concatenate([padsig[(lnconv-1)//2:0:-1],padsig,padsig[-1:len(padsig)-(lnconv-1)//2-1:-1]])
                padsig = np.convolve(padsig,c2,'valid')        
                f_res = padsig>thresh
            X *= (1-f_res) # try calibrate here later
            y_cart = (H2@X.reshape(-1,1)).reshape(Xgrid.shape)
            stop = time.time()
            comp_time = stop-start
            np.save(f"{cur_folder}/processed_data/X_{frame}.npy", X)
            np.save(f"{cur_folder}/processed_data/y_cart_{frame}.npy", y_cart)
        else:
            y_cart = np.load("f{cur_folder}/processed_data/y_cart_{frame}.npy")
        print(f"...done. computation time: {comp_time}")

file_end = time.time()
print(f"done... total time: {file_end-file_start}")
        # # plot 3D point cloud
        # if min([len(xgrid),len(ygrid),len(zgrid)])>2:
        #     print("plotting 3D point cloud...")
        #     th = abs(y_cart)>1500
        #     plt.ion()
        #     fig = plt.figure(figsize=(10,10))
        #     ax = fig.add_subplot(projection='3d')   
        #     ax.scatter(Xgrid[th],Ygrid[th],Zgrid[th])
        #     scatter_arr = np.array([Xgrid[th],Ygrid[th],Zgrid[th]])
        #     np.save(f"{cur_folder}/processed_data/scatter_frame_{frame}.npy", scatter_arr)
        #     # ax.scatter(Xgrid*th,Ygrid*th,Zgrid*th, c='blue')
        #     ax.view_init(0, 0)
        #     if frame == 0:
        #         ax.set_title('Point Cloud')
        #         ax.set_xlabel('x')
        #         ax.set_ylabel('y')
        #         ax.set_label('z')
        #         ax.set_aspect('auto')
        #     plt.savefig(f'{cur_folder}/plots/point_cloud/point_cloud_frame_{frame}.png')
        #     plt.close()
        #     print("...done.")

        
        # # plot PDP
        # print("plotting PDP...")
        # PDP = computePDP(X,Nfft)
        # plt.figure(figsize=(20,5))
        # plt.plot(dist_vec,convert2db(PDP))
        # plt.ylim((-50,50))
        # plt.savefig(f'{cur_folder}/plots/PDP/PDP_frame_{frame}.png')
        # plt.close()
        # print("...done.")
        # print("")

