"""example code for EVK client"""
from struct import unpack_from
import json
import numpy as np
import time
from websocket import create_connection
import subprocess
import datetime
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt


DTYPES = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.float32,
    7: np.float64,
}

ASCII_RS = '\u001e'
ASCII_US = '\u001f'

# for user configuration
target_smat_dir = r'.\Target_Recordings' # directory to target smat recordings
background_smat_dir = r'.\Leakage_Recordings' # directory to background smat recordings
metadata_directory = r'.\AP_012_Package'  # directory for saving frame metadata (freqs,pairs)
save_IQ_dir_background = r'.\Background_PostProcessed'  # directory for saving background IQ frames
save_IQ_dir_target = r'.\Target_PostProcessed' # directory for saving background IQ frames
Target_num_of_frames_to_avg = 12  # num of target frames to perform average on
Background_num_of_frames_to_avg = 1  # num of background frames to perform average on!
Tx_ant_to_plot = 23  # TX antenna to plot
Rx_ant_to_plot = 1  # RX antenna to plot

# Web Socket connection def
def to_message(buffer):
    if isinstance(buffer, str):
        return json.loads(buffer)
    seek = 0
    fields_len = unpack_from('i', buffer, seek + 4)[0]
    fields_split = unpack_from(str(fields_len) + 's', buffer, seek + 8)[0].decode('utf8').split(ASCII_RS)
    msg = {'ID': fields_split[0], 'Payload': dict.fromkeys(fields_split[1].split(ASCII_US))}
    seek += 8 + fields_len
    for key in msg['Payload']:
        seek += np.int32().nbytes
        dtype = DTYPES[np.asscalar(np.frombuffer(buffer, np.int32, 1, seek))]
        seek += np.int32().nbytes
        ndims = np.asscalar(np.frombuffer(buffer, np.int32, 1, seek))
        seek += np.int32().nbytes
        dims = np.frombuffer(buffer, np.int32, ndims, seek)
        seek += ndims * np.int32().nbytes
        data = np.frombuffer(buffer, dtype, np.prod(dims), seek)
        seek += np.prod(dims) * dtype().nbytes
        msg['Payload'][key] = data.reshape(dims) if ndims else np.asscalar(data)
    return msg

# frequency domain to time domain using ifft and zero padding.
def TimeDomainFromS(Sig, Sfreqs, Nfft_and_samp_out, time_method):
    Nfft = Nfft_and_samp_out
    N_samp_out = Nfft
    Y_win = Sig
    y_td = np.fft.ifft(Y_win, n=Nfft, axis=1)
    Ts = 1/(Sfreqs[1]-Sfreqs[0]+(1/(1*np.power(10, 16))))/Nfft
    tvec = np.arange(0, Ts*N_samp_out, Ts)
    Phase_Vec = 1*np.power(10,(2*1j*np.pi*Sfreqs[1]*tvec))
    if time_method == 3:
        sig = np.multiply(y_td, Phase_Vec)
    else:
        error('Unknown time_method = %d.', time_method)
    return sig, tvec


def main():

    # """ connect to server and echoing messages """
    listener = create_connection("ws://127.0.0.1:1234/")
    def listener_def(data_dir):# retrieve current configuration
        listener.send(json.dumps({
            'Type': 'COMMAND',
            'ID': 'SET_PARAMS',
            'Payload': {
                'Cfg.MonitoredRoomDims': [-15, 15, 0.2, 15, 0, 1.8],
                'Cfg.Common.sensorOrientation.mountPlane': 'xz',
                'Cfg.Common.sensorOrientation.transVec(3)': 1.7,
                'Cfg.imgProcessing.substractionMode': 7.0,
                'MPR.save_dir': data_dir,
                'MPR.read_from_file': 1.0,
        }}))

    def STOP():# Send STOP command to engine
        listener.send(json.dumps({
            'Type': 'COMMAND',
            'ID': 'STOP',
            'Payload': {}
        }))

    def START():# Send Start command to engine
        listener.send(json.dumps({
            'Type': 'COMMAND',
            'ID': 'START',
            'Payload': {}
        }))
    def listener_send_params():# define required outputs
        listener.send(json.dumps({
        'Type': 'COMMAND',

        
        'ID': 'SET_OUTPUTS',
        'Payload': {
            'binary_outputs': ['I', 'Q', 'pairs', 'freqs'],
        }
        }))
    Target_index = 0 # Target frames counter
    Backgroung_index = 0 # Background frames counter
    listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
    listener_def(target_smat_dir)
    listener_send_params()
    START()
    listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
    while Target_index <= Target_num_of_frames_to_avg: # do for target frames until reaches num of target to avg
        buffer = listener.recv() # get data from engine
        data = to_message(buffer) # extract message
        print(data['ID'])
        if data['ID'] == 'BINARY_DATA':
            iMat = data['Payload']['I'] # extract I data
            qMat = data['Payload']['Q'] # extract Q data
            name = "IQ_" + str(Target_index)
            iqMat = iMat + qMat*1j # I and Q becomes I+jQ
            df = pd.DataFrame(iqMat)
            df.to_csv(save_IQ_dir_target + '\\' + name + ".csv")
            Target_index += 1
            listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
        if data['ID'] == 'GET_STATUS':
            print(data['Payload']['status'])
        if data['ID'] == 'SET_PARAMS':
            print("CONFIGURATION:")
            for key in data['Payload']:
                print(key, data['Payload'][key])
    STOP()
    listener_def(background_smat_dir) # set directory
    listener_send_params()
    START()
    listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
    while Backgroung_index <= Background_num_of_frames_to_avg: # do for background frames until reaches num of background to avg
        Break = 0
        buffer = listener.recv()
        data = to_message(buffer)
        print(data['ID'])
        if data['ID'] == 'BINARY_DATA':
            #Pairs, Freqs
            if Break == 0:
                name = "Pairs"
                PairsMat = data['Payload']['pairs'] # extract tx-rx pairs data
                with open(metadata_directory+'//' + name + ".csv", "w+") as my_csv:
                    csvWriter = csv.writer(my_csv)
                    csvWriter.writerows(PairsMat) # save pairs to
                name = "Freqs"
                FreqsMat = data['Payload']['freqs'] # extract frequency steps data
                with open(metadata_directory+'//' + name + ".csv", "w+") as my_csv:
                    csvWriter = csv.writer(my_csv)
                    csvWriter.writerows(map(lambda x: [x], FreqsMat)) # save frequencies to csv
            iMat = data['Payload']['I']
            qMat = data['Payload']['Q']
            name = "IQ_" + str(Backgroung_index)
            iqMat = iMat + qMat*1j
            df = pd.DataFrame(iqMat)
            df.to_csv(save_IQ_dir_background + '\\' + name + ".csv")
            Backgroung_index += 1
            listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
        if data['ID'] == 'GET_STATUS':
            print(data['Payload']['status'])
        if data['ID'] == 'SET_PARAMS':
            print("CONFIGURATION:")
            for key in data['Payload']:
                print(key, data['Payload'][key])
    STOP()
    pairs_data = genfromtxt(metadata_directory + '\Pairs.csv', delimiter=',') # get pairs data
    freqs_data = genfromtxt(metadata_directory + '\Freqs.csv', delimiter=',') # get frequency steps data
    rb = 0
# insert all background frames to 3d array of frames for averaging
    background_frames_3d = None
    while rb <= Background_num_of_frames_to_avg:
        with open(save_IQ_dir_background + '\\' + 'IQ_' + "%d" %rb + ".csv", 'r') as dest_b:
            data_iter = csv.reader(dest_b, delimiter=',', quotechar='"')
            data = [data[1:] for data in data_iter][1:]
        background_frames = np.asarray(data).astype(np.complex)
        if background_frames_3d is not None:
            background_frames_3d = np.dstack((background_frames_3d, background_frames))
        else:
            background_frames_3d = background_frames
        rb= rb+1

    rt = 0
    target_frames_3d = None
# insert all target frames to 3d array of frames for averaging
    while rt <= Target_num_of_frames_to_avg:
        with open(save_IQ_dir_target + '\\' + 'IQ_' + "%d" %rt + ".csv", 'r') as dest_t:
            data_iter = csv.reader(dest_t, delimiter=',', quotechar='"')
            data = [data[1:] for data in data_iter][1:]
        target_frames = np.asarray(data).astype(np.complex)
        if target_frames_3d is not None:
            target_frames_3d = np.dstack((target_frames_3d, target_frames))
        else:
            target_frames_3d = target_frames
        rt = rt+1
    Target_avg = np.average(target_frames_3d, axis=2) # average on target frames
    Background_avg = np.average(background_frames_3d, axis=2) # average on background  frames
    iq_to_time_domain = Target_avg - Background_avg # subtract background from target
    st_avg, tvec = TimeDomainFromS(iq_to_time_domain, freqs_data, np.power(2,13), 3) # convert to time domain
    pair_location = np.argwhere((pairs_data[:, 0] == Tx_ant_to_plot) & (pairs_data[:, 1] == Rx_ant_to_plot)) # find required tx-rx pair to plot
# Plot
    plt.figure
    plt.plot((1.5*np.float_power(10,10))*tvec, 10 * np.log10(st_avg[pair_location[0]].transpose()))
    plt.grid()
    plt.show()
    listener.close()

if __name__ == '__main__':
    main()