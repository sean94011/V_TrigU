import numpy as np
import vtrigU as vtrig
from math import ceil, log
from scipy.constants import c
import os
import time
import datetime

def main():
    for old_data in os.listdir('./parameters'):
        os.remove(os.path.join('./parameters',old_data))
    # Radar Setup
    start_freq = 62.0*1000
    stop_freq = 69.0*1000
    num_freq_step = 150
    rbw = 10
    # FFT Bins
    range_Nfft = 512
    angle_Nfft = [64, 64]
    # Doppler Setup
    doppler_window_size = 15
    doppler_Nfft = None
    if doppler_Nfft is None:
        doppler_Nfft = 2**(ceil(log(doppler_window_size,2))+1)
    
    print('Setting up the radar...')
    # initialize the device
    vtrig.Init()
    # set setting structure
    vtrigSettings = vtrig.RecordingSettings(
            vtrig.FrequencyRange(start_freq, # Start Frequency (in MHz)
                                 stop_freq, # Stop  Frequency (in MHz) (66.5 for 5m) (68.0 for 3m)
                                 num_freq_step),      # Number of Frequency Points (Maximum: 150)
            rbw,                           # RBW (in KHz)
            vtrig.VTRIG_U_TXMODE__LOW_RATE  # Tx Mode (LOW: 20 Tx, MED: 10 Tx, HIGH: 4 Tx)
            ) 

    # validate settings
    vtrig.ValidateSettings(vtrigSettings)

    # apply settings
    vtrig.ApplySettings(vtrigSettings)
    print('Done')

    # Calibrating Doppler frame duration
    print('Calibrating Frame Duration for Doppler...')
    frame_duration = []
    for i in range(10):
        data_queue = sorted(os.listdir('./data_queue'))
        start = time.time()
        vtrig.Record()
        rec = vtrig.GetRecordingResult()
        rec_arr = rec2arr(rec)
        pro_arr = rec_arr - rec_arr
        if len(data_queue) >= doppler_window_size*2:
            os.remove(os.path.join('./data_queue',data_queue[0]))
        np.save(f'./data_queue/{datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}_{time.time_ns()}.npy',pro_arr)
        frame_duration.append(time.time()-start)
    frame_duration = np.mean(frame_duration)
    d = frame_duration
    # d = 0.2 #0.24
    # d = 1/fs
    doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    doppler_freq = doppler_freq[doppler_freq>=0]

    for old_data in os.listdir('./data_queue'):
        os.remove(os.path.join('./data_queue',old_data))
    print('Done')
    # get antenna pairs and convert to numpy matrix
    TxRxPairs = np.array(vtrig.GetAntennaPairs(vtrigSettings.mode))

    # get used frequencies in Hz
    freq = np.array(vtrig.GetFreqVector_MHz()) * 1e6

    # define constants
    Ts = 1/range_Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
    time_vec = np.linspace(0,Ts*(range_Nfft-1),num=range_Nfft)
    dist_vec = time_vec*(c/2) # distance in meters

    # Parameter Setup
    if angle_Nfft[0] == 64:
        x_offset_shift = -11
        x_ratio = 20/(34.2857)
    elif angle_Nfft[0] == 512: #(Nfft=512)
        x_offset_shift = -90
        x_ratio = 20/30
    else: #(Nfft=512)
        x_offset_shift = 0
        x_ratio = 1

    if angle_Nfft[1] == 64:
        y_offset_shift = 27
        y_ratio = 20/29
    elif angle_Nfft[1] == 512: #(Nfft=512)
        y_offset_shift = 220 
        y_ratio = 20/25
    else: #(Nfft=512)
        y_offset_shift = 0 
        y_ratio = 1

    
    # Data Formation
    AoD_vec = (np.linspace(-90,90,angle_Nfft[0]))*x_ratio
    AoA_vec = (np.linspace(-90,90,angle_Nfft[1]))*y_ratio
    
    
    print("Saving parameters to './parameters' ...")
    save_params(
        # Radar Setup
        start_freq = start_freq,
        stop_freq = stop_freq,
        num_freq_step = num_freq_step,
        rbw = rbw,
        # Data Formation
        TxRxPairs = TxRxPairs,
        freq = freq,
        dist_vec = dist_vec,
        AoD_vec = AoD_vec,
        AoA_vec = AoA_vec,
        doppler_freq = doppler_freq,
        # FFT Bins
        range_Nfft = range_Nfft,
        angle_Nfft = angle_Nfft,
        doppler_Nfft = doppler_Nfft,
        # Data Calibration
        x_offset_shift = x_offset_shift,
        y_offset_shift = y_offset_shift,
        x_ratio =  x_ratio,
        y_ratio = y_ratio,
        doppler_window_size = doppler_window_size
    )
    print('Done')
    return

def rec2arr(rec):
    recArr = []
    for key in rec.keys():
        recArr.append(rec[key])
    return np.array(recArr)

def save_params(
        # Radar Setup
        start_freq,
        stop_freq,
        num_freq_step,
        rbw,
        # Data Formation
        TxRxPairs,
        freq,
        dist_vec,
        AoD_vec,
        AoA_vec,
        doppler_freq,
        # FFT Bins
        range_Nfft,
        angle_Nfft,
        doppler_Nfft,
        # Data Calibration
        x_offset_shift,
        y_offset_shift,
        x_ratio,
        y_ratio,
        doppler_window_size
):
    np.save('./parameters/start_freq.npy', start_freq)
    np.save('./parameters/stop_freq.npy', stop_freq)
    np.save('./parameters/num_freq_step.npy', num_freq_step)
    np.save('./parameters/rbw.npy', rbw)
    np.save('./parameters/TxRxPairs.npy', TxRxPairs)
    np.save('./parameters/freq.npy', freq)
    np.save('./parameters/dist_vec.npy', dist_vec)
    np.save('./parameters/AoD_vec.npy', AoD_vec)
    np.save('./parameters/AoA_vec.npy', AoA_vec)
    np.save('./parameters/doppler_freq.npy', doppler_freq)
    np.save('./parameters/range_Nfft.npy', range_Nfft)
    np.save('./parameters/angle_Nfft.npy', angle_Nfft)
    np.save('./parameters/doppler_Nfft.npy', doppler_Nfft)
    np.save('./parameters/x_offset_shift.npy', x_offset_shift)
    np.save('./parameters/y_offset_shift.npy', y_offset_shift)
    np.save('./parameters/x_ratio.npy', x_ratio)
    np.save('./parameters/y_ratio.npy', y_ratio)
    np.save('./parameters/ant_loc.npy',ants_locations())
    np.save('./parameters/doppler_window_size.npy',doppler_window_size)

    return





def load_params():
    parameters = {}
    parameters['start_freq'] = np.load('./parameters/start_freq.npy')
    parameters['stop_freq'] = np.load('./parameters/stop_freq.npy')
    parameters['num_freq_step'] = np.load('./parameters/num_freq_step.npy')
    parameters['rbw'] = np.load('./parameters/rbw.npy')
    parameters['TxRxPairs'] = np.load('./parameters/TxRxPairs.npy')
    parameters['freq'] = np.load('./parameters/freq.npy')
    parameters['dist_vec'] = np.load('./parameters/dist_vec.npy')
    parameters['AoD_vec'] = np.load('./parameters/AoD_vec.npy')
    parameters['AoA_vec'] = np.load('./parameters/AoA_vec.npy')
    parameters['doppler_freq'] = np.load('./parameters/doppler_freq.npy')
    parameters['range_Nfft'] = np.load('./parameters/range_Nfft.npy')
    parameters['angle_Nfft'] = np.load('./parameters/angle_Nfft.npy')
    parameters['doppler_Nfft'] = np.load('./parameters/doppler_Nfft.npy')
    parameters['x_offset_shift'] = np.load('./parameters/x_offset_shift.npy')
    parameters['y_offset_shift'] = np.load('./parameters/y_offset_shift.npy')
    parameters['x_ratio'] = np.load('./parameters/x_ratio.npy')
    parameters['y_ratio'] = np.load('./parameters/y_ratio.npy')
    parameters['ant_loc'] = np.load('./parameters/ant_loc.npy')
    parameters['doppler_window_size'] = np.load('./parameters/doppler_window_size.npy')

    return parameters

def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def ants_locations():
    return np.array([[-0.0275, -0.0267, 0], # tx
                     [-0.0253, -0.0267, 0],
                     [-0.0231, -0.0267, 0],
                     [-0.0209, -0.0267, 0],
                     [-0.0187, -0.0267, 0],
                     [-0.0165, -0.0267, 0],
                     [-0.0143, -0.0267, 0],
                     [-0.0122, -0.0267, 0],
                     [-0.0100, -0.0267, 0],
                     [-0.0078, -0.0267, 0],
                     [-0.0056, -0.0267, 0],
                     [-0.0034, -0.0267, 0],
                     [-0.0012, -0.0267, 0],
                     [ 0.0009, -0.0267, 0],
                     [ 0.0031, -0.0267, 0],
                     [ 0.0053, -0.0267, 0],
                     [ 0.0075, -0.0267, 0],
                     [ 0.0097, -0.0267, 0],
                     [ 0.0119, -0.0267, 0],
                     [ 0.0141, -0.0267, 0],
                     [ 0.0274, -0.0133, 0], # rx
                     [ 0.0274, -0.0112, 0],
                     [ 0.0274, -0.0091, 0],
                     [ 0.0274, -0.0070, 0],
                     [ 0.0274, -0.0049, 0],
                     [ 0.0274, -0.0028, 0],
                     [ 0.0274, -0.0007, 0],
                     [ 0.0275,  0.0014, 0],
                     [ 0.0275,  0.0035, 0],
                     [ 0.0275,  0.0056, 0],
                     [ 0.0275,  0.0078, 0],
                     [ 0.0275,  0.0099, 0],
                     [ 0.0275,  0.0120, 0],
                     [ 0.0274,  0.0141, 0],
                     [ 0.0274,  0.0162, 0],
                     [ 0.0275,  0.0183, 0],
                     [ 0.0275,  0.0204, 0],
                     [ 0.0275,  0.0225, 0],
                     [ 0.0275,  0.0246, 0],
                     [ 0.0275,  0.0267, 0]])




if __name__ == '__main__':
    main()