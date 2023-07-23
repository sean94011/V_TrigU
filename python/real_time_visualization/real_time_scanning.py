# Import Libraries
import datetime
import time
import numpy as np
import vtrigU as vtrig
import os
import sys
from parameter_setup import load_params, rec2arr



def main():
    # load parameters
    params = load_params()
    data_queue_size = 200
    if data_queue_size is None:
        data_queue_size = params['doppler_window_size']*2
    print('Setting up the radar...')
    # initialize the device
    vtrig.Init()
    # set setting structure
    vtrigSettings = vtrig.RecordingSettings(
            vtrig.FrequencyRange(params['start_freq'], # Start Frequency (in MHz)
                                 params['stop_freq'], # Stop  Frequency (in MHz) (66.5 for 5m) (68.0 for 3m)
                                 params['num_freq_step']),      # Number of Frequency Points (Maximum: 150)
            params['rbw'],                           # RBW (in KHz)
            vtrig.VTRIG_U_TXMODE__LOW_RATE  # Tx Mode (LOW: 20 Tx, MED: 10 Tx, HIGH: 4 Tx)
            ) 

    # validate settings
    vtrig.ValidateSettings(vtrigSettings)

    # apply settings
    vtrig.ApplySettings(vtrigSettings)
    print('Done')
    # Record the calibration frames
    recalibrate = input('Recalibrate? (Y/n)')
    if recalibrate == 'Y':
        # time.sleep(10)
        print("Collecting calibration data...")
        nrecs = 30
        calFrame = []
        for i in range(nrecs):
            vtrig.Record()
            rec = vtrig.GetRecordingResult()
            recArr = rec2arr(rec)
            calFrame.append(recArr)
        calFrame = np.array(calFrame)
        cal_arr = np.mean(calFrame, axis=0)
        np.save('./parameters/cal_arr.npy', cal_arr)
    else:
        print("Loading previously collected calibration data...")
        cal_arr = np.load('./parameters/cal_arr.npy')
    # calArr = calFrame[-1,:,:]
    print('Done')

    for old_data in os.listdir('./data_queue'):
        os.remove(os.path.join('./data_queue',old_data))

    print("Start Scanning...") 
    while(True):
        data_queue = sorted(os.listdir('./data_queue'))
        start = time.time()
        vtrig.Record()
        rec = vtrig.GetRecordingResult()
        rec_arr = rec2arr(rec)
        pro_arr = rec_arr - cal_arr
        if len(data_queue) > data_queue_size:
            os.remove(os.path.join('./data_queue',data_queue[0]))
        np.save(f'./data_queue/{datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}_{time.time_ns()}.npy',pro_arr)
        print('Scanning Frame Duration', time.time()-start, '[s]')
    
if __name__ == '__main__':
    main()
