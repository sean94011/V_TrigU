import unittest
import itertools
import ctypes
import argparse
import sys
from imp import load_source
from os.path import join

def DefaultModulePath():
    if sys.platform == 'win32':
        modulePath = join('C:/', 'Program Files', 'Vayyar', 'vtrigU', 'python', 'vtrigU.py')
    elif sys.platform.startswith('linux'):
        modulePath = join('/usr', 'share', 'vtrigU', 'python', 'vtrigU.py')
    else:
        raise BaseException('Unsupported platform: ' + sys.platform)
    return modulePath

def Import_vtrigU():
    global vtrig
    vtrig = load_source('vtrigU', DefaultModulePath())

if __name__ == '__main__':
    Import_vtrigU()

    vtrig.Init() 
    # apply settings:
    vtrigSettings = vtrig.RecordingSettings(
        vtrig.FrequencyRange(65.0*1000, 66.0*1000, 21), # 101 points, from 65.0-66.0 GHz
        30.0, # RBW (in KHz)
        vtrig.VTRIG_U_TXMODE__LOW_RATE #
        ) 
    vtrig.ApplySettings(vtrigSettings)

    vtrig.Record() # one recording
    
    # modify settings
    vtrigSettings.rbw_khz = 30.5
    vtrigSettings.mode = vtrig.VTRIG_U_TXMODE__MED_RATE
    vtrig.ApplySettings(vtrigSettings)
    
    # record a bunch of times
    for i in range(10):
        vtrig.Record()  
    
    actual_freqs = vtrig.GetFreqVector_MHz()
    pair_list = vtrig.GetAntennaPairs(vtrigSettings.mode)

    print("FREQUENCIES: ")
    print(actual_freqs)
    print("------")
    recording = vtrig.GetRecordingResult()
    for pair_id in range(len(pair_list)):
        curPair = pair_list[pair_id]
        print(str(curPair) + ":" + str(recording[curPair]))
