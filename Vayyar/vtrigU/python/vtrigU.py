from __future__ import unicode_literals
from sys import platform, maxsize
from ctypes import *
from os.path import exists, join
from collections import namedtuple
from struct import calcsize
from itertools import product

# Error Codes
VTRIG_U_RES_SUCCESS = 0
VTRIG_U_RES_USERERR__NOT_INITIALIZED=1
VTRIG_U_RES_USERERR__NO_ACTIVE_APPLICATION=2
VTRIG_U_RES_USERERR__NO_SETTINGS_APPLIED=3
VTRIG_U_RES_USERERR__NO_RECORDING=4
VTRIG_U_RES_INPUTERR__OUT_OF_RANGE=5
VTRIG_U_RES_INPUTERR__INVALID_SETTINGS=6
VTRIG_U_RES_INPUTERR__BAD_RESULT_SIZE=7
VTRIG_U_RES_INSTRUMENT_NOT_FOUND=8
VTRIG_U_RES_DEVICE_ERROR=9
VTRIG_U_RES_INIT_FAILED=10
VTRIG_U_RES_BAD_CONFIG=11
VTRIG_U_RES_GENERAL_ERROR=12

# TX Modes
VTRIG_U_TXMODE__HIGH_RATE=0  # 4TX: high framerate, low resolution, 2D
VTRIG_U_TXMODE__MED_RATE=1   # 10TX: medium framerate, medium resolution
VTRIG_U_TXMODE__LOW_RATE=2   # 20Tx: low framerate, high resolution, 3D

# Signal calibration
VTRIG_U_SIGCALIB__NO_CALIBRATION=0
VTRIG_U_SIGCALIB__DEFAULT_CALIBRATION=1

vtrigUErrStrings = {
    VTRIG_U_RES_SUCCESS:"Operation successful",
    VTRIG_U_RES_USERERR__NOT_INITIALIZED:"An attempt was made to use the sdk before a call to vtrigU_Init()",
    VTRIG_U_RES_USERERR__NO_ACTIVE_APPLICATION:"An attempt was made to use the sdk without vtrigU application active",
    VTRIG_U_RES_USERERR__NO_SETTINGS_APPLIED:"Attempted use of setting-dependent methods, before a call to ApplySettings()",
    VTRIG_U_RES_USERERR__NO_RECORDING:"Attempted use of vtrigU_WriteRecording(), before anything is recorded",
    VTRIG_U_RES_INPUTERR__OUT_OF_RANGE:"A provided parameter value was out of its allowed range",
    VTRIG_U_RES_INPUTERR__INVALID_SETTINGS:"Recording Settings are not valid -- outside of allowed range, or not matching allowed step",
    VTRIG_U_RES_INPUTERR__BAD_RESULT_SIZE:"Provided size for a requested result parameter does not match actual result size",
    VTRIG_U_RES_INSTRUMENT_NOT_FOUND:"The SDK is unable to communicate with the instrument",
    VTRIG_U_RES_DEVICE_ERROR:"Device could not connect",
    VTRIG_U_RES_INIT_FAILED:"Initialization failed",
    VTRIG_U_RES_BAD_CONFIG:"Bad configuration",
    VTRIG_U_RES_GENERAL_ERROR:"A general error occurred. Inspect the error logs for more information"                                                            
}

vtrigUModeNames = {
    VTRIG_U_TXMODE__HIGH_RATE:"High-Rate",
    VTRIG_U_TXMODE__MED_RATE:"Medium-Rate",
    VTRIG_U_TXMODE__LOW_RATE:"Low-Rate"
}

class vtrigUError(Exception):
    """ vtrigU's specific Exception object.
        Args:
            message:        Short explanation about the occured exception.
            code:           Code number of the exception.
    """
    def __init__(self, ctxt, code):
        super(Exception, self).__init__(ctxt + ": " + vtrigUErrStrings[code])
        self.code = code

# Make sure python is 64 bits
if (platform == 'win32') and  (8 * calcsize('P') != 64):
    raise vtrigUError('Python must be 64bits to use this library', VTRIG_U_RES_GENERAL_ERROR)

def _GetDefaultPaths(): 
    depLibPaths = []
    if platform == 'win32':
        defaultBinPath = join('C:/', 'Program Files', 'Vayyar', 'vtrigU', 'bin')
        libPath = join(defaultBinPath, 'vtrigU.dll')
        configFilePath = join(defaultBinPath, '.config')
        depLibPaths = [ join(defaultBinPath, 'Qt5Core.dll'), join(defaultBinPath, 'libusb-1.0.dll')]
    elif platform.startswith('linux'):
        libPath = join('/usr', 'lib', 'vtrigU', 'libvtrigU.so')     
        configFilePath = join('/etc', 'vtrigU.conf')
    else:
        return None, None, None
    return libPath, configFilePath, depLibPaths
_defaultLibPath, _defaultConfigFilePath, _depLibPaths = _GetDefaultPaths()

def _InitModule(libPath, depLibPaths):
    """Must be called before using vtrigU functions.
        Args:
            libPath:        Full path to vtrigU shared library. If not set, will use default path.
            depLibPaths:    List of full paths to shared libraries that vtrigU depends on.
                            This parameter is only necessary if shared libraries are not installed on your path,
                            and/or you want to use them from some other location.
    """
    for dlPath in depLibPaths:
        if not exists(dlPath):
            raise ValueError('Could not load library at:', dlPath)
        CDLL(dlPath, mode=RTLD_GLOBAL)

    if not exists(libPath):
        raise ValueError('Could not load vtrigU library at:', libPath)
    global _vtrig

    _vtrig = CDLL(libPath)

def IsModuleInitialized():
    if '_vtrig' in globals():
        return not (_vtrig == None)
    else:
        return False;

def _RaiseIfErr(funcName, res):
    """ Raises customized vtrigUError in case encounter one.
    """
    if res != VTRIG_U_RES_SUCCESS:
        raise vtrigUError(funcName, res)

def _GetLastErrString():
    _vtrig.vtrigU_GetLastErrString.restype = c_char_p
    return _vtrig.vtrigU_GetLastErrString().decode('utf-8')
    
def _SetConfigFile(path):
    """ Obtains Sets location of vtrigU configuration file, if moved from
        default.
        Args:
            path           (Optional) config file path. Uses default location
                            if no path is given.
    """
    _RaiseIfErr(_SetConfigFile.__name__, _vtrig.vtrigU_SetConfigFile(path.encode('ascii')))
    
def Init(libPath = _defaultLibPath, depLibPaths = _depLibPaths, configPath = _defaultConfigFilePath):
    _InitModule(libPath, depLibPaths)
    assert(IsModuleInitialized())
    _SetConfigFile(configPath)
    _RaiseIfErr(Init.__name__, _vtrig.vtrigU_Init())
    
class _Ctypes_FrequencyRange(Structure):
    _fields_ = [("freqStartMHz", c_double), ("freqStopMHz", c_double), ("numFreqPoints", c_int)]
    
class _Ctypes_RecordingSettings(Structure):
    _fields_ = [("freqRange", _Ctypes_FrequencyRange), ("rbw_khz", c_double), ("mode", c_int)]

class FrequencyRange:
    def __init__(self, freqStartMHz, freqStopMHz, numFreqPoints):
        self.freqStartMHz = freqStartMHz
        self.freqStopMHz = freqStopMHz
        self.numFreqPoints = numFreqPoints

    def __repr__(self):
        return "<FrequencyRange: [%f-%f] MHz, %d points>" % (self.freqStartMHz, self.freqStopMHz, self.numFreqPoints)

class RecordingSettings:
    def __init__(self, freqRange, rbw_khz, mode):
        self.freqRange = freqRange
        self.rbw_khz = rbw_khz
        self.mode = mode
    def __repr__(self):
        return "<vtrigUSettings: %s, rbw: %f KHz, mode: %s>" % (self.freqRange, self.rbw_khz, vtrigUModeNames[self.mode])

def __PySettings2CSettings(settings):
    return _Ctypes_RecordingSettings(
        _Ctypes_FrequencyRange(
            settings.freqRange.freqStartMHz,
            settings.freqRange.freqStopMHz,
            settings.freqRange.numFreqPoints),
        settings.rbw_khz, settings.mode)

def __CSettings2PySettings(c_settings):
    return RecordingSettings(
        FrequencyRange(
            c_settings.freqRange.freqStartMHz,
            c_settings.freqRange.freqStopMHz,
            c_settings.freqRange.numFreqPoints),
        c_settings.rbw_khz, settings.mode)
      
def ApplySettings(settings):
    ValidateSettings(settings)
    c_settings = __PySettings2CSettings(settings)
    _RaiseIfErr(ApplySettings.__name__, _vtrig.vtrigU_ApplySettings(c_settings))

def GetSettings():
    c_settings = _Ctypes_RecordingSettings()
    _RaiseIfErr(GetSettings.__name__, _vtrig.vtrigU_GetSettings(c_settings))
    return __CSettings2PySettings(c_settings)
    
def GetFreqVector_MHz():
    nFreqs = c_int()
    _RaiseIfErr(GetFreqVector_MHz.__name__, _vtrig.vtrigU_GetFreqVectorSizeDouble(byref(nFreqs)))
    
    buf = (c_double * nFreqs.value)()
    _vtrig.vtrigU_GetFreqVector_MHz.argtypes = [c_int, POINTER(c_double)]
    _RaiseIfErr(GetFreqVector_MHz.__name__, _vtrig.vtrigU_GetFreqVector_MHz(nFreqs, buf))

    return list(buf)

def GetAntennaPairs(txMode=None):
    if txMode is None:
        txMode = GetSettings().mode
    nPairs = c_int()
    _RaiseIfErr(GetAntennaPairs.__name__, _vtrig.vtrigU_GetNPairs(txMode, byref(nPairs)))
    

    buf_tx = (c_int * nPairs.value)()
    buf_rx = (c_int * nPairs.value)()
    _vtrig.vtrigU_GetAntennaPairs.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_int)]
    _RaiseIfErr(GetAntennaPairs.__name__, _vtrig.vtrigU_GetAntennaPairs(txMode, nPairs, buf_tx, buf_rx))

    return list(zip(list(buf_tx), list(buf_rx)))

def GetPairId(tx, rx, txMode=None):
    if txMode is None:
        txMode = GetSettings().mode
    pairId = c_int()
    _RaiseIfErr(GetPairId.__name__, _vtrig.vtrigU_GetPairId(txMode, tx, rx, byref(pairId)))
    return pairId

def Record():
    _RaiseIfErr(Record.__name__, _vtrig.vtrigU_Record())

class _ctypes_vtrigU_Complex(Structure):
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]

def _resbufComplex_2Python(resultBuffer, txrxPair_i, freq_i):
    res_real = resultBuffer[txrxPair_i][2*freq_i]
    res_imag = resultBuffer[txrxPair_i][2*freq_i+1]
    return float(res_real) + (float(res_imag)*1j)
 
class _ctypes_vtrigU_RecordingResult(Structure):
    _fields_ = [
        ('resultBuffer', POINTER(POINTER(c_double))),
        ('txMode', c_int),
        ('nTxRxPairs', c_int),
        ('nFrequenciesMeasured', c_int)
    ]
            
def GetRecordingResult():
    freqVector = GetFreqVector_MHz()
    nFreqs = len(freqVector)
    
    c_resStruct = _ctypes_vtrigU_RecordingResult()
    _vtrig.vtrigU_InitResultStructure_ForCurrentSettings(byref(c_resStruct))
    res_code_getResult = _vtrig.vtrigU_GetRecordingResult(byref(c_resStruct))
    if res_code_getResult != VTRIG_U_RES_SUCCESS:
        _vtrig.vtrigU_FreeResultStructure(byref(c_resStruct))
        _RaiseIfErr(GetRecordingResult.__name__, res_code_getResult)

    txrxPairs = GetAntennaPairs(c_resStruct.txMode)
    nTxRxPairs = len(txrxPairs)
    resMap = { txrxPairs[txrxPair_i] : [_resbufComplex_2Python(c_resStruct.resultBuffer,txrxPair_i, freq_i) for freq_i in range(nFreqs)] for txrxPair_i in range(nTxRxPairs) }

    _vtrig.vtrigU_FreeResultStructure(byref(c_resStruct))

    return resMap
    
def EnterStandbyMode():
    _RaiseIfErr(EnterStandbyMode.__name__, _vtrig.vtrigU_EnterStandbyMode())

VTRIG_U_SETTINGS__VALID=0
VTRIG_U_SETTINGS__FREQ_RANGE_MALFORMED=1
VTRIG_U_SETTINGS__FREQ_OUT_OF_RANGE=2
VTRIG_U_SETTINGS__FREQ_RANGE_TOO_SMALL=3
VTRIG_U_SETTINGS__NPOINTS_OUT_OF_RANGE=4
VTRIG_U_SETTINGS__RBW_OUT_OF_RANGE=5

vtrigUSettingsErrStrings = {
	VTRIG_U_SETTINGS__VALID:"Settings are valid",
    VTRIG_U_SETTINGS__FREQ_RANGE_MALFORMED:"freqStop is not greater than freqStart",
	VTRIG_U_SETTINGS__FREQ_OUT_OF_RANGE:"Frequency start/stop are out of range",
    VTRIG_U_SETTINGS__FREQ_RANGE_TOO_SMALL:"Frequency range length (freqStop - freqStart) is too short",
    VTRIG_U_SETTINGS__NPOINTS_OUT_OF_RANGE:"Number of frequency points is out of range",
	VTRIG_U_SETTINGS__RBW_OUT_OF_RANGE:"RBW is out of range"
}

class vtrigUSettingsError(Exception):
    """ Exception thrown by ValidateSettings if settings are invalid.
        Args:
            message:        Short explanation about the occured exception.
            code:           Code number of the exception.
    """
    def __init__(self, code):
        super(Exception, self).__init__("Invalid settings: " + vtrigUSettingsErrStrings[code])
        self.code = code

def ValidateSettings(settings):
    c_settings = __PySettings2CSettings(settings)
    errval = _vtrig.vtrigU_ValidateSettings(c_settings)
    if errval != VTRIG_U_SETTINGS__VALID:
        raise vtrigUSettingsError(errval)

    
class _ctypes_vtrigU_FrequencyLimits(Structure):
    _fields_ = [
        ("min_MHz", c_double), ("max_MHz", c_double), ("nPointsMin", c_int), ("nPointsMax", c_int)
    ]
class FrequencyLimits:
    def __init__(self, c_struct):
        self.min_MHz = c_struct.min_MHz
        self.max_MHz = c_struct.max_MHz
        self.nPointsMin = c_struct.nPointsMin
        self.nPointsMax = c_struct.nPointsMax
    def __repr__(self):
        return "<FrequencyLimits: [%f-%f] MHz, #Points: [%d-%d]>" % (self.min_MHz, self.max_MHz, self.nPointsMin, self.nPointsMax)

class _ctypes_vtrigU_RbwLimits(Structure):
    _fields_ = [("min_KHz", c_double), ("max_KHz", c_double)]

class RbwLimits:
    def __init__(self, c_struct):
        self.min_KHz = c_struct.min_KHz
        self.max_KHz = c_struct.max_KHz
    def __repr__(self):
        return "<RbwLimits: [%f-%f] KHz>" % (self.min_KHz, self.max_KHz)

def GetFrequencyLimits():
    _vtrig.vtrigU_GetFrequencyLimits.restype = _ctypes_vtrigU_FrequencyLimits
    return FrequencyLimits(_vtrig.vtrigU_GetFrequencyLimits())

def GetRbwLimits():
    _vtrig.vtrigU_GetRbwLimits.restype = _ctypes_vtrigU_RbwLimits
    return RbwLimits(_vtrig.vtrigU_GetRbwLimits())
