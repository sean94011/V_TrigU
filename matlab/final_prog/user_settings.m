function curSettings = user_settings()
    % *Customized Setting
    % Frequency Band: 62-69 GHZ
    curSettings.freqRange.freqStartMHz  = input("Please enter your desired start frequency in MHz (62000-69000 MHZ) : ");
    curSettings.freqRange.freqStopMHz   = input("Please enter your desired start frequency in MHz (62000-69000 MHZ, at least 150 MHz from start freq): ");
    % Frequency Points 2-151
    curSettings.freqRange.numFreqPoints = input("Please enter your desired number of frequency points (2-150): ");
    % RBW 10-800 KHz
    curSettings.rbw_khz = input("Please enter your desired RBW KHz (10-800 KHz): ");
    % Recording Profiles:
    % 1. LOW_RATE:  20 Tx, High Resolution 3D Imaging
    % 2. MED_RATE:  10 Tx, Medium Resolution 3D Imaging
    % 3. HIGH_RATE:  4 Tx, 2D Imaging, People Tracking
    recordingProfile = input("Please enter your desired recording frame rate as -> 'low', 'med', 'high': ",'s');
    if recordingProfile == "high"
        curSettings.txMode = vtrigU.TxMode.HIGH_RATE; 
    elseif recordingProfile == "med"
        curSettings.txMode = vtrigU.TxMode.MED_RATE; 
    else
        curSettings.txMode = vtrigU.TxMode.LOW_RATE; 
    end
    disp('Using Customized Setting...');
end

