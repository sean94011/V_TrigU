function curSettings = default_settings()
    % *Default Setting
    disp('Using Default Setting...');
    curSettings.freqRange.freqStartMHz  = 62*1000;
    curSettings.freqRange.freqStopMHz   = 69*1000;
    curSettings.freqRange.numFreqPoints = 150;
    curSettings.rbw_khz = 800;
    curSettings.txMode = vtrigU.TxMode.LOW_RATE;
end

