%% Reset Workspace & Clear Data
clear all;
close all;

%% Load Library

% add DOTNET assembly
NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

% Load Module
import vtrigU.*;

%% Initialize & Set up the Radar

% initialize the device
                                                    
vtrigU.vtrigU.Init();

% Set setting structure:
customSetting = input("Default Setting: 62-69 GHz, 100 freq points, RBW 100 KHz, LOW Recording Frame Rate\nDo you want to use YOUR OWN setting ('y'/'n')? ",'s');
filter_type = input("What type of filter do you want to apply? (matched, none) ","s");
if customSetting == 'y'
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
else
    % *Default Setting
    disp('Using Default Setting...');
    curSettings.freqRange.freqStartMHz  = 62*1000;
    curSettings.freqRange.freqStopMHz   = 69*1000;
    curSettings.freqRange.numFreqPoints = 150;
    curSettings.rbw_khz = 800;
    curSettings.txMode = vtrigU.TxMode.LOW_RATE;
end

% Apply the above setting
vtrigU.ApplySettings(curSettings.freqRange.freqStartMHz,curSettings.freqRange.freqStopMHz, curSettings.freqRange.numFreqPoints, curSettings.rbw_khz, curSettings.txMode );

% Validate settings
curSettings = vtrigU.GetSettings();
vtrigU.ValidateSettings(curSettings);

%% Record the Scanning
fig = figure(1);
first_iter = true;
[time, PDP, BR_response] = single_frame_record(curSettings,filter_type)
while(1)
    [time, PDP, BR_response] = record(curSettings,filter_type);
    figure(fig);
    plot(time,20*log10(abs(PDP./max(abs(PDP)))));
    hold on;
    plot(time, 20*log10(abs(BR_response./max(abs(BR_response)))));
    hold off;
    if first_iter
        ylim([0 2]);
        xlabel('Distance[m]');
        ylabel('Normalized amplitude[dB]');
        legend('Normalized non - Coherent summation','Normalized Coherent summation');
    end
    first_iter = false;
end
