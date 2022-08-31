%% Reset Workspace & Clear Data
clear;
close all;

%% Load Library

% add DOTNET assembly
NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

% Load Module
import vtrigU.*;

%% Initialize & Set up the Radar

% initialize the device
vtrigU.Init();

% Set setting structure:
% Frequency Band: 62-69 GHZ
curSettings.freqRange.freqStartMHz  = input("Please enter the desired start frequency in MHz (62000-69000 MHZ) : ");
curSettings.freqRange.freqStopMHz   = input("Please enter the desired start frequency in MHz (62000-69000 MHZ): ");
curSettings.freqRange.numFreqPoints = input("Please enter the desired number of frequency points: ");
% RBW 10-800 KHz
curSettings.rbw_khz = input("Please enter the desired RBW KHz: ");
% Recording Profiles:
% 1. LOW_RATE:  20 Tx, High Resolution 3D Imaging
% 2. MED_RATE:  10 Tx, Medium Resolution 3D Imaging
% 3. HIGH_RATE:  4 Tx, 2D Imaging, People Tracking
recordingProfile = input("Please enter the desired recording frame rate as -> 'low', 'med', 'high': ");
if recordingProfile == "high"
    curSettings.txMode = vtrigU.TxMode.HIGH_RATE; 
elseif recordingProfile == "med"
    curSettings.txMode = vtrigU.TxMode.MED_RATE; 
else
    curSettings.txMode = vtrigU.TxMode.LOW_RATE; 
end


% Apply the above setting
vtrigU.ApplySettings(curSettings.freqRange.freqStartMHz,curSettings.freqRange.freqStopMHz, curSettings.freqRange.numFreqPoints, curSettings.rbw_khz, curSettings.txMode );

% Validate settings
curSettings = vtrigU.GetSettings();
vtrigU.ValidateSettings(curSettings);

%Get antenna pairs and convert to matlab matrix
ants_pairs = vtrigU.GetAntennaPairs(curSettings.txMode);
TxRxPairs = zeros(ants_pairs.Length,2);
for ii = 1: ants_pairs.Length
    TxRxPairs(ii,1) = double(ants_pairs(ii).tx);
    TxRxPairs(ii,2) = double(ants_pairs(ii).rx);
end

%Get used frequencies in Hz
Freqs = double(vtrigU.GetFreqVector_MHz())*1e6;

%% Record the Scanning

% Do a single record
vtrigU.Record();

%Read  calibrated result and convert to matlab complex matrix
Record_result = vtrigU.GetRecordingResult(SignalCalibration.DEFAULT_CALIBRATION);
Smat = double(Record_result);
Smat = Smat(1:2:end) + 1i*Smat(2:2:end);
Smat = reshape(Smat,size(TxRxPairs,1),size(Freqs,2));

%% Data Processing

%convert to complex time domain signal
Nfft = 2^(ceil(log2(size(Freqs,2)))+1);
Smat_td = ifft(Smat,Nfft,2);
Ts = 1/Nfft/(Freqs(2)-Freqs(1)+1e-16); %Avoid nan checks
time_vector = 0:Ts:Ts*(Nfft-1);

%Create and show power delay profile - non coherent summation
PDP = mean(abs(Smat_td),1);
figure(1);ylim([0 2]);plot(time_vector*1.5e8,20*log10(abs(PDP./max(abs(PDP)))));
xlabel('Distance[m]');ylabel('Normalized amplitude[dB]');


%get antennas locations from script
vtrigU_ants_location;

%Create a steering vector
theta = 0.0; %sin(theta) = x/R;
phi = 0.0;   %sin(phi) = y/R;
K_vec_x = 2*pi*Freqs*sin(theta)/3e8;
K_vec_y = 2*pi*Freqs*sin(phi)/3e8;

%Create a steering matrix for all pairs location
H = zeros(size(TxRxPairs,1),size(Freqs,2));
for ii = 1: size(TxRxPairs,1)
    D = VtrigU_ants_location(TxRxPairs(ii,1),:)+VtrigU_ants_location(TxRxPairs(ii,2),:);
    H(ii,:) = exp(2*pi*1i*(K_vec_x*D(1)+K_vec_y*D(2)));
end

%calculate and plot the steering response
BR_response = ifft(mean(H.*Smat),Nfft,2);
hold on;
plot(time_vector*1.5e8, 20*log10(abs(BR_response./max(abs(BR_response)))));
legend('Normalized non - Coherent summation','Normalized Coherent summation');

