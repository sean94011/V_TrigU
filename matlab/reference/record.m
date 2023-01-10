function [time, PDP, BR_response] = record(curSettings, filter_type)
%% Load Library

% add DOTNET assembly
NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

% Load Module
import vtrigU.*;

%% Get Radar Parameters

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
if filter_type == "matched"
    b = conj(Smat(end:-1:1));
    Smat = filter(b,1,Smat);
end
    Smat = Smat(1:2:end) + 1i*Smat(2:2:end);
    Smat = reshape(Smat,size(TxRxPairs,1),size(Freqs,2));

[rows, cols] = size(Smat);
Smat = Smat ./ repmat(Smat(25,:),[rows,1]);
% for i = 25:1:71
%     refDiv_Smat = refDiv_Smat + Smat ./ repmat(Smat(i,:),[rows,1]);
% end
% Smat = Smat ./ 48;

%% Data Processing

%convert to complex time domain signal
Nfft = 2^(ceil(log2(size(Freqs,2)))+1);
Smat_td = ifft(Smat,Nfft,2);
Ts = 1/Nfft/(Freqs(2)-Freqs(1)+1e-16); %Avoid nan checks
time_vector = 0:Ts:Ts*(Nfft-1);
time = time_vector*1.5e8;

%Create and show power delay profile - non coherent summation
PDP = mean(abs(Smat_td),1);


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
end

