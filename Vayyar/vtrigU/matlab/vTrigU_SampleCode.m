%add DOTNET assembly
NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

%init the device
vtrigU.vtrigU.Init();
%Set setting structure
mySettings.freqRange.freqStartMHz = 62*1000;
mySettings.freqRange.freqStopMHz = 68*1000;
mySettings.freqRange.numFreqPoints = 101;
mySettings.rbw_khz = 100;
mySettings.txMode = vtrigU.TxMode.LOW_RATE;
vtrigU.vtrigU.ApplySettings(mySettings.freqRange.freqStartMHz,mySettings.freqRange.freqStopMHz, mySettings.freqRange.numFreqPoints, mySettings.rbw_khz, mySettings.txMode );
 
%Validate settings
mySettings = vtrigU.vtrigU.GetSettings();
vtrigU.vtrigU.ValidateSettings(mySettings);

%Get antenna pairs and convert to matlab matrix
ants_pairs = vtrigU.vtrigU.GetAntennaPairs(mySettings.txMode);
TxRxPairs = zeros(ants_pairs.Length,2);
for ii = 1: ants_pairs.Length
    TxRxPairs(ii,1) = double(ants_pairs(ii).tx);
    TxRxPairs(ii,2) = double(ants_pairs(ii).rx);
end

%Get used frequencies in Hz
Freqs = double(vtrigU.vtrigU.GetFreqVector_MHz())*1e6;

% Do a single record
vtrigU.vtrigU.Record();

%Read  calibrated result and convert to matlab complex matrix
Record_result = vtrigU.vtrigU.GetRecordingResult(vtrigU.SignalCalibration.DEFAULT_CALIBRATION);
Smat = double(Record_result);
Smat = Smat(1:2:end) + 1i*Smat(2:2:end);
Smat = reshape(Smat,size(TxRxPairs,1),size(Freqs,2));

%%
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
hold on;plot(time_vector*1.5e8,20*log10(abs(BR_response./max(abs(BR_response)))));legend('Normalized non - Coherent summation','Normalized Coherent summation');

