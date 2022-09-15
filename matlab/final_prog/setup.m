function [Nfft,dist_vec] = setup()
    global TxRxPairs freq xgrid ygrid zgrid Xgrid Ygrid Zgrid H2;
    %% Load Library
    % add DOTNET assembly
    NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);
    
    %% Setup

    %Validate settings
    curSettings = vtrigU.vtrigU.GetSettings();
    vtrigU.vtrigU.ValidateSettings(curSettings);
    
    %Get antenna pairs and convert to matlab matrix
    ants_pairs = vtrigU.vtrigU.GetAntennaPairs(curSettings.txMode);
    TxRxPairs = zeros(ants_pairs.Length,2);
    for ii = 1: ants_pairs.Length
        TxRxPairs(ii,1) = double(ants_pairs(ii).tx);
        TxRxPairs(ii,2) = double(ants_pairs(ii).rx);
    end
    
    %Get used frequencies in Hz
    freq = double(vtrigU.vtrigU.GetFreqVector_MHz())*1e6;
    
    %get antennas locations from script
    vtrigU_ants_location;
    
    %Define constants
    N_txrx = size(TxRxPairs,1);
    N_freq = length(freq);
    c = 3e8; %SOL (m/s)
    
    Nfft = 2^(ceil(log2(size(freq,2)))+1);
    Ts = 1/Nfft/(freq(2)-freq(1)+1e-16); %Avoid nan checks
    time_vec = (0:Ts:Ts*(Nfft-1));
    dist_vec = time_vec*1.5e8; %distance in meters
    
    %%
    xgrid = [0.18,0.18];%-0.3:0.1:0.3;
    ygrid = -0.3:0.03:0.3;
    zgrid = 0.1:0.02:1;
    [Xgrid,Ygrid,Zgrid]=meshgrid(xgrid,ygrid,zgrid);
    
    src = reshape(cat(4,Xgrid,Ygrid,Zgrid),[],3);
    src2 = permute(src,[3,2,4,1]);
    
    % Use Radar equation to find total loss
    % Pe/Ps = (Gtx*Grx*RCS*lambda^2)/((4pi)^3*|Rtx|^2*|Rrx|^2)
    c = physconst('lightspeed'); %(m/s)
    Rvec = src2-VtrigU_ants_location;
    
    Rmag = rssq(Rvec,2);
    Rtheta = atan2(rssq(Rvec(:,1:2,:,:),2),Rvec(:,3,:,:));
    Rphi = atan2(Rvec(:,2,:,:),Rvec(:,1,:,:));
    Sphase = 2*pi*Rmag.*freq/c; %Electrical Length in Radians
    RCS = 1; %m^2
    lambda = c./freq; csf = sqrt(RCS).*lambda./((4*pi).^(3/2));
    Smag = 10^(5.8/20)*RadiationPattern(Rtheta,Rphi)./Rmag;
    H2 = zeros(length(TxRxPairs),length(freq),1,length(src2));
    for ii = 1:length(TxRxPairs)
        tx = TxRxPairs(ii,1); rx = TxRxPairs(ii,2);
        H2(ii,:,:,:) = 1./(csf.*Smag(tx,:,:,:).*Smag(rx,:,:,:).*...
                  exp(-1j.*(Sphase(tx,:,:,:)+Sphase(rx,:,:,:))));
    end
    H2 = reshape(permute(H2,[4,1,2,3]),length(src2),[]); %xyz x txrx x freq

end

