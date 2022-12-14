function [X_RF, X, y_cart, PDP] = single_frame_record(idx)
    global TxRxPairs freq Xgrid Ygrid Zgrid H2 Nfft f_res;
    %% Load Library
    % add DOTNET assembly
    NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

    %% Start Recording
    vtrigU.vtrigU.Record();
    rec = double(vtrigU.vtrigU.GetRecordingResult(vtrigU.SignalCalibration.DEFAULT_CALIBRATION));
    X = rec;
    N_freq = length(freq);
    
    %Convert and reshape result to matlab complex matrix
    smat_size = [size(TxRxPairs,1),size(freq,2),2];
    my_perms = [1,3,2];
    X = reshape(X,smat_size(my_perms));
    X = ipermute(X,my_perms);
    X = X(:,:,1)+ 1j*X(:,:,2);
    X_RF = X;
    
    if idx == 1
        % Identify resonant frequencies 
        thresh = 3;
        lnconv = min(max(floor(N_freq/8)*2+1,floor(50/(freq(2)-freq(1)))*2+1),...
                 floor(3*N_freq/8)*2+1); %conv length between 1/4 and 3/4 N_freq
        c2 = -ones(lnconv,1)/(lnconv-1);
        c2((lnconv+1)/2) = 1;
        padsig = 20*log10(rssq(X,1));
        padsig = [padsig((lnconv-1)/2:-1:1),padsig,padsig(end:-1:end-(lnconv-1)/2+1)]; 
        padsig = conv(padsig,c2,'valid');        
        f_res = padsig>thresh;
    end

    %Remove resonant frequencies
    X = X .* (1-f_res);
    
    %convert to complex time domain signal
    x = ifft(X,Nfft,2);
   
    y_cart = reshape(H2*reshape(X,[],1),size(Xgrid));

    %Create and show power delay profile - non coherent summation
    PDP = rssq(x,1);
end


