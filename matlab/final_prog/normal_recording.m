function normal_recording()
    %% Reset Workspace & Clear Data
    clear;
    close all;
    %% Load Library
    % add DOTNET assembly
    NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

    % Load Module
    import vtrigU.*;

    % add plotting functions
    addpath('plot\')
    
    %% Setup Radar
    % initialize the device                                            
    vtrigU.Init();

    % Set setting structure:
    customSetting = input("Default Setting:\n   62-69 GHz,\n   100 freq points,\n   RBW 100 KHz,\n   LOW Recording Frame Rate\nDo you want to use YOUR OWN setting ('y'/'n')? ",'s');
    if customSetting == 'y'
        curSettings = user_settings();
    else
        curSettings = default_settings();
    end

    % Apply the above setting
    vtrigU.ApplySettings(curSettings.freqRange.freqStartMHz,curSettings.freqRange.freqStopMHz, curSettings.freqRange.numFreqPoints, curSettings.rbw_khz, curSettings.txMode)
    global TxRxPairs freq Nfft;
    [Nfft, dist_vec] = setup();
    smat_size = [size(TxRxPairs,1),size(freq,2),2];
    
    %get antennas locations from script
    vtrigU_ants_location;

    %% Start Recording

    %Start Recording
    nfig = 5; fig = zeros(1,nfig);
    for ii = 1:nfig
        fig(ii) = figure();
    end

    while(1)
        first_iter=true;
        input('continue?');
        % Record the Scanning
        nrecs = input('How many recordings do you want to make? ');
        recs = zeros([size(TxRxPairs,1),size(freq,2),nrecs]);

        for idx = 1:nrecs
            recs = zeros([size(TxRxPairs,1),size(freq,2),nrecs]);
            [X_RF, X, y_cart, PDP] = single_frame_record(idx);
            recs(:,:,idx) = X;
            if idx == 1
                channel_response_plot(smat_size, X_RF, fig(1));
            end

            % Plot
            figure(fig(2)); plot(dist_vec,20*log10(abs(PDP./max(abs(PDP)))));
            point_cloud_plot(y_cart, first_iter)
            yz_slice_plot(y_cart, fig(3), first_iter)
            xz_slice_plot(y_cart, fig(4), first_iter)
            xy_slice_plot(y_cart, fig(5), first_iter)
            first_iter=false;
        end
    end
end

