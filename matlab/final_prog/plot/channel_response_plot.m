function channel_response_plot(smat_size, X_RF, fig)
    global freq
    figure(fig);
    clf; hold on;
    for ll = 1:smat_size(1)
        plot(freq,20*log10(abs(X_RF(ll,:))));
    end
    plot(freq,20*log10(rssq(X_RF,1))-10*log10(smat_size(1)),'k.','LineWidth',4);
    title('Channel response (per channel and rms average)')
end