function xy_slice_plot(y_cart, fig, first_iter)
    global TxRxPairs freq xgrid ygrid zgrid Xgrid Ygrid Zgrid H2;
    %% Plot X-Y Slice
    if and(min([length(xgrid),length(ygrid)])>2,length(zgrid)<=2)
        y_xy = 20*log10(rssq(y_cart(:,:,find(zgrid>=zgrid(1),1):find(zgrid>=zgrid(end),1)),3));
        figure(fig(4));ax=pcolor(squeeze(Xgrid(:,:,1)),squeeze(Ygrid(:,:,1)),squeeze(y_xy));
        set(ax,'EdgeColor', 'none');
        if first_iter
            set(gca,'NextPlot','replacechildren');
            title('xy view');xlabel('x');ylabel('y');daspect([1,1,1]);%caxis([-20,20]);
        end
    end
end

