function xz_slice_plot(y_cart, fig, first_iter)
    global xgrid ygrid zgrid Xgrid Ygrid Zgrid;
    %% Plot X-Z Slice
    if and(min([length(xgrid),length(zgrid)])>2,length(ygrid)<=2)
        y_xz = 20*log10(rssq(y_cart(find(ygrid>=ygrid(1),1):find(ygrid>=ygrid(end),1),:,:),1));
        figure(fig);ax=pcolor(squeeze(Xgrid(1,:,:)),squeeze(Zgrid(1,:,:)),squeeze(y_xz));
        set(ax,'EdgeColor', 'none');
        if first_iter 
            set(gca,'NextPlot','replacechildren');
            title('xz view');xlabel('x');ylabel('z');daspect([1,1,1]);%caxis([-20,20]);
        end
    end
end

