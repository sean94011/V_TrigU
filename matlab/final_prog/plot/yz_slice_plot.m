function yz_slice_plot(y_cart, fig, first_iter)
    global xgrid ygrid zgrid Xgrid Ygrid Zgrid;
    %% Plot Y-Z Slice
    if and(min([length(ygrid),length(zgrid)])>2,length(xgrid)<=2)
        y_yz = 20*log10(rssq(y_cart(:,find(xgrid>=xgrid(1),1):find(xgrid>=xgrid(end),1),:),2));
        figure(fig(2));ax=pcolor(squeeze(Ygrid(:,1,:)),squeeze(Zgrid(:,1,:)),squeeze(y_yz));
        set(ax,'EdgeColor', 'none');
        if first_iter 
            set(gca,'NextPlot','replacechildren');
            title('yz view');xlabel('y');ylabel('z');daspect([1,1,1]);%caxis([-20,20]); 
        end
    end
end

