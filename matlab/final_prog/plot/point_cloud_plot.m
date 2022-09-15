function point_cloud_plot(y_cart, first_iter)
    global xgrid ygrid zgrid Xgrid Ygrid Zgrid;
    if min([length(xgrid),length(ygrid),length(zgrid)])>2
        th = abs(y_cart)>1;
        figure(fig);
        scatter3(Xgrid(th),Ygrid(th),Zgrid(th),20*log10(abs(y_cart(th))),20*log10(abs(y_cart(th))))
        if first_iter 
            title('Point Cloud');xlabel('x');ylabel('y');zlabel('z');daspect([1,1,1]);
            axis([xgrid(1),xgrid(end),ygrid(1),ygrid(end),zgrid(1),zgrid(end),-20,20]);
            set(gca,'NextPlot','replacechildren') ;
        end
    end
end

