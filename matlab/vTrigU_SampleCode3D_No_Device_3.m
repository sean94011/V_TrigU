%Set setting structure
freqStartMHz = 62*1000;
freqStopMHz = 69*1000;
numFreqPoints = 150;
rbw_khz = 80; %dummy variable
txMode = 20; %num tx antenna

%assert txMode is legal: 4=HIGH_RATE, 10=MED_RATE, 20=LOW_RATE
assert(sum(txMode==[4,10,20])>0)

    
%get antennas locations from script
vtrigU_ants_location;

%Construct antenna pairs matrix
TxRxPairs = [repelem((1:txMode)',20),repmat((21:40)',txMode,1)];

%Get used frequencies in Hz
freq = linspace(freqStartMHz,freqStopMHz,numFreqPoints)*1e6;

%Define constants
N_txrx = size(TxRxPairs,1);
N_freq = length(freq);

Nfft = 2^(ceil(log2(size(freq,2)))+1);
Ts = 1/Nfft/(freq(2)-freq(1)+1e-16); %Avoid nan checks
time_vec = (0:Ts:Ts*(Nfft-1));
dist_vec = time_vec*1.5e8; %distance in meters
%dist_vec(1) = 1E-6; %avoid duplicate points in meshgrid calc

xgrid = [0,0];
ygrid = -2:0.05:2;
zgrid = 0.1:0.05:4;
[Xgrid,Ygrid,Zgrid]=meshgrid(xgrid,ygrid,zgrid);


%% second method
% define xyz locations of image voxels
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

% nemel(H2) = ntx * nrx * nfreq * nx * ny * nz
H2 = zeros(length(TxRxPairs),length(freq),1,length(src2));
for ii = 1:length(TxRxPairs)
    tx = TxRxPairs(ii,1); rx = TxRxPairs(ii,2);
    H2(ii,:,:,:) = 1./(csf.*Smag(tx,:,:,:).*Smag(rx,:,:,:).*...
              exp(-1j.*(Sphase(tx,:,:,:)+Sphase(rx,:,:,:))));
end
H2 = reshape(permute(H2,[4,1,2,3]),length(src2),[]); %xyz x txrx x freq
size(H2)
%%
%Start Recording
%src = [0,1-0.1,1;0,1+0.1,1]; %define the xyz locations of point reflectors
src =  [0,-1,1;0,0,1;0,1,1;...
        0,-1,2;0,0,2;0,1,2;...
        0,-1,3;0,0,3;0,1,3];
%src = reshape(cat(3,Xgrid(3,1:2:end,1:20:end),Ygrid(3,1:2:end,1:20:end),Zgrid(3,1:2:end,1:20:end)),[],3);
Pn = -80;
rec = GetRecordingResultSim(src,TxRxPairs,freq,Pn,VtrigU_ants_location);
% X = recs(:,:,10);
X = recs;

%convert to complex time domain signal
x = ifft(X,Nfft,2);

y_cart2 = reshape(H2*reshape(X,[],1),size(Xgrid));

%Create and show power delay profile - non coherent summation
PDP = rssq(x,1);
figure();plot(dist_vec,20*log10(abs(PDP./max(abs(PDP)))));
title('Power Delay Profile - non coherent summation');
xlabel('Distance[m]');ylabel('Normalized amplitude[dB]'); 

% % Plot a 3D Point Cloud
% th = abs(y_cart)>1;
% figure();
% scatter3(Xgrid(th),Ygrid(th),Zgrid(th),20*log10(abs(y_cart(th))),20*log10(abs(y_cart(th)))) 
% title('Point Cloud');xlabel('x');ylabel('y');zlabel('z');daspect([1,1,1]);
% axis([xgrid(1),xgrid(end),ygrid(1),ygrid(end),zgrid(1),zgrid(end),-20,20]);

% Plot a yz slice of the 3D Image
%y_yz = 20*log10(rssq(y_cart2(:,find(xgrid>=0,1):find(xgrid>=0,1),:),2));
y_yz = rssq(y_cart2(:,find(xgrid>=0,1):find(xgrid>=0,1),:),2).^2;
figure();ax=surf(squeeze(Ygrid(:,1,:)),squeeze(Zgrid(:,1,:)),squeeze(y_yz));
%set(ax,'EdgeColor', 'none');
title('yz view');xlabel('y');ylabel('z');%daspect([1,1,10]);%caxis([-20,20]); 

% y_xz = 20*log10(rssq(y_cart2(find(ygrid>=-0.1,1):find(ygrid>=0.1,1),:,:),1));
% figure();ax=pcolor(squeeze(Xgrid(1,:,:)),squeeze(Zgrid(1,:,:)),squeeze(y_xz));
% set(ax,'EdgeColor', 'none');
% title('xz view');xlabel('x');ylabel('z');daspect([1,1,1]);%caxis([-20,20]);
% 
% 
% %y_xy = 20*log10(rssq(y_cart2(:,:,find(zgrid>=0.5,1):find(zgrid>=0.5,1)),3));
% y_xy = rssq(y_cart2(:,:,find(zgrid>=0.5,1):find(zgrid>=0.5,1)),3).^2;
% figure();ax=pcolor(squeeze(Xgrid(:,:,1)),squeeze(Ygrid(:,:,1)),squeeze(y_xy));
% set(ax,'EdgeColor', 'none');
% title('xy view');xlabel('x');ylabel('y');daspect([1,1,1]);
% mx = max(reshape(y_xy,[],1)); caxis([mx-30,mx]);




