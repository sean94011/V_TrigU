function [X] = GetRecordingResultSim(src,txrx,freq,Pn,ant_loc)
% GetRecordingResultSim Simulate IMAGEVK recording result
% X = GetRecordingResultSim(src,txrx,freq, Pn)
% Returns a recording result in matlab array format for reflectors defined
% at src and antenna locations defined at txrx for frequencies freq with 
% AWGN power Pn defined in dB relative to 1
% 
% src is a vector of [x,y,z] or [x,y,z,RCS] locations size is num_src, 4
% txrx are the tx and rx antennas used
% freq are specified as a row vector
% X is size(length(txrx),len(freq))

%vtrigU_ants_location;

src2 = permute(src,[3,2,4,1]);

% Use Radar equation to find total loss
% Pe/Ps = (Gtx*Grx*RCS*lambda^2)/((4pi)^3*|Rtx|^2*|Rrx|^2)
c = physconst('lightspeed'); %(m/s)
Rvec = src2-ant_loc;%VtrigU_ants_location;

Rmag = rssq(Rvec,2);
Rtheta = atan2(rssq(Rvec(:,1:2,:,:),2),Rvec(:,3,:,:));
Rphi = atan2(Rvec(:,2,:,:),Rvec(:,1,:,:));
Sphase = 2*pi*Rmag.*freq/c; %Electrical Length in Radians
RCS = 1; %m^2
lambda = c./freq; csf = sqrt(RCS).*lambda./((4*pi).^(3/2));
Smag = 10^(5.8/20)*RadiationPattern(Rtheta,Rphi)./Rmag;
N = 1/sqrt(2)*10.^(Pn/20)*(randn(length(txrx),length(freq))+1j*randn(length(txrx),length(freq)));
S = zeros(size(N));

for ii = 1:length(txrx)
    tx = txrx(ii,1); rx = txrx(ii,2);
    
    Smag(tx,:,:,:);
    Smag(rx,:,:,:);
    S(ii,:) = sum(csf.*Smag(tx,:,:,:).*Smag(rx,:,:,:).*...
              exp(-1j.*(Sphase(tx,:,:,:)+Sphase(rx,:,:,:))),4);
end

X = S + N;
end