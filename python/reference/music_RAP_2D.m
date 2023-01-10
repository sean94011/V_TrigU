function [indices_New,Ahat_New]=music_RAP_2D(Y, K, res,ts,slope,carrier_freq,antenna_dist,numRX, smoothing)
%smoothing=> 0: no smoothing, =1 forward smoothing, =2 spatial smoothing

%ToF and AoA
L = size(Y,2);
N = size(Y,1);
% K = 1e2;
% Fs = 5209e3;
endRange=25;%1;%
% Initialisation
%dig_freq = 0:res:endRange;

angleSpacing=5;%0.5;%
tauSpacing=res;%0.1;%
ranges=[-40,40;0,endRange];
AoA = ranges(1,1):angleSpacing:ranges(1,2); %AoA
ToF = ranges(2,1):tauSpacing:ranges(2,2);

GridPts=[size(ToF,2),size(AoA,2)];
[ToFind,AoAind] = ind2sub(GridPts,1:prod(GridPts));
zt = ranges(2,1) + (ToFind-1)*tauSpacing;
qt = ranges(1,1) + (AoAind-1)*angleSpacing;

%AoA, distance
ppp=[qt(:)'; zt(:)'];

indices_New=[];
c=3e8;
if smoothing ==2
   
    [R]=spatialSmooth2D(Y,round((N/numRX)/2),numRX);
    arraySize=size(R,1);
else
    arraySize=N;

    %[Y,beta,f] = sanitize(Y,arraySize);
    %x=fft(Y);
    %figure;plot(abs(x));
    % Estimating the auto correlation matrix
    R=(1/L)*(Y * Y');
end

% Forming the steering vectors for 2D search
n = 0:(arraySize/2)-1;
n=[n,n];
Z=exp(1j * ((2 * pi * slope* n' * 2*ppp(2,:) * ts)/c));
anglePhase=exp(1j * (2 * pi * antenna_dist *sind(ppp(1,:))*(carrier_freq/ c)));
angledummy = repmat(anglePhase,arraySize/2,1);
Z(arraySize/2+1:arraySize,:)=angledummy .* Z(arraySize/2+1:arraySize,:);

%forward backward smoothing
if smoothing==1
    J = fliplr(eye(size(R,1)));
    R = R + J * conj(R) * J;
end

% Eigen value decompostion of the auto correlation matrix
[Q,D] = eig(R);
[D,I] = sort(diag(D));
I = flipud(I);
Q = Q(:, I);
Q = Q(:,1:K);

for ind=1:K
    [indices,Ahat]=RAPMusic(Z,Q,indices_New,n,ind,ToF,AoA,res,endRange,slope,ts,c,carrier_freq,arraySize,antenna_dist,N);
    indices_New(ind,:)=indices;
    Ahat_New(:,ind)=Ahat;

end

% (indices_New*11.16)
% diff(indices_New*11.16)
% 
% sort(indices_New*11.16)
% diff(sort(indices_New*11.16))
  
end


function [indices,Ahat]=RAPMusic(SS,NN,indices_New,n,ind,dig_freq,AoA,res,endRange,slope,ts,c,carrier_freq,arraySize,antenna_dist,N)
   maxCorr = [];
 if ind>1
       Ahat = [];
       for x=1:ind-1
            indices=indices_New(x,:);
            [Ahat(:,x)] = calculateSteeringVec2D(indices(:,2),indices(:,1),slope,n,ts,c,carrier_freq,arraySize,antenna_dist);
       end
       PerpAhat = eye(size(Ahat,1)) - Ahat*pinv(Ahat'*Ahat)*Ahat';
 else
      Ahat = [];
      PerpAhat = eye(size(NN,1),'like',2+1i);
  end
  NN = PerpAhat*NN;
  Ps=(NN*NN');
  
  SSp=PerpAhat*SS;
  PsA = Ps*SSp;
  music_spec_num = sum((SSp').*(PsA.'),2);
  %music_spec_den = sum(conj(SSp).*(SSp),2);
  music_spec_den = sum((SSp').*(SSp.'),2);
  PP = abs(music_spec_num./music_spec_den);

  [maxCorr,loopVarMax] = max(PP);
  %figure;plot(PP);
  [ToF_idx,AoA_idx ] = ind2sub([size(dig_freq,2) size(AoA,2)],loopVarMax); %median(maxCorr));%
  indices=[ AoA(AoA_idx) dig_freq(ToF_idx) maxCorr];
  
  n = 0:N-1;
  Ahat=calculateSteeringVec2D(indices(:,2),indices(:,1),slope,n,ts,c,carrier_freq,arraySize,antenna_dist);
end

function Z=calculateSteeringVec2D(d,AoA,slope,n,ts,c,carrier_freq,arraySize,antenna_dist)
Z=exp(1j * ((2 * pi * slope* n' * 2*d * ts)/c));
anglePhase=exp(1j * (2 * pi * antenna_dist *sind(AoA)*(carrier_freq/ c)));
angledummy = repmat(anglePhase,arraySize/2,1);
Z(arraySize/2+1:arraySize,:)=angledummy .* Z(arraySize/2+1:arraySize,:);


Z=transpose(Z);
end