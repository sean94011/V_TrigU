function [F] = RadiationPattern(theta,phi)
% RadiationPattern Generates the radiation pattern of the IMAGEVK-74 Antenna
% F = RadiationPattern(theta,phi) Returns a radiation pattern F for vector of
% theta phi directions D. The E-Plane is the xz-plane (phi = 0) and the H-Plane is
% the yz-plane (phi = pi/2). 

F = sqrt(cos(theta/2).^2.*(sin(phi).^2+(cos(theta).*cos(phi)).^2));