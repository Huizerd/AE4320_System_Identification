%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xdot = kf_calclinFx(x) Calculates the system dynamics MATRIX F 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = kf_calclinFx(t, x, u)

    
    % system dynamics go here!
    xdot(1) = -1*x;
%     xdot(1) = -0.3*cos(x)^3;
%     xdot(1) = -0.1*x^3 + .5*x^2;

