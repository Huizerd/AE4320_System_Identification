%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xdot = kf_calcFx(x) Calculates the system dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = kf_calcFx(t, x, u)

    n = size(x, 1);
    xdot = zeros(n, 1);
    
    % system dynamics go here!
    if (n == 1)
%         xdot(1) = -1*x;
        xdot(1) = -0.3*cos(x)^3;
    elseif (n == 2)
%         xdot(1) = -0.3*cos(x(1))^3;
%         xdot(2) = sin(x(2));
%         xdot(1) = x(2)*cos(x(1))^3;
%         xdot(2) = sin(x(2));
        xdot(1) = x(2)*cos(x(1))^3;
        xdot(2) = -x(2);
%         xdot(1) = x(2)*cos(x(1))^3;
%         xdot(2) = 0;
    end
