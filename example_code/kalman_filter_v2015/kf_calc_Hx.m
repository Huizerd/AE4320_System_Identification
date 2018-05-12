%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H = kf_calcDHx(x) Calculates the Jacobian of the output dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hx = kf_calcDHx(t, x, u)

    n = size(x, 1);
    
    % Calculate Jacobian matrix of output dynamics
    if (n == 1)
%         Hx(1) = 0;
        Hx = 3*x^2;
    elseif (n == 2)
%         Hx = [3*x(1)^2 -2*x(2); 
%                0        -1];
%         Hx = [x(1)/(x(1)^2 + x(2)^2)^(1/2)    x(2)/(x(1)^2 + x(2)^2)^(1/2);];
        Hx = [2*x(1)    2*x(2)];
%         Hx = [1    1];
               
    end
