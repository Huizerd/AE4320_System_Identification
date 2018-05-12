%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F = kf_calcDFx(x) Calculates the Jacobian of the system dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DFx = kf_calcDFx(t, x, u)
    
    n = size(x, 1);
    
    % Calculate Jacobian matrix of system dynamics
    if (n == 1)
    %     DFx = -1;
        DFx = 0.9 * cos(x)^2 * sin(x); 
    elseif (n == 2)
%         DFx   = [0.9 * cos(x(1))^2 * sin(x(1)) 0;
%                  0                             cos(x(2))];
%         DFx   = [x(2) * cos(x(1))^2 * sin(x(1)) cos(x(1))^3;
%                  0                             cos(x(2))];
        DFx   = [x(2) * cos(x(1))^2 * sin(x(1)) cos(x(1))^3;
                 0                             -1];
%         DFx   = [x(2) * cos(x(1))^2 * sin(x(1)) cos(x(1))^3;
%                  0                             0];
    end
    

