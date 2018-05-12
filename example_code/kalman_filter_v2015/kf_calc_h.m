%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% zpred = kf_calcHx(x) Calculates the output dynamics equation h(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zpred = kf_calcHx(t, x, u)
    
    n = size(x, 1);
%     zpred = zeros(n, 1);
    
    % output dynamics go here!
    if (n == 1)
%         zpred(1) = 3*x;
        zpred(1) = x^3;
    elseif (n == 2)
%         zpred(1) = sqrt(x(1)^2 + x(2)^2);
%         zpred(2) = -x(2);
%         zpred(1) = sqrt(x(1)^2 + x(2)^2);
        zpred(1) = (x(1)^2 + x(2)^2);
%         zpred(1) = (x(1) + x(2));
    end
    