%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the symbolic construction of an augmented state observation matrix
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

% define variables
syms('Vx', 'Vz', 'Ax', 'Lx', 'Az', 'Lz', 'g', 'th', 'q', 'Lq', 'ze');

% define state vector
x = [Vx; Vz; th; ze];
xaug = [x; Lx; Lz; Lq];
xaug0 = [100; 20; .01; 1240; 1; 1; 1]; % initial augmented state

naugstates = length(xaug);

% define state transition function
faug = [(Ax-Lx) - g*sin(th) - (q-Lq)*Vz;
     (Az-Lz) + g*cos(th) + (q-Lq)*Vx;
     (q-Lq);
     -Vx*sin(th)+Vz*cos(th);
     0;
     0;
     0;];
 
% define state observation function
haug = [sqrt(Vx^2+Vz^2); 
     th; 
     -ze];
 
% use the kf_calcNonlinObsRank function to calculate the rank of the observation matrix
rankObsaug = kf_calcNonlinObsRank(faug, haug, xaug, xaug0);
if (rankObsaug >= naugstates)
    fprintf('Augmented Observability matrix is of Full Rank: the augmented state is Observable!\n');
else
    fprintf('Augmented Observability matrix is of NOT Full Rank: the augmented state is NOT Observable!\n');
end



