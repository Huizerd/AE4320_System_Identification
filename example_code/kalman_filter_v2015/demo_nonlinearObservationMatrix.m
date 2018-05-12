%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the symbolic construction of the state observation matrix
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
x0 = [100; 20; .01; 1240]; % initial state

xaug = [x; Ax; Lx; Az; Lz; g; q; Lq;];
xaug0 = [x0; rand; rand; rand; rand; 9.81; rand; rand;];

nstates = length(x); % length of state vector

% define state transition function
f = [(Ax-Lx) - g*sin(th) - (q-Lq)*Vz;
     (Az-Lz) + g*cos(th) + (q-Lq)*Vx;
     (q-Lq);
     -Vx*sin(th)+Vz*cos(th)];
 
% define state observation function
h = [sqrt(Vx^2+Vz^2); 
     th; 
     -ze];


% Calculate the first Lie derivative
%   calculate Jacobian of state observation function
Hx = simplify(jacobian(h, x));
%   construct complete Lie derivative
Hxf = Hx * f;

% Calculate the second Lie derivative
%   calculate Jacobian of Hx * f
Hxxf = simplify(jacobian(Hxf, x));
%   construct complete Lie derivative
Hxxff = Hxxf * f;

% Calculate the third Jacobian of the second Lie derivative
%   calculate Jacobian of Hx * f
Hxxxff = simplify(jacobian(Hxxff, x));

% Construct observability matrix:
Ox = [Hx; Hxxf; Hxxxff];

% calculate rank by substituting initial state values 
Obsnum = (subs(Ox, x, x0));
Obsnum2 = (subs(Ox, xaug, xaug0));

rankObs = double(rank(Obsnum));
rankObs2 = double(rank(Obsnum2));
fprintf('\t-> Rank of Observability matrix is %d\n', rankObs);
if (rankObs >= nstates)
    fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
else
    fprintf('Observability matrix is NOT of Full Rank: the state is NOT Observable!\n');
end




