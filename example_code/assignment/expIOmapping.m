% expIOmapping.m
%
%   Description:
%       This file shows how the simNet.m function can be used to simulate
%       the output of two network types: feedforward and RBF.
%
%       The file also demonstrates how the entire IO mapping can be
%       represented in MATLAB.
%
%   Construction date:  17-06-2009
%   Last updated:       17-06-2009
%
%   E. de Weerdt
%   TUDelft, Faculty of Aerospace Engineering, ASTI & Control and
%   Simulation Division

%%   Clearing workspace
clear all
close all
clc

%%    Loading networks
%---------------------------------------------------------
load NetExampleFF netFF
load NetExampleRBF netRBF

%%   Creating IO data points
%---------------------------------------------------------
evalres     = 0.05;
minXI       = -1*ones(1,2);
maxXI       = 1*ones(1,2);
[xxe yye]   = ndgrid((minXI(1):evalres:maxXI(1))', (minXI(2):evalres:maxXI(2)'));
Xeval       = [xxe(:), yye(:)]';

%%   Simulating the network output
%---------------------------------------------------------
yFF     = simNet(netFF,Xeval);
yRBF    = simNet(netRBF,Xeval);

%%   Plotting results
%---------------------------------------------------------
%   ... creating triangulation (only needed for plotting)
TRIeval     = delaunayn(Xeval',{'Qbb','Qc','QJ1e-6'});

%   ... viewing angles
az = 140;
el = 36;

%   ... creating figure for FF network
plotID = 1012;
figure(plotID);
trisurf(TRIeval, Xeval(1, :)', Xeval(2, :)', yFF', 'EdgeColor', 'none'); 
grid on;
view(az, el);
titstring = sprintf('Feedforward neural network - IO mapping');
xlabel('x_1');
xlabel('x_2');
zlabel('y');
title(titstring);

%   ... set fancy options for plotting FF network
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(yFF);
shading interp;
lighting phong;
drawnow();

%   ... creating figure for RBF network
plotID = 1;
figure(plotID);
trisurf(TRIeval, Xeval(1, :)', Xeval(2, :)', yRBF', 'EdgeColor', 'none'); 
grid on;
view(az, el);
titstring = sprintf('RBF neural network - IO mapping');
xlabel('x_1');
xlabel('x_2');
zlabel('y');
title(titstring);

%   ... set fancy options for plotting FF network
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(yRBF);
shading interp;
lighting phong;
drawnow();

%----------------------------- end of file --------------------------------