
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of least squares estimators
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


printfigs = 0;
figpath = '';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% measurement vector
Y = [-1 2 5 1]';
% state vector
x = [0 1 2 3]';

xval = (0:.01:3)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create linear regression model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create regression matrix X such that p(x,theta) = X * theta

X = [ones(size(x)) x]; % linear polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2
Xval = [ones(size(xval)) xval];

% X = [ones(size(x)) x x.^2]; % quadratic polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2
% Xval = [ones(size(xval)) xval xval.^2];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do ordinary least squares estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

COV = inv(X'*X);
chat = COV * X' * Y;

Yval = Xval * chat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [50 200 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 6);
axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Measurement Data');
if (printfigs == 1)
    fpath = sprintf('fig_demoOLSdata');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

%%
plotID = 1001;
figure(plotID);
set(plotID, 'Position', [50 600 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, Yval, 'b', 'linewidth', 2);
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 6);
axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('p(x,\theta)');
title('Data and OLS model');
legend('OLS model', 'data', 'Location', 'northeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoOLS');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end





