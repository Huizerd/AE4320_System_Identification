% Plotting script for FF neural network with Levenberg-Marquardt
%   optimization.
%
% . - 15.07.2018

% Fig 1: output hypothesis by FF neural network

% Get hypothesis
Y_hat = sim_net(net, data.X);

% And plot
plot_hypothesis(data.X, data.Y, Y_hat, save_fig, 'FF Network - Levenberg-Marquart');