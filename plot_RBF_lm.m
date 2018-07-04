% Plotting script for RBF neural network with Levenberg-Marquardt
%   optimization.
%
% Jesse Hagenaars - 14.06.2018

% Fig 1: output hypothesis by RBF neural network

% Get hypothesis
Y_hat = sim_net(net, data.X);

% And plot
plot_hypothesis(data.X, data.Y, Y_hat, save_fig, 'RBF Network - Levenberg-Marquart');