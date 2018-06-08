function FF_net = build_FF_net(N_neurons, activation_fns, input_range, train_algo, epochs, goal, min_grad, mu)
% BUILD_FF_NET Builds a struct containing the configuration of a
%   feed-forward neural network. Also checks for irregularities in the
%   input, and throws an error.
%
% Inputs:
% - N_neurons: number of neurons per layer (also input & output layer)
% - activation_fns: activation functions per layer
% - input_range: bounds on the input space (per input neuron)
% - train_algo: algorithm used for training
% - epochs: max number of epochs to train for
% - goal: desired performance, training stops once reached
% - min_grad: minimum gradient, training stops when abs(grad) below this
% - mu: learning rate, adapted during training
%
% Outputs:
% - FF_net: struct containing the network configuration
%
% Jesse Hagenaars - 02.06.2018

%%% Discrepancy tests %%%

% Hidden layers & assigned activation functions
if length(activation_fns) ~= length(N_neurons) - 1
    error('Discrepancy in number of hidden layers and assigned activation functions')
end

% Input neurons and bounds
if ~all(size(input_range) == [N_neurons(1) 2])
    error('Incorrect size for bounds on input space')
end

%%% Construct network %%%

FF_net = struct();
FF_net.name = 'feedforward';
FF_net.W = {};
FF_net.b = {};
FF_net.range = input_range;
FF_net.activation_fns = activation_fns;
FF_net.train_algo = train_algo;
FF_net.train_param = struct('epochs', epochs, 'goal', goal, 'min_grad', min_grad, 'mu', mu);

% Initialize with random weights
for i = 1:length(N_neurons) - 1
    
    % Shape [# next neurons, # current neurons]
    FF_net.W{i} = randn(N_neurons(i + 1), N_neurons(i));
    FF_net.b{i} = ones(N_neurons(i + 1), 1);
                    
end



