% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Part 3: RBF neural network
%
% Jesse Hagenaars - 11.05.2018

clear; close all; clc;
rng('default')

% Plot/save figures or not
do_plot = 0;
save_fig = 0;

%%% TO DO %%%
% - Check influence of pinv vs inv
% - Regularization? --> early stopping
% - Adaptive learning rate?
% - sum of dE_dy_hat_k, dv_k_dphi_j, dv_j_dw_ij or not?
% - NN notation!


%% Load & split reconstructed dataset

% State vector X, output vector Y
load('data/F16_reconstructed', 'X', 'Y')

% Sizes of train, validation, test
p_train = 0.7;
p_val = 0.15;
p_test = 0.15;
shuffle = 1;

% Split into training, validation and testing sets
[X_train, X_val, X_test, Y_train, Y_val, Y_test] = split_data(X, Y, p_train, p_val, p_test, shuffle);

% Put in struct
data = struct('X', X, 'X_train', X_train, 'X_val', X_val, 'X_test', X_test, 'Y', Y, 'Y_train', Y_train, 'Y_val', Y_val, 'Y_test', Y_test);


%% Part 3.1: Formulate RBF network + linear regression

%%% Network parameters %%%
        
% Layer sizes
N_input = 3;  % alpha, beta, V
N_hidden = 50;
N_output = 1;  % C_m

% Input bounds
R_input = [-ones(N_input, 1) ones(N_input, 1)];

%%% Training parameters %%%

% Number of epochs
N_epochs = 0;

% Training algorithm
train_algo = 'linregress';

% Learning rate
mu = 0;

% Goal (stop training when reached)
goal = 0;

% Minimal gradient (stop training when below)
min_grad = 0;

% Activation functions
activation_fn = {'radbas', 'purelin'};

%%% Create network %%%

net = RBF_net(N_input, N_hidden, N_output, R_input, N_epochs, train_algo, mu, goal, min_grad, activation_fn);

%%% Train network %%%

net = train_net(net, data);

%%% Plot results %%%

if do_plot
    plot_RBF_linregress
end


%% Part 3.2: Train RBF network with Levenberg-Marquardt

%%% Network parameters %%%
        
% Layer sizes
N_input = 3;  % alpha, beta, V
N_hidden = 50;
N_output = 1;  % C_m

% Input bounds
R_input = [-ones(N_input, 1) ones(N_input, 1)];

%%% Training parameters %%%

% Number of epochs
N_epochs = 100;

% Training algorithm
train_algo = 'lm';

% Learning rate
mu = 0.0001;

% Goal (stop training when reached)
goal = 0;

% Minimal gradient (stop training when below)
min_grad = 1e-10;

% Activation functions
activation_fn = {'radbas', 'purelin'};

%%% Create network %%%

net = RBF_net(N_input, N_hidden, N_output, R_input, N_epochs, train_algo, mu, goal, min_grad, activation_fn);

%%% Train network %%%

net = train_net(net, data);

%%% Plot results %%%

if do_plot
    plot_RBF_lm
end

%% Feed-forward neural network

%%% Network parameters %%%

% Number of layers and neurons
N_neurons = [4 3 3 3];  % input, hidden_1, ... , hidden_N, output

% Activation functions --> check with definition in assignment
activation_fns = {'tansig', 'tansig', 'purelin'};  % hidden_1, ... , hidden_N, output

% Bounds on input space
input_range = [-ones(N_neurons(1), 1) ones(N_neurons(1), 1)];

%%% Training parameters %%%

% Levenberg-Marquardt backpropagation
train_algo = 'trainlm';

% Low-level parameters
epochs = 100;  % max epochs during training
goal = 0;  % training stops if goal reached
min_grad = 1e-10;  % training stops if abs(gradient) lower than this
mu = 0.001;  % learning rate, adapted during training --> different structure?

%%% Build network %%%

FF_net = build_FF_net(N_neurons, activation_fns, input_range, train_algo, epochs, goal, min_grad, mu);

%%% Train network %%%



%%% Evaluate network %%%


