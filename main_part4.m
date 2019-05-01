% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Part 4: FF neural network
%
% . - 15.07.2018

clear; close all; clc;
rng('default')

% Plot/save figures or not
do_plot = 1;
save_fig = 0;

%%% TO DO %%%
% - weight initialization: between -1 and 1? Smart? --> 
%   https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e


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


%% Part 4.1: Formulate FF network + backprop

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
train_algo = 'backprop';

% Learning rate
mu = 0.01;
mu_inc = 1;  % increase factor for LM
mu_dec = 0.9;  % decrease factor for LM & backprop
mu_max = 1e10;  % max learning rate for LM

% Goal (stop training when reached)
goal = 0;

% Minimal gradient (stop training when below)
min_grad = 1e-10;

% Activation functions
activation_fn = {'tansig', 'purelin'};

%%% Create network %%%

net = FF_net(N_input, N_hidden, N_output, R_input, N_epochs, train_algo, mu, mu_inc, mu_dec, mu_max, goal, min_grad, activation_fn);

%%% Train network %%%

net = train_net(net, data);

%%% Plot results %%%

if do_plot
    plot_FF_linregress
end


%% Part 4.2: Train FF network with Levenberg-Marquardt

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
mu_inc = 10;  % increase factor for LM
mu_dec = 0.1;  % decrease factor for LM
mu_max = 1e10;  % max learning rate for LM

% Goal (stop training when reached)
goal = 0;

% Minimal gradient (stop training when below)
min_grad = 1e-10;

% Activation functions
activation_fn = {'tansig', 'purelin'};

%%% Create network %%%

net = FF_net(N_input, N_hidden, N_output, R_input, N_epochs, train_algo, mu, mu_inc, mu_dec, mu_max, goal, min_grad, activation_fn);

%%% Train network %%%

net = train_net(net, data);

%%% Plot results %%%

if do_plot
    plot_FF_lm
end

