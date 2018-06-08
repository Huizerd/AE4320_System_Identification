% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Part 2: State & parameter estimation
%
% Jesse Hagenaars - 11.05.2018

clear; close all; clc;
rng('default')

% Plot/save figures or not
do_plot = 1;
save_fig = 0;

%%% TO DO %%%
% - 2.5 in function
% - theta_hat very small?


%% Load data

data_name = 'data/F16traindata_CMabV_2018';

[~, Z_k, U_k] = load_F16_data(data_name);


%% Given parameters

% Time vector, samples at dt = 0.01 s
dt = 0.01;
t = 0:0.01:0.01*(size(Z_k, 2)-1);

% System/process noise statistics
Ew = [0 0 0 0];  % noise bias, E{w(t)}, given in assignment
sigma_w = [1e-3 1e-3 1e-3 0];  % noise std. dev., [u v w C_m_alpha_up], given in assignment

% Measurement/sensor noise statistics:
Ev = [0 0 0];  % noise bias, E{v(t)}, given in assignment
sigma_v = [0.01 0.0058 0.112];  % noise std. dev., [alpha_m, beta_m, V_m], given in assignment


%% Description of state transition and measurement equation

% State equation: x_dot(t) = f(x(t), u(t), t)
%   with state vector: x(t) = [u v w C_alpha_up]'
%   and input vector: u(t) = [u_dot v_dot w_dot]'

% Measurement equation: z_n(t) = h(x(t), u(t), t)
%   with measurement vector: z_n(t) = [alpha_m, beta_m, V_m]'
%   combined with white noise: z(t_k) = z_n(t_k) + v(t_k)


%% Observability check --> proves convergence of KF

check_observability


%% Part 2.3: Perform Kalman filtering
% IEKF because it works very well if main non-linearities are in the
%   measurement equation --> this is the case here (slide 20)

% Do IEKF
% - X = X_hat_k1_k1
% - Y = Z_k1_k
[X, Y_biased, IEKF_count] = do_IEKF(U_k, Z_k, dt, sigma_w, sigma_v);


%% Part 2.4: Reconstruct alpha_true

% Correct alpha for bias
Y = Y_biased;
Y(1, :) = Y(1, :) ./ (1 + X(4, :));

% Plot results if set
if do_plot
    plot_IEKF
end

% Transpose data (more suited from here on)
X = X';
Y = Y';


%% Split data

% Fraction training data
p_train = 0.7;

% Whether or not to randomly shuffle data
shuffle = 1;

% Split states, measurements
[X_train, X_test, Y_train, Y_test] = split_data(X, Y, p_train, shuffle);


%% Part 2.5: Formulate regression problem + LS estimator & estimate coeff.

% Look at:
% - Influence of estimator on accuracy of fit
% - Influence of oder of polynomial on accuracy of fit
% - Statistical validation --> parameter (co)variances
% - Model-error-based validation --> analysis of model residuals

% Steps:
% 1. Use a-priori knowledge
% 2. Determine model structure --> polynomial
% 3. Determine estimator --> WLS (for now) --> does it violate assumption 
%    of uncorrelated residuals?
% 4. Validation: ...

% Order of polynomial model
order = 5;

% Create regression matrix
A_train = create_regression_matrix(X_train, order);

% Perform OLS
[Y_hat_train, ~] = do_OLS(A_train, Y_train);

% Plot results if set
if do_plot
    plot_OLS
end


%% Part 2.6: Influence of polynomial model order on accuracy of fit
% --> can be deleted: same as below

% % Maximum model order to test
% max_order = 10;
% 
% % Get MSE as function of order
% MSE = MSE_for_order(X_train, Y_train, max_order);
% 
% % Plot results if set
% if do_plot
%     plot_order
% end


%% Part 2.7: Model validation --> also includes 2.6

%%% Model order selection %%%
% DISCLAIMER: due to rapidly increasing computational complexity as order
%   increases, I only fit up to 15th-order models. The optimal order came 
%   out as 13, which might be a local minimum.

% Maximum model order to test
max_order = 5;

% Number of cross-validation folds
folds = 10;

% Get optimal OLS estimator and MSE for plotting
[OLSE_opt, MSE] = optimize_model(X_train, X_test, Y_train, Y_test, max_order, folds);

% Plot results if set
if do_plot
    plot_model_select
end

%%% Model-error-based validation %%%

% Perform validation
[epsilon, epsilon_ac, lags, conf_95] = model_error_validation(X_test, Y_test, OLSE_opt);

% Plot results if set
if do_plot
    plot_model_error_validation
end

%%% Statistical validation %%%

% Get parameter covariance and variance
theta_hat_Cov = statistical_validation(X_test, Y_test, OLSE_opt);
sigma2_theta_hat = diag(theta_hat_Cov);

% Plot results if set
if do_plot
    plot_statistical_validation
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


