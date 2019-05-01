% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Part 2: State & parameter estimation
%
% . - 11.05.2018

clear; close all; clc;
rng('default')

% Plot/save figures or not
do_plot = 1;
save_fig = 0;

%%% TO DO %%%
% - Train error increasing?
% - Standardize/normalize data?


%% Load data

data_name = 'data/F16traindata_CMabV_2018';

[C_m, Z_k, U_k] = load_F16_data(data_name);


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
[X_hat_k1_k1, Z_k1_k_biased, IEKF_count] = do_IEKF(U_k, Z_k, dt, sigma_w, sigma_v);


%% Part 2.4: Reconstruct alpha_true

% Correct alpha for bias
Z_k1_k = Z_k1_k_biased;
Z_k1_k(1, :) = Z_k1_k(1, :) ./ (1 + X_hat_k1_k1(4, :));

% Plot results if set
if do_plot
    plot_IEKF
end


%% Save & split data

% Save reconstructed data to .mat file
save_data(Z_k1_k', C_m')

% Transpose & rename data (more suited from here on)
% Measurements as states, C_m as output
X = Z_k1_k';
Y = C_m';

% Fraction training data
p_train = 0.7;
p_val = 0;
p_test = 0.3;

% Whether or not to randomly shuffle data
shuffle = 1;

% Split states, output
[X_train, ~, X_test, Y_train, ~, Y_test] = split_data(X, Y, p_train, p_val, p_test, shuffle);


%% Part 2.5: Formulate regression problem + LS estimator & estimate coeff.

% Order of polynomial model
order = 10;

% Get OLS estimator based on training data
[~, theta_hat] = lin_regress(X_train, Y_train, order);

% Get hypothesis for all data
A = create_regression_matrix(X, order);
Y_hat = A * theta_hat;

% Plot results if set
if do_plot
    plot_OLS
end


%% Part 2.7: Model validation --> also includes 2.6 (influence of order)

%%% Model order selection %%%

% Maximum model order to test
max_order = 15;

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

