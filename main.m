% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Jesse Hagenaars - 11.05.2018

clear; close all; clc;
rng('default')

% Save figures or not
save_fig = 1;

%%% TO DO %%%
% - Legend only in 1st graph?
% - Check BLUEness OLS: computation of sigma^2/covariance
% - Uselessness of WLS/GLS?
% - Check accuracy problems due to singularity
% - Look into colours vs --/.- for plotting
% - Maybe make 'accuracy of fit' quantitative as well? --> model residual
% - Whiteness of residual --> GLS?


%% Load data

data_name = 'data/F16traindata_CMabV_2018';

[Cm, Z_k, U_k] = load_F16_data(data_name);


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


%% IEKF

% IEKF because it works very well if main non-linearities are in the
%   measurement equation --> this is the case here (slide 20)

[X_hat_k1_k1, Z_k1_k, IEKF_count] = do_IEKF(U_k, Z_k, dt, sigma_w, sigma_v);

% Correct alpha for bias
Z_k1_k_corr = Z_k1_k;
Z_k1_k_corr(1, :) = Z_k1_k(1, :) ./ (1 + X_hat_k1_k1(4, :));

%% Plotting IEKF

plot_IEKF


%% Parameter estimation

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

%%% Create model %%%

% Split into identification and validation data
split_idx = floor(size(X_hat_k1_k1, 2) / 2);  % 50:50
[states_id, states_val] = split_data(X_hat_k1_k1', split_idx);
[Y, Y_val] = split_data(Z_k1_k_corr', split_idx);
[t_id, t_val] = split_data(t', split_idx);

% Evaluate various orders
orders = [1 2 3];

% Autocorrelation 95% confidence intervals
conf_95 = 1.96 / sqrt(size(Y, 1));
conf_95_val = 1.96 / sqrt(size(Y_val, 1));

% Number of lags
num_lags = 500;

% Create matrices to store
Y_ORD_OLS = zeros([size(Y) length(orders)]);
Y_VAL_ORD_OLS = zeros([size(Y_val) length(orders)]);
EPSILON_ORD_OLS = zeros([size(Y) length(orders)]);
EPSILON_VAL_ORD_OLS = zeros([size(Y_val) length(orders)]);
SIGMA2_ORD_OLS = cell(2, length(orders));
MEAN_EPSILON_ORD_OLS = cell(2, length(orders));
MEAN_EPSILON_VAL_ORD_OLS = cell(2, length(orders));
AC_EPSILON_ORD_OLS = zeros(num_lags * 2 + 1, size(Y, 2), length(orders));
AC_EPSILON_VAL_ORD_OLS = zeros(num_lags * 2 + 1, size(Y, 2), length(orders));

for i = 1:length(orders)
    
    % Create regression matrix X, for identification and validation
    X = create_poly_model(states_id, orders(i));
    X_val = create_poly_model(states_val, orders(i));
     
    %%% Ordinary least squares %%%

    % OLS assumptions (necessary to make OLS BLUE):
    % 1. residual variance is constant (E{epsilon' * epsilon} = sigma^2 * I)
    % 2. residual is zero-mean white noise

    [Y_OLS, Y_val_OLS, epsilon_OLS, epsilon_val_OLS, sigma2_OLS] = do_OLS(X, X_val, Y, Y_val);
    
    % Compute autocorrelation of residuals
    [ac_epsilon_OLS, ~] = xcorr(epsilon_OLS - mean(epsilon_OLS, 1), num_lags, 'coeff');
    [ac_epsilon_val_OLS, lags] = xcorr(epsilon_val_OLS - mean(epsilon_val_OLS, 1), num_lags, 'coeff');
    
    % Store
    Y_ORD_OLS(:, :, i) = Y_OLS;
    Y_VAL_ORD_OLS(:, :, i) = Y_val_OLS;
    EPSILON_ORD_OLS(:, :, i) = epsilon_OLS;
    EPSILON_VAL_ORD_OLS(:, :, i) = epsilon_val_OLS;
    SIGMA2_ORD_OLS{1, i} = orders(i);
    SIGMA2_ORD_OLS{2, i} = sigma2_OLS;
    MEAN_EPSILON_ORD_OLS{1, i} = orders(i);
    MEAN_EPSILON_ORD_OLS{2, i} = mean(epsilon_OLS, 1);
    MEAN_EPSILON_VAL_ORD_OLS{1, i} = orders(i);
    MEAN_EPSILON_VAL_ORD_OLS{2, i} = mean(epsilon_val_OLS, 1);
    AC_EPSILON_ORD_OLS(:, :, i) = ac_epsilon_OLS(:, [1 5 9]);  % only autocorrelation
    AC_EPSILON_VAL_ORD_OLS(:, :, i) = ac_epsilon_val_OLS(:, [1 5 9]);  % only autocorrelation

    %%% Weighted least squares %%%

    % NOT OF VERY MUCH USE? --> since we don't have variance per sample point

    % Differences with OLS:
    % - Assumes epsilon is not white
    % - Includes a-priori information in W --> std. dev. of sensor noise

    % [Y_WLS, Y_val_WLS, epsilon_WLS, epsilon_val_WLS, sigma2_WLS] = do_WLS(X, X_val, Y, Y_val, sigma_v);
    
    %%% Generalized Least Squares

end

%% Plotting parameter estimation

plot_param_est


%% RBF neural network

%%% Network parameters %%%

% Number of layers and neurons
N_hidden_layers = 2;
N_neurons = [4 3 3 3];  % input, hidden_1, ... , hidden_N, output

% Check.. --> write function for this?
if length(N_neurons) ~= N_hidden_layers + 2
    warning('Discrepancy in number of hidden layers and assigned neurons')
end

% Activation functions --> check with definition in assignment
activation_functions = {'radbas', 'radbas', 'purelin'};  % hidden_1, ... , hidden_N, output

% Check..
if length(activation_functions) ~= N_hidden_layers + 1
    warning('Discrepancy in number of hidden layers and assigned activation functions')
end



%%% Training parameters %%%



%%% Build network %%%



%%% Evaluate network %%%


