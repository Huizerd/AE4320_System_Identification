function [epsilon, epsilon_ac, lags, conf_95] = model_error_validation(X_test, Y_test, OLSE)
% MODEL_ERROR_VALIDATION Performs model-error-based validation of the OLS
%   estimator. Two assumptions need to be proven for the OLS estimator to
%   be BLUE:
%
%       1. E{epsilon} = 0 (residual is zero-mean white noise)
%       2. E{epsilon * epsilon'} = sigma^2 * I (residuals have constant
%          variance for all outputs and are uncorrelated)
%
% Inputs:
% - X_test: state vector for testing, shape (N, N_states)
% - Y_test: output vector for testing, shape (N, N_out)
% - OLSE: struct containing theta_hat and the model order for the OLS
%   estimator
%
% Outputs:
% - epsilon: model residuals
% - epsilon_ac: autocorrelation of model residuals
% - lags: lags for which the autocorrelation was calculated
% - conf_95: 95% confidence interval of model residuals
%
% Jesse Hagenaars - 07.06.2018

%%% Assumption 1 %%%

% Get hypothesis
A = create_regression_matrix(X_test, OLSE.order_opt);
Y_hat = A * OLSE.theta_hat;

% Calculate residuals
epsilon = Y_hat - Y_test;

%%% Assumption 2 %%%

% Compute autocorrelation of residuals
[epsilon_ac, lags] = xcorr(epsilon - mean(epsilon, 1));

% Normalize
epsilon_ac = epsilon_ac ./ max(epsilon_ac, 1);

% 95% confidence interval (two std. dev.)
conf_95 = sqrt(1 / size(epsilon_ac, 1)) * [-2; 2];

end