function theta_hat_Cov = statistical_validation(X_test, Y_test, OLSE)
% STATISTICAL_VALIDATION Performs statistical validation of the parameters
%   of the OLS estimator. Model-error-based validation showed that the OLS
%   estimator was not BLUE, since residual variance was not constant
%   (heteroscedasticity). This means that:
%
%       E{epsilon' * epsilon} ~= sigma^2 * I
%
%   and that the covariance of the OLS estimator parameters has to be
%   computed using:
%
%       Cov{theta_hat} = inv(A' * A) * A' * E{e * e'} * A * inv(A' * A)
%
%   where e = epsilon (to fit equation on line). See slide 20 for
%   derivation.
%
% Inputs:
% - X_test: state vector for testing, shape (N, N_states)
% - Y_test: measurement vector for testing, shape (N, N_meas)
% - OLSE: struct containing theta_hat and the model order for the OLS
%   estimator
%
% Outputs:
% - theta_hat_Cov: covariance matrix of the OLS estimator parameters
%
% Jesse Hagenaars - 07.06.2018

% Get hypothesis
A = create_regression_matrix(X_test, OLSE.order_opt);
Y_hat = A * OLSE.theta_hat;

% Calculate residuals
epsilon = Y_hat - Y_test;

% Make use of MATLAB's pinv() to prevent numerical instability
theta_hat_Cov = pinv(A) * (epsilon * epsilon') * A * pinv(A) / A';

end

