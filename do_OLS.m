function [Y_OLS, Y_val_OLS, epsilon, epsilon_val, sigma2] = do_OLS(X, X_val, Y, Y_val)
% DO_OLS Performs an ordinary least squares for X, which is split into
% identification and validation datasets.
%
% Inputs:
% - X: model identification dataset
% - X_val: model validation dataset
% - Y: measurements for model identification
% - Y_val: measurements for model validation
%
% Outputs:
% - Y: OLS estimate based on identification data
% - Y_val: OLS estimate based on validation data
% - epsilon: model residual of estimate based on identification data
% - epsilon_val: model residual of estimate based on validation data
% - sigma2: variance of the OLS estimator
%
% Jesse Hagenaars - 01.06.2018

% Covariance and variance of OLS
cov = inv(X' * X);  % --> shouldn't this also include sigma^2?
sigma2 = diag(cov);

% OLS estimator
theta_hat = cov * X' * Y;

% Calculate OLS estimation
Y_OLS = X * theta_hat;
Y_val_OLS = X_val * theta_hat;

% Calculate model residuals --> noisy since Y is noisy
epsilon = Y - Y_OLS;
epsilon_val = Y_val - Y_val_OLS;

% Check BLUEness
sigma2_I = mean(epsilon' * epsilon);
sigma2_hat = (epsilon' * epsilon) ./ (size(X, 1) - size(X, 2));

end
