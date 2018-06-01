function [Y_WLS, Y_val_WLS, epsilon, epsilon_val, sigma2] = do_WLS(X, X_val, Y, Y_val, sigma_v)
% DO_WLS Performs a weighted least squares for X, which is split into
% identification and validation datasets.
%
% Inputs:
% - X: model identification dataset
% - X_val: model validation dataset
% - Y: measurements for model identification
% - Y_val: measurements for model validation
% - sigma_v: sensor noise for each sample
%
% Outputs:
% - Y: WLS estimate based on identification data
% - Y_val: WLS estimate based on validation data
% - epsilon: model residual of estimate based on identification data
% - epsilon_val: model residual of estimate based on validation data
% - sigma2: variance of the WLS estimator
%
% Jesse Hagenaars - 01.06.2018

% W is diagonal matrix of sensor noise variances of states (slide 67)
% How does this work for multiple states? --> it doesn't: variance per
%   sample
W = eye(size(X, 1));
W_inv = inv(W);

% Covariance and variance of WLS
cov = inv(X' * W_inv * X);
sigma2 = diag(cov);

% WLS estimator
theta_hat = cov * X' * W_inv * Y;

% Calculate WLS estimation
Y_WLS = X * theta_hat;
Y_val_WLS = X_val * theta_hat;

% Calculate model residuals
epsilon = Y - Y_WLS;
epsilon_val = Y_val - Y_val_WLS;

end
