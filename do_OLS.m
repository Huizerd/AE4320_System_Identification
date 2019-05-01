function [Y_hat, theta_hat] = do_OLS(A, Y)
% DO_OLS Performs an ordinary least squares for state vector X.
%
% Inputs:
% - A: regression matrix
% - Y: measurement vector, shape(N, N_meas)
%
% Outputs:
% - Y_hat: OLS estimate (hypothesis)
% - theta_hat: OLS estimator
%
% . - 01.06.2018

% OLS estimator
% Use pinv instead of inv(X' * X) * X' to prevent numerical instability
theta_hat = pinv(A) * Y;

% Calculate OLS estimate
Y_hat = A * theta_hat;

end
