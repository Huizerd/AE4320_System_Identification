function [Y_hat, theta_hat] = lin_regress(X, Y, order)
% LIN_REGRESS Performs linear regression using OLS for a certain polynomial
%   model order.
%
% Inputs:
% - X: state vector, shape (N, N_states)
% - Y: output vector, shape (N, N_out)
% - order: order of the polynomial model to use
%
% Outputs:
% - Y_hat: OLS estimate (hypothesis), shape (N, N_out)
% - theta_hat: OLS estimator
%
% . - 08.06.2018

% Create regression matrix
A = create_regression_matrix(X, order);

% Perform OLS
[Y_hat, theta_hat] = do_OLS(A, Y);

end