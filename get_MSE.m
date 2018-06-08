function MSE = get_MSE(Y_hat, Y)
% GET_MSE Computes the mean-squared error between 2 (measurement) vectors.
%
% Inputs:
% - Y_hat: estimated vector, shape (N, N_meas)
% - Y: true vector, shape (N, N_meas)
%
% Outputs:
% - MSE: mean-squared error
%
% Jesse Hagenaars - 07.06.2018

N = size(Y, 1);

% Compute residuals
epsilon = Y - Y_hat;

% Compute MSE
% Element-wise multiplication + sum to prevent for-loop
MSE = 1 / N * (epsilon' * epsilon);

% Select only diagonal
MSE = diag(MSE);

end