function [X_train, X_test, Y_train, Y_test] = unfold_data(X_cv, Y_cv, test_fold)
% UNFOLD_DATA Merges folded dataset for cross-validation into train and
%   test data.
%
% Inputs:
% - X_cv: folded state vector, shape (N/folds, N_states, folds)
% - Y_cv: folded output vector, shape (N/folds, N_out, folds)
% - test_fold: fold to use for testing
%
% Outputs:
% - X_train: state vector for training, shape (N - (N/folds), N_states)
% - X_test: state vector for testing, shape (N/folds, N_states)
% - Y_train: output vector for training, shape (N - (N/folds), N_out)
% - Y_test: output vector for testing, shape (N/folds, N_out)
%
% Jesse Hagenaars - 07.06.2018

N_states = size(X_cv, 2);
N_out = size(Y_cv, 2);

% Select test data
X_test = X_cv(:, :, test_fold);
Y_test = Y_cv(:, :, test_fold);

% Delete test data from train data
X_cv(:, :, test_fold) = [];
Y_cv(:, :, test_fold) = [];

% Merge training data
X_train = reshape(permute(X_cv, [2 1 3]), N_states, [])';
Y_train = reshape(permute(Y_cv, [2 1 3]), N_out, [])';

end