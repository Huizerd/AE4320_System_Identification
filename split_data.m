function [X_train, X_val, X_test, Y_train, Y_val, Y_test] = split_data(X, Y, p_train, p_val, p_test, shuffle)
% SPLIT_DATA Splits a dataset into training, validation and test data.
%
% Inputs:
% - X: state vector to split, shape (N, N_states)
% - Y: output vector to split, shape (N, N_out)
% - p_train: portion of the data to become training data
% - p_train: portion of the data to become training data
% - p_train: portion of the data to become training data
% - shuffle: whether or not to randomly shuffle data
%
% Outputs:
% - X_train: state vector, train split
% - X_test: state vector, test split
% - Y_train: output vector, train split
% - Y_test: output vector, test split
%
% . - 31.05.2018

N = size(X, 1);

% Set shuffle order
if shuffle
    new_order = randperm(N);
else
    new_order = 1:N;
end

% Apply shuffle
X = X(new_order, :);
Y = Y(new_order, :);

% Get indices for end training data / begin test data
idx_train = fix(p_train * N);
idx_val = fix(p_val * N) + idx_train;
idx_test = fix(p_test * N) + idx_val;

X_train = X(1:idx_train, :);
X_val = X(idx_train + 1:idx_val, :);
X_test = X(idx_val + 1:idx_test, :);
Y_train = Y(1:idx_train, :);
Y_val = Y(idx_train + 1:idx_val, :);
Y_test =Y(idx_val + 1:idx_test, :);

end