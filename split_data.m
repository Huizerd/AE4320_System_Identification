function [X_train, X_test, Y_train, Y_test] = split_data(X, Y, p_train, shuffle)
% SPLIT_DATA Splits a dataset into training data and test data. No
%   shuffling, training data first.
%
% Inputs:
% - X: state vector to split, shape (t, N_states)
% - Y: measurement vector to split, shape (t, N_meas)
% - p_train: portion of the data to become training data
% - shuffle: whether or not to randomly shuffle data
%
% Outputs:
% - X_train: state vector, train split
% - X_test: state vector, test split
% - Y_train: measurement vector, train split
% - Y_test: measurement vector, test split
%
% Jesse Hagenaars - 31.05.2018

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

X_train = X(1:idx_train, :);
X_test = X(idx_train + 1:end, :);
Y_train = Y(1:idx_train, :);
Y_test = Y(idx_train + 1:end, :);

end