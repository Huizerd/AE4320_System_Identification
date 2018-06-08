function [OLSE_opt, MSE] = optimize_model(X_train, X_test, Y_train, Y_test, max_order, folds)
% OPTIMIZE_MODEL Optimizes the OLS order.
%
% Inputs:
% - X_train: state vector for training, shape (N_train, N_states)
% - X_test: state vector for testing, shape (N_test, N_states)
% - Y_train: measurement vector for training, shape (N_train, N_meas)
% - Y_test: measurement vector for testing, shape (N_test, N_meas)
% - max_order: maximum polynomial model order
% - folds: number of folds for cross-validation
%
% Outputs:
% - OLSE_opt: struct containing the optimal OLS estimator
% - MSE: array containing mean-squared error for training, validation,
%   testing, shape (max_order, N_meas, 3)
%
% Jesse Hagenaars - 07.06.2018

N_meas = size(Y_train, 2);

% Split data into folds for cross-validation
[X_cv, Y_cv] = fold_data(X_train, Y_train, folds);

% MSE for train, validation, test
MSE = zeros(max_order, folds, N_meas, 3);

% Loop over orders
for o = 1:max_order
    % Loop over folds
    for f = 1:folds
        
        % Unfold into train and test data
        [x_train, x_val, y_train, y_val] = unfold_data(X_cv, Y_cv, f);
        
        % Set test data
        x_test = X_test;
        y_test = Y_test;
        
        % Create regression matrix
        A_train = create_regression_matrix(x_train, o);
        A_val = create_regression_matrix(x_val, o);
        A_test = create_regression_matrix(x_test, o);
        
        % Perform OLS
        [y_hat_train, theta_hat] = do_OLS(A_train, y_train);
        y_hat_val = A_val * theta_hat;
        y_hat_test = A_test * theta_hat;
        
        % Get MSE
        MSE(o, f, :, 1) = get_MSE(y_hat_train, y_train);
        MSE(o, f, :, 2) = get_MSE(y_hat_val, y_val);
        MSE(o, f, :, 3) = get_MSE(y_hat_test, y_test);
        
    end   
end

% Average over folds
MSE = squeeze(mean(MSE, 2));

% Average over measurements
MSE_avg = squeeze(mean(MSE, 3));

% Determine optimal order from minimal test error
order_opt = find(MSE_avg(:, 3) == min(MSE_avg(:, 3)));

% Now train with this optimal model order on all training data
A_opt = create_regression_matrix(X_train, order_opt);
[~, theta_hat_opt] = do_OLS(A_opt, Y_train);

% Create struct to return
OLSE_opt.theta_hat = theta_hat_opt;
OLSE_opt.order_opt = order_opt;

end
        
        
        