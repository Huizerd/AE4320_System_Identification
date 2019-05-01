% Object definition for the feed-forward neural network.
% Makes sense since network can be seen as 'object'.
%
% . - 15.07.2018

classdef FF_net
    % FF_NET Defines a feed-forward neural network.
    
    properties
        
        name = 'ff';
        
        %%% Network parameters %%%
        
        % Layer sizes
        N_input; N_hidden; N_output;
        
        % All weights
        N_weights;
        
        % Weights (no biases)
        W_1; W_2;
        
        % Input bounds
        R_input;
        
        %%% Training parameters %%%
        
        % Number of epochs
        N_epochs;
        
        % Training algorithm
        train_algo;
        
        % Learning rate
        mu; mu_inc; mu_dec; mu_max;
        
        % Goal (stop training when reached)
        goal;
        
        % Minimal gradient (stop training when below)
        min_grad;
        
        % Activation functions
        activation_fn;
        
        %%% Data processing %%%
        
        % Settings for preprocessing
        preprocess;
        
        % Settings for postprocessing
        postprocess;
        
        %%% Results %%%
        
        % Struct to store results
        results;
        
    end
    
    methods
        
        function obj = FF_net(N_input, N_hidden, N_output, R_input, N_epochs, train_algo, mu, mu_inc, mu_dec, mu_max, goal, min_grad, activation_fn)
            % FF_NET Constructor method for the radial basis function
            %   neural network.
            %
            % Inputs:
            % - N_input: number of input neurons
            % - N_hidden: number of hidden neurons
            % - N_output: number of output neurons
            % - R_input: range of the input values for each input neuron,
            %   shape (N_input, 2)
            % - N_epochs: number of epochs for training
            % - train_algo: training algorithm to use
            % - mu: (adaptive) learning rate
            % - mu_inc: learning rate increase factor (for LM)
            % - mu_dec: learning rate decrease factor (for LM & backprop)
            % - mu_max: maximum learning rate (for LM)
            % - goal: goal to trigger training stop
            % - min_grad: minimal gradient to trigger training stop
            % - activation_fn: activation functions to use for hidden &
            %   output layers
            %
            % Outputs:
            % - obj: object containing the FF network
            %
            % . - 15.07.2018
            
            % Neurons
            obj.N_input = N_input;
            obj.N_hidden = N_hidden;
            obj.N_output = N_output;
            
            % Weights
            obj.N_weights = N_hidden * (N_input + N_output);
            
            % Initialize weights using normally distributed white noise
            obj.W_1 = randn(N_input, N_hidden);
            obj.W_2 = randn(N_hidden, N_output);
            
            % Set input range
            obj.R_input = R_input;
            
            % Training parameters
            obj.N_epochs = N_epochs;
            obj.train_algo = train_algo;
            obj.mu = mu;
            obj.mu_inc = mu_inc;
            obj.mu_dec = mu_dec;
            obj.mu_max = mu_max;
            obj.goal = goal;
            obj.min_grad = min_grad;
            obj.activation_fn = activation_fn;     
            
        end
        
        function obj = train_net(obj, data)
            % TRAIN_NET Trains the network using a certain method. Choose
            %   from:
            %    - backprop: regular backpropagation
            %    - lm: Levenberg-Marquardt, following implementation of
            %      MATLAB's trainlm
            %
            % Inputs:
            % - obj: object containing the FF network
            % - data: struct containing training, validation and testing
            %   data
            %
            % Outputs:
            % - obj: object containing the (trained) FF network
            %
            % . - 15.07.2018
            
            % Preprocess X_train & Y_train
            [obj, X_train_pp, Y_train_pp] = data_preprocessing(obj, data.X_train, data.Y_train);
            
            % Choose appropriate training algorithm
            switch obj.train_algo
                
                case 'backprop'
                    
                    % Initialize arrays to store MSE, averaged later to
                    %   store in object (1 column for train, val, test)
                    MSE = zeros(obj.N_epochs, 3);
                    
                    % Initialize array for cost function
                    E = zeros(obj.N_epochs, 1);
                    
                    % Early stopping
                    early_stop = 0;
                    
                    % Loop over epochs
                    for e = 1:obj.N_epochs
                        
                        % Get loss (based on MSE, no regularization)
                        Y_hat_train_pp = fw_prop(obj, X_train_pp);
                        error = get_MSE(Y_hat_train_pp, Y_train_pp);
                        
                        % Store error in cost array
                        E(e) = error;
                        
                        % Compute gradients
                        grad = compute_gradients(obj, X_train_pp, Y_train_pp);
                        
                    end
                    
                case 'lm'
                    
                    % Initialize arrays to store MSE, averaged later to
                    %   store in object (1 column for train, val, test)
                    MSE = zeros(obj.N_epochs, 3);
                    
                    % Initialize array for cost function
                    E = zeros(obj.N_epochs, 1);
                    
                    % Early stopping
                    early_stop = 0;
                    
                    % Loop over epochs
                    for e = 1:obj.N_epochs
                        
                        % Get loss (based on MSE, no regularization)
                        % When using Levenberg-Marquardt, MSE or SSE has to
                        %   be used
                        Y_hat_train_pp = fw_prop(obj, X_train_pp);
                        error = get_MSE(Y_hat_train_pp, Y_train_pp);
                        
                        % Store error in cost array
                        E(e) = error;
                        
                        % Compute Jacobian
                        [J, error_vec] = compute_jacobian(obj, X_train_pp, Y_train_pp);
                        
                        % Estimate Hessian
                        H = J' * J;
                        
                        % Flatten weights
                        W = reshape([obj.W_1' obj.W_2], 1, obj.N_weights);
                        
                        % Update the weights according to:
                        %   x_k1 = x_k - [J' * J + mu * I]^-1 * J' * error
                        W_new = W - ((H + obj.mu * eye(size(H))) \ (J' * error_vec))';
                        
                        % Store new weights in new object
                        obj_new = obj;
                        W_new_reshape = reshape(W_new, obj_new.N_hidden, obj_new.N_input + obj_new.N_output);
                        obj_new.W_1 = W_new_reshape(:, 1:obj_new.N_input)';
                        obj_new.W_2 = W_new_reshape(:, end);
                        
                        % Get error for new weights
                        Y_hat_train_pp_new = fw_prop(obj_new, X_train_pp);
                        error_new = get_MSE(Y_hat_train_pp_new, Y_train_pp);
                        
                        % If new error > old error: increase learning rate,
                        %   try again (repeat)
                        while error_new > error
                            
                            % Increase learning rate
                            obj.mu = obj.mu * obj.mu_inc;
                            
                            % Update the weights
                            W_new = W - ((H + obj.mu * eye(size(H))) \ (J' * error_vec))';
                            
                            % Store new weights in new object
                            obj_new = obj;
                            W_new_reshape = reshape(W_new, obj_new.N_hidden, obj_new.N_input + obj_new.N_output);
                            obj_new.W_1 = W_new_reshape(:, 1:obj_new.N_input)';
                            obj_new.W_2 = W_new_reshape(:, end);
                            
                            % Get error for new weights
                            Y_hat_train_pp_new = fw_prop(obj_new, X_train_pp);
                            error_new = get_MSE(Y_hat_train_pp_new, Y_train_pp);
                            
                        end
                        
                        % Save updated weights to object
                        obj.W_1 = W_new_reshape(:, 1:obj.N_input)';
                        obj.W_2 = W_new_reshape(:, end);
                        
                        % Decrease learning rate
                        obj.mu = obj.mu * obj.mu_dec;
                        
                        % Simulate for training, validation, testing
                        % No pre- or postprocessing needed, is done inside
                        %   sim_net
                        Y_hat_train = sim_net(obj, data.X_train);
                        Y_hat_val = sim_net(obj, data.X_val);
                        Y_hat_test = sim_net(obj, data.X_test);

                        % Store MSE
                        MSE(e, 1) = get_MSE(Y_hat_train, data.Y_train);
                        MSE(e, 2) = get_MSE(Y_hat_val, data.Y_val);
                        MSE(e, 3) = get_MSE(Y_hat_test, data.Y_test);
                        
                        % Determine stopping criteria for LM
                        % Taken from MATLAB's trainlm
                        [stop, early_stop, obj] = stop_criteria_lm(obj, e, E, MSE, early_stop);
                        
                        if stop
                            break
                        end
                    end              
            end
            
        end
        
        function [stop, early_stop, obj] = stop_criteria_lm(obj, e, E, MSE, early_stop)
            % STOP_CRITERIA_LM Checks if criteria for stopping are
            %   fulfilled, where criteria are based on MATLAB's trainlm.
            %
            % Inputs:
            % - obj: object containing the RBF network
            % - e: current epoch number
            % - E: cost function
            % - MSE: array with MSE for train, val, test in each column
            % - early_stop: counter for early stopping
            %
            % Outputs:
            % - stop: whether to stop or not
            % - early_stop: updated counter for early stopping
            % - obj: updated object containing the RBF network
            %
            % . - 15.07.2018
            
            % Compute gradients
            E_grad = gradient(E);
            
            % Increment early stop: if validation error has increased 5
            %   times, stop early
            if e > 1 && MSE(e, 2) > MSE(e-1, 2)
                early_stop = early_stop + 1;
            else
                early_stop = 0;
            end
            
            % Set stop
            stop = 1;
            
            % Case 1: max number of epochs reached
            if e == obj.N_epochs
                obj.results.stop_criteria = 'max epochs reached';
            % Case 2: cost goal reached
            elseif E(end) < obj.goal
                obj.results.stop_criteria = 'goal reached';
            % Case 3: gradient too low
            elseif e > 2 && abs(E_grad(end)) < obj.min_grad
                obj.results.stop_criteria = 'too low gradient';
            % Case 4: learning rate too high
            elseif obj.mu > obj.mu_max
                obj.results.stop_criteria = 'exceeded mu_max';
            % Case 5: early stopping
            elseif early_stop == 5
                obj.results.stop_criteria = 'early stop';
                e = e - 5;
            % No stop needed
            else
                stop = 0;
            end
            
            % Store results
            obj.results.MSE.train = MSE(:, 1);
            obj.results.MSE.val = MSE(:, 2);
            obj.results.MSE.test = MSE(:, 3);
            obj.results.epoch_optimal = e;
            
        end 
        
        function [v_j, R] = compute_v_j(obj, X)
            % COMPUTE_V_J Calculates the output of the input layer, defined
            %   as:
            %       
            %       v_j = sum_i(w_ij * (x_i - c_ij)^2)
            %
            % Also outputs squared distances (needed for backprop).
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            %
            % Outputs:
            % - v_j: output vector of the input layer
            % - R: squared distance matrix, 3rd dimension contains
            %   alpha, beta, V for alpha
            %
            % . - 15.07.2018            
            
            % Squared distances from data points to centers
            % 3rd dimension contains alpha, beta, V
            R(:, :, 1) = (X(:, 1) - obj.centers(:, 1)').^2;
            R(:, :, 2) = (X(:, 2) - obj.centers(:, 2)').^2;
            R(:, :, 3) = (X(:, 3) - obj.centers(:, 3)').^2;
            
            % Compute v by summing and multiplying distances with weights
            v_j = obj.W_1(1, :) .* R(:, :, 1) + ...
                  obj.W_1(2, :) .* R(:, :, 2) + ...
                  obj.W_1(3, :) .* R(:, :, 3);
            
        end
        
        function Y_hat = sim_net(obj, X)
            % SIM_NET Simulates the FF network to obtain a hypothesis.
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            % - Y: output vector, shape (N, N_out)
            %
            % Outputs:
            % - Y_hat: FF network hypothesis
            %
            % . - 15.07.2018
            
            % Preprocess state vector X
            [obj, X_pp] = data_preprocessing(obj, X);
            
            % Compute outputs of input layer
            [v_j, ~] = compute_v_j(obj, X_pp);

            % Compute outputs of FF activation function
            phi_j = exp(-v_j);
            
            % Compute outputs hidden layer: v_k = sum_j(a_jk * phi_j),
            %   where amplitude a_jk is determined by the hidden layer
            %   weights W_2
            v_k = phi_j * obj.W_2;
            
            % Output neuron is purelin: y_hat_k = v_k
            Y_hat_pp = v_k;
            
            % Postprocess the (preprocessed) FF network hypothesis
            %   Y_hat_pp
            Y_hat = data_postprocessing(obj, Y_hat_pp);
            
        end
        
        function [Y_hat, phi_j, v_j, R] = fw_prop(obj, X)
            % FW_PROP Propagates forward through the FF network. Equal to
            %   sim_net, but without the pre- and postprocessing. Also
            %   gives back all intermediate layer outputs (for backprop).
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            %
            % Outputs:
            % - Y_hat: RBF network hypothesis
            % - phi_j: output vector of the FF activation function
            % - v_j: output vector of the input layer
            % - R: squared distance matrix, 3rd dimension contains
            %   alpha, beta, V for alpha
            %
            % . - 15.07.2018
            
            % Compute outputs of input layer
            [v_j, R] = compute_v_j(obj, X);

            % Compute outputs of FF activation function
            phi_j = exp(-v_j);
            
            % Compute outputs hidden layer: v_k = sum_j(a_jk * phi_j),
            %   where amplitude a is determined by the hidden layer
            %   weights W_2
            v_k = phi_j * obj.W_2;
            
            % Output neuron is purelin: y_hat_k = v_k
            Y_hat = v_k;
            
        end
        
        function [obj, X_pp, varargout] = data_preprocessing(obj, X, varargin)
            % DATA_PREPROCESSING Preprocesses input data and optionally
            %   output data (for training, for example). For the
            %   preprocessing, normalization (map to [-1, 1]) is chosen 
            %   over standardization (zero mean, std. dev. 1), since the
            %   data contains no 'real' outliers that could skew the
            %   inliers.
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            % - varargin: variable-length list of arguments, used to pass
            %   the output vector Y, with shape (N, N_out), in case of
            %   training
            %
            % Outputs:
            % - obj: object containing the FF network
            % - X_pp: preprocessed state vector, shape (N, N_states)
            % - varargout: variable-length list of arguments, used to pass
            %   the preprocessed output vector Y_pp, with shape (N, N_out),
            %   in case of training
            %
            % . - 15.07.2018
            
            % If no settings available (no preprocessing done before)
            if isempty(obj.preprocess)
                
                % Map values to [-1, 1], save settings
                % Done in row-wise fashion, so transpose
                [X_pp, obj.preprocess] = mapminmax(X');
                
                % And transpose back
                X_pp = X_pp';
                
            % If settings are available (preprocessing already done before)    
            else
                
                % Apply settings to map to [-1, 1]
                X_pp = mapminmax('apply', X', obj.preprocess);
                
                % Transpose back
                X_pp = X_pp';
                
            end
            
            % If specified, also preprocess output vector
            if ~isempty(varargin)
                
                % Check if postprocessing settings available
                % Postprocessing because these settings will later be used
                %   to 'reverse' the hypothesis Y_hat_pp to Y_hat
                if isempty(obj.postprocess)
                    
                    % Map values to [-1, 1], save settings
                    % Done in row-wise fashion, so transpose
                    [Y_pp, obj.postprocess] = mapminmax(varargin{1}');
                    
                    % Pass Y_pp as optional output, transpose back
                    varargout{1} = Y_pp';
                    
                % If settings are available (postprocessing already done
                %   before)    
                else
                    
                    % Apply settings to map to [-1, 1]
                    Y_pp = mapminmax('apply', varargin{1}', obj.postprocess);
                    
                    % Pass Y_pp as optional output, transpose back
                    varargout{1} = Y_pp';
                    
                end           
            end         
        end
        
        function Y_hat = data_postprocessing(obj, Y_hat_pp)
            % DATA_POSTPROCESSING Postprocesses the RBF network hypothesis.
            %   In other words, the normalization carried out by the
            %   preprocessing is 'reversed'.
            %
            % Inputs:
            % - obj: object containing the RBF network
            % - Y_hat_pp: normalized/preprocessed RBF network hypothesis
            %
            % Outputs:
            % - Y_hat: 'normal'/postprocessed RBF network hypothesis
            %
            % . - 14.06.2018
            
            % As with preprocessing, mapminmax() is used
            % Done in row-wise fashion, so transpose
            Y_hat = mapminmax('reverse', Y_hat_pp', obj.postprocess);
            
            % And transpose back
            Y_hat = Y_hat';
            
        end
        
        function [J, error] = compute_gradients(obj, X, Y)
            % COMPUTE_GRADIENTS Calculates the Jacobian matrix, which
            %   contains the cost function gradient of each data point
            %   w.r.t. each weight.
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            % - Y: output vector, shape (N, N_out)
            %
            % Outputs:
            % - J: Jacobian matrix, shape (N, N_weights)
            % - error: error per sample
            %
            % . - 15.07.2018
            
            % Forward propagation
            [Y_hat, phi_j, ~, R] = fw_prop(obj, X);
            
            %%% Gradient w.r.t. hidden layer weights, w_jk %%%
            
            % Get number of points
            N = size(X, 1);
            
            % Cost function w.r.t hypothesis
%             dE_dy_hat_k = 1 / N * (Y - Y_hat) * -1;
            dE_dy_hat_k = (Y - Y_hat) * -1;
            
            % Hypothesis w.r.t. hidden layer output
            dy_hat_k_dv_k = 1;  % since purelin
            
            % Hidden layer output w.r.t hidden layer weights
            % Note that w_jk = obj.W_2 (just for notation)
            dv_k_dw_jk = phi_j;
            
            % Result
            dE_dw_jk = dE_dy_hat_k .* dy_hat_k_dv_k .* dv_k_dw_jk;
            
            %%% Gradient w.r.t. input layer weights, w_ij %%%
            
            % Hidden layer output w.r.t. hidden layer activation function
            dv_k_dphi_j = obj.W_2;
            
            % Hidden layer activation function w.r.t. input layer output
            % Derivative of exp(-v_j) = -exp(-v_j) = -phi_j
            dphi_j_dv_j = -phi_j;
            
            % Input layer output w.r.t. input layer weights
            % Note that w_ij = obj.W_1 (just for notation)
            % Derivative of v_j = sum_i((x_i - c_ij)^2) = distance.^2
            dv_j_dw_ij = R.^2;
            
            % Result
            dE_dw_ij(:, :, 1) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 1);
            dE_dw_ij(:, :, 2) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 2);
            dE_dw_ij(:, :, 3) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 3);
            
            %%% Compute Jacobian %%%
            
            J = [reshape(dE_dw_ij, N, obj.N_input*obj.N_hidden) dE_dw_jk];
            
            %%% Compute error per sample %%%
            
            % Based on cost function J = 1 / (2 * N) * sum((Y - Y_hat).^2)                        
            error = 1 / 2 * (Y - Y_hat).^2;
            
        end
        
        function [J, error] = compute_jacobian(obj, X, Y)
            % COMPUTE_JACOBIAN Calculates the Jacobian matrix, which
            %   contains the cost function gradient of each data point
            %   w.r.t. each weight.
            %
            % Inputs:
            % - obj: object containing the FF network
            % - X: state vector, shape (N, N_states)
            % - Y: output vector, shape (N, N_out)
            %
            % Outputs:
            % - J: Jacobian matrix, shape (N, N_weights)
            % - error: error per sample
            %
            % . - 15.07.2018
            
            % Forward propagation
            [Y_hat, phi_j, ~, R] = fw_prop(obj, X);
            
            %%% Gradient w.r.t. hidden layer weights, w_jk %%%
            
            % Get number of points
            N = size(X, 1);
            
            % Cost function w.r.t hypothesis
%             dE_dy_hat_k = 1 / N * (Y - Y_hat) * -1;
            dE_dy_hat_k = (Y - Y_hat) * -1;
            
            % Hypothesis w.r.t. hidden layer output
            dy_hat_k_dv_k = 1;  % since purelin
            
            % Hidden layer output w.r.t hidden layer weights
            % Note that w_jk = obj.W_2 (just for notation)
            dv_k_dw_jk = phi_j;
            
            % Result
            dE_dw_jk = dE_dy_hat_k .* dy_hat_k_dv_k .* dv_k_dw_jk;
            
            %%% Gradient w.r.t. input layer weights, w_ij %%%
            
            % Hidden layer output w.r.t. hidden layer activation function
            dv_k_dphi_j = obj.W_2;
            
            % Hidden layer activation function w.r.t. input layer output
            % Derivative of exp(-v_j) = -exp(-v_j) = -phi_j
            dphi_j_dv_j = -phi_j;
            
            % Input layer output w.r.t. input layer weights
            % Note that w_ij = obj.W_1 (just for notation)
            % Derivative of v_j = sum_i((x_i - c_ij)^2) = distance.^2
            dv_j_dw_ij = R.^2;
            
            % Result
            dE_dw_ij(:, :, 1) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 1);
            dE_dw_ij(:, :, 2) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 2);
            dE_dw_ij(:, :, 3) = dE_dy_hat_k .* dy_hat_k_dv_k * dv_k_dphi_j' .* dphi_j_dv_j .* dv_j_dw_ij(:, :, 3);
            
            %%% Compute Jacobian %%%
            
            J = [reshape(dE_dw_ij, N, obj.N_input*obj.N_hidden) dE_dw_jk];
            
            %%% Compute error per sample %%%
            
            % Based on cost function J = 1 / (2 * N) * sum((Y - Y_hat).^2)                        
            error = 1 / 2 * (Y - Y_hat).^2;
            
        end  
    end  
end
        
        