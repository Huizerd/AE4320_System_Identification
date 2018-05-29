% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Jesse Hagenaars - 11.05.2018

clear; clc
rng('default')

%% Load data

data_name = 'data/F16traindata_CMabV_2018';

% [Cm, alpha_m, beta_m, V_m, u_dot, v_dot, w_dot] = load_F16_data(data_name);
[Cm, Z_k, U_k] = load_F16_data(data_name);

%% Description of state transition and observation equation

% State equation: x_dot(t) = f(x(t), u(t), t)
% with state vector: x(t) = [u v w C_alpha_up]'
% and input vector: u(t) = [u_dot v_dot w_dot]'

% Observation equation: z_n(t) = h(x(t), u(t), t)
% with observation vector: z_n(t) = [alpha_m, beta_m, V_m]'
% combined with white noise: z(t_k) = z_n(t_k) + v(t_k)


%% Observability check --> proves convergence of KF

% Causes crash! --> update first
% check_observability


%% IEKF
% IEKF because it works very well if main non-linearities are in the
% observation equation --> this is the case here (slide 20, lec. state est.)

%%% Set simulation parameters %%%

dt = 0.01;
N = size(U_k, 2);
epsilon = 1e-10;
iter_max = 100;  % max iterations for IEKF
doIEKF = 1;

%%% Set initial values for states and noise statistics %%%

fprintf('\nV_m(1): %.2f m/s\nalpha_m(1): %.4f rad\nbeta_m(1): %.4f rad\n', ...
        Z_k(3, 1), Z_k(1, 1), Z_k(2, 1))  % provide intuition about initial states
x_0 = [150; 0; 0; 1];  % initial state
Ex_0 = [100; 10; 10; 1];  % initial estimate, x_hat(0|0) == E{x_0} (or x_hat_0_0)

N_states = length(x_0);
N_input = 3;  % u_dot, v_dot, w_dot

% Initial estimate for the state prediction covariance matrix
sigma_x_0 = [10 1 1 1];
P_0_0      = diag(sigma_x_0.^2);  % different from definition on slide 34/38?

% System noise statistics
Ew = [0 0 0 0];  % noise bias, E{w(t)}, given in assignment
sigma_w = [1e-3 1e-3 1e-3 0];  % noise variance, [u v w C_m_alpha_up], given in assignment
Q = diag(sigma_w.^2);  % system noise covariance matrix
w_k = diag(sigma_w) * randn(N_states, N)  + diag(Ew) * ones(N_states, N);  % discretized matrix

% Measurement noise statistics:
Ev = [0 0 0];  % noise bias, E{v(t)}, given in assignment
sigma_v = [0.01 0.0058 0.112]; % noise variance, [alpha_m, beta_m, V_m], given in assignment
R = diag(sigma_v.^2);  % measurement noise covariance matrix
N_obs = length(sigma_v);  % number of measurements/sensors
v_k = diag(sigma_v) * randn(N_obs, N)  + diag(Ev) * ones(N_obs, N);  % discretized matrix

G = eye(N_states);  % system noise input matrix, N_states x N_states (= N_systemnoise)
B = [eye(N_input); 0 0 0];  % input matrix, N_states x N_input

%%% Calculate batch with measurement data %%%
% Not necessary? --> since we already have input & measurement

% Real simulated state-variable and measurements data
% x = x_0;
% X_k = zeros(N_states, N);
% Z_k = zeros(N_obs, N);
% U_k = zeros(N_input, N);

% for i = 1:N
%     
%     % Calculate state equation
%     dx = calculate_f(0, x, U_k(:, i));
%     x = x + (dx + w_k(:, i)) * dt;
%     X_k(:, i) = x;
%     
%     % Calculate observation equation
%     Z_k(:, i) = calculate_h(0, x, U_k(:,i)) + v_k(:,i);
%
% end

%%% Define arrays to store results, set initial conditions %%%

X_HAT_K1_K1 = zeros(N_states, N);
P_K1_K1 = zeros(N_states, N);
SIGMA_X_COR = zeros(N_states, N);  % --> look into estimation error and X_k above!
Z_K1_K = zeros(N_obs, N);
IEKF_COUNT = zeros(N, 1);  % IEKF iterations for each data point

% Can also be seen as x_hat(k|k) and P(k|k)
x_hat_k1_k1 = Ex_0;  % x_hat(0|0) == E{x_0} (or x_hat_0_0)
P_k1_k1 = P_0_0;  % P(0|0) == E{(x_hat_0_0 - x_0) * (x_hat_0_0 - x_0)'}

%%% Run the IEKF %%%

% Timespan for ode45
ti = 0; 
tf = dt;

% Run the filter through all N samples
for k = 1:N
    
    % Predict x_hat(k+1|k) --> linear, so no non-linear prediction needed?
    % Method 1 - C.C. de Visser
    % [t, x_hat_k1_k] = rk4(@calculate_f, x_hat_k1_k1, U_k(:, k), [ti tf]);
    % Method 2 - use dummy fn. to allow extra arguments
    [t, x_hat_k1_k] = ode45(@(t, x) calculate_f(t, x, U_k(:, k)), [ti tf], x_hat_k1_k1);
    t = t(end); x_hat_k1_k = x_hat_k1_k(end, :)';  % select only last
    
    % Predict z(k+1|k)
    z_k1_k = calculate_h(0, x_hat_k1_k1, U_k(:, k));
    Z_K1_K(:, k) = z_k1_k;

    % Calculate the Jacobian of f(x(t), u(t), t)
    % Ain't the Jacobian 0? Since x_dot is fully determined by input -->
    % independent of x (and for Jacobian we take derivative w.r.t. x)
    % Is this really perturbation? --------v
    Fx = calculate_Fx(0, x_hat_k1_k, U_k(:,k));  % perturbation of f(x(t), u(t), t)
    
    % Calculate Phi(k+1|k) and Gamma(k+1|k) (discretized state transition &
    % input matrix
    % [dummy, Psi] = c2d(Fx, B, dt); --> not used
    [Phi, Gamma] = c2d(Fx, G, dt);   
    
    % Calculate P(k+1|k) (prediction covariance matrix)
    P_k1_k = Phi * P_k1_k1 * Phi' + Gamma * Q * Gamma'; 
    % P_pred = diag(P_k1_k); --> not used
    % stdx_pred = sqrt(diag(P_k1_k)); --> not used

    % Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if doIEKF

        % Set iteration parameters & initial conditions
        eta_i1 = x_hat_k1_k;  % initialize with state estimation x_hat(k+1|k)
        iter_error = 2 * epsilon;
        iter_N = 0;
        
        while iter_error > epsilon
            
            if iter_N >= iter_max
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', iter_max);
                break
            end
            
            iter_N = iter_N + 1;
            eta_i = eta_i1;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx = calculate_Hx(0, eta_i, U_k(:,k));  % perturbation of h(x(t), u(t), t)
            
            % -- up to here --
            
            % Check observability of state
            if (k == 1 && iter_N == 1)
                rankHF = kf_calcObsRank(Hx, Fx);
                if (rankHF < n)
                    warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
                end
            end
            
            % The innovation matrix
            Ve  = (Hx*P_k1_k*Hx' + R);

            % calculate the Kalman gain matrix
            K       = P_k1_k * Hx' / Ve;
            % new observation state
            z_p     = kf_calc_h(0, eta_i, U_k(:,k)) ;%fpr_calcYm(eta1, u);

            eta_i1    = x_kk_1 + K * (Z_k(:,k) - z_p - Hx*(x_kk_1 - eta_i));
            iter_error     = norm((eta_i1 - eta_i), inf) / norm(eta_i, inf);
        end

        IEKF_COUNT(k)    = iter_N;
        x_k_1k_1          = eta_i1;

    else
        % Correction
        Hx = kf_calc_Hx(0, x_kk_1, U_k(:,k)); % perturbation of h(x,u,t)
        % Pz(k+1|k) (covariance matrix of innovation)
        Ve = (Hx*P_k1_k * Hx' + R); 

        % K(k+1) (gain)
        K = P_k1_k * Hx' / Ve;
        % Calculate optimal state x(k+1|k+1) 
        x_k_1k_1 = x_kk_1 + K * (Z_k(:,k) - z_kk_1); 

    end    
    
    P_k_1k_1 = (eye(n) - K*Hx) * P_k1_k * (eye(n) - K*Hx)' + K*R*K';  
    P_cor = diag(P_k_1k_1);
    stdx_cor = sqrt(diag(P_k_1k_1));

    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    XX_k1k1(:,k) = x_k_1k_1;
%     PP_k1k1(k,:) = P_k_1k_1;
    STDx_cor(:,k) = stdx_cor;
end

time2 = toc;

% calculate state estimation error (in real life this is unknown!)
EstErr = (XX_k1k1-X_k);

fprintf('IEKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr)), N, time2);





