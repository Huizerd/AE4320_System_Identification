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
% x_0 = [150; 0; 0; 1];  % initial state
Ex_0 = [100; 10; 10; 1];  % initial estimate, x_hat(0|0) == E{x_0} (or x_hat_0_0)

N_states = length(Ex_0);
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
% B = [eye(N_input); 0 0 0];  % input matrix, N_states x N_input --> not used

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

tic;

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
    % Is this really perturbation? --------v --> yes, see slide 81
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

            % Construct the Jacobian Hx = d/dx(h(x(t), u(t), t)) with h(x) the observation matrix 
            Hx = calculate_Hx(0, eta_i, U_k(:,k));  % perturbation of h(x(t), u(t), t)
            
            % Check observability of state --> check this part, might be
            % wrong
            if k == 1 && iter_N == 1
                
                % Why HF? Also, check inputs & function
                rankHF = calculate_rank(Fx, Hx);
                
                if rankHF < N_states
                    warning('The current state is not observable --> rank of observability matrix is %d, should be %d', rankHF, N_states);
                end
                
            end
            
            % The innovation matrix (or covariance matrix of innovation, Pz(k+1|k)?)
            % --> part of Kalman gain recalculation, see slide 110
            Ve  = (Hx * P_k1_k * Hx' + R);  % shouldn't R be updated? R = R_k1 on slide --> no, constant noise

            % Calculate the Kalman gain matrix
            K_k1 = P_k1_k * Hx' / Ve;  % is division the same as inverse? --> yes, and better
            
            % New observation state
            z_p = calculate_h(0, eta_i, U_k(:,k));
            
            % Iterative version of the measurement update equation
            eta_i1 = x_hat_k1_k + K_k1 * (Z_k(:,k) - z_p - Hx * (x_hat_k1_k - eta_i));
            
            % Compute error to check stop criteria
            iter_error = norm((eta_i1 - eta_i), inf) / norm(eta_i, inf);
            
        end
        
        % Update count and state estimate
        IEKF_COUNT(k) = iter_N;
        x_hat_k1_k1 = eta_i1;
        
        % Calculate covariance matrix of state estimation error, IEKF
        % (slide 112)
        P_k1_k1 = (eye(N_states) - K_k1 * Hx) * P_k1_k * (eye(N_states) - K_k1 * Hx)' + K_k1 * R * K_k1';
    
    % EKF
    else
        
        % Correction
        Hx = calculate_Hx(0, x_hat_k1_k, U_k(:,k));  % perturbation of h(x,u,t)
        
        % Pz(k+1|k) (covariance matrix of innovation) --> check name
        Ve = (Hx * P_k1_k * Hx' + R); 

        % K(k+1) Kalman gain matrix
        K_k1 = P_k1_k * Hx' / Ve;
        
        % Calculate optimal state x_hat(k+1|k+1)    
        x_hat_k1_k1 = x_hat_k1_k + K_k1 * (Z_k(:,k) - z_k1_k);  % in slides: z(k+1) - h(x_hat(k+1|k), u(k+1)) --> different? no, correct (see z_k1_k above)
        
        % Calculate covariance matrix of state estimation error, EKF
        % (slide 89)
        P_k1_k1 = (eye(N_states) - K_k1 * Hx) * P_k1_k;
        
    end  
    
    P_cor = diag(P_k1_k1);
    sigma_x_cor = sqrt(diag(P_k1_k1));

    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    X_HAT_K1_K1(:,k) = x_hat_k1_k1;
%     P_K1_K1(k,:) = P_k1_k1;
    SIGMA_X_COR(:,k) = sigma_x_cor;
    
end

time_end = toc;

% Calculate state estimation error (in real life this is unknown!) --> and
% in our case?
% estimation_error = (X_HAT_K1_K1-X_k);
estimation_error = 0;

fprintf('IEKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(estimation_error)), N, time_end);


%% Plotting

figure
plot(U_k')

figure
plot(X_HAT_K1_K1')
 
figure
plot(Z_K1_K')

figure
plot(Z_k(1, :)')
hold on
plot(Z_k(1, 100:end)' ./ (1 + X_HAT_K1_K1(4, 100:end)'))
% plot(atan(X_HAT_K1_K1(3, :)' ./ X_HAT_K1_K1(1, :)') .* (1 + X_HAT_K1_K1(4, :)'))

figure
plot(IEKF_COUNT)

