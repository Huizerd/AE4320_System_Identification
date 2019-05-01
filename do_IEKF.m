function [X_HAT_K1_K1, Z_K1_K, IEKF_COUNT] = do_IEKF(U_k, Z_k, dt, sigma_w, sigma_v)
% DO_IEKF Estimates states using an iterated extended Kalman filter (IEKF).
%
% Inputs:
% - U_k: input vector
% - Z_k: measurement vector
% - dt: used sample time
% - sigma_w: system noise std. dev.
% - sigma_v: measurement noise std. dev.
%
% Outputs:
% - X_HAT_K1_K1: one-step-ahead optimal state estimation vector
% - Z_K1_K: one-step-ahead measurement prediction vector
% - IEKF_COUNT: vector containing number of IEKF iterations per sample
%
% . - 31.05.2018

%%% Set simulation parameters %%%

N = size(U_k, 2);
epsilon = 1e-10;
iter_max = 100;  % max iterations for IEKF

%%% Set initial values for states and noise statistics %%%

Ex_0 = [Z_k(3, 1); 0.5; 0.5; 0.5];  % initial estimate, x_hat(0|0) == E{x_0} (or x_hat_0_0)

N_states = length(Ex_0);
N_input = 3;  % u_dot, v_dot, w_dot

% Initial estimate for the state prediction covariance matrix
P_0_0 = eye(N_states) * 0.1;

% System noise statistics
Q = diag(sigma_w.^2);  % system noise covariance matrix

% Measurement noise statistics:
R = diag(sigma_v.^2);  % measurement noise covariance matrix
N_meas = length(sigma_v);  % number of measurements/sensors

G = eye(N_states);  % system noise input matrix, N_states x N_states

%%% Define arrays to store results, set initial conditions %%%

X_HAT_K1_K1 = zeros(N_states, N);
P_K1_K1 = zeros(N_states, N);
SIGMA_X_COR = zeros(N_states, N);  % --> look into estimation error and X_k above!
Z_K1_K = zeros(N_meas, N);
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
    
    % Predict x_hat(k+1|k)
    % Use dummy fn. to allow extra arguments
    [t, x_hat_k1_k] = ode45(@(t, x) calculate_f(t, x, U_k(:, k)), [ti tf], x_hat_k1_k1);
    t = t(end); x_hat_k1_k = x_hat_k1_k(end, :)';  % select only last
    
    % Predict z(k+1|k)
    z_k1_k = calculate_h(0, x_hat_k1_k1, U_k(:, k));
    Z_K1_K(:, k) = z_k1_k;

    % Calculate the Jacobian of f(x(t), u(t), t) --> trivial, since x_dot
    %   is fully determined by input, so independent of x
    Fx = calculate_Fx(0, x_hat_k1_k, U_k(:,k));  % perturbation of f(x(t), u(t), t), see slide 81
    
    % Calculate Phi(k+1|k) and Gamma(k+1|k) (discretized state transition &
    %   input matrix
    [Phi, Gamma] = c2d(Fx, G, dt);   
    
    % Calculate P(k+1|k) (prediction covariance matrix)
    P_k1_k = Phi * P_k1_k1 * Phi' + Gamma * Q * Gamma'; 

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

        % Construct the Jacobian Hx = d/dx(h(x(t), u(t), t)) with h(x) the
        %   observation matrix 
        Hx = calculate_Hx(0, eta_i, U_k(:,k));  % perturbation of h(x(t), u(t), t)

        % Calculate the covariance matrix of innovation, P_z(k+1|k)
        %   --> part of Kalman gain recalculation, see slide 110
        P_z  = (Hx * P_k1_k * Hx' + R);  % R is constant, since constant noise

        % Calculate the Kalman gain matrix
        K_k1 = P_k1_k * Hx' / P_z;

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

    % Calculate covariance matrix of state estimation error, IEKF version
    %   (slide 112)
    P_k1_k1 = (eye(N_states) - K_k1 * Hx) * P_k1_k * (eye(N_states) - K_k1 * Hx)' + K_k1 * R * K_k1';
    
    % Calculate std. dev. of correction
    sigma_x_cor = sqrt(diag(P_k1_k1));

    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % Store results
    X_HAT_K1_K1(:,k) = x_hat_k1_k1;
    P_K1_K1(:,k) = diag(P_k1_k1);  % store only diagonal
    SIGMA_X_COR(:,k) = sigma_x_cor;
    
end

end