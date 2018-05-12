% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Jesse Hagenaars - 11.05.2018

clear; clc
rng('default')

%% Load data

data_name = 'data/F16traindata_CMabV_2018';

[Cm, alpha_m, beta_m, V_m, u_dot, v_dot, w_dot] = load_F16_data(data_name);


%% Description of state transition and observation equation

% State transition equation: x_dot(t) = f(x(t), u(t), t)
% with state vector: x(t) = [u v w C_alpha_up]'
% and input vector: u(t) = [u_dot v_dot w_dot]'

% Observation equation: z_n(t) = h(x(t), u(t), t)
% with observation vector: z_n(t) = [alpha_m, beta_m, V_m]'
% combined with white noise: z(t_k) = z_n(t_k) + v(t_k)


%% Observability check --> proves convergence of KF

check_observability


%% IEKF

%%% Set simulation parameters %%%

dt             = 0.01;
N              = 1000;
epsilon        = 1e-10;
max_iterations = 100;  % for IEKF

%%% Set initial values for states and noise statistics %%%

fprintf('\nV_m(1): %.2f m/s\nalpha_m(1): %.4f rad\nbeta_m(1): %.4f rad\n', ...
        V_m(1), alpha_m(1), beta_m(1))  % provide intuition about initial states
x_0      = [150; 0; 0; 1];    % initial state
x_hat_00 = [100; 10; 10; 1];  % initial estimate, x_hat(k|k) at k = 0, E{x_0} = x_hat_00

N_states = length(x_0);
N_input = 3;  % u_dot, v_dot, w_dot

% Initial estimate for the state prediction covariance matrix
sigma_x_0 = [10 1 1 1];
P_00      = diag(sigma_x_0.^2);  % different from definition on slide 34/38?

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

% Real simulated state-variable and measurements data
x = x_0;
X_k = zeros(n, N);
Z_k = zeros(nm, N);
U_k = zeros(m, N);
for i = 1:N
    dx = kf_calc_f(0, x, U_k(:,i));%subs(x);
    x = x + (dx + w_k(:,i)) * dt;
    
    X_k(:,i) = x; % store state
    Z_k(:,i) = kf_calc_h(0, x, U_k(:,i)) + v_k(:,i); % calculate measurement 
end

XX_k1k1 = zeros(n, N);
PP_k1k1 = zeros(n, N);
STDx_cor = zeros(n, N);
z_pred = zeros(nm, N);
IEKFitcount = zeros(N, 1);

x_k_1k_1 = Ex_0; % x(0|0)=E{x_0}
P_k_1k_1 = P_0; % P(0|0)=P(0)







