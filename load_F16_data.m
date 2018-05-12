function [Cm, alpha_m, beta_m, V_m, u_dot, v_dot, w_dot] = load_F16_data(filename)

load(filename, 'Cm', 'Z_k', 'U_k')

% Measurements Z_k = Z(t) + v(t)
alpha_m = Z_k(:,1);  % measured angle of attack
beta_m  = Z_k(:,2);  % measured angle of sideslip
V_m     = Z_k(:,3);  % measured velocity

% Input to Kalman filter
u_dot = U_k(:,1);  % perfect accelerometer du/dt data
v_dot = U_k(:,2);  % perfect accelerometer dv/dt data
w_dot = U_k(:,3);  % perfect accelerometer dw/dt data

end