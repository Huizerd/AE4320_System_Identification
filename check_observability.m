function check_observability
% CHECK_OBSERVABILITY Checks the observability of a non-linear system,
%   defined as:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Jesse Hagenaars - 11.05.2018

% Define symbols
syms('u', 'v', 'w','C_alpha_up', 'u_dot', 'v_dot', 'w_dot')

% State vector: x(t) = [u v w C_alpha_up]'
x  = [u; v; w; C_alpha_up];
x_0 = [100; 10; 10; 1];  % some random values

N_states = length(x);

% Input vector: u(t) = [u_dot v_dot w_dot]' not declared due to conflict
%   with u (velocity in x)

% State transition equation: x_dot(t) = f(x(t), u(t), t) = [u(t) 0]'
f = [u_dot; v_dot; w_dot; 0];

% Measurement equation: z_n(t) = h(x(t), u(t), t) derived from definitions
%   of measured/true angles and velocities
h = [atan(w/u) * (1 + C_alpha_up);
     atan(v/sqrt(u^2 + w^2));
     sqrt(u^2 + v^2 + w^2)];

% Compute rank of the observability matrix
rank = calculate_rank_nonlin(x, x_0, f, h);

if rank >= N_states
    fprintf('\nObservability matrix is of full rank: the system is observable!\n');
else
    fprintf('\nObservability matrix is NOT of full rank: the system is NOT observable!\n');
end

end