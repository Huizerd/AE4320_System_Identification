function X_dot = calculate_f(t, X, U)
% CALCULATE_F Calculates the state (transition) matrix f, as defined in
% the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t and X are included as arguments for the function to work
% with ode45.
%
% Inputs:
% - t: time
% - X: state vector
% - U: input vector
%
% Outputs:
% - X_dot: state derivative vector
%
% Jesse Hagenaars - 26.05.2018

% f is trivial: fully defined by system input
X_dot = [U; 0];

end
