function x_dot = calculate_f(t, x, u)
% CALCULATE_F Calculates the state (transition) matrix f, as
% defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t and x are included as arguments for the function to work with
% ode45.
% Jesse Hagenaars - 26.05.2018

% f is trivial: fully defined by system input
x_dot = [u; 0];

end
