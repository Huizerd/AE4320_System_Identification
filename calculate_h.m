function z_n = calculate_h(t, x, u)
% CALCULATE_H Calculates observation matrix h, as
% defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
% Note that t is included as argument for the function to work with
% ode45.
% Jesse Hagenaars - 26.05.2018

% h is non-trivial
z_n = [atan(u(3) / u(1)) * (1 + x(4));
       atan(u(2) / sqrt(u(1)^2 + u(3)^2));
       sqrt(u(1)^2 + u(2)^2 + u(3)^2)];

end
