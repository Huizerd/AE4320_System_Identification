function Fx = calculate_Fx(t, x, u)
% CALCULATE_Fx Calculates the Jacobian of f, as defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t, x and u are included as arguments for the function to work
% with ode45.
% Jesse Hagenaars - 27.05.2018

% Fx = [1 0 0 0;
%       0 1 0 0;
%       0 0 1 0;
%       0 0 0 0];

% Because d(x_dot)/dx = 0 (partial derivative)
Fx = zeros(4);

end
