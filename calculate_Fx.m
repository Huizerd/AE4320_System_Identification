function Fx = calculate_Fx(t, X, U)
% CALCULATE_Fx Calculates the Jacobian of f, as defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t, X and U are included as arguments for the function to work
% with ode45.
%
% Inputs:
% - t: time
% - X: state vector
% - U: input vector
%
% Outputs:
% - Fx: Jacobian of f
%
% Jesse Hagenaars - 27.05.2018

% Because d(x_dot)/dx = 0 (partial derivative)
Fx = zeros(4);

end
