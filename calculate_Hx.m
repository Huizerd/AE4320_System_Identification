function Hx = calculate_Hx(t, X, U)
% CALCULATE_Hx Calculates the Jacobian of h, as defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t and U are included as arguments for the function to work
% with ode45.
%
% Inputs:
% - t: time
% - X: state vector
% - U: input vector
%
% Outputs:
% - Hx: Jacobian of h
%
% Jesse Hagenaars - 27.05.2018

% For readability
u = X(1); v = X(2); w = X(3); C = X(4);

Hx = [-w * (1 + C) / (u^2 + w^2) 0 u * (1 + C) / (u^2 + w^2) atan(w / u);
      -v * u / (sqrt(u^2 + w^2) * (u^2 + v^2 + w^2)) sqrt(u^2 + w^2) / (u^2 + v^2 + w^2) -v * w / (sqrt(u^2 + w^2) * (u^2 + v^2 + w^2)) 0;
      u / sqrt(u^2 + v^2 + w^2) v / sqrt(u^2 + v^2 + w^2) w / sqrt(u^2 + v^2 + w^2) 0];

end
