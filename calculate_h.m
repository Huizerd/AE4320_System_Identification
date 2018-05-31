function Z_n = calculate_h(t, X, U)
% CALCULATE_H Calculates observation matrix h, as defined in the system:
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
% - Z_n: measurement vector
%
% Jesse Hagenaars - 26.05.2018

% For readability
u = X(1); v = X(2); w = X(3); C = X(4);

% h is non-trivial
Z_n = [atan(w / u) * (1 + C);
       atan(v / sqrt(u^2 + w^2));
       sqrt(u^2 + v^2 + w^2)];

end
