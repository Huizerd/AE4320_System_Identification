function Hx = calculate_Hx(t, x, u)
% CALCULATE_Fx Calculates the Jacobian of h, as defined in the system:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% Note that t and x are included as arguments for the function to work
% with ode45.
% Jesse Hagenaars - 27.05.2018

% For readability
u = x(1); v = x(2); w = x(3); C = x(4);

Hx = [-w * (1 + C) / (u^2 + w^2) 0 u * (1 + C) / (u^2 + w^2) atan(w / u);
      -v * u / (sqrt(u^2 + w^2) * (u^2 + v^2 + w^2)) sqrt(u^2 + w^2) / (u^2 + v^2 + w^2) -v * w / (sqrt(u^2 + w^2) * (u^2 + v^2 + w^2)) 0;
      u / sqrt(u^2 + v^2 + w^2) v / sqrt(u^2 + v^2 + w^2) w / sqrt(u^2 + v^2 + w^2) 0];

end
