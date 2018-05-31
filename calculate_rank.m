function rank_obs = calculate_rank(Fx, Hx)
% CALCULATE_RANK Calculates the rank of the observability matrix of a
% discretized linear system, defined as:
%
%       x_dot(t) = Fx * x(t) + G * u(t)
%       z_n(t) = Hx * x(t) + D * u(t)
%
% Inputs are the Jacobians of f (Fx) and h (Hx). --> Or not Jacobians?
%
% Jesse Hagenaars - 31.05.2018

N_states = size(Fx, 1);

% Why?
F = eye(size(Fx));

% Observability matrix
R = [];

for i = 1:N_states-1
   R = [R; Hx * F];
   % Why?
   F = F * Fx;
end

% Why?
R = [R; Hx * F];

rank_obs = rank(R);

end
