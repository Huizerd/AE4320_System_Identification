function rank_obs = calculate_rank_nonlin(x, x_0, f, h)
% CALCULATE_RANK Calculates the rank of the observability matrix of a
%   non-linear system, defined as:
%
%       x_dot(t) = f(x(t), u(t), t)
%       z_n(t) = h(x(t), u(t), t)
%
% . - 11.05.2018

N_states = length(x);
N_obs    = length(h);

% Compute Jacobian of the observation equation
H_x = simplify(jacobian(h, x));

% Initialize observability matrix
obs_matrix = zeros(N_obs*N_states, N_states);

% Convert to symbolic matrix, rational mode
obs = sym(obs_matrix, 'r');

% Substitute Jabobian, since H_x = d_x*h
obs(1:N_obs, :) = H_x;

% Substitute x0 for x
obs_num = subs(obs, x, x_0);

% Evaluate rank of the observability matrix
rank_obs = double(rank(obs_num));

if rank_obs >= N_states
    return
end

% Multiply with f to get 1st Lie derivative
LfH_x = simplify(H_x * f);

% Fill the observability matrix
for i = 2:N_states
    
    % Compute next order Jacobian and fill in
    LfH_x = jacobian(LfH_x, x);
    obs((i-1)*N_obs+1:i*N_obs, :) = LfH_x;
    
    % Substitute x0 for x and evaluate rank again
    obs_num = subs(obs, x, x_0);
    rank_obs = double(rank(obs_num));
    
    if rank_obs >= N_states
        return
    end
    
    % Next Lie derivative
    LfH_x = (LfH_x * f);
    
end

end
