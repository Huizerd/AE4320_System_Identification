function X = create_poly_model(x, order)
% CREATE_POLY_MODEL Creates a regression matrix X for a polynomial model of
%   a certain order, such that p(x, theta) = X * theta, where all states
%   are represented in each order of the polynomial.
%
% Inputs:
% - x: state vector, shape (t, N_states)
% - order: order of the desired polynomial model
%
% Outputs:
% - X: regression matrix
%
% Jesse Hagenaars - 31.05.2018

X = ones(size(x, 1), 1);

for o = 1:order
    
    % Concatenate next order
    X = [X x.^o];
    
end
 
end