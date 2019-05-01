function [Cm, Z_k, U_k] = load_F16_data(filename)
% LOAD_F16_DATA Loads F-16 data from .mat file.
%
% Inputs:
% - filename: name of data file
%
% Outputs:
% - Cm: vector containg the pitching moment coefficient
% - Z_k: measurement vector
% - U_k: input vector
%
% . - 31.05.2018

load(filename, 'Cm', 'Z_k', 'U_k')

% Transpose
Cm = Cm';
Z_k = Z_k';
U_k = U_k';

end