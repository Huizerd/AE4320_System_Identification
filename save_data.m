function save_data(Z_k, C_m)
% SAVE_DATA Saves reconstructed F-16 dataset to .mat file.
%
% Inputs:
% - Z_k: (corrected) measurement vector, shape (N_meas, N)
% - C_m: moment coefficient vector, shape (1, N)
%
% Jesse Hagenaars - 08.06.2018

% Measurements as states, C_m as output
X = Z_k;
Y = C_m;

save('data/F16_reconstructed', 'X', 'Y')

end