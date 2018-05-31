% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Parameter estimation
%
% Jesse Hagenaars - 11.05.2018

clear; clc
rng('default')


%% Load data

data_name = 'data/F16traindata_CMabV_2018';

% [Cm, alpha_m, beta_m, V_m, u_dot, v_dot, w_dot] = load_F16_data(data_name);
[Cm, Z_k, U_k] = load_F16_data(data_name);