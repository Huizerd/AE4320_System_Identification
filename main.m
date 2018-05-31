% Assignment AE4320 System Identification of Aerospace Vehicles
% Neural Networks
%
% Jesse Hagenaars - 11.05.2018

clear; close all; clc;
rng('default')


%% Load data

data_name = 'data/F16traindata_CMabV_2018';

[Cm, Z_k, U_k] = load_F16_data(data_name);

% Time vector, samples at dt = 0.01 s
t = 0:0.01:0.01*(size(Z_k, 2)-1);


%% Description of state transition and measurement equation

% State equation: x_dot(t) = f(x(t), u(t), t)
% with state vector: x(t) = [u v w C_alpha_up]'
% and input vector: u(t) = [u_dot v_dot w_dot]'

% Measurement equation: z_n(t) = h(x(t), u(t), t)
% with measurement vector: z_n(t) = [alpha_m, beta_m, V_m]'
% combined with white noise: z(t_k) = z_n(t_k) + v(t_k)


%% Observability check --> proves convergence of KF

check_observability


%% IEKF

% IEKF because it works very well if main non-linearities are in the
% measurement equation --> this is the case here (slide 20)

[X_HAT_K1_K1, Z_K1_K, IEKF_COUNT] = do_IEKF(U_k, Z_k);

%% Plotting IEKF

% Figures: measurements over time, bias over time, IEKF iterations,
% alpha vs beta

set(0, 'DefaultAxesTickLabelInterpreter','latex')
set(0, 'DefaultLegendInterpreter','latex')
set(0, 'DefaultFigurePosition', [150 150 720 800])

% Fig 1: measurements
% Raw measurements: red, after KF: blue, true alpha: green
subplot(3, 1, 1); hold on
plot(t, Z_k(1, :), 'r')
plot(t, Z_K1_K(1, :), 'b')
plot(t, Z_K1_K(1, :) ./ (1 + X_HAT_K1_K1(4, :)), 'g')
ylabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered', 'Bias-corrected'}, 'Location', 'northwest')
legend('boxoff')
grid on

subplot(3, 1, 2); hold on
plot(t, Z_k(2, :), 'r')
plot(t, Z_K1_K(2, :), 'b')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered'}, 'Location', 'northwest')
legend('boxoff')
grid on

subplot(3, 1, 3); hold on
plot(t, Z_k(3, :), 'r')
plot(t, Z_K1_K(3, :), 'b')
xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('$V$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered'}, 'Location', 'northwest')
legend('boxoff')
grid on

figure_name = 'figures/measurements_separate';
set(gcf, 'Renderer', 'Painters')
savefig([figure_name '.fig'])
print('-painters', '-depsc', [figure_name '.eps'])

set(0, 'DefaultFigurePosition', [150 150 720 300])

% Fig 2: IEKF iterations --> x-axis = time?
figure
plot(t, IEKF_COUNT, 'b')
xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('N [-]', 'Interpreter', 'Latex')
title('IEKF Iterations', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

figure_name = 'figures/IEKF_count';
set(gcf, 'Renderer', 'Painters')
savefig([figure_name '.fig'])
print('-painters', '-depsc', [figure_name '.eps'])

set(0, 'DefaultFigurePosition', [150 150 720 800])

figure; hold on
plot(Z_k(1, :), Z_k(2, :), 'r')
plot(Z_K1_K(1, :), Z_k(2, :), 'b')
plot(Z_K1_K(1, :) ./ (1 + X_HAT_K1_K1(4, :)), Z_K1_K(2, :), 'g')
xlabel('$\alpha$ [s]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\alpha$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered', 'Bias-corrected'}, 'Location', 'northeast')
legend('boxoff')
grid on

figure_name = 'figures/a_vs_b';
set(gcf, 'Renderer', 'Painters')
savefig([figure_name '.fig'])
print('-painters', '-depsc', [figure_name '.eps'])
