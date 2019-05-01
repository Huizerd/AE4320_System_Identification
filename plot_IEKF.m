% Plotting script for IEKF.
%
% . - 01.06.2018

% Fig 1: measurements
% Raw measurements: red, after KF: blue, true alpha: green

set(0, 'DefaultAxesTickLabelInterpreter','Latex')
set(0, 'DefaultLegendInterpreter','Latex')
set(0, 'DefaultFigurePosition', [150 150 720 800])

subplot(3, 1, 1); hold on
plot(t, Z_k(1, :), 'r')
plot(t, Z_k1_k_biased(1, :), 'b')
plot(t, Z_k1_k(1, :), 'g')
ylabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered', 'Bias-corrected'}, 'Location', 'northwest')
legend('boxoff')
grid on

subplot(3, 1, 2); hold on
plot(t, Z_k(2, :), 'r')
plot(t, Z_k1_k(2, :), 'b')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered'}, 'Location', 'northwest')
legend('boxoff')
grid on

subplot(3, 1, 3); hold on
plot(t, Z_k(3, :), 'r')
plot(t, Z_k1_k(3, :), 'b')
xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('$V$ [m/s]', 'Interpreter', 'Latex')
title('Measurements: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered'}, 'Location', 'northwest')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/measurements_separate';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 2: IEKF iterations --> x-axis = time?
set(0, 'DefaultFigurePosition', [150 150 720 300])

figure
plot(t, IEKF_count, 'b')
xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('N [-]', 'Interpreter', 'Latex')
title('IEKF Iterations', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

if save_fig
    figure_name = 'figures/IEKF_count';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end
    
% Fig 3: upwash over time
figure
plot(t, X_hat_k1_k1(4, :), 'b')
xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('$C_{\alpha_{up}}$', 'Interpreter', 'Latex')
title('States: $C_{\alpha_{up}}$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

if save_fig
    figure_name = 'figures/upwash';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end
    
% Fig 4: alpha vs beta
set(0, 'DefaultFigurePosition', [150 150 720 800])

figure; hold on
plot(Z_k(1, :), Z_k(2, :), 'r')
plot(Z_k1_k_biased(1, :), Z_k1_k_biased(2, :), 'b')
plot(Z_k1_k(1, :), Z_k1_k(2, :), 'g')
xlabel('$\alpha$ [s]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Measurements: $\alpha$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Raw measurement', 'Kalman-filtered', 'Bias-corrected'}, 'Location', 'northeast')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/a_vs_b';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end