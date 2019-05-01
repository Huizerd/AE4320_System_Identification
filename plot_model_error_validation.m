% Plotting script for model validation.
%
% . - 07.06.2018

% Fig 1: histogram of residuals to check normal distribution with zero mean
set(0, 'DefaultFigurePosition', [150 150 720 500])

bins = [101 101 301];

figure; hold on
% subplot(3, 1, 1); hold on
histogram(epsilon(:, 1), bins(1), 'FaceColor', 'b')
xlabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
ylabel('\# Residuals [-]', 'Interpreter', 'Latex')
title('Model Residual: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
axis([-0.02 0.02 0 200])
grid on

% subplot(3, 1, 2); hold on
% histogram(epsilon(:, 2), bins(2), 'FaceColor', 'b')
% xlabel('Residual $\beta$ [rad]', 'Interpreter', 'Latex')
% ylabel('\# Residuals [-]', 'Interpreter', 'Latex')
% title('Model Residual: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
% axis([-0.004 0.004 0 1000])
% grid on
% 
% subplot(3, 1, 3); hold on
% histogram(epsilon(:, 3), bins(3), 'FaceColor', 'b')
% xlabel('Residual $V$ [m/s]', 'Interpreter', 'Latex')
% ylabel('\# Residuals [-]', 'Interpreter', 'Latex')
% title('Model Residual: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
% axis([-0.1 0.1 0 1000])
% grid on

if save_fig
    figure_name = 'figures/model_residuals';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 2: normalized autocorrelation of residuals and 95% bounds to check
%   correlation
figure; hold on

% subplot(3, 1, 1); hold on
plot(lags, epsilon_ac(:, 1), 'bo-')
plot(lags, conf_95(1) * ones(1, length(lags)), 'r--')
plot(lags, conf_95(2) * ones(1, length(lags)), 'r--')
xlabel('Lag [-]', 'Interpreter', 'Latex')
ylabel('Normalized Autocorrelation $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual Normalized Autocorrelation: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Normalized autocorrelation', '95\% confidence interval'}, 'Location', 'northeast')
legend('boxoff')
grid on

% subplot(3, 1, 2); hold on
% plot(lags, epsilon_ac(:, 2), 'b')
% plot(lags, conf_95(1) * ones(1, length(lags)), 'r--')
% plot(lags, conf_95(2) * ones(1, length(lags)), 'r--')
% ylabel('Normalized Autocorrelation [-]', 'Interpreter', 'Latex')
% title('Model Residual Normalized Autocorrelation: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
% legend({'Normalized autocorrelation', '95\% confidence interval'}, 'Location', 'northeast')
% legend('boxoff')
% grid on
% 
% subplot(3, 1, 3); hold on
% plot(lags, epsilon_ac(:, 3), 'b')
% plot(lags, conf_95(1) * ones(1, length(lags)), 'r--')
% plot(lags, conf_95(2) * ones(1, length(lags)), 'r--')
% xlabel('Lag [-]', 'Interpreter', 'Latex')
% ylabel('Normalized Autocorrelation [-]', 'Interpreter', 'Latex')
% title('Model Residual Normalized Autocorrelation: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
% legend({'Normalized autocorrelation', '95\% confidence interval'}, 'Location', [0.76 0.257 0 0])
% legend('boxoff')
% grid on

if save_fig
    figure_name = 'figures/autocorrelation_residuals';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 3: residuals vs fitted values to check constant variance
set(0, 'DefaultFigurePosition', [150 150 900 800])

figure;
subplot(2, 2, 1); hold on
scatter(X_test(:, 1), epsilon(:, 1), 'b')
xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 2); hold on
scatter(X_test(:, 2), epsilon(:, 1), 'b')
xlabel('$\beta$ [rad]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 3); hold on
scatter(X_test(:, 3), epsilon(:, 1), 'b')
xlabel('$V$ [m/s]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $V$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 4); hold on
scatter(Y_test(:, 1), epsilon(:, 1), 'b')
xlabel('$C_m$ [-]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

if save_fig
    figure_name = 'figures/residual_variance';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

