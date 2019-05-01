% Plotting script for statistical validation.
%
% . - 07.06.2018

% Fig 1: OLS estimator parameter values and variance
set(0, 'DefaultFigurePosition', [150 150 900 400])

figure;
subplot(1, 2, 1); hold on
bar(OLSE_opt.theta_hat)
xlabel('Index [-]', 'Interpreter', 'Latex')
ylabel('$\hat{\theta}_i$ [-]', 'Interpreter', 'Latex')
title('Parameter Value of OLS Estimator', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(1, 2, 2); hold on
plot(1:size(sigma2_theta_hat, 1), sigma2_theta_hat, 'b')
xlabel('Index [-]', 'Interpreter', 'Latex')
ylabel('$\sigma^2_{\hat{\theta}_i}$ [-]', 'Interpreter', 'Latex')
title('Parameter Variance of OLS Estimator', 'Interpreter', 'Latex', 'FontSize', 12)
grid on