% Plotting script for parameter estimation
%
% Jesse Hagenaars - 01.06.2018

% Figures: influence of estimator, influence of order, estimator
% covariances, model residuals, autocorrelation of residuals

% Fig 1: influence of various estimators
figure; hold on
plot(Z_k1_k_corr(1, :), Z_k1_k_corr(2, :), 'g')
plot(Y_OLS(:, 1), Y_OLS(:, 2), 'b')
plot(Y_val_OLS(:, 1), Y_val_OLS(:, 2), 'r')
xlabel('$\alpha$ [s]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Influence of Various Estimators: $\alpha$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Bias-corrected', 'OLS: identification', 'OLS: validation'}, 'Location', 'northeast')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/estimator_influence';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 2a: influence of polynomial model order: alpha vs beta
figure; hold on
plot(Z_k1_k_corr(1, :), Z_k1_k_corr(2, :), 'g')

% Plot styles & legend entries for looping
patterns = {'-', '-.', '--', ':'};  % solid, dash-dot, dashed, dotted
legend_entries = {'Bias-corrected'};

% Plot orders
for i = 1:length(orders)
    plot(Y_ORD_OLS(:, 1, i), Y_ORD_OLS(:, 2, i), ['b' patterns{i}])
    plot(Y_VAL_ORD_OLS(:, 1, i), Y_VAL_ORD_OLS(:, 2, i), ['r' patterns{i}])
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation'];
end

xlabel('$\alpha$ [s]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Influence of Model Order: $\alpha$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'northeast')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/order_influence_avsb';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 2b: influence of polynomial model order: model residuals
figure
subplot(3, 1, 1); hold on

% Legend entries for looping
legend_entries = {};

for i = 1:length(orders)
    plot(t_id, EPSILON_ORD_OLS(:, 1, i), ['b' patterns{i}])
    plot(t_val, EPSILON_VAL_ORD_OLS(:, 1, i), ['r' patterns{i}])
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation'];
end
    
ylabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
title('Influence of Model Order on Residual: $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

subplot(3, 1, 2); hold on

for i = 1:length(orders)
    plot(t_id, EPSILON_ORD_OLS(:, 2, i), ['b' patterns{i}])
    plot(t_val, EPSILON_VAL_ORD_OLS(:, 2, i), ['r' patterns{i}])
end

ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Influence of Model Order on Residual: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

subplot(3, 1, 3); hold on

for i = 1:length(orders)
    plot(t_id, EPSILON_ORD_OLS(:, 3, i), ['b' patterns{i}])
    plot(t_val, EPSILON_VAL_ORD_OLS(:, 3, i), ['r' patterns{i}])
end

xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('$V$ [rad]', 'Interpreter', 'Latex')
title('Influence of Model Order on Residual: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/order_influence_residual';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

% Fig 3: estimator covariances --> table
disp(sigma2_OLS')

% Fig 4: autocorrelation of model residuals
figure
subplot(3, 1, 1); hold on
plot(lags, ones(size(lags)) * conf_95, 'k:')
plot(lags, ones(size(lags)) * -conf_95, 'k:')

% Legend entries for looping
legend_entries = {'95\% confidence interval', '95\% confidence interval'};

for i = 1:length(orders)
    plot(lags, AC_EPSILON_ORD_OLS(:, 1, i), ['b' patterns{i}])
    plot(lags, AC_EPSILON_VAL_ORD_OLS(:, 1, i), ['r' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_ORD_OLS{2, i}(1), ['c' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_VAL_ORD_OLS{2, i}(1), ['m' patterns{i}])
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification mean: ' num2str(MEAN_EPSILON_ORD_OLS{2, i}(1)) ' rad'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation mean: ' num2str(MEAN_EPSILON_VAL_ORD_OLS{2, i}(1)) ' rad'];
end
    
ylabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
title('Model Residual Autocorrelation: $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

subplot(3, 1, 2); hold on
plot(lags, ones(size(lags)) * conf_95, 'k:')
plot(lags, ones(size(lags)) * -conf_95, 'k:')

% Reset
legend_entries = {'95\% confidence interval', '95\% confidence interval'};

for i = 1:length(orders)
    plot(lags, AC_EPSILON_ORD_OLS(:, 2, i), ['b' patterns{i}])
    plot(lags, AC_EPSILON_VAL_ORD_OLS(:, 2, i), ['r' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_ORD_OLS{2, i}(2), ['c' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_VAL_ORD_OLS{2, i}(2), ['m' patterns{i}])
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification mean: ' num2str(MEAN_EPSILON_ORD_OLS{2, i}(2)) ' rad'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation mean: ' num2str(MEAN_EPSILON_VAL_ORD_OLS{2, i}(2)) ' rad'];
end

ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('Model Residual Autocorrelation: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

subplot(3, 1, 3); hold on
plot(lags, ones(size(lags)) * conf_95, 'k:')
plot(lags, ones(size(lags)) * -conf_95, 'k:')

% Reset
legend_entries = {'95\% confidence interval', '95\% confidence interval'};

for i = 1:length(orders)
    plot(lags, AC_EPSILON_ORD_OLS(:, 3, i), ['b' patterns{i}])
    plot(lags, AC_EPSILON_VAL_ORD_OLS(:, 3, i), ['r' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_ORD_OLS{2, i}(3), ['c' patterns{i}])
    plot(lags, ones(size(lags)) * MEAN_EPSILON_VAL_ORD_OLS{2, i}(3), ['m' patterns{i}])
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': identification mean: ' num2str(MEAN_EPSILON_ORD_OLS{2, i}(3)) ' rad'];
    legend_entries{size(legend_entries, 2) + 1} = ['Order ' num2str(orders(i)) ': validation mean: ' num2str(MEAN_EPSILON_VAL_ORD_OLS{2, i}(3)) ' rad'];
end

xlabel('$t$ [s]', 'Interpreter', 'Latex')
ylabel('$V$ [rad]', 'Interpreter', 'Latex')
title('Model Residual Autocorrelation: $V$', 'Interpreter', 'Latex', 'FontSize', 12)
legend(legend_entries, 'Location', 'southwest')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/residual_autocorrelation';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

