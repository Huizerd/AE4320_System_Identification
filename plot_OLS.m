% Plotting script for OLS.
%
% Jesse Hagenaars - 06.06.2018

set(0, 'DefaultAxesTickLabelInterpreter','Latex')
set(0, 'DefaultLegendInterpreter','Latex')
set(0, 'DefaultFigurePosition', [150 150 720 800])

% Fig 1: measurement estimations by OLS
figure; hold on
plot3(Y(:, 1), Y(:, 2), Y(:, 3), 'b')
scatter3(Y_hat_train(:, 1), Y_hat_train(:, 2), Y_hat_train(:, 3), 'r')
% scatter(X_train(:, 4), Y_train(:, 3), 'b')
% scatter(X_train(:, 4), Y_train_OLS(:, 3), 'r')
% scatter(Y_train(:, 1), Y_train(:, 2), 'b')
% scatter(Y_train_OLS(:, 1), Y_train_OLS(:, 2), 'r')
xlabel('$\alpha$ [s]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
title('OLS Estimation: $\alpha$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Bias-corrected', 'OLS estimate'}, 'Location', 'northeast')
legend('boxoff')
grid on

if save_fig
    figure_name = 'figures/OLS_estimation';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end