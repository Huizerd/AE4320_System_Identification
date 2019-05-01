% Plotting script for OLS mean-squared error (training, validation,
%   testing) as a function of polynomial model order.
%
% . - 07.06.2018

% Fig 1: MSE (train, val, test) as function of model order
set(0, 'DefaultFigurePosition', [150 150 720 300])
figure

order = 1:size(MSE, 1);

% subplot(3, 1, 1); hold on
hold on
plot(order, MSE(:, 1, 1), 'b')
plot(order, MSE(:, 1, 2), 'r')
plot(order, MSE(:, 1, 3), 'g')
% ylabel('MSE \big[rad$^2$\big]', 'Interpreter', 'Latex')
xlabel('Order [-]', 'Interpreter', 'Latex')
ylabel('MSE [-]', 'Interpreter', 'Latex')
title('MSE vs Model Order: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Training', 'Validation', 'Testing'}, 'Location', 'northeast')
legend('boxoff')
grid on

% subplot(3, 1, 2); hold on
% plot(order, MSE(:, 2, 1), 'b')
% plot(order, MSE(:, 2, 2), 'r')
% plot(order, MSE(:, 2, 3), 'g')
% ylabel('MSE \big[rad$^2$\big]', 'Interpreter', 'Latex')
% title('MSE vs Model Order: $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
% legend({'Training', 'Validation', 'Testing'}, 'Location', 'northeast')
% legend('boxoff')
% grid on
% 
% subplot(3, 1, 3); hold on
% plot(order, MSE(:, 3, 1), 'b')
% plot(order, MSE(:, 3, 2), 'r')
% plot(order, MSE(:, 3, 3), 'g')
% xlabel('Order [-]', 'Interpreter', 'Latex')
% ylabel('MSE \big[m$^2$/s$^2$\big]', 'Interpreter', 'Latex')
% title('MSE vs Model Order: V', 'Interpreter', 'Latex', 'FontSize', 12)
% legend({'Training', 'Validation', 'Testing'}, 'Location', 'northeast')
% legend('boxoff')
% grid on

if save_fig
    figure_name = 'figures/model_selection';
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end
