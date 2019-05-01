function plot_hypothesis(X, Y, Y_hat, save_fig, h_type)
% PLOT_HYPOTHESIS Plots the hypothesis as a surface. Makes use of 2 of the
%   3 states: alpha & beta.
%
% Inputs:
% - X: state vector, shape (N, N_states)
% - Y: output vector, shape (N, N_out)
% - Y_hat: hypothesis vector, shape (N, N_out)
% - save_fig: switch whether or not to save figures
% - h_type: string containing the type of hypothesis (for figure title)
%
% . - 08.06.2018

% Creating triangulation
TRIeval = delaunayn(X(:, 1:2), {'Qbb','Qc','QJ1e-6'});

% Viewing angles
az = 140;
el = 36;

% Create figure
set(0, 'DefaultAxesTickLabelInterpreter','Latex')
set(0, 'DefaultLegendInterpreter','Latex')
set(0, 'DefaultFigurePosition', [150 150 720 800])
figure

% Plot true data
plot3(X(:, 1), X(:, 2), Y, '.k');
grid on; hold on
view(az, el);

% Plot hypothesis surface
trisurf(TRIeval, X(:,1), X(:,2), Y_hat, 'EdgeColor', 'none'); 

% Configure plot
xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
ylabel('$\beta$ [rad]', 'Interpreter', 'Latex')
zlabel('$C_m [-]$', 'Interpreter', 'Latex')
legend({'True', 'Hypothesis'}, 'Location', 'northeast')
legend('boxoff')
xlim([-.17 .78])
ylim([-.52 .52])
title([h_type ' Hypothesis: $C_m$'], 'Interpreter', 'Latex', 'FontSize', 12)

% Set fancy options for plotting 
set(gcf,'Renderer','OpenGL');
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
drawnow();

if save_fig
    figure_name = ['figures/' h_type '_hypothesis'];
    set(gcf, 'Renderer', 'Painters')
    savefig([figure_name '.fig'])
    print('-painters', '-depsc', [figure_name '.eps'])
end

end