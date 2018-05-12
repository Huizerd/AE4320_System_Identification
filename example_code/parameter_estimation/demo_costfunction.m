
close all;

printfigs = 1;
figpath = '';


% data locations series
x = (-1.5:.01:1.5)';

% regressor matrix (model structure!)
X = [x.^2 x ones(size(x))];


% Objective function parameters
A = 0.8;
B = 0;
C = -0.5;
Func = X * [A; B; C];

% Initial (very bad) estimate of Func
A0 = -0.8;
B0 = 0;
C0 = 0.5;
InitialModel = X * [A0; B0; C0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial model quality assessement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% linear error
err1 = Func-InitialModel;
% quadratic error
err2 = (Func-InitialModel).^2;
% absolute error
err3 = abs(Func-InitialModel);

% linear cost function
costfn1 = sum(err1);
% quadratic cost function
costfn2 = sum(err2);


% running linear cost function
costfn1run = cumsum(err1);
% running quadratic cost function
costfn2run = cumsum(err2);
% running abs cost function
costfn3run = cumsum(err3);



fprintf('Initial model linear cost function value: %2.2f\n', costfn1);
fprintf('Initial model quadratic cost function value: %2.2f\n', costfn2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Linear cost function optimization
params1 = fminsearch(@(c)sum((Func - X*c)), [A0; B0; C0]); % quadratic cost function
% Evaluate Least Squares Model
Model1 = X*params1;

% Quadratic cost function optimization
params2 = fminsearch(@(c)sum((Func - X*c).^2), [A0; B0; C0]); % quadratic cost function
% params2 = fminsearch(@(c)sum(sign(Func - X*c)), [A0; B0; C0]); % signum cost function
% params2 = fminsearch(@(c)sum(abs(Func - X*c)), [A0; B0; C0]); % absolute value cost function

% Evaluate Least Squares Model
Model2 = X*params2;

% print results
fprintf('Expected results for objective function: A = %2.2f, B = %2.2f, C = %2.2f\n', A, B, C);
fprintf('Linear cost function optimization results: A = %d, B = %d, C = %d\n', params1(1), params1(2), params1(3));
fprintf('Quadratic cost function optimization results: A = %2.2f, B = %2.2f, C = %2.2f\n', params2(1), params2(2), params2(3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;

plotID = 1001;
figure(plotID);
set(plotID, 'Position', [50 200 1200 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
subplot(1,3,1);
hold on;
grid on;
plot(x, Func, 'linewidth', 2);
plot(x, InitialModel, 'r', 'linewidth', 2);
xlabel('x');
ylabel('f(x)');
title('Function and initial model');
legend('Function', 'Initial Model', 'Location', 'northeast');
subplot(1,3,2);
hold on;
grid on;
plot(x, err1, 'r', 'linewidth', 2);
plot(x, err2, 'k', 'linewidth', 2);
plot(x, err3, 'g');
xlabel('x');
ylabel('\epsilon');
title('Cost function values');
legend('Linear error', 'Quadratic error', 'Absolute error', 'Location', 'northeast');
subplot(1,3,3);
hold on;
grid on;
plot(x, costfn1run, 'r', 'linewidth', 2);
plot(x, costfn2run, 'k', 'linewidth', 2);
plot(x, costfn3run, 'g');
xlabel('x');
ylabel('\Sigma_i^N \epsilon_i');
title('Cumulative cost function value');
legend('Cumulative linear error', 'Cumulative quadratic error', 'Cumulative absolute error', 'Location', 'northwest');
if (printfigs == 1)
    fpath = sprintf('fig_democostfn');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

% plotID = 1002;
% figure(plotID);
% set(plotID, 'Position', [50 600 800 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% subplot(1,2,1);
% hold on;
% grid on;
% plot(x, Func, 'linewidth', 3);
% plot(x, Model1, 'r', 'linewidth', 1);
% xlabel('x');
% ylabel('f(x)');
% title('Function and Model');
% legend('Function', 'Model (lin. cost fn.)', 'Location', 'northeast');
% subplot(1,2,2);
% hold on;
% grid on;
% plot(x, Func, 'linewidth', 3);
% plot(x, Model2, 'r', 'linewidth', 1);
% xlabel('x');
% ylabel('f(x)');
% title('Function and Model');
% legend('Function', 'Model (quad. cost fn.)', 'Location', 'northeast');
% if (printfigs == 1)
%     fpath = sprintf('fig_democostfnAbs');
%     savefname = strcat(figpath, fpath);
%     print(plotID, '-dpng', '-r300', savefname);
% end

% 
% plotID = 2001;
% figure(plotID);
% set(plotID, 'Position', [800 200 800 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% subplot(1,2,1);
% hold on;
% grid on;
% plot(x, Func, 'linewidth', 3);
% plot(x, Model1, 'r', 'linewidth', 1);
% xlabel('x');
% ylabel('f(x)');
% title('Function and Model');
% legend('Function', 'Model (lin. cost fn.)', 'Location', 'northeast');
% subplot(1,2,2);
% hold on;
% grid on;
% plot(x, Func, 'linewidth', 3);
% plot(x, Model2, 'r', 'linewidth', 1);
% xlabel('x');
% ylabel('f(x)');
% title('Function and Model');
% legend('Function', 'Model (quad. cost fn.)', 'Location', 'northeast');
% if (printfigs == 1)
%     fpath = sprintf('fig_democostfnAbs');
%     savefname = strcat(figpath, fpath);
%     print(plotID, '-dpng', '-r300', savefname);
% end


