%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the Extended Kalman Filter for nonlinear systems
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;


% Use these commands to initialize the randomizer to a fixed (reproducable) state.
% rng('default'); % init randomizer (default, fixed)-> version 2014a,b
% RandStream.setDefaultStream(RandStream('mt19937ar','seed', 300));-> version 2013a,b

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 1;
dt = 0.01;
N = 1000;

printfigs = 1;
figpath = '';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ex_0    = 10; % initial estimate of optimal value of x_k_1k_1
x_0     = 5; % initial state
u       = 0; % system input

B = [0]; % input matrix
G = [1]; % noise input matrix

% Initial estimate for covariance matrix
stdx_0  = 10;
P_0     = stdx_0^2;

% System noise statistics:
Ew = 0; % bias
stdw = 1; % noise variance
Q = stdw^2;
w_k = stdw * randn(N, n) + Ew;

% Measurement noise statistics:
Ev = 0; % bias
stdv = 1; % noise variance
R = stdv^2;
v_k = stdv * randn(N, n) + Ev;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate batch with measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

% Real simulated state-variable and measurements data:
x = x_0;
X_k = zeros(N, 1);
Z_k = zeros(N, 1);
U_k = zeros(N, 1);
for i = 1:N
    dx = kf_calc_lin_F(0, x, U_k(i));%subs(x);
    x = x + (dx + w_k(i,:)) * dt;
    
    X_k(i,:) = x; % store state
    Z_k(i,:) = kf_calc_lin_H(0, x, U_k(i)) + v_k(i,:); % calculate measurement 
end

XX_k1k1 = zeros(N, n);
PP_k1k1 = zeros(N, n);
STDx_cor = zeros(N, n);
z_pred = zeros(N, n);

x_k_1k_1 = Ex_0; % x(0|0)=E{x_0}
P_k_1k_1 = P_0; % P(0|0)=P(0)

time1 = toc;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Linear Kalman Filter (KF)
ti = 0; 
tf = dt;
n = 1;%length(x_k); % n: state dimension

tic;

% Calculate discrete state transition matrix and prediction (Time and State invariant!)
F = kf_calc_lin_F(0, 1, u);
[dummy, Psi] = c2d(F, B, dt);   
[Phi, Gamma] = c2d(F, G, dt);   
% Calculate discrete state observation matrix (Time invariant!)
H = kf_calc_lin_H(0, 1, 0);

% Run the filter through all N samples
for k = 1:N

    % x(k+1|k) (prediction)
    x_kk_1 = Phi*x_k_1k_1 + Psi*U_k(k);

    % P(k+1|k) (prediction covariance matrix)
    P_kk_1 = Phi*P_k_1k_1*Phi' + Gamma*Q*Gamma'; 
    P_pred = diag(P_kk_1);
    stdx_pred = sqrt(diag(P_kk_1));

    % K(k+1) (gain)
    K = P_kk_1 * H' / (H*P_kk_1*H' + R);
    % Calculate optimal state x(k+1|k+1) 
    x_k_1k_1 = x_kk_1 + K * (Z_k(k) - H*x_kk_1); 

    % P(k|k) (correction)
    P_k_1k_1 = (eye(n) - K*H) * P_kk_1;
    P_cor = diag(P_k_1k_1);
    stdx_cor = sqrt(diag(P_k_1k_1));
    
    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    XX_k1k1(k) = x_k_1k_1;
    PP_k1k1(k) = P_k_1k_1;
    STDx_cor(k) = stdx_cor;
end

time2 = toc;

% calculate state estimation error (in real life this is unknown!)
EstErr = XX_k1k1-X_k;

fprintf('KF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr)), N, time2);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [1 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(Z_k, 'k');
title('True state (blue) and Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateMeasurement');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1001;
figure(plotID);
set(plotID, 'Position', [1 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
%plot(z_pred, 'r');
title('True state (blue) and Estimated state (red)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateEstimates');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1002;
figure(plotID);
set(plotID, 'Position', [500 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
plot(Z_k, 'k');
title('True state (blue), Estimated state (red), Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2001;
figure(plotID);
set(plotID, 'Position', [800 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
title('State estimation error');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 2002;
figure(plotID);
set(plotID, 'Position', [800 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
axis([0 50 min(EstErr) max(EstErr)]);
title('State estimation error (Zoomed in)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2003;
figure(plotID);
set(plotID, 'Position', [1000 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
title('State estimation error with STD of Innovation');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2004;
figure(plotID);
set(plotID, 'Position', [1000 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
axis([0 50 min(EstErr) max(EstErr)]);
title('State estimation error');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end